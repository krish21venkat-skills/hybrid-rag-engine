from __future__ import annotations

import json
import logging
import re
import time
import uuid
from .logging_utils import stage

from dataclasses import replace
from typing import TypedDict
from .llm import LLMUnavailable, llm_chat

import ollama
from langgraph.graph import END, START, StateGraph

from .config import Settings, load_settings
from .pipeline import RAGPipeline

MAX_Q = 500
TRACE_FIELDS = {"retrieve": ("top1",), "judge": ("said", "quote_ok"), "fit": ("fit_blocked",),
                "rewrite": ("tries",), "abstain": ("decision",)}
log = logging.getLogger(__name__)
NO_ANSWER = "I couldn't find this in the documents."

PROMPT = """You answer a question using only the context.
Reply with JSON only: {{"answer_found": true or false, "quote": "<one sentence copied from the context, or empty>", "answer": "<short answer, or empty>"}}
Rules:
- answer_found is true if the context contains a rule, number or statement that directly answers the question, even when the question words it differently. Example: a question about a "huge delay" can be answered by a rule about "delays longer than 2 hours".
- answer_found is false if the context only covers the same topic, or names the subject, but does not give the specific thing asked. Example: a context that says a feature exists does not answer a question about its price.
- The quote must be the single sentence that supports your answer, copied word for word from the context.
- The answer uses only facts from the context.

Context:
{context}

Question: {question}
JSON:"""

REWRITE = ("Rewrite the question as a short English search query using the vocabulary a company policy "
           "or technical document would use (formal terms, no slang). If the question is not in English, "
           "translate it. Return only the query.\n\nQuestion: ")

EXTRACT = """Answer the question using only the sentence, in 8 words or fewer. If the sentence does not give the specific thing the question asks for, reply NONE.

Question: {question}
Sentence: {quote}
Answer:"""

NOT_FOUND = re.compile(
    r"\b(?:not|no)\b[^.]{0,60}?\b(?:mention|specif|stated|provided|available|information)\w*", re.I
)

TIME = "second|minute|hour|day|week|month|year"
UNITS = ["days", "weeks", "months", "years", "hours", "minutes", "characters",
         "connections", "requests", "orders", "times", "attempts", "items"]
FREQ = re.compile(r"\b(every|each|daily|weekly|monthly|yearly|annual\w*|twice|once|per|hourly|times|"
                  r"a (?:year|month|week|day))\b", re.I)


def supported(quote: str, context: str) -> bool:
    """Quote must be (nearly) made of words that appear in the context."""
    qt = re.findall(r"\w+", quote.lower())
    if len(qt) < 3:
        return False
    ct = set(re.findall(r"\w+", context.lower()))
    return sum(t in ct for t in qt) / len(qt) >= 0.9


def unit_ok(question: str, text: str) -> bool:
    """No-LLM check: does the answer use the unit or kind of fact the question asks for?"""
    q, t = question.lower(), text.lower()
    if re.search(r"\bhow many\b", q):
        found = [(q.find(u), u) for u in UNITS if re.search(rf"\b{u}\b", q)]
        if found and not re.search(rf"\b{min(found)[1][:-1]}s?\b", t):
            return False
    if re.search(r"\bhow (long|soon|quickly|early)\b", q) and not re.search(rf"\b({TIME})s?\b", t):
        return False
    if re.search(r"\bhow often\b", q) and not FREQ.search(t):
        return False
    return True

STOP = set("a an the of to in on for is are was were be do does did what who when where which how "
           "i my me we our you your it its and or if can will with by at as from this that "
           "get much many long often".split())

def closest_sentence(question: str, hits) -> str:
    """Context sentence sharing the most content words with the question (no LLM). '' if weak."""
    qw = {w for w in re.findall(r"\w+", question.lower()) if w not in STOP}
    best, best_n = "", 1                      # need at least 2 shared words
    for h in hits:
        for s in re.split(r"(?<=[.!?])\s+|\n+", h.text):
            s = " ".join(s.split())
            if len(s) < 20:
                continue
            n = len(qw & set(re.findall(r"\w+", s.lower())))
            if n > best_n:
                best, best_n = s, n
    return best

class State(TypedDict):
    question: str
    query: str
    hits: list
    top1: float
    tries: int
    said: bool
    quote_ok: bool
    quote: str
    answer: str
    decision: str          # "answer" | "abstain"
    fit_blocked: bool
    related: str
    partial: str
    trace_id: str

class GraphRAG:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or load_settings()
        # gate off inside the pipeline: scoring decisions are made in the graph
        self.rag = RAGPipeline(replace(self.s, min_rerank_score=-1.0))
        self.graph = self._build()

    # ---- nodes ----
    def _retrieve(self, st: State):
        hits = self.rag.retrieve(st["query"], self.s.top_k)
        top1 = hits[0].score if hits else 0.0
        log.info("retrieve try=%d top1=%.4f query=%r", st["tries"], top1, st["query"])
        return {"hits": hits, "top1": top1}

    def _judge(self, st: State):
        context = "\n\n".join(h.text for h in st["hits"])
        resp = llm_chat(
            model=self.s.llm_model, format="json",
            options={"temperature": 0, "num_ctx": 4096, "num_predict": 200},
            messages=[{"role": "user",
                       "content": PROMPT.format(context=context, question=st["question"])}],
        )
        try:
            out = json.loads(resp["message"]["content"])
        except json.JSONDecodeError:
            out = {}
        said = bool(out.get("answer_found"))
        quote = str(out.get("quote", ""))
        answer = str(out.get("answer", "")).strip()
        quote_ok = supported(quote, context)
        ok = said and quote_ok and bool(answer) and not NOT_FOUND.search(answer)
        log.info("judge answer_found=%s quote_ok=%s", said, quote_ok)
        related = "" if ok else closest_sentence(st["question"], st["hits"])
        return {"said": said, "quote_ok": quote_ok, "quote": quote,
                "answer": answer if ok else "", "decision": "answer" if ok else "",
                "related": related}

    def _fit(self, st: State):
        """Does the quoted sentence give the specific thing asked? Extraction prompt plus unit check."""
        r = llm_chat(model=self.s.llm_model, options={"temperature": 0, "num_predict": 24},
                        messages=[{"role": "user",
                                        "content": EXTRACT.format(question=st["question"], quote=st["quote"])}])
        direct = not r["message"]["content"].strip().upper().startswith("NONE")
        unit = unit_ok(st["question"], st["answer"] + " " + st["quote"])
        log.info("fit direct=%s unit=%s", direct, unit)
        if direct and unit:
            return {"fit_blocked": False}
        return {"answer": "", "decision": "", "fit_blocked": True}

    def _rewrite(self, st: State):
        msg = llm_chat(model=self.s.llm_model, options={"temperature": 0, "num_predict": 40},
                          messages=[{"role": "user", "content": REWRITE + st["question"]}])
        q = msg["message"]["content"].strip().strip('"') or st["question"]
        log.info("rewrite -> %r", q)
        return {"query": q, "tries": st["tries"] + 1}

    def _abstain(self, st: State):
        rel = st.get("related", "")
        partial = (f'I couldn\'t find a direct answer. The closest passage is: "{rel}" '
                   f"(it may be about something different, so check the source)." if rel else "")
        return {"answer": NO_ANSWER, "decision": "abstain", "partial": partial}

    # ---- routing ----
    def _route_gate(self, st: State):
        if st["top1"] >= self.s.abstain_below:
            return "judge"
        return "rewrite" if st["tries"] < self.s.max_retries else "abstain"   # low score: rewrite (if enabled), else abstain

    def _route_judge(self, st: State):
        return "fit" if st["decision"] == "answer" else "abstain"

    def _route_fit(self, st: State):
        return END if st["decision"] == "answer" else "abstain"

    def _traced(self, name, fn):
        def run(st):
            t0 = time.perf_counter()
            out = fn(st)
            fields = {k: out[k] for k in TRACE_FIELDS.get(name, ()) if k in out}
            stage(log, st["trace_id"], name, ms=(time.perf_counter() - t0) * 1000, **fields)
            return out
        return run

    def _build(self):
        g = StateGraph(State)
        g.add_node("retrieve", self._traced("retrieve", self._retrieve))
        g.add_node("judge", self._traced("judge", self._judge))
        g.add_node("fit", self._traced("fit", self._fit))
        g.add_node("rewrite", self._traced("rewrite", self._rewrite))
        g.add_node("abstain", self._traced("abstain", self._abstain))
        g.add_edge(START, "retrieve")
        g.add_conditional_edges("retrieve", self._route_gate, ["judge", "rewrite", "abstain"])
        g.add_conditional_edges("judge", self._route_judge, ["fit", "abstain"])
        g.add_conditional_edges("fit", self._route_fit, [END, "abstain"])
        g.add_edge("rewrite", "retrieve")
        g.add_edge("abstain", END)
        return g.compile()

    def ask(self, question: str) -> State:
        question = " ".join((question or "").split())
        state = {
            "question": question, "query": question, "hits": [], "top1": 0.0, "tries": 0,
            "said": False, "quote_ok": False, "quote": "", "answer": "", "decision": "",
            "fit_blocked": False, "related": "", "partial": "", "trace_id": uuid.uuid4().hex[:8],
        }
        if not question:
            return {**state, "decision": "invalid", "answer": "Please enter a question."}
        if len(question) > MAX_Q:
            return {**state, "decision": "invalid",
                    "answer": f"Question is too long (max {MAX_Q} characters)."}
        tid, t0 = state["trace_id"], time.perf_counter()
        stage(log, tid, "start", q=repr(question[:80]))
        try:
            res = self.graph.invoke(state)
        except LLMUnavailable:
            log.error("trace=%s LLM unavailable after retries", tid)
            res = {**state, "decision": "error",
                   "answer": "The language model is not responding. Check that Ollama is running and try again."}
        stage(log, tid, "done", decision=res["decision"], top1=f"{res['top1']:.3f}",
              total_ms=(time.perf_counter() - t0) * 1000)
        return res


if __name__ == "__main__":
    import sys
    from .logging_utils import setup_logging
    setup_logging("INFO")
    res = GraphRAG().ask(" ".join(sys.argv[1:]))
    print("\n" + res["answer"],
          f"\n(decision={res['decision']} fit_blocked={res['fit_blocked']} tries={res['tries']} top1={res['top1']:.3f})")