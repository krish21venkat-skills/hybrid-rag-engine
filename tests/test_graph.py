from types import SimpleNamespace

import pytest

from hybridrag import graph as G
from hybridrag.graph import GraphRAG, NO_ANSWER, closest_sentence, supported, unit_ok

CONTEXT = "Employees get 10 days of annual leave per year. Contractors do not receive paid leave."
QUOTE = "Employees get 10 days of annual leave per year."


def hit(text=CONTEXT, score=0.5):
    return SimpleNamespace(text=text, score=score)


class FakeRag:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, k=5):
        return self.hits


def make_graph(hits, abstain_below=0.001, max_retries=0):
    g = GraphRAG.__new__(GraphRAG)          # skip __init__: no models
    g.s = SimpleNamespace(abstain_below=abstain_below, max_retries=max_retries,
                          top_k=5, llm_model="fake")
    g.rag = FakeRag(hits)
    g.graph = g._build()
    return g


def fake_chat(monkeypatch, judge=None, fit="8 days"):
    """Judge call has format='json'; every other call is the fit check."""
    calls = []

    def chat(model=None, messages=None, **kw):
        calls.append("judge" if kw.get("format") == "json" else "fit")
        if kw.get("format") == "json":
            import json
            return {"message": {"content": json.dumps(judge or {})}}
        return {"message": {"content": fit}}

    monkeypatch.setattr(G.ollama, "chat", chat)
    return calls


# ---- pure helpers ----
def test_supported():
    assert supported(QUOTE, CONTEXT)
    assert not supported("Contractors get unlimited holidays abroad", CONTEXT)
    assert not supported("too short", CONTEXT)


def test_unit_ok():
    assert unit_ok("How many days of leave?", "10 days per year")
    assert not unit_ok("How many days of leave?", "ten per year")
    assert unit_ok("How often is it reviewed?", "reviewed every quarter")
    assert not unit_ok("How often is it reviewed?", "reviewed by the manager")


def test_closest_sentence_needs_two_shared_words():
    assert "10 days" in closest_sentence("How many annual leave days do employees get?", [hit()])
    assert closest_sentence("What is the VPN port?", [hit()]) == ""


# ---- routing ----
def test_gate_abstains_without_calling_llm(monkeypatch):
    calls = fake_chat(monkeypatch)
    st = make_graph([hit(score=0.0005)]).ask("anything")
    assert st["decision"] == "abstain" and st["answer"] == NO_ANSWER
    assert st["partial"] == "" and calls == []


def test_answer_path(monkeypatch):
    calls = fake_chat(monkeypatch, judge={"answer_found": True, "quote": QUOTE, "answer": "10 days"})
    st = make_graph([hit()]).ask("How many days of annual leave do employees get?")
    assert st["decision"] == "answer" and st["answer"] == "10 days"
    assert calls == ["judge", "fit"]


def test_not_found_guard(monkeypatch):
    fake_chat(monkeypatch, judge={"answer_found": True, "quote": QUOTE,
                                  "answer": "The documents do not mention this."})
    st = make_graph([hit()]).ask("How many days of annual leave do employees get?")
    assert st["decision"] == "abstain"


def test_unsupported_quote_abstains(monkeypatch):
    fake_chat(monkeypatch, judge={"answer_found": True, "quote": "Everyone gets unlimited holidays abroad",
                                  "answer": "unlimited"})
    st = make_graph([hit()]).ask("How many days of annual leave do employees get?")
    assert st["decision"] == "abstain"


def test_fit_blocks_and_stays_plain(monkeypatch):
    fake_chat(monkeypatch, judge={"answer_found": True, "quote": QUOTE, "answer": "10 days"}, fit="NONE")
    st = make_graph([hit()]).ask("How many days of annual leave do employees get?")
    assert st["decision"] == "abstain" and st["fit_blocked"] is True
    assert st["partial"] == ""


def test_judge_refusal_gives_partial(monkeypatch):
    fake_chat(monkeypatch, judge={"answer_found": False})
    st = make_graph([hit()]).ask("Can annual leave days be cashed out by employees?")
    assert st["decision"] == "abstain" and st["answer"] == NO_ANSWER
    assert QUOTE in st["partial"]


def test_bad_json_from_judge_abstains(monkeypatch):
    def chat(model=None, messages=None, **kw):
        return {"message": {"content": "not json"}}
    monkeypatch.setattr(G.ollama, "chat", chat)
    st = make_graph([hit()]).ask("How many days of annual leave do employees get?")
    assert st["decision"] == "abstain"