"""Print the partial message for a few refused questions."""
from hybridrag.graph import GraphRAG

QUESTIONS = [
    "do i get paid if i take a sick day as contractor",
    "Can I carry forward my sick leave to next year?",
    "Can unused sick leave be cashed out?",
    "What is the IT Security desk's email address?",
    "What is the penalty for submitting an expense claim late?",
    "What is the name of the engagement manager for contractors?",
]

g = GraphRAG()
for q in QUESTIONS:
    st = g.ask(q)
    print(f"\nQ: {q}\n decision={st['decision']}")
    print(" partial:", st.get("partial") or "(plain refusal)")
    if st["decision"] == "answer":
        print(" ANSWERED:", st["answer"])