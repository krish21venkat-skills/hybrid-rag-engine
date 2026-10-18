from hybridrag.partial import has_multi_part_signal, split_question


def test_single_question_skips_llm():
    def boom(_: str) -> str:
        raise AssertionError("LLM should not be called")

    assert split_question("What is the notice period?", boom) == [
        "What is the notice period?"
    ]


def test_multi_part_is_split():
    fake = lambda _: '["What is the notice period?", "What is the penalty?"]'
    q = "What is the notice period and what is the penalty?"
    assert split_question(q, fake) == [
        "What is the notice period?",
        "What is the penalty?",
    ]


def test_bad_llm_output_falls_back():
    q = "What is the notice period and the penalty?"
    assert split_question(q, lambda _: "not json") == [q]


def test_signal_detection():
    assert has_multi_part_signal("A? B?")
    assert not has_multi_part_signal("What is the fee")