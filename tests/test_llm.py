import pytest

from hybridrag import llm


class Flaky:
    def __init__(self, fails):
        self.fails, self.n = fails, 0

    def chat(self, **kw):
        self.n += 1
        if self.n <= self.fails:
            raise ConnectionError("refused")
        return {"message": {"content": "ok"}}


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    c = Flaky(fails=2)
    monkeypatch.setattr(llm, "_client", c)
    assert llm.llm_chat(model="m", messages=[])["message"]["content"] == "ok"
    assert c.n == 3


def test_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    c = Flaky(fails=99)
    monkeypatch.setattr(llm, "_client", c)
    with pytest.raises(llm.LLMUnavailable):
        llm.llm_chat(model="m", messages=[])
    assert c.n == 3