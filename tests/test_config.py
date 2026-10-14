from hybridrag.config import load_settings


def test_env_overrides_yaml(tmp_path, monkeypatch):
    f = tmp_path / "s.yaml"
    f.write_text("chunk_size: 500\ntop_k: 3\n", encoding="utf-8")
    monkeypatch.setenv("RAG_TOP_K", "7")
    s = load_settings(f)
    assert s.chunk_size == 500 and s.top_k == 7