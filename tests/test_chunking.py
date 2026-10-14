from hybridrag.chunking import Chunker, clean
from hybridrag.config import Settings
from hybridrag.loaders import Page


def test_clean_rejoins_hyphenated_breaks():
    assert clean("inter-\nnational") == "international"


def test_heading_kept_and_short_chunks_dropped():
    md = "# Refunds\n\n" + ("Refunds are issued within 14 days of approval. " * 5) + "\n\n# Tiny\n\nhi"
    chunks = Chunker(Settings()).split(Page(md, "policy.md"))
    assert chunks and "Refunds" in chunks[0].text
    assert all("Tiny" not in c.text for c in chunks)


def test_ids_are_stable():
    page = Page("# A\n\n" + "Some reasonably long paragraph of text. " * 6, "a.md")
    c = Chunker(Settings())
    assert [x.id for x in c.split(page)] == [x.id for x in c.split(page)]