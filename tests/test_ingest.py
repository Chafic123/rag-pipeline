"""Unit tests for chunking and document loading. Fast, no network, no API key."""
import pytest

from app.rag.ingest import chunk_text, load_chunks


def alphabet_text(length: int) -> str:
    """Varied text, so overlap checks compare real content rather than repeated characters."""
    return "".join(chr(ord("a") + i % 26) for i in range(length))


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world", chunk_size=500, overlap=100) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("", chunk_size=500, overlap=100) == []


def test_chunks_never_exceed_chunk_size():
    chunks = chunk_text(alphabet_text(1234), chunk_size=500, overlap=100)
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_each_chunk_starts_with_the_previous_chunks_tail():
    chunks = chunk_text(alphabet_text(1234), chunk_size=500, overlap=100)
    for previous, current in zip(chunks, chunks[1:]):
        assert current[:100] == previous[-100:]


def test_chunks_rebuild_the_original_text():
    text = alphabet_text(1234)
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    rebuilt = chunks[0] + "".join(chunk[100:] for chunk in chunks[1:])
    assert rebuilt == text


def test_overlap_must_be_smaller_than_chunk_size():
    # Without this guard the sliding window would never move forward (infinite loop).
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)


def test_load_chunks_skips_drafts_and_unsupported_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("bravo", encoding="utf-8")
    (tmp_path / "_draft.txt").write_text("skip me", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    chunks, sources = load_chunks(tmp_path)
    assert chunks == ["alpha", "bravo"]
    assert sources == ["a.txt", "b.md"]
