"""Search collapse (3D-Stories fork): each text appears once in search results.

The best-ranked hit of an identical text stays and gains ``also_in`` (the
other chats holding that text); later copies drop out, so the top-N slice
backfills from the rest of the candidate pool. ``MEMPALACE_SEARCH_COLLAPSE=0``
restores upstream behaviour.
"""

from __future__ import annotations

import pytest

from mempalace.palace import get_collection
from mempalace.searcher import (
    _collapse_identical_hits,
    _search_collapse_enabled,
    search_memories,
)

REPEATED = "quasar token rotation runbook: rotate the quasar signing key, then restart quasar"
DISTINCT = [
    "quasar token rotation postmortem: the relay cached the old key for an hour",
    "quasar token rotation checklist for the staging cluster and its replicas",
    "quasar token rotation cadence: every ninety days, with a seven day overlap",
]


def _hit(drawer_id, text, source):
    return {
        "drawer_id": drawer_id,
        "text": text,
        "source_file": source.rsplit("/", 1)[-1],
        "source_path": source,
        "matched_via": "drawer",
    }


@pytest.fixture(autouse=True)
def _collapse_on(monkeypatch):
    monkeypatch.delenv("MEMPALACE_SEARCH_COLLAPSE", raising=False)


def test_collapse_keeps_first_ranked_copy_and_notes_other_chats():
    hits = [
        _hit("a", "same text", "/chats/a.jsonl"),
        _hit("u", "unique text", "/chats/u.jsonl"),
        _hit("b", "same text", "/chats/b.jsonl"),
        _hit("c", "same text", "/chats/c.jsonl"),
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["a", "u"]
    assert result[0]["also_in"] == {
        "count": 2,
        "sources": ["/chats/b.jsonl", "/chats/c.jsonl"],
    }
    assert "also_in" not in result[1]


def test_collapse_drops_same_chat_repeat_without_a_note():
    hits = [
        _hit("a0", "same text", "/chats/a.jsonl"),
        _hit("a1", "same text", "/chats/a.jsonl"),
        _hit("u", "unique text", "/chats/u.jsonl"),
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["a0", "u"]
    assert "also_in" not in result[0]


def test_collapse_counts_every_other_chat_but_lists_three_once_each():
    hits = [
        _hit("a", "same text", "/chats/a.jsonl"),
        _hit("b0", "same text", "/chats/b.jsonl"),
        _hit("b1", "same text", "/chats/b.jsonl"),
        _hit("c", "same text", "/chats/c.jsonl"),
        _hit("a1", "same text", "/chats/a.jsonl"),
        _hit("d", "same text", "/chats/d.jsonl"),
        _hit("e", "same text", "/chats/e.jsonl"),
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["a"]
    assert result[0]["also_in"] == {
        "count": 4,
        "sources": ["/chats/b.jsonl", "/chats/c.jsonl", "/chats/d.jsonl"],
    }


def test_collapse_passes_non_string_and_blank_text_through():
    hits = [
        {"drawer_id": "n0", "text": None, "source_path": "/chats/a.jsonl"},
        {"drawer_id": "n1", "text": None, "source_path": "/chats/b.jsonl"},
        {"drawer_id": "m", "source_path": "/chats/c.jsonl"},
        _hit("w0", "   ", "/chats/d.jsonl"),
        _hit("w1", "   ", "/chats/e.jsonl"),
        _hit("e0", "", "/chats/f.jsonl"),
        _hit("e1", "", "/chats/g.jsonl"),
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["n0", "n1", "m", "w0", "w1", "e0", "e1"]
    assert all("also_in" not in h for h in result)


def test_collapse_does_not_mutate_dropped_copies_or_reorder():
    hits = [
        _hit("u1", "first unique", "/chats/u1.jsonl"),
        _hit("a", "same text", "/chats/a.jsonl"),
        _hit("u2", "second unique", "/chats/u2.jsonl"),
        _hit("b", "same text", "/chats/b.jsonl"),
    ]
    dropped = hits[3]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["u1", "a", "u2"]
    assert "also_in" not in dropped


@pytest.mark.parametrize("value", ["0", "false", "FALSE", " no ", "off", "Off"])
def test_kill_switch_spellings_disable_collapse(monkeypatch, value):
    monkeypatch.setenv("MEMPALACE_SEARCH_COLLAPSE", value)
    hits = [
        _hit("a", "same text", "/chats/a.jsonl"),
        _hit("b", "same text", "/chats/b.jsonl"),
    ]

    assert _search_collapse_enabled() is False
    result = _collapse_identical_hits(hits)
    assert [h["drawer_id"] for h in result] == ["a", "b"]
    assert all("also_in" not in h for h in result)


@pytest.mark.parametrize("value", ["1", "", "true", "yes", "on", "anything"])
def test_other_switch_values_keep_collapse_on(monkeypatch, value):
    monkeypatch.setenv("MEMPALACE_SEARCH_COLLAPSE", value)

    assert _search_collapse_enabled() is True


def test_switch_unset_keeps_collapse_on():
    assert _search_collapse_enabled() is True


def _seed(palace_path):
    sources = [f"/chats/copy-{i}.jsonl" for i in range(4)]
    ids = [f"copy_{i}" for i in range(4)] + [f"distinct_{i}" for i in range(3)]
    docs = [REPEATED] * 4 + DISTINCT
    metas = [
        {"wing": "sessions", "room": "technical", "source_file": s, "chunk_index": 0}
        for s in sources
    ] + [
        {
            "wing": "sessions",
            "room": "technical",
            "source_file": f"/chats/distinct-{i}.jsonl",
            "chunk_index": 0,
        }
        for i in range(3)
    ]
    get_collection(palace_path).upsert(ids=ids, documents=docs, metadatas=metas)
    return sources


def _assert_collapsed(result, sources):
    assert "error" not in result, result
    hits = result["results"]
    texts = [h["text"] for h in hits]
    assert len(hits) == 3
    assert len(set(texts)) == 3
    kept = [h for h in hits if h["text"] == REPEATED]
    assert len(kept) == 1
    note = kept[0]["also_in"]
    assert note["count"] == 3
    assert len(note["sources"]) == 3
    assert set(note["sources"]) == set(sources) - {kept[0]["source_path"]}


def test_vector_search_returns_each_text_once_and_backfills(palace_path):
    sources = _seed(palace_path)

    result = search_memories("quasar token rotation", palace_path, n_results=3)

    _assert_collapsed(result, sources)


def test_vector_disabled_fallback_returns_each_text_once(palace_path):
    sources = _seed(palace_path)

    result = search_memories(
        "quasar",
        palace_path,
        n_results=3,
        vector_disabled=True,
        collection_name="mempalace_drawers",
    )

    _assert_collapsed(result, sources)


@pytest.mark.parametrize("vector_disabled", [False, True])
def test_kill_switch_restores_repeats_end_to_end(palace_path, monkeypatch, vector_disabled):
    _seed(palace_path)
    monkeypatch.setenv("MEMPALACE_SEARCH_COLLAPSE", "0")

    result = search_memories(
        "quasar",
        palace_path,
        n_results=7,
        vector_disabled=vector_disabled,
        collection_name="mempalace_drawers",
    )

    assert "error" not in result, result
    texts = [h["text"] for h in result["results"]]
    assert texts.count(REPEATED) == 4
    assert all("also_in" not in h for h in result["results"])
