"""A line starting with ``---`` is ordinary text and must reach a drawer (local patch).

``_chunk_by_exchange`` used to end the AI response at such a line and then skip
every line up to the next ``>`` turn, so a Markdown divider in a reply, or a
pasted block with a ``---`` line in a multi-line user message, silently dropped
the rest of that exchange.
"""

from mempalace.convo_miner import chunk_exchanges

TRANSCRIPT = "\n".join(
    [
        "> First question about the cache layer and its eviction policy?",
        "Part one of the answer explains the LRU choice in some detail.",
        "",
        "---",
        "",
        "Part two, after a Markdown divider, explains the TTL choice.",
        "",
        "> Second question, with a pasted front-matter block below it:",
        "---",
        "title: pasted note",
        "---",
        "the pasted body the user wants remembered",
        "Answer two discusses the pasted note at length and in full.",
        "",
        "> Third question to close the conversation, about monitoring?",
        "Answer three: export hit rate and eviction count as metrics.",
        "",
    ]
)


def test_text_after_a_dash_line_reaches_a_drawer():
    chunks = chunk_exchanges(TRANSCRIPT, chunk_size=2000, min_chunk_size=10)
    stored = "\n".join(c["content"] for c in chunks)
    for needle in (
        "Part two, after a Markdown divider",
        "title: pasted note",
        "the pasted body the user wants remembered",
        "Answer two discusses the pasted note",
    ):
        assert needle in stored, f"dropped: {needle!r}"


def test_a_dash_line_does_not_start_a_new_exchange():
    chunks = chunk_exchanges(TRANSCRIPT, chunk_size=2000, min_chunk_size=10)
    assert [c["content"].split("\n", 1)[0][:14] for c in chunks] == [
        "> First questi",
        "> Second quest",
        "> Third questi",
    ]
