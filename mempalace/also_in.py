"""The ``Also in`` note of a collapsed search hit (3D-Stories fork, sysop#297).

A module of its own, imports included, so the hub-forwarded command line can
print the note without importing the local search stack.
"""

from pathlib import Path


def also_in_note(hit: dict) -> str:
    """``Also in: 2 other chats (a.jsonl, b.jsonl)`` for a collapsed hit, else ``""``."""
    note = hit.get("also_in")
    if not note:
        return ""
    count = note["count"]
    names = ", ".join(Path(source).name for source in note["sources"])
    more = ", ..." if count > len(note["sources"]) else ""
    return f"Also in: {count} other chat{'' if count == 1 else 's'} ({names}{more})"
