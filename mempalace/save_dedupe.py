"""Save-time skip of boilerplate and same-chat repeats (3D-Stories fork, sysop#297).

A chat is still cut into the same pieces with the same numbers, so drawer ids and
bookmarks keep their meaning; a skipped piece only leaves a gap. A piece is
skipped when it is boilerplate, or when an earlier piece of the same chat has
identical text, so the first copy is the one kept. The decision reads nothing but
the pieces' text and the reviewed boilerplate list, never another chat's rows,
so a chat never depends on a row it does not own.

``MEMPALACE_SAVE_DEDUPE=0`` stores every piece again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# A piece that starts with one of these, after an optional "> " user-turn quote,
# is boilerplate every session repeats.
BOILERPLATE_MARKERS = (
    "<herdr_model_guidance",
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<command-message>",
    "Base directory for this skill:",
)

_OFF_VALUES = ("0", "false", "no", "off")
_LIST_FILE = "boilerplate.json"
_LIST_VERSION = 1
_HEX64 = re.compile(r"[0-9a-f]{64}")


def save_dedupe_enabled() -> bool:
    raw = os.environ.get("MEMPALACE_SAVE_DEDUPE")
    return raw is None or raw.strip().lower() not in _OFF_VALUES


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_boilerplate_hashes() -> frozenset:
    """Fingerprints from the reviewed list in ``<config dir>/boilerplate.json``.

    The file is ``{"version": 1, "sha256": ["<64 lowercase hex>", ...]}``. A
    missing file, or any doubt about its shape, means an empty list: the safe
    direction is to store more, never less.
    """
    from .config import _default_config_dir

    path = _default_config_dir() / _LIST_FILE
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return frozenset()
    except (OSError, ValueError):
        logger.warning("boilerplate list %s is unreadable; using none", path)
        return frozenset()
    entries = data.get("sha256") if isinstance(data, dict) else None
    version = data.get("version") if isinstance(data, dict) else None
    if (
        isinstance(version, bool)
        or version != _LIST_VERSION
        or not isinstance(entries, list)
        or not all(isinstance(h, str) and _HEX64.fullmatch(h) for h in entries)
    ):
        logger.warning("boilerplate list %s is malformed; using none", path)
        return frozenset()
    return frozenset(entries)


def is_boilerplate(text: str, listed: frozenset) -> bool:
    body = text.lstrip()
    if body.startswith("> "):
        body = body[2:]
    return body.startswith(BOILERPLATE_MARKERS) or text_sha(text) in listed


def plan_skips(
    contents: Iterable[str],
    *,
    prior: frozenset = frozenset(),
    listed: Optional[frozenset] = None,
    keep_one: bool = True,
) -> tuple:
    """Positions in ``contents`` that are not stored, and whether keep-one fired.

    ``contents`` are the chat's pieces in order. ``prior`` holds fingerprints the
    palace already stores for earlier pieces of the same chat (the bookmark save
    passes those that match its new pieces; a full read passes none). A piece is
    skipped when it is boilerplate or its text was already seen. ``listed``
    defaults to the reviewed list on disk.

    With ``keep_one`` (a full read of the whole chat), a chat always keeps at
    least one piece: when every piece would be skipped, the first is kept and the
    second value is True, so the caller can refuse to bookmark a state a later
    save could not reproduce. The bookmark save passes ``keep_one=False``: the
    chat already stores the piece before its bookmark.
    """
    contents = list(contents)
    if not contents or not save_dedupe_enabled():
        return set(), False
    if listed is None:
        listed = load_boilerplate_hashes()
    seen = set(prior)
    skip = set()
    for pos, text in enumerate(contents):
        sha = text_sha(text)
        if sha in seen or is_boilerplate(text, listed):
            skip.add(pos)
            continue
        seen.add(sha)
    if keep_one and len(skip) == len(contents):
        skip.discard(0)
        return skip, True
    return skip, False
