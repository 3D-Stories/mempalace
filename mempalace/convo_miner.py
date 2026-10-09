#!/usr/bin/env python3
"""
convo_miner.py — Mine conversations into the palace.

Ingests chat exports (Claude Code, ChatGPT, Slack, plain text transcripts).
Normalizes format, chunks by exchange pair (Q+A = one unit), files to palace.

Same palace as project mining. Different ingest strategy.
"""

import errno
import os
import sys
import json
import codecs
import math
import hashlib
import logging
import stat
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

from .backends import PalaceNotFoundError
from .collision_scan import assert_no_collisions
from .ids import (
    ID_RECIPE,
    make_convo_drawer_id,
    make_convo_sentinel_id,
    make_exchange_drawer_id,
)
from .normalize import UnparsedCodexTranscriptError, normalize_conversations
from .entities import entities_metadata
from .palace import (
    NORMALIZE_VERSION,
    SKIP_DIRS,
    _metadata_matches_extract_mode,
    _validate_palace_fts5_after_mine,
    file_already_mined,
    get_collection,
    mine_lock,
    mine_palace_lock,
    prefetch_content_hashes,
    prefetch_mined_set,
)

logger = logging.getLogger("mempalace_mcp")


# Cached hall keywords — avoids re-reading config per drawer
_HALL_KEYWORDS_CACHE = None


def _detect_hall_cached(content: str) -> str:
    """Route content to a hall using cached keywords. Same logic as miner.detect_hall."""
    global _HALL_KEYWORDS_CACHE
    if _HALL_KEYWORDS_CACHE is None:
        from .config import MempalaceConfig

        _HALL_KEYWORDS_CACHE = MempalaceConfig().hall_keywords
    content_lower = content[:3000].lower()
    scores = {}
    for hall, keywords in _HALL_KEYWORDS_CACHE.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[hall] = score
    return max(scores, key=scores.get) if scores else "general"


def file_conversation_exchange(
    collection,
    *,
    wing: str,
    room: str,
    text: str,
    source_file: str,
    agent: str,
    authored_at: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Optional[str]:
    """File one verbatim conversation exchange as a single drawer.

    Canonical write path for live agent integrations (e.g. Hermes) and
    their backfills — both must route here so routing, normalization,
    and metadata conventions stay identical between live and historical
    ingest. Builds the same metadata the convo miner writes so hallway
    traversal, entity search, and since/before date filters see
    integration drawers exactly like mined ones.

    ``wing`` and ``room`` are validated with the same ``sanitize_name``
    rules the MCP write tools apply, but a failed name falls back
    (``wing_general`` / ``conversations``) instead of erroring: this
    path files *live* turns, and dropping a turn over a config typo
    would break the verbatim / 100%-recall promise. The fallback is
    logged at warning level so the misconfiguration is visible.

    ``extra_metadata`` lets callers append integration-specific fields
    (e.g. ``source`` / ``session_id``); keys that collide with the
    canonical fields are ignored, so it cannot be used to overwrite or
    drop them. Returns the drawer id, or None when ``text`` is empty
    after stripping.
    """
    from .config import sanitize_name

    text = (text or "").strip()
    if not text:
        return None
    try:
        wing = sanitize_name(wing, "wing")
    except ValueError:
        logger.warning(
            "file_conversation_exchange: invalid wing %r — filing under wing_general", wing
        )
        wing = "wing_general"
    try:
        room = sanitize_name(room, "room")
    except ValueError:
        logger.warning(
            "file_conversation_exchange: invalid room %r — filing under conversations", room
        )
        room = "conversations"
    filed_at = datetime.now().isoformat()
    drawer_id = make_exchange_drawer_id(wing, room, source_file, filed_at, text)
    metadata = {
        "wing": wing,
        "room": room,
        "hall": _detect_hall_cached(text),
        "source_file": source_file,
        "chunk_index": 0,
        "added_by": agent,
        "filed_at": filed_at,
        "entities": entities_metadata(text),
        "authored_at": authored_at if authored_at is not None else filed_at,
        "ingest_mode": "convos",
        "extract_mode": "exchange",
        "normalize_version": NORMALIZE_VERSION,
        "id_recipe": ID_RECIPE,
    }
    if extra_metadata:
        for key, value in extra_metadata.items():
            metadata.setdefault(key, value)
    collection.upsert(ids=[drawer_id], documents=[text], metadatas=[metadata])
    return drawer_id


# File types that might contain conversations
CONVO_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}

# Directories inside conversation sources that never hold conversations.
# ``tool-results``: Claude Code pages large tool outputs to
# ``<session>/tool-results/*.txt`` inside ``~/.claude/projects/<slug>/``.
# They are raw machine dumps referenced from the transcript JSONL — mining
# them stores megabytes of command output as "memories" (field measurement:
# 12.8k drawers from tool-results files on one palace; a single file
# produced 3.6k). Extends the generic SKIP_DIRS set for the convo scanner
# only — project mining semantics are unchanged.
CONVO_SKIP_DIRS = SKIP_DIRS | {"tool-results"}

MIN_CHUNK_SIZE = 30
CHUNK_SIZE = 800  # chars per drawer — align with miner.py
_LINE_GROUP_SIZE = 25  # lines per fallback group when no paragraph breaks
_LINE_FALLBACK_MIN_NEWLINES = 20  # trigger line-group fallback above this newline count
DRAWER_UPSERT_BATCH_SIZE = 1000
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB — skip files larger than this.
# Matches miner.py at 500 MB. Long Claude Code sessions, multi-year
# ChatGPT exports, and lifetime Slack dumps routinely exceed 10 MB; the
# cap at that level silently dropped them with `continue`. Per-drawer
# size is bounded by CHUNK_SIZE, but larger source files still produce
# more drawers and therefore more embedding/storage work — and content
# is normalized and loaded fully into memory before chunking, so memory
# use also scales with source size.


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
        return True
    except (OSError, ValueError):
        return False


def _is_regular_source_file(filepath: Path, root: Path) -> bool:
    if not _path_within_root(filepath, root):
        return False
    # O_NONBLOCK keeps the S_ISREG verdict below reachable: a blocking open
    # of a FIFO waits in the kernel for a writer, so a named pipe called
    # ``session.jsonl`` would hang this check instead of failing it. See the
    # matching comment in ``miner._read_text_no_follow``, including why the
    # EAGAIN branch re-checks the type and then opens without the flag.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = -1
    try:
        try:
            fd = os.open(filepath, flags)
        except OSError as exc:
            if exc.errno != errno.EAGAIN or not stat.S_ISREG(os.lstat(filepath).st_mode):
                raise
            fd = os.open(filepath, flags & ~getattr(os, "O_NONBLOCK", 0))
        st = os.fstat(fd)
        return stat.S_ISREG(st.st_mode) and st.st_size <= MAX_FILE_SIZE
    except OSError:
        return False
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass


def _register_file(
    collection,
    source_file: str,
    wing: str,
    agent: str,
    extract_mode: str,
    content_hash: Optional[str] = None,
):
    """Write a sentinel so file_already_mined() returns True for 0-chunk files.

    Without this, files that normalize to nothing or produce zero chunks are
    re-read and re-processed on every mine run because nothing was written to
    ChromaDB on the first pass.

    Stamps source_mtime like every real drawer does, so a file that later
    grows past the min-chunk-size floor (e.g. a short session that gets
    extended) is correctly detected as changed on the next mine instead of
    being skipped forever by this sentinel.

    Also used to register a file recognized as a content-duplicate of an
    already-mined transcript under a different path (see
    ``prefetch_content_hashes``) — stamping it here means the next run skips
    it via the cheap mtime check instead of re-normalizing and re-hashing it.
    """
    try:
        source_mtime = os.path.getmtime(source_file)
    except OSError:
        source_mtime = None
    sentinel_id = make_convo_sentinel_id(source_file, extract_mode)
    meta = {
        "wing": wing,
        "room": "_registry",
        "source_file": source_file,
        "added_by": agent,
        "filed_at": datetime.now().isoformat(),
        "ingest_mode": "registry",
        "extract_mode": extract_mode,
        "normalize_version": NORMALIZE_VERSION,
        "id_recipe": ID_RECIPE,
    }
    if source_mtime is not None:
        meta["source_mtime"] = source_mtime
    if content_hash is not None:
        meta["content_hash"] = content_hash
    collection.upsert(
        documents=[f"[registry] {source_file}"],
        ids=[sentinel_id],
        metadatas=[meta],
    )


def _source_file_existing(collection, source_file: str, extract_mode: str) -> dict[str, dict]:
    """Map drawer_id -> stored metadata for one source file and extraction mode.

    Legacy conversation drawers did not carry extract_mode; treat those as
    exchange-mode rows so schema rebuilds can still clean them up without
    deleting newer general-mode drawers for the same transcript.
    """
    existing: dict[str, dict] = {}
    offset = 0
    while True:
        batch = collection.get(
            where={"source_file": source_file},
            limit=1000,
            offset=offset,
            include=["metadatas"],
        )
        batch_ids = batch.get("ids") or []
        metadatas = batch.get("metadatas") or []
        for drawer_id, meta in zip(batch_ids, metadatas):
            if _metadata_matches_extract_mode(meta or {}, extract_mode):
                existing[drawer_id] = meta or {}
        if not batch_ids:
            break
        offset += len(batch_ids)
    return existing


def _source_file_delete_ids(collection, source_file: str, extract_mode: str) -> list[str]:
    """Collect drawer IDs for one source file and extraction mode."""
    return list(_source_file_existing(collection, source_file, extract_mode))


# =============================================================================
# CHUNKING — exchange pairs for conversations
# =============================================================================


def chunk_exchanges(
    content: str,
    chunk_size: int = None,
    min_chunk_size: int = None,
) -> list:
    """
    Chunk by exchange pair: one > turn + AI response = one unit.
    Falls back to paragraph chunking if no > markers.

    Optional params override module-level defaults when provided.

    Raises ``ValueError`` if ``chunk_size`` is not a positive integer or
    ``min_chunk_size`` is negative. A non-positive ``chunk_size`` would
    cause ``_chunk_by_exchange`` below to loop forever — ``content[:0]``
    is empty, ``content[0:]`` is the whole string, and the remainder
    never shrinks.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if min_chunk_size is None:
        min_chunk_size = MIN_CHUNK_SIZE

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if min_chunk_size < 0:
        raise ValueError(f"min_chunk_size must be >= 0, got {min_chunk_size}")

    lines = content.split("\n")
    quote_lines = sum(1 for line in lines if line.strip().startswith(">"))

    if quote_lines >= 3:
        return _chunk_by_exchange(lines, chunk_size, min_chunk_size)
    else:
        return _chunk_by_paragraph(content, chunk_size, min_chunk_size)


def _chunk_by_exchange(lines: list, chunk_size: int, min_chunk_size: int) -> list:
    """One user turn (>) + the AI response that follows = one or more chunks.

    The full AI response is preserved verbatim.  When the combined
    user-turn + response exceeds chunk_size the response is split across
    consecutive drawers so nothing is silently discarded.
    """
    chunks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(">"):
            user_turn = line.strip()
            i += 1

            ai_lines = []
            while i < len(lines):
                next_line = lines[i]
                # Local patch: a ``---`` line no longer ends the response. It is
                # ordinary text (a Markdown divider, a pasted front-matter block)
                # and ending here skipped every line up to the next ``>`` turn.
                if next_line.strip().startswith(">"):
                    break
                # Preserve the line as-is — blank lines and indentation carry meaning
                # (paragraph breaks, list/code structure) and must survive verbatim.
                ai_lines.append(next_line)
                i += 1

            # Join on newline (not space) so line structure, blank lines, and
            # indentation reach the drawer unchanged. Trim only trailing blank
            # lines produced by the loop stopping at the next `>` turn.
            ai_response = "\n".join(ai_lines).rstrip("\n")
            content = f"{user_turn}\n{ai_response}" if ai_response else user_turn

            _emit_bounded(chunks, content, chunk_size, min_chunk_size)
        else:
            i += 1

    return chunks


def _emit_bounded(
    chunks: list,
    content: str,
    chunk_size: int,
    min_chunk_size: int,
) -> None:
    """Append ``content`` as one or more drawers, none exceeding ``chunk_size``.

    The ``min_chunk_size`` floor gates the WHOLE call (drops the input if
    its stripped length is at or below the floor, treated as noise). Once
    the input passes the floor, every slice is emitted verbatim so a
    small trailing remainder is preserved instead of silently dropped.
    The index-based loop avoids the O(N^2) repeated-substring allocation
    of a ``while content: content = content[chunk_size:]`` shape.
    """
    if len(content.strip()) <= min_chunk_size:
        return
    for i in range(0, len(content), chunk_size):
        chunks.append({"content": content[i : i + chunk_size], "chunk_index": len(chunks)})


def _chunk_by_paragraph(content: str, chunk_size: int, min_chunk_size: int) -> list:
    """Fallback: chunk by paragraph breaks."""
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # If no paragraph breaks and long content, chunk by line groups
    if len(paragraphs) <= 1 and content.count("\n") > _LINE_FALLBACK_MIN_NEWLINES:
        lines = content.split("\n")
        for i in range(0, len(lines), _LINE_GROUP_SIZE):
            group = "\n".join(lines[i : i + _LINE_GROUP_SIZE]).strip()
            _emit_bounded(chunks, group, chunk_size, min_chunk_size)
        return chunks

    for para in paragraphs:
        _emit_bounded(chunks, para, chunk_size, min_chunk_size)

    return chunks


# =============================================================================
# ROOM DETECTION — topic-based for conversations
# =============================================================================

TOPIC_KEYWORDS = {
    "technical": [
        "code",
        "python",
        "function",
        "bug",
        "error",
        "api",
        "database",
        "server",
        "deploy",
        "git",
        "test",
        "debug",
        "refactor",
    ],
    "architecture": [
        "architecture",
        "design",
        "pattern",
        "structure",
        "schema",
        "interface",
        "module",
        "component",
        "service",
        "layer",
    ],
    "planning": [
        "plan",
        "roadmap",
        "milestone",
        "deadline",
        "priority",
        "sprint",
        "backlog",
        "scope",
        "requirement",
        "spec",
    ],
    "decisions": [
        "decided",
        "chose",
        "picked",
        "switched",
        "migrated",
        "replaced",
        "trade-off",
        "alternative",
        "option",
        "approach",
    ],
    "problems": [
        "problem",
        "issue",
        "broken",
        "failed",
        "crash",
        "stuck",
        "workaround",
        "fix",
        "solved",
        "resolved",
    ],
}


def detect_convo_room(content: str) -> str:
    """Score conversation content against topic keywords."""
    content_lower = content[:3000].lower()
    scores = {}
    for room, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[room] = score
    if scores:
        return max(scores, key=scores.get)
    return "general"


# =============================================================================
# PALACE OPERATIONS
# =============================================================================


# =============================================================================
# SCAN FOR CONVERSATION FILES
# =============================================================================


def scan_convos(convo_dir: str, include_subagents: bool = False) -> list:
    """Find all potential conversation files.

    Skips symlinks and oversized files. Each skipped symlink is logged to
    ``sys.stderr`` with a ``  SKIP: <relative-path> (symlink)`` line so the
    caller can tell why an apparent conversation directory yielded no files.

    By default, directories named ``subagents`` are skipped: Claude Code
    records Explore/Plan/Grep subagent transcripts there, and on typical
    workspaces they outnumber main session files by one to two orders of
    magnitude. Pass ``include_subagents=True`` to mine them anyway.

    The match is case-insensitive on the directory name only (``subagents``
    or ``Subagents``), so directories like ``mysubagents`` or
    ``subagentsbackup`` are not affected.
    """
    # A direct conversation file is a valid source. For a file, feed only
    # its basename through the existing directory validation loop.
    requested_path = Path(convo_dir).expanduser()
    single_file = requested_path.is_file()
    convo_path = (requested_path.parent if single_file else requested_path).resolve()
    scan_entries = (
        [(str(convo_path), [], [requested_path.name])] if single_file else os.walk(convo_path)
    )
    files = []
    for root, dirs, filenames in scan_entries:
        dirs[:] = [
            d
            for d in dirs
            if d not in CONVO_SKIP_DIRS and (include_subagents or d.lower() != "subagents")
        ]
        for filename in filenames:
            if filename.endswith(".meta.json"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() in CONVO_EXTENSIONS:
                # Skip symlinks and oversized files
                if filepath.is_symlink():
                    rel = filepath.relative_to(convo_path).as_posix()
                    try:
                        print(f"  SKIP: {rel} (symlink)", file=sys.stderr)
                    except OSError:
                        pass
                    continue
                # Skip files exceeding size limit, or those whose stat() raises
                # (permission denied, racing delete, broken symlink that
                # survived the earlier is_symlink check). Both branches log
                # to stderr to match the SKIP: (symlink) line above; silent
                # drops at this gate were the original #923 complaint.
                try:
                    file_stat = filepath.stat()
                    # Drop non-regular entries (FIFO, socket, device node)
                    # before any reader touches them — see the matching
                    # gate in ``miner.scan_project``.
                    if not stat.S_ISREG(file_stat.st_mode):
                        print(
                            f"  SKIP: {filepath.name} (not a regular file)",
                            file=sys.stderr,
                        )
                        continue
                    file_size = file_stat.st_size
                    if file_size > MAX_FILE_SIZE:
                        print(
                            f"  SKIP: {filepath.name} ({file_size / (1024 * 1024):.1f} MB)"
                            f" exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit",
                            file=sys.stderr,
                        )
                        continue
                except OSError as exc:
                    # Prefer ``exc.strerror`` so the path isn't duplicated in
                    # the output (see the matching comment in
                    # ``miner.scan_project``).
                    print(
                        f"  SKIP: {filepath.name} (stat error: {exc.strerror or exc})",
                        file=sys.stderr,
                    )
                    continue
                if not _is_regular_source_file(filepath, convo_path):
                    continue
                files.append(filepath)
    return files


# =============================================================================
# MINE CONVERSATIONS
# =============================================================================


def _extract_authored_at(filepath):
    """Most-recent message timestamp in a transcript, used as the drawer's authored date.

    Both Claude Code and Codex JSONL transcripts carry a top-level ISO-8601
    ``timestamp`` on each line. We take the max so ``authored_at`` reflects when the
    content was actually written, independent of when it was mined (``filed_at``).
    This restores chronology: a session from days ago keeps its real date even when
    re-mined today, instead of every drawer collapsing to ingest time. Returns None
    for formats without per-line timestamps (e.g. plain ``.md``).
    """
    path = Path(filepath)
    if path.suffix != ".jsonl":
        return None
    latest = None
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts = json.loads(line).get("timestamp")
                except (ValueError, TypeError, AttributeError):
                    continue
                # ISO-8601 timestamps are strings; guard against a non-string
                # ``timestamp`` so a malformed line can't raise TypeError on compare.
                if isinstance(ts, str) and (latest is None or ts > latest):
                    latest = ts
    except OSError:
        return None
    return latest


def _convo_drawer_meta(
    chunk,
    chunk_room,
    *,
    wing,
    source_file,
    agent,
    filed_at,
    authored_at,
    extract_mode,
    chunk_total,
    chunk_hash,
    source_mtime,
    content_hash,
) -> dict:
    """Metadata for a new or changed convo drawer (shared by the full and bookmark paths)."""
    meta = {
        "wing": wing,
        "room": chunk_room,
        "hall": _detect_hall_cached(chunk["content"]),
        "source_file": source_file,
        "chunk_index": chunk["chunk_index"],
        "added_by": agent,
        "filed_at": filed_at,
        "entities": entities_metadata(chunk["content"]),
        "authored_at": authored_at if authored_at is not None else filed_at,
        "ingest_mode": "convos",
        "extract_mode": extract_mode,
        "normalize_version": NORMALIZE_VERSION,
        "id_recipe": ID_RECIPE,
        "chunk_total": chunk_total,
        "chunk_hash": chunk_hash,
    }
    if source_mtime is not None:
        meta["source_mtime"] = source_mtime
    # Stamp content_hash only on chunk 0 so multi-conversation
    # privacy-export hashes are not O(N²)-duplicated across every
    # chunk row. ``prefetch_content_hashes`` still finds them — it
    # scans all drawers and splits comma-joined hash fields.
    if content_hash is not None and chunk.get("chunk_index", 0) == 0:
        meta["content_hash"] = content_hash
    return meta


def _upsert_convo_drawers(collection, to_upsert: list) -> int:
    """Upsert (drawer_id, content, meta) rows in bounded batches; returns rows written.

    Batched so large transcripts keep most of the embedding speedup without one
    huge Chroma/SQLite request.
    """
    written = 0
    for batch_start in range(0, len(to_upsert), DRAWER_UPSERT_BATCH_SIZE):
        batch = to_upsert[batch_start : batch_start + DRAWER_UPSERT_BATCH_SIZE]
        batch_ids = [drawer_id for drawer_id, _, _ in batch]
        batch_docs = [content for _, content, _ in batch]
        batch_metas = [meta for _, _, meta in batch]
        assert_no_collisions(list(zip(batch_ids, batch_metas)), collection)
        try:
            collection.upsert(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas,
            )
            written += len(batch_docs)
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise
    return written


def _file_chunks_locked(
    collection,
    source_file,
    chunks,
    wing,
    room,
    agent,
    extract_mode,
    authored_at=None,
    content_hash=None,
):
    """Lock the source file and file its chunks incrementally.

    Combines the per-file serialization that prevents concurrent agents from
    duplicating work (via mine_lock) with an incremental re-mine contract
    (#2403): drawer ids are deterministic over
    ``(source_file, extract_mode, chunk_index)``, so a re-mine of a changed
    source — a Claude Code session appends to its own transcript every turn,
    and /compact or /clear can rewrite one in place — only re-embeds the
    chunks whose content actually changed. Unchanged chunks get a cheap
    metadata-only refresh (``source_mtime`` / ``chunk_total``) so the
    completion check in ``file_already_mined`` still sees one full
    current-mtime group. Existing drawers are never deleted before the new
    set is fully written: orphaned ids (a shrunk/rewritten source, or an id
    recipe migration) are deleted last, after every upsert and metadata
    touch has succeeded, so a crash mid-operation leaves everything the
    palace previously held (append-only ingest; CLAUDE.md "Incremental
    only"). Any interrupted pass leaves mixed mtime groups, each short of
    its ``chunk_total`` — ``file_already_mined`` returns False and the next
    mine repairs (#2183).

    Returns (drawers_added, room_counts_delta, skipped) where drawers_added
    counts only new/changed drawers actually upserted this pass.
    """
    room_counts_delta: dict = defaultdict(int)
    drawers_added = 0
    with mine_lock(source_file):
        # Re-check after lock — another agent may have just finished this file
        # at the current schema/mtime. A stale hit here returns False, so we
        # still fall through to the incremental path below.
        if file_already_mined(collection, source_file, check_mtime=True, extract_mode=extract_mode):
            return 0, room_counts_delta, True

        # Snapshot what the palace already holds for this source+mode. A
        # failed snapshot must abort this file's mine attempt rather than
        # fall through to a blind full upsert: without the snapshot the pass
        # cannot tell changed chunks from unchanged ones or find orphaned
        # ids, and would degrade to the very purge/rebuild churn this path
        # exists to avoid (#105 — convo_miner's own instance of the same
        # swallow already fixed for miner.py at #23). Returning here leaves
        # the old drawers' stored mtime untouched, so the next mine still
        # sees a mismatch and retries.
        try:
            existing = _source_file_existing(collection, source_file, extract_mode)
        except Exception as exc:
            print(
                f"  ! [skip] existing-drawer snapshot failed for {source_file!r} "
                f"({exc!r}); leaving existing drawers untouched, will retry "
                f"on the next mine",
                file=sys.stderr,
            )
            logger.debug("Existing-drawer snapshot failed for %s", source_file, exc_info=True)
            return 0, room_counts_delta, True

        # One filed_at per source file so all transcript drawers of a pass
        # share an ingest timestamp.
        #
        # Every drawer of this pass carries ``chunk_total`` so
        # ``file_already_mined`` / ``prefetch_mined_set`` can tell a complete
        # multi-batch mine from one that crashed mid-file (#2183). Without it
        # a stable mtime + any surviving drawer permanently skips the file
        # and the missing exchanges never come back.
        filed_at = datetime.now().isoformat()
        try:
            source_mtime = os.path.getmtime(source_file)
        except OSError:
            source_mtime = None
        chunk_total = len(chunks)

        # Partition the target set: a chunk whose drawer already exists with
        # the same content hash at the current schema keeps its embedding and
        # only needs its completion metadata refreshed; everything else is
        # (re-)upserted. Drawers written before ``chunk_hash`` existed have no
        # hash to compare, count as changed once, and migrate themselves.
        to_upsert: list = []  # (drawer_id, content, meta)
        to_touch: list = []  # (drawer_id, refreshed_meta)
        new_ids: set = set()
        for chunk in chunks:
            chunk_room = chunk.get("memory_type", room) if extract_mode == "general" else room
            if extract_mode == "general":
                room_counts_delta[chunk_room] += 1
            drawer_id = make_convo_drawer_id(
                wing, chunk_room, source_file, extract_mode, chunk["chunk_index"]
            )
            new_ids.add(drawer_id)
            chunk_hash = hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()
            prev = existing.get(drawer_id)
            if (
                prev is not None
                and prev.get("chunk_hash") == chunk_hash
                and prev.get("normalize_version", 1) >= NORMALIZE_VERSION
            ):
                touched = dict(prev)
                touched["chunk_total"] = chunk_total
                if source_mtime is not None:
                    touched["source_mtime"] = source_mtime
                else:
                    # A stale stored mtime with no current one to replace it
                    # would let an old group satisfy a future completion check.
                    touched.pop("source_mtime", None)
                # Refresh the chunk-0 conversation-hash stamp so cross-file
                # dedup keeps seeing conversations added since the last pass.
                if content_hash is not None and chunk.get("chunk_index", 0) == 0:
                    touched["content_hash"] = content_hash
                to_touch.append((drawer_id, touched))
                continue
            meta = _convo_drawer_meta(
                chunk,
                chunk_room,
                wing=wing,
                source_file=source_file,
                agent=agent,
                filed_at=filed_at,
                authored_at=authored_at,
                extract_mode=extract_mode,
                chunk_total=chunk_total,
                chunk_hash=chunk_hash,
                source_mtime=source_mtime,
                content_hash=content_hash,
            )
            to_upsert.append((drawer_id, chunk["content"], meta))

        # Batch into bounded requests so large transcripts keep most of the
        # embedding speedup without one huge Chroma/SQLite request.
        #
        # No cleanup on failure: nothing was purged, so everything the palace
        # held before this pass is still there, and the partially written
        # pass leaves mixed mtime groups that each fall short of their
        # ``chunk_total`` — ``file_already_mined`` stays False and the next
        # mine retries (#2183 / #2403).
        drawers_added += _upsert_convo_drawers(collection, to_upsert)
        for batch_start in range(0, len(to_touch), DRAWER_UPSERT_BATCH_SIZE):
            batch = to_touch[batch_start : batch_start + DRAWER_UPSERT_BATCH_SIZE]
            collection.update(
                ids=[drawer_id for drawer_id, _ in batch],
                metadatas=[meta for _, meta in batch],
            )

        # Delete orphaned drawers LAST — ids the current source no longer
        # produces (shrunk/rewritten file, id recipe migration). Deleting
        # them only after the full new set is in place means an interrupted
        # pass never leaves the palace with less than it had; the worst
        # crash window leaves transient duplicates that the next
        # content-change mine sweeps. A failed delete is logged, not fatal,
        # for the same reason.
        stale_ids = [drawer_id for drawer_id in existing if drawer_id not in new_ids]
        if stale_ids:
            try:
                collection.delete(ids=stale_ids)
            except Exception:
                logger.warning(
                    "Failed to delete %d orphaned convo drawers for %s; "
                    "they remain as duplicates until the next content change",
                    len(stale_ids),
                    source_file,
                    exc_info=True,
                )
    return drawers_added, room_counts_delta, False


def _is_ai_tool_path(path: Path) -> bool:
    """Return True when `path` lives inside a known AI-tool storage dir.

    Detected paths (exact-segment match — substrings like `.gemini-backup`
    or `.codex-archive` do NOT match):
      - any segment ``.codex`` (Codex CLI sessions / archives)
      - any segment ``.gemini`` (Gemini CLI sessions under ~/.gemini/tmp/...)
      - the consecutive segment pair ``.claude/projects`` (Claude Code).
        ``.claude`` alone is NOT matched — that is the settings/config dir,
        not a conversation source.

    Used by ``_resolve_wing`` to default the destination wing to
    ``wing_api`` when the user hasn't passed an explicit ``--wing``.
    """
    try:
        parts = path.resolve().parts
    except (OSError, RuntimeError):
        return False

    if ".codex" in parts:
        return True
    if ".gemini" in parts:
        return True
    for i in range(len(parts) - 1):
        if parts[i] == ".claude" and parts[i + 1] == "projects":
            return True
    return False


def _split_new_and_duplicate_conversations(
    conversations: list,
    wing: str,
    source_file: str,
    mined_content_hashes: dict,
) -> tuple:
    """Hash each conversation and split them into (new, duplicate) lists.

    A conversation is a duplicate when its hash is already registered under
    a *different* source_file in the same wing — mining the same transcript
    into a second wing is a deliberate re-file, not a repeat, so the lookup
    is scoped to (wing, hash). Returns ([(hash, text), ...] new, [(hash,
    dup_source_file), ...] duplicates).
    """
    new_items = []
    duplicates = []
    for conversation in conversations:
        content_hash = _conversation_hash(conversation)
        dup_source = mined_content_hashes.get((wing, content_hash))
        if dup_source is None or dup_source == source_file:
            new_items.append((content_hash, conversation))
        else:
            duplicates.append((content_hash, dup_source))
    return new_items, duplicates


def _is_unchanged_since_last_mine(source_file: str, mined_mtimes: dict) -> bool:
    """True iff source_file was mined at the current schema AND its on-disk
    mtime still matches what was stored -- the mtime-aware replacement for
    "we've seen this source_file before" (transcripts are not immutable).

    False (re-mine) whenever the file isn't in mined_mtimes at all, its
    stored mtime is None (never recorded -- pre-mtime-tracking drawer, or
    getmtime failed when it was written), or getmtime fails right now
    (treat as changed rather than silently trusting stale data).
    """
    if source_file not in mined_mtimes:
        return False
    stored_mtime = mined_mtimes[source_file]
    if stored_mtime is None:
        return False
    try:
        current_mtime = os.path.getmtime(source_file)
    except OSError:
        return False
    return abs(stored_mtime - current_mtime) < 0.001


def _resolve_wing(convo_path: Path, wing: Optional[str]) -> str:
    """Determine the destination wing for ``mine_convos``.

    Precedence (first match wins):

      1. Explicit ``wing`` argument from the user — always wins, even on
         an AI-tool path. Empty string is treated as "no wing".
      2. AI-tool path detection — defaults to ``wing_api`` so Claude
         Code / Codex / Gemini conversations group under a single wing
         dedicated to API-sourced content.
      3. Basename fallback — sanitized via ``config.normalize_wing_name``
         (lowercase, spaces/hyphens collapsed to underscores). Shared
         single source of truth with ``cmd_init``,
         ``room_detector_local``, and ``miner.load_config`` so all
         wing-slug producers stay in sync (per #1194 consolidation).
    """
    from .config import normalize_wing_name

    if wing:
        return wing
    if _is_ai_tool_path(convo_path):
        return "wing_api"
    return normalize_wing_name(convo_path.name)


def mine_convos(
    convo_dir: str,
    palace_path: str,
    wing: str = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
    include_subagents: bool = False,
):
    """Mine a directory of conversation files into the palace.

    extract_mode:
        "exchange" — default exchange-pair chunking (Q+A = one unit)
        "general"  — general extractor: decisions, preferences, milestones, problems, emotions
    include_subagents:
        False (default) — skip Claude Code ``subagents/`` directories
        True            — also mine subagent transcripts

    The real work is in :func:`_mine_convos_impl`; this wrapper holds the
    per-palace flock around it so two concurrent ``mempalace mine --mode
    convos`` invocations against the same palace can't pile up. This
    mirrors the pattern in :func:`mempalace.miner.mine`. The lock is
    non-blocking: ``MineAlreadyRunning`` propagates to the CLI (which
    renders a holder-aware message and exits non-zero) or to in-process
    callers that expect to coexist with another writer.

    Dry-run skips the lock — it never writes to the palace and so cannot
    corrupt anything, and skipping the lock lets dry-run probes coexist
    with a live mine.

    Chunking parameters (chunk_size, min_chunk_size) are read from
    MempalaceConfig inside :func:`_mine_convos_impl` so `config.json`
    governs both this path and the project-file miner in `miner.py`.
    """
    if dry_run:
        return _mine_convos_impl(
            convo_dir,
            palace_path,
            wing=wing,
            agent=agent,
            limit=limit,
            dry_run=dry_run,
            extract_mode=extract_mode,
            include_subagents=include_subagents,
        )

    with mine_palace_lock(palace_path):
        return _mine_convos_impl(
            convo_dir,
            palace_path,
            wing=wing,
            agent=agent,
            limit=limit,
            dry_run=dry_run,
            extract_mode=extract_mode,
            include_subagents=include_subagents,
        )


def _compute_hallways_for_wing_safe(wing, collection, drawers_filed, config=None):
    """Auto-populate the associative graph from the entities just mined.

    Best-effort: hallway computation must never fail an otherwise-good mine, and is
    skipped when nothing new was filed.
    """
    if drawers_filed <= 0:
        return None
    try:
        from .hallways import compute_hallways_for_wing

        # Local patch (step 3): compute_hallways_for_wing catches its own fetch error
        # and returns [], so watch the fetch to report that as a failure too.
        watched = _FetchWatch(collection) if collection is not None else None
        compute_hallways_for_wing(wing, col=watched, config=config)
        return not (watched is not None and watched.failed)
    except Exception as exc:
        print(f"  (hallways skipped: {exc})")
        return False


def _normalize_convo_conversations(
    filepath: Path,
    source_file: str,
    cfg_min_chunk_size: int,
    collection,
    wing: str,
    agent: str,
    extract_mode: str,
    dry_run: bool,
) -> Optional[list]:
    """Normalize a transcript file into its individual conversations,
    registering it as filed when there's nothing worth mining. Returns None
    when the caller should skip the file (normalize failed, or normalized
    content is too short to chunk).

    Kept as separate conversations rather than joined into one string so
    dedup can hash and skip per conversation — a Claude.ai privacy export
    bundles every conversation into a single file, and hashing the joined
    bundle means one new conversation added to a re-export changes the
    whole-file hash and hides the conversations that didn't change.
    """
    try:
        conversations = [c for c in normalize_conversations(str(filepath)) if c]
    except UnparsedCodexTranscriptError as exc:
        logger.warning("Skipping %s: %s; source remains eligible for retry", filepath, exc)
        return None
    except (OSError, ValueError):
        if not dry_run:
            _register_file(collection, source_file, wing, agent, extract_mode)
        return None

    total_len = sum(len(c.strip()) for c in conversations)
    if not conversations or total_len < cfg_min_chunk_size:
        if not dry_run:
            _register_file(collection, source_file, wing, agent, extract_mode)
        return None

    return conversations


def _open_convo_collection(
    palace_path: str,
    *,
    dry_run: bool,
):
    """Open the conversation collection without creating it during dry-run."""
    if not dry_run:
        return get_collection(palace_path)

    try:
        return get_collection(
            palace_path,
            create=False,
            read_only=True,
        )
    except PalaceNotFoundError:
        # A missing palace or uninitialized collection represents empty
        # prior state to a dry-run. Do not create either one.
        return None


def _mine_convos_impl(
    convo_dir: str,
    palace_path: str,
    wing: str = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
    include_subagents: bool = False,
):
    from .config import MempalaceConfig

    palace_config = MempalaceConfig(palace_path=palace_path)
    cfg_chunk_size = palace_config.chunk_size
    # Only override convo_miner's MIN_CHUNK_SIZE when the user has set
    # min_chunk_size explicitly. min_chunk_size_explicit returns the
    # validated value or None — None keeps convo's lower 30-char floor
    # (more permissive than the 50-char project default, so short
    # exchanges aren't dropped). Using the validated accessor (not raw
    # _file_config) means a garbage/negative/bool config value can't
    # TypeError the length gate below or ValueError out of
    # chunk_exchanges and abort convo ingest.
    explicit_min = palace_config.min_chunk_size_explicit
    cfg_min_chunk_size = explicit_min if explicit_min is not None else MIN_CHUNK_SIZE

    convo_path = Path(convo_dir).expanduser().resolve()
    wing = _resolve_wing(convo_path, wing)

    files = scan_convos(convo_dir, include_subagents=include_subagents)

    print(f"\n{'=' * 55}")
    print("  MemPalace Mine -- Conversations")
    print(f"{'=' * 55}")
    print(f"  Wing:    {wing}")
    print(f"  Source:  {convo_path}")
    limit_suffix = f" (limit: {limit} new)" if limit > 0 else ""
    print(f"  Files:   {len(files)}{limit_suffix}")
    print(f"  Palace:  {palace_path}")
    if dry_run:
        print("  DRY RUN -- nothing will be filed")
    print(f"{'-' * 55}\n")

    collection = _open_convo_collection(
        palace_path,
        dry_run=dry_run,
    )

    if _maybe_fast_path(
        collection,
        files,
        palace_path,
        wing,
        agent,
        extract_mode,
        dry_run,
        cfg_chunk_size,
        cfg_min_chunk_size,
        palace_config,
    ):
        return

    # Bulk pre-fetch already-mined source_file -> stored mtime in one
    # paginated pass instead of `len(files)` separate WHERE-source_file
    # queries. On a 150k-drawer palace each per-file query costs ~2s, so a
    # 2000-file sweep used to spend >1h just deciding to skip.
    # prefetch_mined_set() does the same decisions in a single scan; loop
    # body becomes an O(1) dict lookup + a cheap local mtime comparison.
    # content_hash -> source_file for transcripts already filed. Repeated
    # exports from Claude/ChatGPT commonly land under a new filename each
    # run even when the conversation itself is unchanged, so the
    # source_file-keyed skip above ("mined_mtimes") never recognizes them —
    # this catches the same conversation reappearing at a new path.
    # Local patch (step 3): a one-log save scopes both to that log, see _mine_prefetch.
    one_log, mined_mtimes, mined_content_hashes = _mine_prefetch(collection, files, extract_mode)

    total_drawers = 0
    files_mined = 0
    files_skipped = 0
    files_processed = 0
    room_counts = defaultdict(int)

    for i, filepath in enumerate(files, 1):
        files_processed = i
        source_file = str(filepath)

        # Skip only if already filed at the current NORMALIZE_VERSION AND
        # unchanged on disk since. Transcripts are NOT assumed immutable:
        # a Claude Code session keeps appending to the same file while
        # active, and /compact or /clear can rewrite one in place -- so
        # "we've seen this source_file before" alone is not sufficient.
        # Falling through re-mines: _file_chunks_locked purges this
        # source_file's stale drawers before inserting fresh ones, so this
        # never leaves duplicates behind.
        if _is_unchanged_since_last_mine(source_file, mined_mtimes) or _bookmark_unchanged(
            collection, palace_path, source_file, extract_mode, wing, dry_run
        ):
            files_skipped += 1
            continue

        if not _is_regular_source_file(filepath, Path(convo_dir).expanduser().resolve()):
            files_skipped += 1
            continue

        bm_snap = _bookmark_snapshot(filepath, len(files), palace_path, extract_mode, wing, dry_run)

        conversations = _normalize_convo_conversations(
            filepath,
            source_file,
            cfg_min_chunk_size,
            collection,
            wing,
            agent,
            extract_mode,
            dry_run,
        )
        if conversations is None:
            continue

        # Hash and dedup per conversation, not per file: a Claude/ChatGPT
        # privacy export bundles every conversation into one file, so a
        # re-export that adds one new conversation changes the whole-file
        # hash and would hide the conversations that didn't change if we
        # hashed the joined bundle. Conversations whose hash is already
        # filed under a different source_file in this wing are dropped;
        # the rest are re-joined and mined as usual.
        if mined_content_hashes is None:
            mined_content_hashes = _content_hashes_of(
                collection, conversations, extract_mode, source_file
            )
        new_items, duplicates = _split_new_and_duplicate_conversations(
            conversations, wing, source_file, mined_content_hashes
        )
        if not new_items:
            if not dry_run:
                _register_file(collection, source_file, wing, agent, extract_mode)
            dup_source = duplicates[0][1]
            print(
                f"  = [{i:4}/{len(files)}] {filepath.name[:50]:50} "
                f"duplicate of {Path(dup_source).name}"
            )
            files_skipped += 1
            continue

        content = "\n\n".join(text for _, text in new_items)
        content_hash = ",".join(h for h, _ in new_items)

        # Chunk — either exchange pairs or general extraction
        if extract_mode == "general":
            from .general_extractor import extract_memories

            chunks = extract_memories(content, chunk_size=cfg_chunk_size)
            # Each chunk already has memory_type; use it as the room name
        else:
            chunks = chunk_exchanges(
                content,
                chunk_size=cfg_chunk_size,
                min_chunk_size=cfg_min_chunk_size,
            )

        if not chunks:
            if not dry_run:
                _register_file(collection, source_file, wing, agent, extract_mode)
            continue

        # Detect room from content (general mode uses memory_type instead)
        if extract_mode != "general":
            room = detect_convo_room(content)
        else:
            room = None  # set per-chunk below

        if dry_run:
            if extract_mode == "general":
                from collections import Counter

                type_counts = Counter(c.get("memory_type", "general") for c in chunks)
                types_str = ", ".join(f"{t}:{n}" for t, n in type_counts.most_common())
                print(f"    [DRY RUN] {filepath.name} -> {len(chunks)} memories ({types_str})")
            else:
                print(f"    [DRY RUN] {filepath.name} -> room:{room} ({len(chunks)} drawers)")
            total_drawers += len(chunks)
            # Track room counts
            if extract_mode == "general":
                for c in chunks:
                    room_counts[c.get("memory_type", "general")] += 1
            else:
                room_counts[room] += 1
            files_mined += 1
            if limit > 0 and files_mined >= limit:
                break
            continue

        if extract_mode != "general":
            room_counts[room] += 1

        # Lock + purge stale + file fresh chunks. Lock serializes concurrent
        # agents; purge removes pre-v2 drawers so the schema bump applies.
        drawers_added, room_delta, skipped = _file_chunks_locked(
            collection,
            source_file,
            chunks,
            wing,
            room,
            agent,
            extract_mode,
            authored_at=_extract_authored_at(filepath),
            content_hash=content_hash,
        )
        if skipped:
            files_skipped += 1
            continue
        _write_bookmark_after_full(
            single_conversation=not duplicates and len(new_items) == 1,
            palace_path=palace_path,
            source_file=source_file,
            wing=wing,
            extract_mode=extract_mode,
            room=room,
            content=content,
            chunks=chunks,
            snap=bm_snap,
            chunk_size=cfg_chunk_size,
            min_chunk_size=cfg_min_chunk_size,
        )
        for r, n in room_delta.items():
            room_counts[r] += n

        for h, _ in new_items:
            mined_content_hashes[(wing, h)] = source_file
        total_drawers += drawers_added
        files_mined += 1
        print(f"  + [{i:4}/{len(files)}] {filepath.name[:50]:50} +{drawers_added}")
        if limit > 0 and files_mined >= limit:
            break

    if not dry_run:
        _post_mine_steps(palace_path, wing, collection, total_drawers, palace_config, one_log)

    _print_convo_summary(files_processed, files_skipped, total_drawers, room_counts)


def _print_convo_summary(files_processed, files_skipped, total_drawers, room_counts) -> None:
    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"  Files processed: {files_processed - files_skipped}")
    print(f"  Files skipped (already filed): {files_skipped}")
    print(f"  Drawers filed: {total_drawers}")
    if room_counts:
        print("\n  By room:")
        for room, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {room:20} {count} files")
    print('\n  Next: mempalace search "what you\'re looking for"')
    print(f"{'=' * 55}\n")


# =============================================================================
# BOOKMARKED RE-MINE — local patch, 3D-Stories/claude-skills tools/mempalace-hub
# =============================================================================
#
# A live Claude Code session appends to its JSONL log, and the Stop hook
# re-mines that log every 15 exchanges. The full path re-reads and re-chunks
# the whole log, restamps every drawer, scans the whole palace twice and
# recomputes the wing's hallways — 98.8 s for one 324 MB log on a 58k-drawer
# palace (measured 2026-10-03), all of it inside the hub's exclusive lock.
#
# The fast path keeps one bookmark per log: the byte offset of the JSONL line
# that started the LAST user message, and how many chunks precede it. Exchange
# chunking restarts at every user turn and nothing after a user message can
# change the transcript before it, so every chunk before the bookmark is final.
# The next mine reads only from the bookmark, re-chunks only the open exchange
# and what follows, and writes only new or changed drawers. Whatever it cannot
# prove unchanged declines to the full path, which then writes a fresh
# bookmark. A bookmark is only ever written after the full path's own chunks
# agree with this parser's, chunk for chunk. Before every use it must hold that:
#   - a sha256 of ALL the log's bytes up to the bookmarked size still matches;
#   - the palace holds exactly the drawer ids the bookmark expects, no more;
#   - the log has no BOM, no bare CR and only valid UTF-8 (so this byte parser
#     and the full path's universal-newline text reader see the same lines);
#   - spellcheck is off (it rewrites old user turns from a mutable registry);
#     checked before a bookmark is made or resumed. A directory sweep's skip does
#     not check it: the full path skips an unchanged log the same way.
# The bookmark is written only after every write and delete succeeded. An
# interrupted save leaves the OLD bookmark. If it added drawers or failed to
# delete some, the inventory check sees the changed id set and the next save
# takes the full path. If it only rewrote drawers inside the old range, the next
# save re-cuts everything after the old bookmark and rewrites any that differ.
#
# Unchanged drawers keep their old ``source_mtime``/``chunk_total``; the
# bookmark is the completion record for a fast-path log. A bookmark save blanks
# chunk 0's ``content_hash`` (it would advertise an old version of the
# conversation to cross-file dedup); the next full-path mine restores it.
# Bounded limit: until then, another file holding exactly this conversation is
# filed again instead of being skipped as a duplicate.
#
# Kill switch: MEMPALACE_CONVO_BOOKMARKS=0. Post-mine throttle (hallways and
# the SQLite quick_check, on bookmark saves and, since step 3, on every one-log
# save): MEMPALACE_FASTPATH_POSTMINE_SECS (default 1800; 0 = run them on every
# save, as before).
#
# ONE-LOG SAVE — local patch step 3 (2026-10-03). A log the bookmark cannot
# resume (a one-prompt worker log has no user turn to bookmark) still took the
# full path, which read every row of the palace twice (mined set, conversation
# hashes) and rebuilt the wing's hallways: 73 of 81.5 s for a 1.9 MB log on an
# 84k-row palace. A mine of exactly one file now runs the same upstream prefetch
# functions over a _ScopedView: the mined set over that file's rows, and, when the
# file holds ONE conversation (every Claude Code log does), the hashes over the
# file's own rows plus the rows whose content_hash equals that conversation's hash.
# A file holding several conversations keeps the whole-palace hash lookup.
# Known limit: the equality lookup cannot see the hash inside ANOTHER file's
# comma-joined list (a multi-conversation export), so the log is then filed again
# instead of skipped as a duplicate. Duplication, never loss: a one-conversation
# file judged a duplicate is only registered, and none of its drawers is deleted.
# A throttled hallway rebuild that was skipped is marked pending, and the first
# save after the window runs it, even a save that files nothing.


class _ScopedView:
    """A collection whose get() returns only the rows matching one ``where`` filter.

    The upstream prefetches page through ``collection.get(limit, offset, include)``
    while ``offset < collection.count()`` and stop at the first empty page, so over
    this view they decide exactly what their whole-palace scan decides for the rows
    the filter keeps. ``count()`` stays the whole-palace count: an upper bound."""

    def __init__(self, collection, where: dict):
        self._collection = collection
        self._where = where

    def count(self) -> int:
        return self._collection.count()

    def get(self, **kwargs):
        if kwargs.get("where") is not None or kwargs.get("ids") is not None:
            raise TypeError("_ScopedView.get takes no where or ids of its own")
        kwargs["where"] = self._where
        return self._collection.get(**kwargs)


class _FetchWatch:
    """A collection whose get() remembers whether it raised, then re-raises."""

    def __init__(self, collection):
        self._collection = collection
        self.failed = False

    def get(self, **kwargs):
        try:
            return self._collection.get(**kwargs)
        except Exception:
            self.failed = True
            raise


def _mine_prefetch(collection, files, extract_mode: str):
    """-> (one_log, mined_mtimes, mined_content_hashes). A mine of exactly one file
    scopes the mined set to that file's rows and returns None for the hashes: the
    caller fills them after normalizing, from the file's own conversations."""
    if collection is None:
        return False, {}, {}
    if len(files) == 1:
        scoped = _ScopedView(collection, {"source_file": str(files[0])})
        return True, prefetch_mined_set(scoped, extract_mode=extract_mode), None
    return (
        False,
        prefetch_mined_set(collection, extract_mode=extract_mode),
        prefetch_content_hashes(collection, extract_mode=extract_mode),
    )


def _post_mine_steps(palace_path, wing, collection, drawers, palace_config, throttled):
    """Hallways, then the SQLite quick_check. Throttled (bookmark and one-log saves):
    each at most once per MEMPALACE_FASTPATH_POSTMINE_SECS.

    Hallways go before the FTS5 validation: the latter opens a direct sqlite
    connection to the Chroma DB, which can invalidate the live collection handle on
    some Chroma builds and make the hallway fetch fail."""
    _hallway_step(palace_path, wing, collection, drawers, palace_config, throttled)
    if not throttled or _postmine_due(palace_path, "quick_check"):
        _validate_palace_fts5_after_mine(palace_path)
        _postmine_mark(palace_path, "quick_check")


def _hallway_step(palace_path, wing, collection, drawers, palace_config, throttled):
    """Rebuild the wing's hallways when this save filed drawers or, throttled, when an
    earlier save left the rebuild pending. A throttled save inside the window marks
    the rebuild pending instead; a failed rebuild (it raised, or its collection fetch
    failed) stays pending."""
    stamp = _hallway_stamp(wing)
    pending = _bookmark_dir(palace_path) / f"{stamp}-pending.stamp"
    if drawers <= 0 and (not throttled or not pending.exists()):
        return
    if throttled and not _postmine_due(palace_path, stamp):
        if drawers > 0:
            _postmine_mark(palace_path, f"{stamp}-pending")
        return
    # Pending work counts as one drawer: the wrapper skips a rebuild of none.
    ok = _compute_hallways_for_wing_safe(wing, collection, max(drawers, 1), config=palace_config)
    _postmine_mark(palace_path, stamp)
    if ok is False:
        _postmine_mark(palace_path, f"{stamp}-pending")
        return
    try:
        pending.unlink()
    except OSError:
        pass


def _conversation_hash(conversation: str) -> str:
    return hashlib.sha256(conversation.strip().encode("utf-8")).hexdigest()


def _content_hashes_of(collection, conversations: list, extract_mode: str, source_file: str):
    """prefetch_content_hashes() for a one-file mine. One conversation: this file's own
    rows (an older version of it may have stored a comma-joined hash) and the rows
    whose content_hash equals its hash, in palace order, so the first source seen is
    the whole-palace one except for the known limit above. Several conversations: the
    whole palace, as before; one filter value per conversation could pass SQLite's
    variable limit."""
    if len(conversations) != 1:
        return prefetch_content_hashes(collection, extract_mode=extract_mode)
    where = {
        "$or": [
            {"content_hash": _conversation_hash(conversations[0])},
            {"source_file": source_file},
        ]
    }
    return prefetch_content_hashes(_ScopedView(collection, where), extract_mode=extract_mode)


_BOOKMARK_VERSION = 2  # 2: whole-prefix sha256, palace inventory check
# Bumped whenever chunk_exchanges() can cut the same text differently, so a
# bookmark written under the old rule is never resumed under the new one.
_CHUNK_RECIPE = "exchange-no-dash-break-1"
_HASH_BLOCK = 8 * 1024 * 1024
_BOOKMARK_MAX_FILE = 500 * 1024 * 1024  # normalize._read_transcript_file's cap
_POSTMINE_THROTTLE_ENV = "MEMPALACE_FASTPATH_POSTMINE_SECS"
_POSTMINE_THROTTLE_DEFAULT = 1800.0


class _FastPathDecline(Exception):
    """The fast path cannot prove it matches the full path; use the full path."""


def _bookmarks_enabled() -> bool:
    flag = os.environ.get("MEMPALACE_CONVO_BOOKMARKS", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def _bookmark_dir(palace_path: str) -> Path:
    from .config import _default_config_dir

    palace = os.path.abspath(os.path.expanduser(palace_path))
    return (
        _default_config_dir() / "convo_bookmarks" / hashlib.sha256(palace.encode()).hexdigest()[:16]
    )


def _bookmark_path(palace_path: str, source_file: str, extract_mode: str, wing: str) -> Path:
    key = hashlib.sha256(f"{source_file}\0{extract_mode}\0{wing}".encode()).hexdigest()[:32]
    return _bookmark_dir(palace_path) / f"{key}.json"


_BOOKMARK_FIELDS = {
    "v": int,
    "source_file": str,
    "wing": str,
    "extract_mode": str,
    "room": str,
    "normalize_version": int,
    "chunk_recipe": str,
    "chunk_size": int,
    "min_chunk_size": int,
    "dev": int,
    "ino": int,
    "size": int,
    "mtime": (int, float),
    "resume_offset": int,
    "base": int,
    "total": int,
    "prev_hash": str,
    "prefix_sha": str,
    "tool_map": dict,
}


_HEX64 = frozenset("0123456789abcdef")


def _is_hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _bookmark_valid(bm) -> bool:
    """Every field present with its type and in range; anything else is no bookmark."""
    if not isinstance(bm, dict) or bm.get("v") != _BOOKMARK_VERSION:
        return False
    for key, kind in _BOOKMARK_FIELDS.items():
        value = bm.get(key)
        if isinstance(value, bool) or not isinstance(value, kind):
            return False
    if not isinstance(bm.get("id_recipe"), type(ID_RECIPE)) or isinstance(bm["id_recipe"], bool):
        return False
    if "prefix_ts" not in bm or not (bm["prefix_ts"] is None or isinstance(bm["prefix_ts"], str)):
        return False
    if not all(bm[k] for k in ("source_file", "wing", "extract_mode", "room", "chunk_recipe")):
        return False
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in bm["tool_map"].items()):
        return False
    return (
        math.isfinite(bm["mtime"])
        and bm["dev"] >= 0
        and bm["ino"] >= 0
        and bm["normalize_version"] >= 1
        and bm["chunk_size"] > 0
        and bm["min_chunk_size"] >= 0
        and 0 < bm["resume_offset"] <= bm["size"] <= _BOOKMARK_MAX_FILE
        and 1 <= bm["base"] < bm["total"] <= bm["size"]  # every chunk takes at least one byte
        and _is_hex64(bm["prefix_sha"])
        and _is_hex64(bm["prev_hash"])
    )


def _load_bookmark(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            bm = json.load(fh)
    except (OSError, ValueError):
        return None
    try:
        valid = _bookmark_valid(bm)
    except Exception:  # e.g. OverflowError on mtime = 10**400: unjudgeable is invalid
        valid = False
    return bm if valid else None


def _save_bookmark(path: Path, bm: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(bm, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _drop_bookmark(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _hash_through(fh, old_size: int, size: int, keep_from: int) -> tuple:
    """One pass over bytes [0, size): sha256 of the first ``old_size`` bytes, sha256 of
    all ``size`` bytes, and a copy of bytes [keep_from, size)."""
    digest = hashlib.sha256()
    fh.seek(0)
    pos, old_digest, kept = 0, None, []
    while pos < size:
        n = min(_HASH_BLOCK, size - pos)
        if pos < old_size < pos + n:
            n = old_size - pos
        block = fh.read(n)
        if len(block) != n:
            raise _FastPathDecline("the log changed while it was read")
        digest.update(block)
        if pos + n > keep_from:
            kept.append(block[max(0, keep_from - pos) :])
        pos += n
        if pos == old_size:
            old_digest = digest.hexdigest()
    return old_digest, digest.hexdigest(), b"".join(kept)


def _not_plain(data: bytes) -> Optional[str]:
    """Why this byte parser could read ``data`` differently from the full path
    (``utf-8-sig``, ``errors="replace"``, universal newlines) and from
    ``_extract_authored_at`` (``utf-8``, ``errors="ignore"``); None when it cannot.
    Valid UTF-8 everywhere, the unfinished last line included, makes the two error
    modes and this parser's strict decoding identical."""
    if data.startswith(b"\xef\xbb\xbf"):
        return "the log starts with a byte-order mark"
    if data.count(b"\r") != data.count(b"\r\n"):
        return "the log has a bare carriage return"
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        for start in range(0, len(data), _HASH_BLOCK):
            decoder.decode(data[start : start + _HASH_BLOCK])
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return "the log has invalid UTF-8"
    return None


def _spellcheck_active() -> bool:
    try:
        from .spellcheck import _get_speller

        return _get_speller() is not None
    except Exception:
        return True  # unknown: assume it may rewrite old turns


def _inventory_matches(
    collection, bm: dict, source_file: str, wing: str, extract_mode: str
) -> bool:
    """The palace holds exactly the drawers the bookmark expects for this log."""
    try:
        got = collection.get(
            where={
                "$and": [
                    {"source_file": source_file},
                    {"wing": wing},
                    {"ingest_mode": "convos"},
                    {"extract_mode": extract_mode},
                ]
            },
            include=[],
        )
    except Exception:
        return False
    want = {
        make_convo_drawer_id(wing, bm["room"], source_file, extract_mode, i)
        for i in range(bm["total"])
    }
    return set(got.get("ids") or []) == want


def _cc_parse(data: bytes, base_offset: int = 0, seed_tools: Optional[dict] = None) -> dict:
    """Parse Claude Code JSONL bytes exactly as ``normalize._try_claude_code_jsonl``
    does, keeping the byte offset of the line that started each message.

    Also mirrors ``_extract_authored_at`` (newest top-level ``timestamp``) and
    records, at the LAST user message: its index, offset, the tool_use id ->
    name map as it stood just before that line, and the newest timestamp
    strictly before it. The map must be the WHOLE map, not only open calls:
    Claude Code re-appends tool_result lines for calls made long before
    (measured 2026-10-03 on a live log), and the name picks their formatting.
    ``seed_tools`` is that map for the lines before ``base_offset``.
    """
    from .normalize import _extract_content, strip_noise

    messages: list = []
    offsets: list = []
    tool_use_map = dict(seed_tools or {})
    tool_events: list = []  # (line_offset, tool_id, name), in file order
    last_user = None
    max_ts = None
    pos = 3 if base_offset == 0 and data.startswith(b"\xef\xbb\xbf") else 0
    n = len(data)
    while pos < n:
        nl = data.find(b"\n", pos)
        end = n if nl < 0 else nl
        line_offset = base_offset + pos
        line = data[pos:end].decode("utf-8", errors="replace").strip()
        pos = end + 1
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        ts_before = max_ts
        ts = entry.get("timestamp") if isinstance(entry, dict) else None
        if isinstance(ts, str) and (max_ts is None or ts > max_ts):
            max_ts = ts
        if not isinstance(entry, dict):
            continue
        msg_type = entry.get("type", "")
        message = entry.get("message", {})
        if not isinstance(message, dict):
            continue
        msg_content = message.get("content", "")

        if msg_type == "assistant" and isinstance(msg_content, list):
            for block in msg_content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id", "")
                    if tool_id:
                        tool_use_map[tool_id] = block.get("name", "Unknown")
                        tool_events.append((line_offset, tool_id, tool_use_map[tool_id]))

        if msg_type in ("human", "user"):
            is_tool_only = isinstance(msg_content, list) and all(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in msg_content
            )
            text = _extract_content(msg_content, tool_use_map=tool_use_map)
            if text:
                text = strip_noise(text)
            if text:
                if is_tool_only and messages and messages[-1][0] == "assistant":
                    messages[-1] = (messages[-1][0], messages[-1][1] + "\n" + text)
                elif not is_tool_only:
                    messages.append(("user", text))
                    offsets.append(line_offset)
                    last_user = {
                        "index": len(messages) - 1,
                        "offset": line_offset,
                        "ts_before": ts_before,
                    }
        elif msg_type == "assistant":
            text = _extract_content(msg_content, tool_use_map=tool_use_map)
            if text:
                text = strip_noise(text)
            if text:
                if messages and messages[-1][0] == "assistant":
                    messages[-1] = (messages[-1][0], messages[-1][1] + "\n" + text)
                else:
                    messages.append(("assistant", text))
                    offsets.append(line_offset)
    if last_user is not None:
        tools_at = dict(seed_tools or {})
        for off, tool_id, name in tool_events:
            if off >= last_user["offset"]:
                break
            tools_at[tool_id] = name
        last_user["tool_map"] = tools_at
    return {"messages": messages, "offsets": offsets, "last_user": last_user, "max_ts": max_ts}


def _max_ts(*values):
    present = [v for v in values if isinstance(v, str)]
    return max(present) if present else None


def _exchange_chunks(messages: list, chunk_size: int, min_chunk_size: int) -> tuple:
    """Transcript text and exchange chunks for ``messages`` (starting at a user turn)."""
    from .normalize import _messages_to_transcript

    text = _messages_to_transcript(messages)
    return text, _chunk_by_exchange(text.split("\n"), chunk_size, min_chunk_size)


def _bookmark_from_full(
    data: bytes, full_chunks: list, chunk_size: int, min_chunk_size: int
) -> Optional[dict]:
    """Bookmark fields for a log the full path just chunked into ``full_chunks``.

    ``data`` is a byte snapshot of the log taken BEFORE the full path read it,
    so the log may have grown since; only chunks before the snapshot's last user
    turn are compared, and those cannot change with later appends. Returns None
    unless they match the full path's chunks exactly.
    """
    parsed = _cc_parse(data)
    last = parsed["last_user"]
    if last is None or last["index"] < 1:
        return None
    head_text, head = _exchange_chunks(
        parsed["messages"][: last["index"]], chunk_size, min_chunk_size
    )
    base = len(head)
    if base < 1 or len(full_chunks) <= base:
        return None
    if [c["content"] for c in full_chunks[:base]] != [c["content"] for c in head]:
        return None
    tail_text, _tail = _exchange_chunks(
        parsed["messages"][last["index"] :], chunk_size, min_chunk_size
    )
    # chunk_exchanges() chunks by exchange only from 3 quote lines up, and by
    # paragraph below that. The count only grows as the log grows, so a log
    # already at 3 can never flip back; below 3 it still can, so no bookmark.
    quote_lines = sum(
        1 for line in (head_text + "\n" + tail_text).split("\n") if line.strip().startswith(">")
    )
    if quote_lines < 3:
        return None
    first = tail_text.split("\n", 1)[0].strip()
    if not full_chunks[base]["content"].startswith(first[:chunk_size]):
        return None
    return {
        "resume_offset": last["offset"],
        "base": base,
        "tool_map": last["tool_map"],
        "prefix_ts": last["ts_before"],
        "total": len(full_chunks),
        "prev_hash": hashlib.sha256(full_chunks[base - 1]["content"].encode("utf-8")).hexdigest(),
    }


def _fast_tail(tail: bytes, bm: dict, chunk_size: int, min_chunk_size: int) -> dict:
    """Chunks from the bookmark onward, and the bookmark to store after writing them.

    ``tail`` holds the log's bytes from ``bm['resume_offset']`` to the size the
    caller stat'ed. Raises ``_FastPathDecline`` when the tail does not start with
    the bookmarked user turn, or when it cannot be chunked identically.
    """
    resume = bm["resume_offset"]
    parsed = _cc_parse(tail, base_offset=resume, seed_tools=bm["tool_map"])
    msgs = parsed["messages"]
    if not msgs or msgs[0][0] != "user" or parsed["offsets"][0] != resume:
        raise _FastPathDecline("the log no longer starts a user turn at the bookmark")
    _text, chunks = _exchange_chunks(msgs, chunk_size, min_chunk_size)
    if not chunks:
        raise _FastPathDecline("no chunks after the bookmark")
    base = bm["base"]
    out = [{"content": c["content"], "chunk_index": base + j} for j, c in enumerate(chunks)]
    last = parsed["last_user"]
    new = {
        "resume_offset": resume,
        "base": base,
        "tool_map": bm["tool_map"],
        "prefix_ts": bm.get("prefix_ts"),
        "prev_hash": bm["prev_hash"],
    }
    if last["index"] > 0:
        _t2, chunks2 = _exchange_chunks(msgs[last["index"] :], chunk_size, min_chunk_size)
        k = len(chunks) - len(chunks2)
        if k < 1 or [c["content"] for c in chunks[k:]] != [c["content"] for c in chunks2]:
            raise _FastPathDecline("the newest user turn does not start a chunk boundary")
        new.update(
            resume_offset=last["offset"],
            base=base + k,
            tool_map=last["tool_map"],
            prefix_ts=_max_ts(bm.get("prefix_ts"), last["ts_before"]),
            prev_hash=hashlib.sha256(chunks[k - 1]["content"].encode("utf-8")).hexdigest(),
        )
    new["total"] = base + len(chunks)
    return {
        "chunks": out,
        "bookmark": new,
        "authored_at": _max_ts(bm.get("prefix_ts"), parsed["max_ts"]),
    }


def _bookmark_unchanged(
    collection, palace_path: str, source_file: str, extract_mode: str, wing: str, dry_run: bool
) -> bool:
    """True when a valid bookmark records this exact file state, the log's bytes still
    hash to it, and the palace holds exactly its drawers (a directory sweep can then
    skip the log)."""
    if dry_run or collection is None or not _bookmarks_enabled():
        return False
    bm = _load_bookmark(_bookmark_path(palace_path, source_file, extract_mode, wing))
    if bm is None or bm["source_file"] != source_file:
        return False
    try:
        fd = os.open(source_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as fh:
            st = os.fstat(fh.fileno())
            if (st.st_dev, st.st_ino, st.st_size, st.st_mtime) != (
                bm["dev"],
                bm["ino"],
                bm["size"],
                bm["mtime"],
            ):
                return False
            old_sha, _, _ = _hash_through(fh, bm["size"], bm["size"], bm["size"])
    except (OSError, _FastPathDecline):
        return False
    return old_sha == bm["prefix_sha"] and _inventory_matches(
        collection, bm, source_file, wing, extract_mode
    )


def _bookmark_eligible(filepath: Path, n_files: int, extract_mode: str, dry_run: bool) -> bool:
    """Only single-file Claude Code JSONL mines (the Stop/SessionEnd hook shape), and
    only while spellcheck is off."""
    return (
        _bookmarks_enabled()
        and not dry_run
        and n_files == 1
        and extract_mode == "exchange"
        and filepath.suffix == ".jsonl"
        and not _spellcheck_active()
    )


def _write_bookmark_after_full(
    *,
    single_conversation: bool,
    palace_path: str,
    source_file: str,
    wing: str,
    extract_mode: str,
    room: str,
    content: str,
    chunks: list,
    snap,
    chunk_size: int,
    min_chunk_size: int,
) -> bool:
    """Record a bookmark for a log the full path just wrote. Best effort: any doubt writes nothing."""
    if snap is None or not single_conversation:
        return False
    path = _bookmark_path(palace_path, source_file, extract_mode, wing)
    try:
        if len(content) < 3000 or snap.st_size > _BOOKMARK_MAX_FILE:
            return False  # detect_convo_room() reads content[:3000]; below that the room can still change
        fd = os.open(source_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as fh:
            st = os.fstat(fh.fileno())
            if (st.st_dev, st.st_ino) != (snap.st_dev, snap.st_ino) or st.st_size < snap.st_size:
                return False
            data = fh.read(snap.st_size)
        if len(data) != snap.st_size or _not_plain(data) is not None:
            return False
        fields = _bookmark_from_full(data, chunks, chunk_size, min_chunk_size)
        if fields is None:
            return False
        fields.update(
            v=_BOOKMARK_VERSION,
            prefix_sha=hashlib.sha256(data).hexdigest(),
            source_file=source_file,
            wing=wing,
            extract_mode=extract_mode,
            room=room,
            normalize_version=NORMALIZE_VERSION,
            id_recipe=ID_RECIPE,
            chunk_recipe=_CHUNK_RECIPE,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            dev=snap.st_dev,
            ino=snap.st_ino,
            size=snap.st_size,
            mtime=snap.st_mtime,
            written_at=datetime.now().isoformat(),
            origin="full",
        )
        if not _bookmark_valid(fields):
            return False
        _save_bookmark(path, fields)
        return True
    except Exception:
        logger.debug("bookmark not written for %s", source_file, exc_info=True)
        return False


def _blank_stale_conversation_hash(collection, wing, room, source_file, extract_mode) -> None:
    """chunk 0's ``content_hash`` names the conversation as the last FULL mine saw it;
    after a bookmark save that version is gone, so stop advertising it to dedup."""
    first = make_convo_drawer_id(wing, room, source_file, extract_mode, 0)
    got = collection.get(ids=[first], include=["metadatas"])
    metas = got.get("metadatas") or []
    if metas and metas[0] and metas[0].get("content_hash"):
        meta = dict(metas[0])
        meta["content_hash"] = ""
        collection.update(ids=[first], metadatas=[meta])


def _fast_path_locked(
    collection, source_file, wing, agent, extract_mode, bm, chunk_size, min_chunk_size
):
    fd = os.open(source_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise _FastPathDecline("not a regular file")
        if (st.st_dev, st.st_ino) != (bm["dev"], bm["ino"]):
            raise _FastPathDecline("the log was replaced")
        size = st.st_size
        if size < bm["size"]:
            raise _FastPathDecline("the log shrank")
        if size > _BOOKMARK_MAX_FILE:
            raise _FastPathDecline("the log passed the 500 MB cap")
        old_sha, new_sha, tail = _hash_through(fh, bm["size"], size, bm["resume_offset"])
    if old_sha != bm["prefix_sha"]:
        raise _FastPathDecline("bytes before the bookmark changed")
    if not _inventory_matches(collection, bm, source_file, wing, extract_mode):
        raise _FastPathDecline("the palace does not hold exactly the bookmarked drawers")
    if size == bm["size"] and st.st_mtime == bm["mtime"]:
        return {"status": "unchanged", "drawers": 0, "read": 0, "size": size}
    reason = _not_plain(tail)
    if reason:
        raise _FastPathDecline(reason)
    result = _fast_tail(tail, bm, chunk_size, min_chunk_size)
    new_bm = dict(bm)
    new_bm.update(result["bookmark"])

    room = bm["room"]
    base, new_total = bm["base"], new_bm["total"]
    hi = max(bm["total"], new_total)
    ids = {
        i: make_convo_drawer_id(wing, room, source_file, extract_mode, i)
        for i in range(base - 1, hi)
    }
    existing: dict = {}
    id_list = list(ids.values())
    for start in range(0, len(id_list), 1000):
        got = collection.get(ids=id_list[start : start + 1000], include=["metadatas"])
        for drawer_id, meta in zip(got.get("ids") or [], got.get("metadatas") or []):
            existing[drawer_id] = meta or {}
    prev = existing.get(ids[base - 1])
    if prev is None or prev.get("chunk_hash") != bm["prev_hash"]:
        raise _FastPathDecline("the palace no longer holds the chunk before the bookmark")

    filed_at = datetime.now().isoformat()
    to_upsert = []
    for chunk in result["chunks"]:
        drawer_id = (
            ids[chunk["chunk_index"]]
            if chunk["chunk_index"] in ids
            else make_convo_drawer_id(wing, room, source_file, extract_mode, chunk["chunk_index"])
        )
        chunk_hash = hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()
        p = existing.get(drawer_id)
        if (
            p is not None
            and p.get("chunk_hash") == chunk_hash
            and p.get("normalize_version", 1) >= NORMALIZE_VERSION
        ):
            continue
        meta = _convo_drawer_meta(
            chunk,
            room,
            wing=wing,
            source_file=source_file,
            agent=agent,
            filed_at=filed_at,
            authored_at=result["authored_at"],
            extract_mode=extract_mode,
            chunk_total=new_total,
            chunk_hash=chunk_hash,
            source_mtime=st.st_mtime,
            content_hash=None,
        )
        to_upsert.append((drawer_id, chunk["content"], meta))
    stale = [ids[i] for i in range(new_total, hi) if ids[i] in existing]

    drawers = _upsert_convo_drawers(collection, to_upsert)
    if stale:
        try:
            collection.delete(ids=stale)
        except Exception as exc:
            raise _FastPathDecline(f"could not delete {len(stale)} stale drawers: {exc}") from exc
    _blank_stale_conversation_hash(collection, wing, room, source_file, extract_mode)
    new_bm.update(
        prefix_sha=new_sha,
        size=size,
        mtime=st.st_mtime,
        written_at=datetime.now().isoformat(),
        origin="bookmark",
    )
    return {
        "status": "mined",
        "drawers": drawers,
        "read": len(tail),
        "size": size,
        "bookmark": new_bm,
    }


def _try_fast_path(
    collection, filepath: Path, palace_path, wing, agent, extract_mode, chunk_size, min_chunk_size
):
    """Bookmark re-mine of one log. None means "use the full path"."""
    source_file = str(filepath)
    path = _bookmark_path(palace_path, source_file, extract_mode, wing)
    bm = _load_bookmark(path)
    if bm is None:
        return None
    expected = {
        "source_file": source_file,
        "wing": wing,
        "extract_mode": extract_mode,
        "normalize_version": NORMALIZE_VERSION,
        "id_recipe": ID_RECIPE,
        "chunk_size": chunk_size,
        "min_chunk_size": min_chunk_size,
        "chunk_recipe": _CHUNK_RECIPE,
    }
    if any(bm.get(k) != v for k, v in expected.items()):
        return None
    try:
        with mine_lock(source_file):
            out = _fast_path_locked(
                collection,
                source_file,
                wing,
                agent,
                extract_mode,
                bm,
                chunk_size,
                min_chunk_size,
            )
            if out["status"] == "mined":
                _save_bookmark(path, out["bookmark"])
            out["room"] = bm["room"]
            return out
    except (_FastPathDecline, OSError) as exc:
        print(f"  (bookmark not used: {exc}; full read)")
        return None


def _postmine_due(palace_path: str, name: str) -> bool:
    try:
        secs = float(os.environ.get(_POSTMINE_THROTTLE_ENV, _POSTMINE_THROTTLE_DEFAULT))
    except ValueError:
        secs = _POSTMINE_THROTTLE_DEFAULT
    if secs <= 0:
        return True
    import time

    try:
        return time.time() - (_bookmark_dir(palace_path) / f"{name}.stamp").stat().st_mtime >= secs
    except OSError:
        return True


def _postmine_mark(palace_path: str, name: str) -> None:
    try:
        d = _bookmark_dir(palace_path)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.stamp").touch()
    except OSError:
        pass


def _hallway_stamp(wing: str) -> str:
    return "hallways-" + hashlib.sha256(wing.encode()).hexdigest()[:16]


def _maybe_fast_path(
    collection,
    files,
    palace_path,
    wing,
    agent,
    extract_mode,
    dry_run,
    chunk_size,
    min_chunk_size,
    palace_config,
) -> bool:
    """Bookmark fast path (local patch): one live Claude Code log, resumed from its
    bookmark. Skips the palace-wide prefetches; hallways and the quick_check run at
    most every MEMPALACE_FASTPATH_POSTMINE_SECS. True when it handled the mine."""
    if (
        collection is None
        or not files
        or not _bookmark_eligible(Path(files[0]), len(files), extract_mode, dry_run)
    ):
        return False
    fast = _try_fast_path(
        collection,
        Path(files[0]),
        palace_path,
        wing,
        agent,
        extract_mode,
        chunk_size,
        min_chunk_size,
    )
    if fast is None:
        return False
    name = Path(files[0]).name
    if fast["status"] == "unchanged":
        print(f"  = [   1/1] {name[:50]:50} unchanged (bookmark)")
        _hallway_step(palace_path, wing, collection, 0, palace_config, throttled=True)
        _print_convo_summary(1, 1, 0, {})
        return True
    print(f"  + [   1/1] {name[:50]:50} +{fast['drawers']}")
    print(f"    bookmark: read {fast['read'] / 1e6:.1f} of {fast['size'] / 1e6:.1f} MB")
    _post_mine_steps(palace_path, wing, collection, fast["drawers"], palace_config, throttled=True)
    _print_convo_summary(1, 0, fast["drawers"], {fast["room"]: 1} if fast["drawers"] else {})
    return True


def _bookmark_snapshot(filepath: Path, n_files: int, palace_path, extract_mode, wing, dry_run):
    """Before ANY full read of a log: drop its bookmark (the full path rewrites the
    log's drawers, so the bookmark no longer describes them). Then, for an eligible
    log, stat its size so a fresh bookmark can be written after the read."""
    if dry_run:
        return None
    source_file = str(filepath)
    _drop_bookmark(_bookmark_path(palace_path, source_file, extract_mode, wing))
    if not _bookmark_eligible(filepath, n_files, extract_mode, dry_run):
        return None
    try:
        return os.stat(source_file)
    except OSError:
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convo_miner.py <convo_dir> [--palace PATH] [--limit N] [--dry-run]")
        sys.exit(1)
    from .config import MempalaceConfig

    mine_convos(sys.argv[1], palace_path=MempalaceConfig().palace_path)
