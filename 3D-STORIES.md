# 3D-Stories fork of MemPalace

This fork carries the local fixes that the 3D-Stories MemPalace hub runs, as plain commits on a
release branch. Upstream is [MemPalace/mempalace](https://github.com/MemPalace/mempalace).

## Branches and tags

- `develop`: upstream's default branch, kept as forked. Do not commit here.
- `3d-stories/v<upstream version>`: upstream's release tag plus the local commits below. Changes
  land by pull request into this branch.
- `3ds-v<upstream version>-<n>`: the tag the hub installs. A new local commit means a new `<n>`.
  The tag must not start with `v`: upstream's `docker-publish` workflow runs on every `v*` tag.

## The local commits on `3d-stories/v3.10.0`

1. **Backport of upstream PR #2408** by @PostProtoroman, unchanged, with its own test update: a
   re-save of a grown chat log re-embeds only what changed.
2. **Bookmark saves and `---` lines:** a save of a growing Claude Code log reads only the new bytes,
   and a line starting with `---` no longer drops the text after it. The two palace-wide post-save
   steps run at most every `MEMPALACE_FASTPATH_POSTMINE_SECS` (default 1800).
3. **One-log saves:** a save of one file reads only that file's rows for its duplicate checks.
4. **`hallways.skip_wings`:** a listed wing is never walked for hallways, and its old records are
   dropped once. Default: none.
5. **Cheaper bookmark inventory check:** a bookmark save checks the palace's rows for its log by
   `source_file` alone and checks the other three labels in Python, failing closed on any answer
   it cannot read. Bookmarks over 10,000 rows keep the four-way query. It runs under the hub's
   write lock, so every other call waited behind the old 2 to 8 s query.
6. **Saves of logs over 500 MB:** a save with nothing to mine returns before any palace-wide work,
   without even opening the palace. It used to read every drawer's labels and run the quick
   check, under the hub's write lock: 29 to 44 minutes at 706,000 drawers. A log already over the
   500 MB full-read cap when a save starts keeps resuming from its bookmark. A resume reads at
   most 500 MB after the bookmark; that bounds the input, not peak memory, which can be about
   twice that or more. Such a log is never read whole, and a declined resume keeps its bookmark.
   Not fixed here: a log that crosses the cap during a full-path save can still lose its
   bookmark, as before this step.
7. **Search shows each text once:** hits with identical text collapse to the best-ranked one,
   whatever chat they came from, before the top-N cut, so the cut fills from the rest of the
   ranked pool. The kept hit gains `also_in`: how many other chats among the candidates hold the
   text, and the first three of them, counting only chats whose full source path is known. It
   covers `search_memories` (vector, union and date-window searches, and the vector-disabled
   keyword fallback), the command-line `search` (local, and the hub's forwarded output in both
   its shapes) and the MCP tool's command-line mode, and `Layer3` (reached through
   `MemoryStack.search`). Union keeps its whole vector fetch while collapse is on, so it
   can backfill. Where a search fetches only the requested count (the command line without a
   date window, `Layer3`), a list of copies comes back short instead. `MEMPALACE_SEARCH_COLLAPSE=0`
   turns it off. 30.7% of the hub's drawers were exact copies, and one search returned the same
   text in 4 of its top 10 slots.
8. **Saves skip boilerplate and same-chat repeats:** a save that has one new conversation to store
   does not store a piece that starts with one of five fixed markers (`<herdr_model_guidance`,
   `<local-command-caveat>`, `<local-command-stdout>`, `<command-message>`, `Base directory for
   this skill:`), a piece whose SHA-256 is on the reviewed list `<config dir>/boilerplate.json`,
   or a piece whose text an earlier piece of the same chat already holds. The first copy is kept.
   A save with several new conversations of one file (a privacy export) stores every piece,
   because those conversations are joined before chunking. A file's conversations the palace
   already holds are not counted, as before this step. Pieces keep their numbers, so a skipped piece only
   leaves a gap. A chat whose every piece would be skipped keeps its first piece and gets no
   bookmark. Holding the same text as another chat never makes a piece skipped; only boilerplate
   is skipped across chats.
   - The bookmark lists its skipped pieces, and its anchor is the last stored piece before it. A
     bookmark save decides the new pieces against the pieces this chat stores before the
     bookmark, asking by this chat's file, and checks their stored text before trusting them.
     It accepts at most 10,000 rows per 200 texts, asking for one more to see the cap was
     passed. Past the cap it does not decline: it stores the new pieces without the repeat
     check against earlier pieces, so a chat over the 500 MB cap, which has no full read,
     still saves. Boilerplate and repeats inside the new part are still skipped. Past the cap
     it may store repeats a full read would skip, and those stay behind later bookmarks until
     the chat is read in full or a cleanup removes them. Otherwise, under an unchanged list
     and switch, it stores what a full read would. The worst batch on a copy of the hub's
     palace (2026-10-11) was 6,069 rows.
   - Every row a save writes or refreshes carries `chunk_skipped`, 0 included (chromadb merges
     labels, so an omitted one would survive). A row a bookmark save only re-stamps, to clear an
     old conversation hash, keeps its other labels; a missing count there still means 0. The
     "already saved?" checks count stored plus skipped, and only rows with the same label set
     count together.
   - A missing or malformed list means no listed texts. `MEMPALACE_SAVE_DEDUPE=0` stores every
     piece of the saves made after it.
   - Not covered: a change to the list or the switch applies to pieces saved after it. Pieces
     already behind a bookmark keep the decision made when they were saved, until the chat is
     read in full again; making a list change re-read every chat would force full reads that
     chats over the 500 MB cap cannot take. A bookmark written before this step has no skip
     list; its earlier repeats stay until the one-time cleanup removes them.
   - Measured on a copy of the hub's palace (707,738 rows): 130,294 rows (18.4%) were repeats
     inside the same chat, and 40,925 (5.8%) were boilerplate.

Each commit carries its own tests under `tests/`. Measured numbers and the reasons for each step
are in the 3D-Stories `claude-skills` repository, `tools/mempalace-hub/README.md`.

## Install on the hub

```bash
pipx install --force 'git+https://github.com/3D-Stories/mempalace@3ds-v3.10.0-5'
```

Pin the dependencies to the versions already installed, so a reinstall cannot move ChromaDB under
the palace. Restart `mempalace-hub` afterwards, only when no mine is running.

## A new upstream release

1. Fetch upstream tags and create `3d-stories/v<new>` at the new tag.
2. Rebase the local commits onto it, dropping any that upstream now carries (check #2408 first).
3. Run the whole suite (`uv run pytest tests/ --ignore=tests/benchmarks -q`) and compare with the
   same run on the bare upstream tag.
4. Open a pull request into the new branch, then tag `3ds-v<new>-1`.

## Sending fixes upstream

Steps 2 to 8 are this fork's own work. Each can go upstream as a pull request from a branch cut off
upstream `develop`. Step 1 is already upstream as PR #2408.
