---
name: memory-compact
description: Compact and structure a Claude Code auto-memory store so MEMORY.md stays under the harness load cap (first 200 lines / 25KB load each session — content past that SILENTLY never loads) while every memory stays reachable. Splits the index into two tiers — hot MEMORY.md (laws/preferences/live-keys/in-flight/open-research) + cold MEMORY_archive.md (shipped/fixed/forensic/dated, recalled on demand) — and proves "done" with an integrity witness (check_memory.py) that re-derives both caps + the both-files bijection from disk, never from narration. Use when the operator says "compact memory", "MEMORY.md is too big / keeps truncating / shows as large", "tier the memory", "fix what loads for this project", or after any heavy pass that grew the index. Read-only except the memory files it edits; deletions need explicit sign-off.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Edit, Write, Grep, Glob, Bash, AskUserQuestion
argument-hint: "[--dir PATH] [--check]   (--check = witness only, no edits)"
output_root: none
---

# /memory-compact — keep the auto-memory index under the harness cap, provably

The recurring problem this skill packages: **`MEMORY.md` grows past the harness
load cap and silently truncates, so the bottom of the index becomes invisible to
every future session — yet a session still reports "memory looks fine."** That is
the narration-vs-ground-truth gap, inside the memory store itself. The fix is
structural (two tiers by value) and the proof is a witness (re-derive the
invariants from disk, let the exit code be the verdict).

## Finding the store

The Claude Code auto-memory dir for a project is under the Claude Code config
root, keyed by the encoded working-tree path:

```
~/.claude/projects/<encoded-cwd>/memory
```

The encoded form replaces `/` and `:` with `-` (e.g. a tree at `/work/myproj` →
`-work-myproj`). On Windows the root is `%USERPROFILE%\.claude\projects\...`. Pass
`--dir` to point the witness at the exact path. If the store is small (well under
cap) this skill is a no-op — it is here so that when the index grows past the cap,
the fix is a one-command pass, not a guess.

## The one ground-truth fact everything rests on

The Claude Code harness injects **only the first 200 lines OR 25KB of `MEMORY.md`**
(whichever comes first) at session start. Content past that threshold **does not
load** — not truncated-with-a-warning in a way the model can act on, just gone.
A sibling file (`MEMORY_archive.md`, a topic `.md`) is reachable **only** if a
session sees a link to it in the loaded part of `MEMORY.md` and chooses to `Read`
it. So:

- shrinking `MEMORY.md` under the cap is the *only* thing that fixes what loads;
- moving an entry to a sibling file keeps it reachable **iff** the index still
  links it (one hop) — otherwise you have hidden it, which is worse than truncation.

This is verified harness behavior, not a guess. If a future harness changes the
cap, change `--max-lines` / `--max-bytes` in the witness and this doc.

## The two-tier structure (split by value)

A flat index has no notion of *value* — a refuted 3-week-old spike costs the same
always-loaded line as a load-bearing law. Split by a cheap-frequent-vs-rare logic:

- **`MEMORY.md` = HOT** (always loaded): working preferences, architecture/design
  laws, live keys, **in-flight work**, **open research**, live reference cards.
- **`MEMORY_archive.md` = COLD** (recalled on demand): **shipped features** (done +
  in the code), **fixed bugs**, **dated process residue** (handoffs/notes-passes/
  release logs), **forensic-once audits**, **superseded design notes**.

**Default going forward** (also stated in the `MEMORY.md` header so every session
inherits it): a NEW memory that is shipped/fixed/forensic/dated goes to
`MEMORY_archive.md`; only hot-tier kinds get a line in `MEMORY.md`. When a hot
entry's work ships or its conclusion is absorbed into a law, MOVE it cold.

## The invariant that must never break

**The index↔disk bijection spans BOTH files**: every topic `.md` on disk is
referenced **exactly once** across `MEMORY.md` + `MEMORY_archive.md` — no orphan
files (on disk, unreferenced → invisible forever), no orphan links (referenced,
absent → a broken promise), no duplicate references. The witness checks this.

## Procedure

1. **Witness first (baseline).** Run the script — it tells you exactly what is
   broken and by how much. This is also the whole skill if invoked with `--check`.

   ```bash
   python .claude/skills/memory-compact/check_memory.py --dir "<memory dir>"
   ```
   Exit 0 = clean; non-zero = a hard invariant (cap or bijection) failed.

2. **If over the LINE cap** — the structural lever. Trim any multi-line header
   prose to a few lines (the how-to belongs in the topic file, not the
   always-loaded index), then move/delete cold entries until under 200 lines.

3. **If over the BYTE cap** — compress hooks. The detail already lives in the
   topic file, so an index line needs only Title + the one load-bearing verdict +
   the primary handle (SHA / `wf_` id / doc name). Don't shred the dense research
   lines to a lossy stub — ~200–250 chars that keeps the SHA + conclusion beats a
   100-char stub that loses the recall key.

4. **Tiering move (the main lever):** to move an entry cold, copy its index line
   **byte-identical** into the right section of `MEMORY_archive.md`, then delete it
   from `MEMORY.md`. Tiering is *where it lives*, NOT a re-summarize (re-summarize
   is step 3, a separate concern). Keep the `[MEMORY_archive.md](MEMORY_archive.md)`
   pointer in the `MEMORY.md` header so the cold tier stays one hop away.

5. **Deletion needs sign-off.** Hard-deleting an entry (index line **and** topic
   `.md` from disk) is irreversible — only do it with the operator's explicit OK,
   and present the list first (use `AskUserQuestion`). Delete only: a refuted spike
   whose idea is fully captured in a surviving law/entry, an exact duplicate, or a
   superseded "in-flight" dup of a "shipped" entry. **MERGE ≠ DELETE** for a file
   with a refuted *conclusion* but a unique *idea* — keep it, add a ⚠ caveat.

6. **Witness last (the verdict).** Re-run step 1. Exit 0 is the only "done." If a
   prior session said "fixed," do not believe it — the witness is the read-back for
   a store that lives outside any git repo.

## Mechanics that matter (if the store is MULTI-SESSION-HOT)

- If several concurrent sessions may write `MEMORY.md`, the byte count rises
  **between your edits**. You cannot hold a fixed byte target by hand against live
  writers — so **leave margin**: the witness gates at 25_000 bytes, ~600B under the
  real 25_600 cap, precisely so concurrent growth doesn't immediately re-truncate.
  Aim for a comfortable line margin too (~150 lines), not 199.
- The index **self-heals**: another session adds its own file + index line. Do NOT
  race it by adding an index line for a file you didn't create — you collide with
  their pending edit (a duplicate). Only fix an orphan that has sat un-indexed
  across multiple witness runs.
- Edit by **fresh-read → targeted Edit**; an external touch between Read and Edit
  trips the "modified since read" guard. Re-read the region if an Edit fails.
- **Dangling `[[wiki-links]]`** to deleted slugs are TOLERATED (the memory spec
  treats an unresolved `[[name]]` as a write-later marker, not an error), so the
  witness reports them as advisory and does not fail on them unless `--strict`.

## The witness (`check_memory.py`)

Pure stdlib, read-only, ships beside this file. Checks (1) the 200-line / 25KB
cap on `MEMORY.md`, (2) the both-files bijection, (3) dangling wiki-links
(advisory). `--json` for machine output, `--strict` to also fail on dangling
links, `--max-lines`/`--max-bytes` if the harness cap ever changes. Wire it into a
Stop hook to get a verdict automatically, or run it by hand after any pass.

## Caveats / honest limits

- The memory dir lives **outside any git repo**, so nothing here is git-committed;
  the files on disk are the only durability.
- The cap figures (200 / 25KB) are the harness's as of 2026-06. Re-verify with the
  claude-code-guide if behavior seems to differ, and update the witness defaults.
- This skill operates on memory files only — it never touches any code lane.
