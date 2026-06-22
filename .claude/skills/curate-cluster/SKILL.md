---
name: curate-cluster
description: One curation pass over a research/doc cluster — reconcile the project's index doc (e.g. INDEX.md) against the docs and scripts actually on disk (add missing entries in the house format, fix dangling references, refresh counts/date/git-context), gitignore regenerable build artifacts, then commit ONLY the quiescent curation lane (docs/experiments/tools/index) by explicit path — never an actively-built code tree. Concurrency-safe by construction: when the repo is a live multi-session tree, the pass excludes any file-tree a peer is writing right now. Use after a burst of doc-writing, when the index drifts from disk, or on a /loop cadence to keep the cluster clean and indexed.
---

# curate-cluster — the index-and-commit gardening pass

> **What this does.** Keeps the project's index doc true to what is on disk, keeps
> build artifacts out of git, and commits the curation work — **without colliding
> with any other agents editing this repo at the same time.** When the tree is a
> live multi-session checkout (several sessions building or curating in parallel),
> a naive `git add -A && git commit` will snapshot a peer's half-written code.
> This pass is built to never do that: it stages **only by explicit path**.

The shape: **survey disk → reconcile the index → gitignore artifacts → detect
live lanes → commit only the quiescent curation lane → verify.**

---

## The one rule that matters (learned the hard way)

**Markdown mid-edit is self-healing; code mid-edit is not.** Before committing,
find every file-tree that changed in the last ~60 s. Treat them by kind:

- **Code under active construction** (any `*.go` / `*.py` / source tree a peer is
  building) → **exclude it.** It is another worker's lane; its intermediate state
  may not compile and is not yours to commit. Leave it for its builder.
- **Markdown / data / index** (`*.md`, `experiments/**`, `tools/**`) → safe to
  commit even if a peer just touched it: a slightly-stale index entry self-heals
  on the next pass; a mid-write source file does not.

You never need a lock. `git`'s own index serializes the commit; you only need to
choose *what* to stage — and you stage it by explicit path.

---

## Step 1 — Survey the disk

```bash
# top-level docs + scripts that should be indexed
ls *.md *.py 2>/dev/null
# tracked vs untracked
git status --porcelain
```

Map what each kind is: top-level prose lives in `*.md`; models/scripts in `*.py`;
runnable experiments under `experiments/NN-name/`; reusable tooling under
`tools/`; rendered diagrams under `visuals/`. The project's index doc names the
sections.

## Step 2 — Reconcile the index doc against disk

Three checks (all read-only) against the project's index doc (`INDEX.md` here):

```bash
# (a) docs/scripts on disk but NOT referenced in the index → need an entry
for f in $(ls *.md | grep -v '^INDEX.md$') *.py; do
  grep -q "$f" INDEX.md || echo "UNREFERENCED: $f"
done
# (b) filenames referenced in the index that don't exist → dangling (fix/remove)
grep -oE '`[A-Za-z0-9_.-]+\.(md|py)`' INDEX.md | tr -d '`' | sort -u | while read f; do
  [ -e "$f" ] || echo "DANGLING: $f"
done
# (c) counts in the header vs reality
echo "docs on disk: $(ls *.md | grep -v '^INDEX.md$' | wc -l)  scripts: $(ls *.py | wc -l)"
```

A prose phrase like "a `skill.md` file" can show as a false-positive DANGLING —
it is text, not a file reference. Ignore those.

For each **UNREFERENCED** doc, add an entry in the **house format** (match the
neighbours exactly), placed in the right section and cross-linked. Mirror whatever
template the index already uses for its entries — title, filename, a short
description, the load-bearing result, structure, status, and related docs.

Also refresh, when stale: the **Generated** date, the **Index covers: N
documents** count line, the **Git context** blockquote, and any **non-document
artifacts** bullets. Add the new doc to the numbered **Reading order** if it
belongs on a main thread.

## Step 3 — Gitignore regenerable artifacts

Compiled / generated outputs must never be committed — they bloat history and
churn. Confirm `.gitignore` covers them:

```bash
git check-ignore experiments/**/bench.exe   # should print the path (ignored)
```

Add any new generated kind to `.gitignore`, not to a commit.

## Step 4 — Detect live lanes (the concurrency gate)

```bash
# any source tree touched in the last 60s = a peer is building it → exclude
find . -name '*.go' -o -name '*.py' -newermt '-60 seconds' -printf '%p (LIVE)\n' 2>/dev/null
```

If any code lane reports LIVE, **exclude that tree from the commit.** Only commit
code you intend to commit when its build is green — never a red or
actively-written tree.

## Step 5 — Commit the quiescent curation lane (explicit path only)

Stage every doc/index/tool path you curated **by explicit path** — never
`git add -A`, never an exclude-glob sweep. List exactly the files you touched:

```bash
git add -- INDEX.md <each curated doc> <each tool>   # explicit paths only
# guard: confirm no live code lane or ignored artifact slipped in
git diff --cached --name-only | grep -iE '\.exe$|\.dos/' && echo VIOLATION || echo OK
git commit -m "Curate cluster: <one-line of what changed>" -- \
  INDEX.md <each curated doc> <each tool>
```

Keep the commit body to a few bullets: index entries added, counts/context fixed,
artifacts ignored. If you deliberately left an actively-built lane to its builder,
say so in one line. Do **not** add a `Co-Authored-By` trailer.

## Step 6 — Verify (witness, don't assume)

```bash
git log -1 --oneline
git status --short      # remaining = only the peer lanes you deliberately left
```

The pass is done when: the index has no UNREFERENCED/DANGLING entries, the
header counts match disk, no artifact/live-code was committed, and the only
remaining working-tree changes are the lanes you intentionally left to their
builders.

---

## Running it on a cadence

This is a single pass. To keep the cluster continuously clean, drive it with
`/loop` (e.g. `/loop curate-cluster`, self-paced) so it re-surveys after each
burst of doc-writing. Each pass is idempotent — if nothing drifted and nothing
is quiescent-new, it commits nothing and exits.

## Where this misleads / honest limits

- The 60 s liveness window is a heuristic, not a lease. For a hard guarantee use
  a real file-tree lease (e.g. `dos arbitrate`) before committing.
- Entry quality is the model's judgment; the format checks are mechanical but
  "is this the load-bearing number?" is not.
- It assumes the project's house index format. A different cluster needs its own
  entry template in Step 2.
