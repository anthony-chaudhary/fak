---
name: refresh-readme
description: One repeatable pass over README.md — the front door — that keeps ONLY the most important points current and honestly framed. Runs the freshness auditor (tools/readme_freshness_audit.py), turns each FAIL into a required edit and each WARN into a judgment call, applies the three front-page laws (SOTA-vs-us-never-naive, 6th-grade/Feynman-but-accurate, wide-audience), re-stamps the readme-verified marker, and commits ONLY README.md (+ the tool) by explicit path. Use after a release / version bump, after a headline number changes, when a link dies, or on a /loop cadence to keep the front page from rotting. The README's checking layer, the way curate-cluster is the index's.
---

# refresh-readme — keep the front door correct, honest, and small

> **What this does.** `README.md` is the one surface read cold by everyone —
> adopter, reviewer, skeptic — and the one most likely to rot: a link dies, a
> version pin lags `VERSION`, a headline number drifts from
> `BENCHMARK-AUTHORITY.md`, a "we beat naive" claim creeps back into the lead.
> Every other claim surface here has a checking layer (a memory-recall audit for
> memories, a closure audit for issue closes, `BENCHMARK-AUTHORITY` for numbers).
> This is the README's. It makes "keep only the most important points on the
> front page, framed honestly" a **repeatable pass**, not a one-time edit that
> decays the moment the person who did it moves on.

The shape: **run the auditor → fix every FAIL → weigh every WARN → apply the
three laws → re-stamp → commit ONLY the README lane.**

---

## The three front-page laws (the durable policy this pass enforces)

These are not style preferences — they are standing rules for the front page, and
the auditor checks each one. Internalize them; the README is graded against them
every pass.

1. **SOTA-vs-us, never naive.** Every headline number on the front page compares
   `fak` against the *best already-shipped* alternative, not a strawman. "~4×
   vs a tuned warm-cache stack" leads; "~60× vs naive" does not — beating naive
   is easy, and leading with it invites the "you're fighting a strawman"
   dismissal. A naive number may appear as an honest aside that *says* it's not
   the headline; it may never be the bolded lead. → auditor `naive_baseline` FAIL.

2. **6th-grade / Feynman voice, but still accurate.** The front page is the
   audience-widening surface; the deep-dive links are where the jargon lives.
   On the first screen, lead with the plain-English idea, then name the term in
   parens — "a scratchpad of the work-so-far (the *KV cache*)", not "the KV
   cache". Explain by concrete example, the Feynman move, before reaching for the
   abstraction. **Accuracy is not negotiable** — simplify the words, never the
   claim. Every acronym gets a parenthetical on first use. → auditor
   `jargon_density` ADVISORY.

3. **Wide-audience appeal.** The first screen gives each reader a foothold: the
   skeptic (what's real / what's not), the security lead (the lock, not the
   screener), the perf engineer (the reuse win + its fences), the casual reader
   (the 2-minute no-key demo). If a section serves only one audience, ask whether
   it belongs on the front page or behind a "Go deeper" link.

And the size law that wraps all three: **the front page holds only the most
important points.** Before adding anything, ask — *would this earn its place if
the page could hold only ten things?* If not, it belongs in a linked topic doc,
not on the front page. Detail flows OUT to `docs/` and the "Go deeper" table; the
front page stays small.

---

## Step 1 — Run the auditor (it builds your work-list)

From the repo root:

```bash
python tools/readme_freshness_audit.py            # human-readable
python tools/readme_freshness_audit.py --json     # machine-readable (the loop uses this)
```

It checks, and exits non-zero on any **FAIL**:

| check | fires on | severity |
|---|---|---|
| `links` | a local Markdown link whose target is missing on disk | **FAIL** |
| `version_pins` | a `vX.Y.Z` string behind the `VERSION` file | **FAIL** |
| `naive_baseline` | a bolded headline that LEADS with a "naive" baseline (law 1) | **FAIL** |
| `headline_authority` | a bolded multiplier not mirrored in `BENCHMARK-AUTHORITY` | WARN |
| `freshness_stamp` | the `readme-verified` marker absent or older than 14d | WARN |
| `jargon_density` | first-screen expert terms with no plain gloss nearby (law 2) | advisory |

**FAIL = a required edit. WARN = a judgment call. ADVISORY = a nudge.** Voice
(jargon) is never a hard gate — plain-language is writing judgment, not a
mechanical rule.

## Step 2 — Fix every FAIL

- **dead link** — the target moved or was deleted. Repoint it to the current
  path, or drop the link if the doc is gone. (Don't invent a path; verify it
  exists.)
- **stale version pin** — bump it to match `VERSION`. A deliberate forward range
  (`v0.31.x`) on the current minor is fine and passes.
- **naive-lead headline (law 1)** — invert it. Put the SOTA comparison in the
  bold lead; demote the naive number to a plain-prose aside that names itself as
  not-the-headline, or cut it.

## Step 3 — Weigh every WARN, apply laws 2 & 3

- **headline_authority WARN** — a front-page number isn't traceable to
  `BENCHMARK-AUTHORITY`. Either it's stale (fix it to the authority figure) or
  it's a number that shouldn't be on the front page at all (an untraced claim).
  Reconcile against the authority doc; never invent a number to match.
- **jargon ADVISORY** — for each flagged first-screen term, add a one-clause
  plain gloss the first time it appears (law 2). Don't touch the deep-dive links.
- **read the first screen as each audience** (law 3) — does the skeptic, the
  security lead, the perf engineer, and the casual reader each get a foothold in
  the first screen? If a point serves none of them, it's a candidate to move
  behind a link (the size law).

## Step 4 — Re-stamp the freshness marker

After the page is correct, update the stamp near the top of `README.md` to
**today's date and the current `VERSION`**:

```
<!-- readme-verified: YYYY-MM-DD vs VERSION X.Y.Z + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme -->
```

This is the freshness contract: it's how the next reader (and the next audit)
knows the page was checked against reality, and when. Re-run the auditor — it
should now be **green (exit 0)**.

## Step 5 — Commit ONLY the README lane, by explicit path

On a shared tree, HEAD can move under you and peers may have dirty paths. The
commit discipline:

- **Stage by explicit path, never `git add -A`** — commit *your* README, not a
  peer's half-written code:
  ```bash
  git pull --no-rebase --no-edit          # merge integrates fine alongside dirty files
  git commit -F <msgfile> -- README.md    # options BEFORE --, paths AFTER
  ```
- **Doc-only diff → `docs(readme): …` subject**, NOT `fix(`/`feat(`. A
  code-effect prefix on a docs-only diff overclaims — keep the prefix honest to
  what changed.
- **On Windows, pass the message via a file** (`-F`), not a here-string — native
  exe arg passing mangles multi-line quotes.
- **If a peer's `MERGE_HEAD` is set** (`cannot do a partial commit during a
  merge`): **wait for it to clear** — don't abort or work around it. Markdown
  self-heals; re-try the pathspec commit once `MERGE_HEAD` is gone.
- **Stay on the trunk (`main`)** — never branch or worktree to dodge a
  dirty/diverged tree. Push promptly: `git push`.

If a release just happened, this pass typically only needs Step 4 (re-stamp to
the new VERSION) — the auditor catches the bump immediately.

---

## When to run this

- After a `/release` or any `VERSION` bump (the stamp + any pin go stale at once).
- When a headline number changes in `BENCHMARK-AUTHORITY`.
- When a doc the README links to moves or is renamed.
- On a `/loop` cadence to keep the front page from drifting.

The auditor is read-only; this skill's only writes are `README.md` and re-running
the tool. It never edits a deep-dive doc — the front page is the only surface in
scope.
