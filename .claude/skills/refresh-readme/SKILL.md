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

**The size law now has teeth (it used to be aspirational).** Five of the six
substance checks reward *adding* an affordance — put `fak guard` up top, add a
speed number, add a hero result, add a persona router. Optimized to those alone,
the page only ever grows: the git log shows it halved on 2026-07-01 and
immediately regrew, one well-meaning "surface X on the front page" commit at a
time. `front_page_focus` is the counterweight — a line budget, a section-count
budget, and a **single-lead** rule (the one-binary/syscall pitch stated *once* in
the preamble, not restated three times before the reader reaches a section). It
feeds the composite score and the `readme_debt` gate, so a bloated page can no
longer score an A. The two forces now both live in the tool; concision is checked,
not just hoped for.

The three anti-regrowth rules this pass enforces:

1. **Retire before you add.** The page is at its section budget by design. To add
   a section, *fold or cut* one first — do not bump `FRONT_PAGE_SECTION_BUDGET` to
   make room. Bumping a budget to silence the warning is the failure mode; bump it
   only when the page has genuinely, durably earned the slot, and say why in the
   commit.
2. **One lead, not three.** The pitch is derived from
   [`docs/adoption/pitch-ladder.md`](../../../docs/adoption/pitch-ladder.md) rung 1
   and appears *once* above the first `## `. A second "and also, fak is a binary
   you put in front of your agent" paragraph is the confusion `single_lead`
   catches — collapse it into the one lead.
3. **Detail flows OUT, never back IN.** The overflow sink is
   [`docs/README-legacy.md`](../../../docs/README-legacy.md). Narrower-audience or
   deep-dive material moves there and earns a link; it does not migrate back onto
   the front page. If a section keeps wanting to come home, that is a signal to
   write a topic doc, not to re-inline it.

---

## Step 1 — Run the auditor (it builds your work-list)

From the repo root:

```bash
python tools/readme_freshness_audit.py --json > readme-before.json
python tools/readme_freshness_audit.py            # human rendering of the same evidence
```

`readme-before.json` is the mandatory first artifact. Quote every binding `FAIL` check name and
its evidence before editing. Retire all binding FAIL rows before advisory prose polish; an
advisory unglossed term such as `KV cache` must not displace a red `guard_prominence` check. The
current README product defect belongs in its dedicated README issue, not in this skill-only
adjudication.

It checks, and exits non-zero on any **FAIL**:

| check | fires on | severity |
|---|---|---|
| `links` | a local Markdown link whose target is missing on disk | **FAIL** |
| `version_pins` | a `vX.Y.Z` string behind the `VERSION` file | **FAIL** |
| `naive_baseline` | a bolded headline that LEADS with a "naive" baseline (law 1) | **FAIL** |
| `headline_authority` | a bolded multiplier not mirrored in `BENCHMARK-AUTHORITY` | WARN |
| `freshness_stamp` | the `readme-verified` marker absent or older than 14d | WARN |
| `jargon_density` | first-screen expert terms with no plain gloss nearby (law 2) | advisory |
| `front_page_focus` | the page busts the line/section budget or restates the lead ≥3× (the size law) | debt |

**FAIL = a required edit. WARN = a judgment call. ADVISORY = a nudge.** Voice
(jargon) is never a hard gate — plain-language is writing judgment, not a
mechanical rule.

## Step 2 — Fix every FAIL

Work in the exact binding order quoted from `readme-before.json`; do not start WARN/ADVISORY
polish or the separately issue-tracked README product fix while any FAIL remains.

- **dead link** — the target moved or was deleted. Repoint it to the current
  path, or drop the link if the doc is gone. (Don't invent a path; verify it
  exists.)
- **stale version pin** — bump it to match `VERSION`. A deliberate forward range
  (`v0.31.x`) on the current minor is fine and passes.
- **naive-lead headline (law 1)** — invert it. Put the SOTA comparison in the
  bold lead; demote the naive number to a plain-prose aside that names itself as
  not-the-headline, or cut it.

## Step 3 — Weigh every WARN, apply laws 2 & 3

Enter this step only after all binding FAIL rows are green.

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
  fak sync check                          # or fak sync reconcile --apply to integrate trunk safely
  fak commit --path README.md -F <msgfile> [--push]
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
  dirty/diverged tree. Push promptly via `fak sync push` (or `--push`).

If a release just happened, this pass typically only needs Step 4 (re-stamp to
the new VERSION) — the auditor catches the bump immediately.

---

## This is already a durable loop — you don't have to remember to run it

The front page does **not** rely on someone thinking to check it. The checking
half runs on a fixed cadence, and this skill is the acting half it hands work to:

1. **The auditor is a registered gardening loop.** `readme-freshness` is an
   `enabled` entry in [`tools/control_pane.loops.json`](../../../tools/control_pane.loops.json),
   wired to `python tools/readme_freshness_audit.py --json`. It sits alongside the
   other front-door checking layers (`memory-recall-audit`, `docs-scorecard`, …).
2. **It runs daily, unattended.** [`.github/workflows/garden.yml`](../../../.github/workflows/garden.yml)
   runs `fak garden` on a daily `cron` (06:23 UTC), and `fak garden` folds the
   fleet **loop-audit** — which runs every enabled gardening loop once, this one
   included. So the README is re-checked against the filesystem, `VERSION`, and
   `BENCHMARK-AUTHORITY` every day, whether or not anyone asked.
3. **The loop itself is watched.** The `gardening-loops-audit` meta-loop flags
   `readme-freshness` as BROKEN if its status command ever stops running — the
   loop can't silently rot into a no-op.
4. **This skill is the acting half.** When the daily audit returns `ACTION`
   (`ok:false` — a FAIL), that verdict names `/refresh-readme` as the runbook.
   You (or a supervisor turn) run the pass below; the loop closes.

So "a durable README loop that routinely gardens the front page" already exists
end to end: **daily audit → ACTION verdict → this skill → commit the README lane.**
The section below is what you do when the loop hands you an ACTION (or when one of
the triggers fires early).

## When to run this

- After a `/release` or any `VERSION` bump (the stamp + any pin go stale at once).
- When a headline number changes in `BENCHMARK-AUTHORITY`.
- When a doc the README links to moves or is renamed.
- When the daily `readme-freshness` loop (above) returns `ACTION`, or on a manual
  `/loop` cadence to garden the front page ahead of the daily tick.

The auditor is read-only; this skill's only writes are `README.md` and re-running
the tool. It never edits a deep-dive doc — the front page is the only surface in
scope.
