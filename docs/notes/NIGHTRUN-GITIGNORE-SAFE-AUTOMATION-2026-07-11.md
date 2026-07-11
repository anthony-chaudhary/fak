# Nightrun `.gitignore` hygiene, and how a background loop can "detect and add to gitignore" *safely* (2026-07-11)

A worked incident + a design note. Motivating question: *"nightrun dir probably gitignore
— think about processes that detect and add to gitignore as background loops safely."*

Short answer: **the safe process detects and *refuses*; it does not detect and *mutate
`.gitignore` on a loop*.** The nightrun dir must NOT be blanket-ignored, and the leak that
prompted this was not a missing ignore rule — it was a careless bulk `git add` re-committing
files a prior fix had removed. `.gitignore` is powerless against that by construction.

## The incident (witnessed, not hypothesized)

- #4276 (`e8ae57722 fix(nightrun): ignore legacy step-advice stamps`) did the correct cleanup:
  it **`git rm`'d** the 192 tracked `docs/nightrun/stepadvice-*.json` stamps, added the ignore
  rule (`.gitignore:284 docs/nightrun/stepadvice-*.json`), a locking test
  (`cmd/fak/nightrun_legacy_artifacts_test.go`), and a README invariant
  (`docs/nightrun/README.md:180` — *"not publication artifacts and must never return to the
  index"*).
- `e041fdd42 docs(notes): DeepWiki concept study` was meant to add exactly two files
  (`INDEX.md`, `docs/notes/CONCEPT-STUDY-DEEPWIKI-2026-07-11.md`). It also re-added **all 192**
  `stepadvice-*.json` — a broad `git add`/`git add -A` swept them back in, undoing #4276.
- A later still-running legacy guard re-wrote two of them, so they surfaced as tracked-and-dirty
  (`stepadvice-02defed6-…json`, `stepadvice-unknown.json`) — visible churn on the shared trunk.
- A prior turn "fixed" this by appending `/docs/nightrun` **twice** at the end of `.gitignore`.
  That is wrong on two counts (below), and it does not untrack anything.

## Two load-bearing facts about `.gitignore`

1. **`.gitignore` is inert for already-tracked paths.** Ignoring a path that is already in the
   index changes nothing — the file stays tracked and keeps showing as modified. Proof from this
   incident: while the 192 stamps were tracked, `git check-ignore -v <stamp>` returned *no match*
   (git reports a tracked path as not-ignored). After `git rm --cached`, the same path reports
   `.gitignore:284`. So a rule can only ever *prevent a future add*; it can never *clean a leak*.
   The clean-up is always `git rm --cached`, never another ignore line.

2. **A whole-directory ignore over a curated dir is a foot-gun.** `docs/nightrun/` holds
   *tracked, curated* publication snapshots (`README.md`, the `*.md` plans, `collected.jsonl`) —
   the live writers target the gitignored runtime root `.fak/nightrun/` (#3209), and
   `TestLiveNightrunTicksKeepTrackedDocsClean` locks that a tick never dirties the tracked docs.
   A blanket `/docs/nightrun` neither untracks the curated files (fact 1) nor should hide them;
   it only masks the *next* stray add so nobody notices the regression. The correct shape is a
   **scoped** ignore of the regenerable siblings only (`cache-value.jsonl`, `stepadvice-*.json`).

## Why "a background loop that auto-appends to `.gitignore`" is the wrong primitive

Auto-mutating `.gitignore` from a loop concentrates every failure mode this repo has already
paid for:

- **Overbroad patterns eat source.** A loop that infers a pattern from a filename is one bad
  glob away from `*scratchpad*` swallowing a real `.go` file. A human reviewing a one-line
  `.gitignore` diff catches this; a loop committing it does not.
- **It cannot fix what it is aimed at.** By fact 1, appending a rule for an already-tracked leak
  does nothing. The loop would "succeed" (rule added) while the leak persists — the worst kind of
  green: a fix that isn't one.
- **Blanket rules hide curated content** (fact 2), turning a visible regression into a silent one.
- **It races on a shared, hand-curated file.** `.gitignore` here is dense with ordered rules and
  `!re-include` negations; concurrent appends from a fleet corrupt that ordering.
- **Mutation is hard to reverse; refusal is free.** A wrongly-added ignore rule silently changes
  what future commits capture, fleet-wide, until someone notices. A refusal changes nothing.

## The safe pattern (what this repo already converges on)

1. **Prevent at the source — one gitignored runtime root.** Writers target `.fak/…`, never the
   tracked tree (#3209, `cmd/fak/nightrun_ledger_path.go`). No per-artifact ignore rule needed;
   nothing regenerable is ever a candidate for `git add`. This is the real fix — ignore rules are
   only defense-in-depth for legacy writers that predate it.
2. **Detect-and-*refuse* at the commit seam, don't detect-and-mutate.** `tools/check_committed_files.py`
   (FILE_ADMISSION) audits *staged additions* and returns a reason string; it never touches
   `.gitignore`. Refusal is safe because it changes no state and routes to a human/replan.
3. **Scoped, root-anchored, path-exact rules; never a bare dir or bare glob**, and never a pattern
   that also matches a tracked path (check `git ls-files -- <pattern>` is empty first).
4. **Lock coverage with a test.** `TestLegacyStepAdviceStampsAreIgnored` asserts an *untracked*
   probe stamp is ignored — the durable witness that the rule exists and is scoped.

## Recommended hardening (design only — deliberately not shipped in this pass)

FILE_ADMISSION would have caught `e041fdd42` if `_classify` refused **a staged *addition* that a
committed `.gitignore` rule already matches** — i.e. a re-add of a declared-ignorable artifact.
This is the ideal safe detector: it *refuses*, never mutates; it is general (the whole class, not
just stamps); and it closes exactly the hole the incident slipped through.

The trap that makes it non-trivial (and why it is not a one-liner shipped here): the naive check
`git check-ignore --no-index <path>` will **false-positive on every intentionally-tracked file
that lives under a broadly-ignored dir kept alive by a `!re-include`** — this repo has many. So
the rule must be narrow:

- audit **staged additions only** (`--audit-staged`), never the whole tracked tree — an addition
  is the act we want to gate; existing tracked files are settled policy;
- flag a path **only** if an ignore rule matches it **and** no later `!` negation re-includes it;
- keep the `ALLOW_STRAY_FILE=1` override for the rare intentional case.

Scope it, give it its own `check_committed_files_test.py` case (a staged `docs/nightrun/stepadvice-…json`
must be refused; a legitimately re-included tracked path must pass), and warn-soak before block.

## What this pass changed

- Removed the duplicate blanket `/docs/nightrun` from `.gitignore`; added a comment there
  explaining why the dir is *not* blanket-ignored (fact 2), so the next turn doesn't re-add it.
- `git rm --cached docs/nightrun/stepadvice-*.json` (192 files, kept on disk) — re-applies #4276
  and the README's "must never return to the index" invariant; `git check-ignore` now matches, so
  the rule is finally effective.
- Left the tracked publication snapshots (`README.md`, `*.md`, `collected.jsonl`) and all peer WIP
  untouched. The hardening above is filed as a recommendation, not a shipped code change.
