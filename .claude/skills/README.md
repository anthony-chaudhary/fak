# Project skill pack

These are [Claude Code skills](https://docs.claude.com/en/docs/claude-code/skills)
checked into the repo so any agent working in `fak` can invoke them with `/<name>`.
Each skill is one `<name>/SKILL.md` with YAML front-matter; some bundle a small
read-only helper script beside it. The maintenance skills read `.claude/project.yaml`
for this repo's helper-script wiring — the skill text is universal, the helpers are
project-supplied.

## Workflow / analysis (no project contract)

| Skill | What it does |
|---|---|
| [`tail-wag`](tail-wag/SKILL.md) | Find the most "tail wagging the dog" inversion — a peripheral concern driving a core decision. Diagnoses, ranks, proposes a rebalance. |
| [`phased-plan`](phased-plan/SKILL.md) | Ceremony rules for shipping a phase of a phased plan — when to release, the hero-exit rule, the phase-split test. (Auto-loaded; not user-invocable.) |
| [`clean-skill`](clean-skill/SKILL.md) | Audit a skill's per-invocation context use and propose a context-bundling helper + SKILL.md trim. Stops for approval before writing code. |
| [`memory-compact`](memory-compact/SKILL.md) | Keep a Claude Code auto-memory store under the harness load cap (200 lines / 25 KB), tier into hot/cold, prove it with the bundled `check_memory.py` witness. |

## Maintenance (read `.claude/project.yaml`)

| Skill | What it does |
|---|---|
| [`release`](release/SKILL.md) | Full versioned release: decide → bump VERSION → draft notes → commit → push → tag → create the GitHub release page. Wraps `tools/release_*.py`; encodes the ordering gotchas the helpers enforce by refusing. |
| [`refresh-readme`](refresh-readme/SKILL.md) | Keep `README.md` current and honest: run the freshness auditor, fix every FAIL, apply the three front-page laws, re-stamp, commit only the README lane by explicit path. |
| [`issue-triage`](issue-triage/SKILL.md) | Classify + rank the open GitHub issue backlog, propose mechanical gardening moves (mark-stale, close-dormant), apply only on approval. Read-only helper; writes are gated. |
| [`plan-audit`](plan-audit/SKILL.md) | Reconcile every plan-state surface into one completion audit and render a dated snapshot. Read-only — never edits a plan file. |
| [`curate-cluster`](curate-cluster/SKILL.md) | Reconcile a project index doc against disk, gitignore artifacts, and commit only the quiescent curation lane by explicit path — never an actively-built code tree. |
| [`trajectory-audit`](trajectory-audit/SKILL.md) | Sweep recent Claude Code session transcripts for token-weighted cost/efficiency problems (I:O ratio, cache reuse, heaviest sessions). Read-only; wraps `tools/session_audit.py`. |
| [`quality-score`](quality-score/SKILL.md) | The CODE-quality RSI pass: run the code-quality scorecard, retire code-debt worst-first (gofmt, real tests, safe extraction), re-measure to prove the drop, ground the ship in DOS, commit by explicit path. Wraps `tools/code_quality_scorecard.py`. |
| [`appeal-score`](appeal-score/SKILL.md) | The PROSE-voice pass: make a doc read human, not machine. Run the doc-appeal scorecard, retire appeal-debt (em-dash flood, run-ons, walls, stacked contrast frames, LLM-scaffolding) WITHOUT changing a claim/number/link, re-measure, commit the doc lane. Wraps `tools/doc_appeal_scorecard.py`. |

## Conventions

- **Per-project overrides.** A project can ship a wrapper `SKILL.md` at the same
  path to extend a universal skill (e.g. add a registry stamp or a notify step);
  the local copy shadows this one.
- **Commit discipline.** Skills that commit stage **by explicit path** (never
  `git add -A`) and never add a `Co-Authored-By` trailer — matching this repo's
  trunk-only, explicit-pathspec rules in `AGENTS.md`.
- **Helper contract.** The maintenance skills resolve their helper scripts through
  `.claude/project.yaml`; if a key is absent the skill prints one line and stops
  rather than improvising.
