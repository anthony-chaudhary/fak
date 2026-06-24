---
name: persona-score
description: One repeatable pass that keeps fak serving the top-10 personas who land on it — from the free-tier dev who downloads a binary and won't read a word, through the infra engineer who has to operate it, to the researcher who wants to reproduce it. Runs the persona-readiness scorecard (tools/persona_readiness_scorecard.py) over the git-tracked tree, turns each unmet HARD affordance into a required thing to ADD (a prebuilt-binary release, a deployment guide, a determinism witness, a refusal vocabulary, a green gate), retires persona-debt worst-served-first, re-measures + regenerates the snapshot to PROVE the debt dropped, and commits only the scorecard lane by explicit path. The go-to-market counterpart of agent-readiness (one persona: the AI agent), product (the concepts), and industry-score (the field). Use after a change to a persona's entry path (a release, a deploy doc, an integration recipe, the policy spec), when adding a persona to the roster, or on a /loop cadence to keep every front door open.
metadata:
  opencode: claude-only
---

# persona-score — keep fak serving its top-10 personas, and prove it

> **What this does.** A project lives or dies on a question no internals scorecard
> asks: of the kinds of human (and one machine) who actually show up at fak — the
> free-tier dev who will download a binary and run it but not read a word, the app
> developer pointing an agent at it, the infra engineer who has to operate it, the
> security engineer auditing the floor, the researcher reproducing it — **is each
> one served?** Does the first affordance that persona reaches for actually exist in
> the tree? This pass turns that from a pitch into a **repeatable, provable number**,
> the way `agent-readiness` does it for the single persona that is the coding agent.

The shape: **run the scorecard → ADD every missing affordance (HARD defect) → weigh
every SOFT signal → re-measure to prove persona-debt dropped → regenerate the
snapshot → commit only the scorecard lane.**

The headline number is **persona-debt**: unmet HARD affordances + verdict overclaims
+ malformed rows + coverage gaps. Drive it to zero and "fak serves its users"
becomes a number you moved, not a claim you made. The shipped tree is at **0 /
grade A** — this pass is what keeps it there, and the number becomes a regression
sentinel: the day someone deletes the release workflow or the deploy guide, a
persona's affordance fill drops and persona-debt rises.

This skill is one instance of the generic [`scorecard`](../scorecard/SKILL.md)
doctrine. Read that for the anatomy shared by the whole family; this file is the
persona-specific contract.

---

## The one rule that overrides everything: add the affordance, never fake the check

The scorecard reads the **real tracked tree**. Every affordance is a genuine thing a
persona reaches for — a file that must exist, a doc that must mention a token, a
command that must resolve, a CLAIMS section that must be tagged. So:

- **Retire a defect by adding the real thing the persona needs**, not by adding a
  keyword to satisfy a substring match. A `free-tier-dev` gap is fixed by a real
  prebuilt-binary release path, not by typing "download" into the README.
- **Never weaken a verdict to score.** A persona is `served` only when its hard
  affordances are genuinely present; the `verdict_honest` KPI catches a row that
  claims `served` on a missing affordance (an overclaim) — the most important thing
  this scorecard catches. Never paper over a real gap by calling it served.
- **It is read-only.** The tool edits nothing; you add the affordance, then re-run.

If "fixing" a defect would mean gaming the detector instead of adding what the
persona needs, **stop** — that's not a real gap.

---

## The top-10 personas (the roster, declared in `_meta.json`)

Four tiers, ordered by how much effort the persona invests to understand fak:

| Tier | Persona | The affordance they reach for first |
|---|---|---|
| **consume** | `free-tier-dev` | a prebuilt-binary release + install.sh + a no-key first run (won't build, won't read) |
| consume | `app-developer` | an integration index + a drop-in `.mcp.json` + per-harness recipes |
| consume | `backend-integrator` | the `go install …@latest` one-liner + a CLI/config reference + the frozen-ABI seams |
| **operate** | `infra-engineer` | a Dockerfile + a deployment guide + `/metrics` + `/healthz` + rate limiting |
| operate | `security-engineer` | a threat model + the closed refusal vocabulary + a tamper-evident journal + the security CI |
| **evaluate** | `ml-researcher` | a determinism witness + a repro packet + a citation + the research notes |
| evaluate | `benchmark-engineer` | a runnable benchmark command + the bench CI + the industry scorecard |
| evaluate | `decision-maker` | a quotable identity + the honesty ledger + a competitive comparison + a license |
| **build** | `oss-contributor` | CONTRIBUTING + EXTENDING + the leaf scaffold + a green gate + surfaced guardrails |
| build | `ai-agent` | AGENTS.md + llms.txt + a per-agent recipe (depth lives in `agent-readiness`) |

The `ai-agent` row is intentionally thin: it **delegates** the deep audit to
`tools/agent_readiness_scorecard.py` (16 KPIs, friction-debt). Don't duplicate that
work here — keep this row the roster entry.

The affordance **check kinds** (the fixed doctrine): `path_exists`,
`any_path_exists`, `doc_mentions`, `fenced_command`, `command_resolves`,
`claim_section`. Each is a pure predicate over the real tree.

---

## Step 1 — Run the scorecard (it builds your work-list)

From the repo root:

```bash
python tools/persona_readiness_scorecard.py            # human scorecard + work-list
python tools/persona_readiness_scorecard.py --chart    # at-a-glance: who's served, affordance fill
python tools/persona_readiness_scorecard.py --json      # machine payload (the loop / control pane)
python tools/persona_readiness_scorecard.py --critical  # the worst-served personas, worst-first
python tools/persona_readiness_scorecard.py --gaps      # the coverage backlog (unpositioned personas)
```

It scores each persona's met/unmet HARD affordances into a readiness verdict
(served / mostly-served / partially-served / unserved), folds them into a composite
(0–100, A–F) and the **persona-debt** integer, and prints the work-list: every
missing affordance with the thing to add, then the SOFT signals. Read-only.

## Step 2 — Retire persona-debt worst-served-first

Take the worst-served persona first (`--critical` names it). For each missing HARD
affordance, **add the real thing** that persona reaches for (from the roster table).
After a batch, **re-run** and watch the number fall. Capture a baseline to prove it:

```bash
python tools/persona_readiness_scorecard.py --json > /tmp/before.json   # baseline
# … add the affordances (a release path, a deploy guide, a witness) …
python tools/persona_readiness_scorecard.py --compare /tmp/before.json  # the persona-debt delta + the 2x/3x verdict
```

Coverage gaps (`--gaps`) are their own debt: a top-10 persona with no row. Close one
by adding an **honest** row in `tools/persona_readiness_scorecard.data/rows-<tier>.json`
— affordances grounded in real paths, the verdict matching what's actually present.

## Step 3 — Weigh the SOFT signals, then stop

Read the SOFT list once (a Homebrew tap, a k8s manifest file, a citable DOI, a
good-first-issue on-ramp). These lower no grade and are never persona-debt — they're
the forward backlog. Fix the cheap, real ones; don't grind them to zero.

## Step 4 — Re-measure, confirm, regenerate the snapshot

Re-run; state the before/after (e.g. "persona-debt 4 → 0, infra-engineer
partially-served → served"). Then regenerate the committed doc folder so it matches
the tree:

```bash
python tools/persona_readiness_scorecard.py --markdown-dir docs/persona-scorecard --stamp $(date +%F)
```

If you added or removed a scorecard surface, re-fold the portfolio and re-pin:
`python tools/scorecard_control_pane.py --pin` (persona reports `persona_debt`).

## Step 5 — Commit only the scorecard lane, by explicit path

This is a shared trunk; commit *your* lane, never a peer's work:

```bash
git commit -s -F <msgfile> -- \
  tools/persona_readiness_scorecard.py \
  tools/persona_readiness_scorecard_test.py \
  tools/persona_readiness_scorecard.data/ \
  docs/persona-scorecard/
```

- **Stage by explicit path, never `git add -A`** — stage + commit in one shell call
  so a peer's bare commit can't sweep your files.
- **Subject:** a tool/data change → `feat(tools): … (fak tools)`; a docs-only
  snapshot regen → `docs(scorecard): … (fak docs)`. End with the `(fak <leaf>)`
  trailer; lead with a verb.
- **The control pane (`tools/scorecard_control_pane.py`), the baseline
  (`tools/scorecard_baseline.json`), `INDEX.md`, and the skill README are co-edited
  by peers** adding sibling scorecards — if one shows in your diff and you didn't
  change it, exclude it; if a `MERGE_HEAD` is set, wait it out, then commit by path.

---

## When to run this

- After a change to a persona's entry path: a release (`free-tier-dev`), an
  integration recipe (`app-developer`), the deployment guide / observability
  (`infra-engineer`), the policy spec / refusal vocabulary (`security-engineer`), a
  benchmark or the industry scorecard (`benchmark-engineer`), CONTRIBUTING /
  EXTENDING (`oss-contributor`).
- When adding a persona to the roster (position it in `_meta.json` + a `rows-*.json`).
- On a `/loop` cadence to keep every front door open as the repo grows.

The scorecard is fak's go-to-market checking layer, the way `agent-readiness` is its
agent-experience layer and `industry-score` is its competitive one. Same discipline:
a number you can move, and prove you moved.
