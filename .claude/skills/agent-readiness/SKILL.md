---
name: agent-readiness
description: One repeatable pass that keeps fak the path of least resistance for an AI agent — Claude Code, OpenAI Codex, Cursor, an MCP client — to discover, adopt, and build on. Runs the agent-readiness scorecard (tools/agent_readiness_scorecard.py) over the git-tracked tree, turns each HARD defect into a required affordance to ADD (a missing agents.md entry point, a missing harness config, a dead orientation link, no copy-pasteable first command, no install one-liner, an untagged claim, a missing per-agent recipe, a missing leaf scaffold, an unsurfaced guardrail, a missing contributor contract), retires friction-debt worst-step-first, re-measures + regenerates the snapshot to PROVE the debt dropped, and commits only the scorecard lane by explicit path. The INWARD agent-experience counterpart of industry-score (competitive) and repo-hygiene (structure). Use after a change to an agent surface (AGENTS.md, llms.txt, CLAIMS.md, the integration recipes, the guards), when onboarding a new agent harness, or on a /loop cadence to keep the front door agent-friendly.
---

# agent-readiness — keep fak the path of least resistance for an agent, and prove it

> **What this does.** An agent-first project lives or dies on a question no
> human-facing scorecard asks: can an autonomous coding agent that lands in the
> repo cold **discover** what fak is, **want** to adopt it, and **build** on it
> without tripping over a missing affordance or an undocumented guard? This pass
> turns that from a vibe ("we have an AGENTS.md, we're fine") into a **repeatable,
> provable number** — the way `industry-score` keeps the competitive story honest
> and `repo-hygiene` keeps the tree lean.

The shape: **run the scorecard → add every missing affordance (HARD defect) →
weigh every SOFT signal → re-measure to prove friction-debt dropped → regenerate
the snapshot → commit only the scorecard lane.**

The headline number is **friction-debt**: the count of concrete, re-derivable
defects that make fak harder for an agent to find, trust, and build on. Drive it
to zero and "attractive to agents" becomes a number you moved, not a claim you
made. (The shipped tree is at **0 / grade A** — this pass is what keeps it there,
and the stick any forked or adopting repo measures its own gap against.)

---

## The one rule that overrides everything: add the affordance, never fake the check

The scorecard reads the **real tracked tree**. Every KPI is a genuine agent
affordance — a file that must exist, a link that must resolve, a command that must
be paste-able, a claim that must be tagged. So:

- **Retire a defect by adding the real thing an agent reaches for**, not by adding
  a keyword to satisfy a substring match. A `first_command` defect is fixed by a
  fenced, runnable no-key command an agent can actually paste — not by typing the
  words `fak preflight` into prose.
- **Never weaken a claim or a guard to score.** The honesty ledger, the trunk law,
  the leaf/ABI discipline are load-bearing; `guardrails_surfaced` wants them
  *documented*, never relaxed.
- **It is read-only.** The tool edits nothing; you make the edit, then re-run.

If "fixing" a defect would mean gaming the detector instead of adding the
affordance, **stop** — that's not a real gap.

---

## The three steps an agent walks (the groups, and the HARD defects each retires)

Thirteen KPIs, each 0–100, grouped by the step they gate. Retire worst-step-first
(the scorecard names the weakest step in `group_scores`).

| Step | KPI | The affordance to add when it's red |
|---|---|---|
| **discover** | `agents_entrypoint` | An AGENTS.md (the agents.md convention) that states what fak is and carries build + test + run commands. |
| discover | `agent_config` | The zero-setup configs a harness auto-loads: `.mcp.json`, `.cursorrules`, `.github/copilot-instructions.md`. |
| discover | `llms_map` | `llms.txt` (the answer-engine / agent doc-map). |
| discover | `identity_statement` | A one-sentence "fak is a/an …" near the top of AGENTS.md / llms.txt / README an agent can quote. |
| discover | `entry_links_resolve` | Fix any dead local link in AGENTS.md or the integration index — a 404 on the orientation path. |
| **adopt** | `first_command` | A copy-pasteable, **fenced**, no-key/no-model/no-GPU first command (the 30-second proof). |
| adopt | `install_oneliner` | The `go install …@latest` one-liner (the module is at the repo root, so it resolves). |
| adopt | `honesty_ledger` | CLAIMS.md present, every `- [` claim carrying exactly one status tag (the `make claims-lint` rule). |
| adopt | `integration_recipes` | A per-agent recipe under `docs/integrations/` for each family (Claude, Codex/OpenAI, Cursor, MCP). |
| **build** | `extension_scaffold` | The additive path: `tools/new_leaf.py` + `EXTENDING.md` (add a leaf, don't edit core). |
| build | `guardrails_surfaced` | Document each enforced rule in AGENTS.md: trunk-only, commit-by-path, DCO sign-off, tagged claims, leaf/ABI. |
| build | `contributor_contract` | `CONTRIBUTING.md` linked from the entry point + a one-command green gate (`make ci`). |
| build | `machine_consumable` | **SOFT** — how much of the measurement family speaks `--json`. Scores; never hard debt. |

**SOFT signals** (a missing `llms-full.txt`, a tool without `--json`) lower the
score but are **never** friction-debt; weigh them, don't grind on them.

---

## Step 1 — Run the scorecard (it builds your work-list)

From the repo root:

```bash
python tools/agent_readiness_scorecard.py            # human scorecard + friction-debt work-list
python tools/agent_readiness_scorecard.py --json     # machine payload (the loop uses this)
```

It scores the three steps (discover · adopt · build) into a composite (0–100, A–F)
and a **friction-debt** integer, and prints the work-list: every HARD defect with
the affordance to add, then the SOFT signals. Read-only; it never edits the tree.

## Step 2 — Retire friction-debt worst-step-first

Take the weakest step first (the scorecard names it in `group_scores`). For each
HARD defect, add the real affordance from the table above. After a batch, **re-run
the scorecard** and watch the number fall; that loop (add, re-measure, add again)
is the whole method. Capture a before/after baseline so you can prove the drop:

```bash
python tools/agent_readiness_scorecard.py --json > /tmp/before.json   # baseline before the pass
# … add the affordances …
python tools/agent_readiness_scorecard.py --compare /tmp/before.json  # prints the friction-debt delta + the 2x verdict
```

## Step 3 — Weigh the SOFT signals, then stop

Read the SOFT list once; fix the cheap, real ones. Don't chase them to zero — a
token added only to move a metric is the gaming this pass refuses.

## Step 4 — Re-measure, confirm, regenerate the snapshot

Re-run the scorecard; state the before/after (e.g. "friction-debt 6 → 0, adopt
67 → 100"). Then regenerate the committed snapshot so the doc matches the tree:

```bash
python tools/agent_readiness_scorecard.py --markdown --stamp $(date +%F) > docs/AGENT-READINESS-SCORECARD.md
```

If you added or removed a scorecard surface, also re-fold the portfolio:
`python tools/scorecard_control_pane.py` (agent-readiness reports `friction_debt`).

## Step 5 — Commit only the scorecard lane, by explicit path

This is a shared trunk; commit *your* lane, never a peer's work:

```bash
git commit -s -F <msgfile> -- tools/agent_readiness_scorecard.py \
  tools/agent_readiness_scorecard_test.py docs/AGENT-READINESS-SCORECARD.md
```

- **Stage by explicit path, never `git add -A`** — stage + commit in one shell
  call so a peer's bare commit can't sweep your files.
- **Subject:** a code change to the tool → `feat(tools): … (fak tools)`; a
  docs-only snapshot regen → `docs(scorecard): … (fak docs)`. End with the
  `(fak <leaf>)` trailer; lead with a verb.
- **The control pane (`tools/scorecard_control_pane.py`), `INDEX.md`, and this
  skill are co-edited by peers** adding sibling scorecards — if one shows in your
  diff and you didn't change it, exclude it; if a `MERGE_HEAD` is set, wait it out,
  then commit by explicit path.

---

## When to run this

- After a change to any agent surface: AGENTS.md, llms.txt/llms-full.txt,
  CLAIMS.md, the `docs/integrations/` recipes, the surfaced guardrails, `new_leaf.py`.
- When onboarding a new agent harness (add its `agent_config` + a recipe).
- On a `/loop` cadence to keep the front door agent-friendly as the repo grows.

The scorecard is fak's agent-experience checking layer, the way `industry-score`
is its competitive layer and `repo-hygiene` is its structural one. Same discipline:
a number you can move, and prove you moved.
