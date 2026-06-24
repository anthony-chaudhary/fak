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

## Cross-loading into opencode (#422)

opencode scans `**/SKILL.md` too, but its skill loader honors only
`name, description, license, compatibility, metadata` — **every other
frontmatter field is silently dropped into inert `options`.** So these
Claude-only fields lose their meaning when a skill is cross-loaded:

| Field | What it does in Claude | What opencode does |
|---|---|---|
| `allowed-tools` | per-skill tool allowlist (a read-only skill stays read-only) | dropped — scope widens to the invoking *agent's* permission |
| `disable-model-invocation` | gate so only the operator can invoke | dropped — the model can invoke it |
| `user-invocable` | hide an auto-load-only skill from direct invocation | dropped — it becomes directly invocable |
| `argument-hint`, `output_root` | UX / output-path hints | dropped (cosmetic) |

A field is **load-bearing** when dropping it changes the access or invocation
posture — a read-only `allowed-tools` allowlist, or `disable-model-invocation:
true` / `user-invocable: false`. opencode's per-agent `permission:` lives on
the agent, not the skill, so there is no per-skill equivalent.

**You cannot exclude a skill from opencode's scan.** opencode *auto-discovers*
every `.claude/skills/<name>/SKILL.md` by walking up to the worktree root; its
`skills.paths` config only **adds** folders, it never subtracts one. So a
load-bearing skill is **always loaded** under opencode and its frontmatter is
**always dropped** — there is no "exclude from the scan." The boundary has to be
re-expressed in opencode's own access-control surface: the per-agent (or global,
in `opencode.json`) **`permission`** object.

| Claude frontmatter | opencode `permission:` equivalent |
|---|---|
| `disable-model-invocation` / `user-invocable: false` | `permission.skill: { "<name>": "deny" }` — gates invocation by skill name |
| read-only `allowed-tools` (no Write/Edit) | `permission.edit: "deny"` — read-only *agent* (covers edit/write/patch); per-agent, not per-skill |

**Mitigation.** A skill whose Claude-only frontmatter is load-bearing must
acknowledge the gap via the opencode-honored `metadata` field (it survives the
cross-load — though it is inert *data* to opencode, not a directive):

```yaml
metadata:
  opencode: agent-permission  # the gate IS re-expressed in opencode.json `permission`
  # or
  opencode: claude-only       # NOT portable per-skill — documented Claude-only; an
                              # opencode worker loses the boundary unless it runs under
                              # a read-only agent (`permission.edit: "deny"`)
```

This repo's `opencode.json` ports the one fully-portable gate — `phased-plan`
(operator-only) is denied via `permission.skill: { "phased-plan": "deny" }`
(`agent-permission`). The two read-only audit skills (`issue-triage`,
`trajectory-audit`) carry a per-skill read-only boundary that opencode's
per-*agent* permission can't express one-to-one, so they stay `claude-only`: run
them under Claude, or under an opencode agent whose `permission.edit` is `deny`.

**Lint.** `python tools/skill_frontmatter_lint.py` flags every skill whose
Claude-only frontmatter is load-bearing and not yet acknowledged;
`--check` is the CI gate (exit 1 on an unacknowledged one).

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
