---
name: plan-audit
description: Reconcile every plan-state surface a project tracks into one completion audit — how many recent plan tasks shipped, how many remain, and which surfaces disagree. Runs the project's plan-audit helper and renders a dated operator snapshot. Use when the user asks "where do we stand on the plans", "audit plan completion", "what's shipped vs remaining", or wants the portfolio percent-complete picture. Read-only — never edits a plan file.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Grep, Glob
argument-hint: "(no args)"
---

# /plan-audit — Plan-Completion Audit (project-agnostic)

> Wraps `tools/plan_audit.py` + the `plan_docs_glob` declared in
> `.claude/project.yaml`. If a project keeps a *single* plan-state surface (the
> plan docs themselves — no typed registry, no live-percent YAML), then `drift`
> is structurally empty and the helper's `percent_complete` is a **coarse
> header-marker signal** (shipped/built marker present → 100), not a verified
> per-unit census — read the "Work-units rollup" as a floor. The reconcile
> machinery below activates only once a project grows a second surface.

Answer "how much of the plan portfolio is done, and how much is left" with **one reconciled view** instead of the several surfaces that routinely disagree (a typed plan registry, a live-percent state file, the plan-doc glob, git claim/ship history — projects vary in which subset they keep, but the failure mode is the same: they drift).

This skill does not re-implement any per-surface scan. The project's `helpers.plan_audit` script orchestrates whatever surfaces the project has, joins them on plan id, and emits one reconciled JSON. This skill consumes that JSON and renders an operator snapshot.

> ⚓ One surface per metric. This is the **audit** surface. It is **read-only on every plan file.** Drift is *reported*; the operator resolves it by editing the project's plan-state files. The audit itself never writes a plan file.

## Project contract

This skill reads `.claude/project.yaml` at the repo root. Required keys:

- `python` — interpreter path (default: `python`).
- `helpers.plan_audit` — script that emits the reconciled JSON (shape below).
- `plan_docs_glob` — glob for the project's plan documents (default: `docs/*-plan.md`).
- `audits_dir` — where to write `plan-completion-<YYYY-MM-DD>.md` (default: `docs/_audits/`).

If `.claude/project.yaml` is missing or `helpers.plan_audit` is absent, print one line pointing at `.claude/skills/README.md` and stop. Do not improvise file locations.

**No-plans projects.** If `plan_docs_glob` matches zero files, the skill prints `no plan docs found — nothing to audit` and exits 0. A project that does not run phased plans is a supported configuration — the contract just no-ops.

### Expected JSON shape from `helpers.plan_audit`

The helper is invoked as `<python> <helpers.plan_audit> --json` and must emit a single JSON object to stdout with these keys:

| Key | Type | Purpose |
|---|---|---|
| `counts` | object | Headline totals: `total_plans`, `complete`, `in_progress`, `not_started`, plus any project-defined slot/class buckets (e.g. `keep_slots`) |
| `plans` | array of per-plan rows | One row per plan; columns are project-defined but should include at minimum `id`, `name`, and one reconciled `percent_complete`-style value plus the per-surface raw values |
| `drift` | array | Each entry: `{id, name, flags: [string, …]}` — one row per plan with at least one surface disagreement |
| `work_units` | object | Two rollups: `plan_weighted` (every plan = 1 unit scaled by reconciled percent) and `task_weighted` (sum of per-plan phase counts where the helper can parse them). Both carry a `pct_complete`; `task_weighted` additionally carries `coverage_plans` / `coverage_total` so the renderer can label it a floor |

If a project's helper does not yet emit `work_units` (a phased-plan portfolio with no parseable phase counts), the skill still renders `counts`, `plans`, and `drift`, and notes "work-units rollup not available for this project" once in the summary.

### Expected behavior of helper exit codes

- `0` — audit ran cleanly; the JSON is consumable.
- `1` — used by `--check` mode only (drift gate). Means "drift detected, please look." The skill in normal (`--json`/`--markdown`) mode treats `1` as "ran with non-blocking findings" and proceeds.
- `2` — infra error (a surface file missing, a sub-script failed). Skill prints the helper's stderr verbatim and stops; do not guess at numbers by hand.

## Step 1: Run the reconciler

```bash
<python> <helpers.plan_audit> --json
```

Read the JSON inline from the Bash tool result. If the helper exits 2, print its stderr and stop.

If the helper exits 0 with `{"counts": {"total_plans": 0}}` (or equivalent — see "No-plans projects" above), print one line — `no plan docs found — nothing to audit` — and exit 0.

## Step 2: Render the operator snapshot

```bash
<python> <helpers.plan_audit> --markdown --out <audits_dir>/plan-completion-<YYYY-MM-DD>.md
```

Use today's date in UTC. Use `--out` (not a shell `>` redirect) — PowerShell redirects re-encode to UTF-16 and mangle non-ASCII glyphs like `·` / `—`; `--out` writes UTF-8 directly. If a report for today already exists, overwrite it (one audit per day).

If `audits_dir` does not exist, create it (the audit file is the only thing the skill writes — a fresh project may not have the directory yet).

## Step 3: Summarize for the user

In the chat response, lead with the **headline counts** (`total_plans` / `complete` / `in_progress` / `not_started`, plus whichever class/slot buckets the project's `counts` includes), then the **work-units rollup**, then the **drift list** — drift is the load-bearing output. For each drift item, state which surfaces disagree and by how much; do not bury a real discrepancy under the routine ones.

**Work-units rollup** — when `work_units` is present, always include it in the summary:

- **Plan-weighted** — every plan = 1 unit, scaled by its reconciled percent. If the helper exposes a committed-lanes (e.g. KEEP+ACTIVE) subtotal, report both the all-plans figure and the committed figure separately: speculative/DRAFT plans count arithmetically but are not greenlit work, so the committed figure is the honest "live backlog" number.
- **Task-weighted** — sums per-plan phase counts. Phase data typically exists for only a subset of plans (scanners can't parse every notation), so always state the coverage (`coverage_plans` / `coverage_total`) and call it a **floor, not a full census** — do not present the phase total as complete.

Both numbers come straight from `work_units` in the helper output — do not recompute them by hand.

If the user asked specifically about *recent* completion, also run `git log --oneline -50` and name what landed in the last few days — reconciled percent fields lag, the git log does not.

## Step 4: Offer, do not auto-apply

End by offering to reconcile the drift. Resolving it (editing the project's plan-state files, re-running the project's render/reconcile step if it has one) is a separate, user-approved step. The audit itself never writes a plan file.

Wording template:

> Drift in plan `<id>`: surface A says `<value>`, surface B says `<value>`. Git history backs surface B. Want me to fix surface A? (Editing `<file>` is a separate step.)

## Drift thresholds (universal defaults; helper may override)

The helper decides what counts as drift — the skill renders what it reports. As a convention, helpers should flag:

- Reconciled-percent gap **≥ 25pp** between two authoritative surfaces (e.g. typed registry vs live state). Below that is normal ~1-release lag.
- A plan with a registry percent of 0% / no last-shipped-phase but **≥ 3 done claims** in git — the classic "registry never got updated" class.
- A committed-lane plan (KEEP-equivalent) **missing from one of the registries** — the surface that tracks live work cannot see it.
- Membership churn between non-committed states (DRAFT / PARK / TOMB equivalents) is **not** flagged — that is expected.

## Post-ship hook (project decides)

A common pattern: wire `<python> <helpers.plan_audit> --check` to run quietly after each phase ships, so drift is caught at the cause rather than a week later. `--check` exits 1 + one-line stderr on drift, silent otherwise — it does not block the ship, only surfaces a warning.

Wiring is per-project (a git `post-commit` hook, a `phased-plan` ceremony step, a CI gate, etc.). Use the `update-config` skill to register it in `.claude/settings.json` if you want it on the Claude Code event surface. Don't bake the hook wiring into this universal skill.

## Scope boundaries

- **Does not** rank what to ship next — that is the project's planning skill (if any), not this one.
- **Does not** garden the plan portfolio or close stale claims — that is the project's replan/cleanup skill (if any).
- **Does not** decide a plan's classification — that is operator-curated in the project's plan-state files.
- It **only** answers "how complete is each plan, and do the surfaces agree".

## Project-specific extensions

Helpers may expose additional modes (e.g. `--render-plans-yaml` to regenerate a typed registry from plan-doc front-matter). Those are project-local extensions of the orchestrator, not part of the universal contract. The universal contract is `--json` + `--markdown` + `--check`; anything else stays project-local and is invoked by the project's own skill or scripts, not by this universal skill.

If a project's release flow is genuinely different — e.g. it ships a wrapper skill at `<project>/.claude/skills/plan-audit/SKILL.md` to also regenerate a registry, stamp a ledger, or post to a notify channel — that local skill shadows this one and is the right place for that ceremony. See `.claude/skills/README.md` § "Per-project skill overrides".
