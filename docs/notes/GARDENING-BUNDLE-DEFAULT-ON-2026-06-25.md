---
title: "The gardening bundle: turn the cleanup passes we run by hand into one default-on workload"
description: "An inventory of fak's self-maintenance machinery (three overlapping orchestrators + the loop runtime) and a five-rung plan to fold the gardening/cleanup passes into one named bundle that runs by default on the loop kernel."
---

# The gardening bundle — default-on the cleanup we already do

Date: 2026-06-25.

The cleanup machinery is **already built**. What is missing is that almost none of
it runs unless a human types a slash command. We have thirteen deterministic
scorecards folded into one ratchet, a fleet loop catalog of nineteen gardening
checks, a durable loop ledger, and a tunable governor — and the only recurring
workload any OS scheduler actually fires is issue-dispatch. The gardening passes
are dogfood we do not eat on a cadence.

So this is not a "build the machine" memo. It is a **converge-and-arm** memo: name
the bundle once, point it at the runtime we already have, and turn it on by default
with a kill switch. This is the gardening-workload companion to
[`LONG-RUNNING-AGENT-LOOPS-2026-06-25`](LONG-RUNNING-AGENT-LOOPS-2026-06-25.md)
(which builds the generic loop kernel) — the gardening bundle is the **first real
recurring workload** that loop kernel should carry, and the proof that it carries
anything besides dispatch.

## What we actually have (the inventory)

Three orchestrators already exist. Each folds a *different* slice; none of them is
the whole. That fragmentation is the core finding.

| Orchestrator | File | Folds | Run today |
|---|---|---|---|
| **Scorecard control pane** | `tools/scorecard_control_pane.py` | 13 `*-debt` scorecards → one portfolio `total_debt` + per-metric early-warning + CI ratchet | **Default-on in CI** (`.github/workflows/ci.yml:442` runs `--check`). The measurement half is solved. |
| **Fresh status** | `tools/fresh_status.py` (`make status`) | 4 top-level domains (git · benchmarks · work · industry) → one snapshot | Advisory; `make status-check` exists, not in the hard `ci` chain |
| **Fleet control pane** | `tools/fleet_control_pane.py` + `tools/control_pane.loops.json` | 19 gardening *loops* as `(status_cmd, timeout, action, auto_recover)` specs: scorecard ratchet, readme-freshness, docs/seo scorecards, issue-closure-audit, memory-recall-audit, gofmt-debt, public-leak-scan, DOS supervisor health, … | `loop-audit` / `tick` run them; scheduling is per-machine via `register_control_pane_tick.{ps1,sh}` + `release-cadence.yml`. **Only `scan-needles` has `auto_recover`** — the other 18 are status-only. |

The loop **runtime** the bundle would ride is also built:

- `internal/loopmgr` — append-only, SHA-256 hash-chained JSONL ledger (`fak.loop-event.v1`): `armed → fire → admit → start → heartbeat → end → witness → notify → control`, folded by `Summarize()` into per-loop fire/admit/refused/witnessed counts and control state.
- `internal/loopmgr/governor.go` — pure `Admit(loop, policy, now) → Decision` with reason codes `LOOP_PAUSED` / `CADENCE_FLOOR` / `REFUSAL_STORM` / `WITNESS_COLLAPSE`. Operator controls write one policy file (`.fak/loop-policy.json`); no OS-task re-registration.
- `fak loop run|admit|control|status|append` — `run` records fire/admit/start/end automatically; `admit` exits 0/3 so a cron line can gate (`fak loop admit --loop X && <work>`); `control` writes pause/resume/disable/enable policy changes and an audited control row; default ledger `.fak/loops.jsonl`.
- OS schedulers — launchd (`com.fak.dogfood-fleet.plist`, 30 min), systemd (`gcp-dogfood-control-vm.sh`, 30 min), Windows Scheduled Tasks. **All fire issue-dispatch. None fire a gardening loop.**

And the dogfood-as-default pattern is already proven once, for a single tool:
`.github/workflows/dogfood-coverage.yml` does **gate + daily cadence** for
`tools/dogfood_coverage.py` (`--check` exits 1 on HARD debt; `schedule:` re-emits
the payload daily to the step summary). That workflow is the template every
default-on gardening pass should copy. The conceptual sibling
`tools/recent_feature_dogfood.py` already exists to prove *new* features get
exercised; the gardening bundle is its *recurring* analogue — the passes that rot
not because they are new, but because nobody re-runs them.

## The three real gaps

The work decomposes cleanly. These are independent and separately shippable.

1. **No single bundle.** Three orchestrators, three entry points, three mental
   models. "Run the gardening" is not one command. A new contributor (or agent)
   cannot discover the maintenance surface from one verb.

2. **The RSI/fix half never auto-fires.** The fleet catalog runs the *status* side
   of nineteen loops, but eighteen are `auto_recover: false` — they detect drift
   and stop. The *retire-the-debt* side (the `quality-score` / `appeal-score` /
   `curate-cluster` / `memory-compact` / `issue-triage` skills) only runs when a
   human invokes the slash command. Detection is on a cadence; remediation is not.

3. **Nothing arms a gardening loop on a real scheduler.** `fleet_control_pane tick`
   is the runner, but its scheduling is thin and per-machine. No launchd plist,
   systemd timer, Windows task, or `schedule:` workflow fires the bundle. "Default
   on" today means "on if you hand-registered a tick on this box."

## The design: one bundle, ridden on the kernel we have, default-on with a brake

The shape mirrors the scorecard control pane's own success — *one named fold, one
ratchet, one baseline* — and the dogfood-coverage workflow's *gate + cadence*.

**The bundle is a manifest, not new code.** It is the union of the read-only
gardening passes that already emit a control-pane payload, plus the explicit set of
RSI skills allowed to run unattended. Keep the manifest in one tracked file so
adding a pass is a data edit, not a code change — the same discipline the scorecard
family already uses (`SCORECARDS` in the control pane, `control_pane.loops.json` in
the fleet pane).

The default-on contract has three properties, in priority order:

- **Read-only by default; auto-fix is opt-in per pass.** A gardening tick that runs
  unattended must be safe to ignore. The audits (control pane, fresh-status, the
  fleet `loop-audit`) run every tick and only *report*. A pass graduates to auto-fix
  only when (a) its change is mechanical and reversible (gofmt, index-sync,
  baseline re-pin), and (b) the result is witnessed by `dos commit-audit` before it
  is trusted — never on the skill's own say-so. This is the
  [`dos-witness-claim`](../../.claude/skills/dos-witness-claim) discipline applied to
  the fleet's own maintenance.

- **One brake the operator already knows.** Reuse the governor. `FAK_GARDEN=off`
  (env) and `{"loops": {"garden/default": {"paused": true}}}` in
  `.fak/loop-policy.json` both stop it; `min_interval_seconds` floors the cadence so
  ticks can't pile up; `witness_collapse` holds the auto-fix loops if their proof
  rate falls. No new kill-switch vocabulary — the loop kernel's is the brake.

- **The ledger is the proof, the scorecard baseline is the floor.** Every tick
  writes fire/admit/start/end/witness to `.fak/loops.jsonl`, so "did gardening
  actually run, and did it prove its work" is one `fak loop status` query — not a
  self-report. The scorecard ratchet's pinned baseline stays the regression floor;
  the bundle's job is to keep the portfolio at-or-below it and re-pin after a drop.

### Why a bundle and not "just schedule each skill"

Because the failure mode of N independent cron lines is exactly what we have now:
nineteen detached checks, no shared admission, no shared brake, no single answer to
"is the repo getting better or worse this week." The bundle's value is the *fold* —
one verdict, one ledger lane, one policy knob — the same reason the scorecard
control pane beats thirteen separate `--check` invocations. The anti-pattern to
avoid is a fourth orchestrator: the bundle must *converge* the three we have behind
one verb, not add a competitor to them.

## The rollout (five rungs, each independently shippable & green)

Built so each rung leaves the trunk green and is useful alone — the project's
rung discipline. Earlier rungs de-risk later ones.

- **Rung 0 — the read-only bundle (no scheduler, no auto-fix).** One `make garden`
  target (and `tools/garden_bundle.py`) that runs the fast read-only folds —
  `scorecard_control_pane --check`, `fresh_status` — and prints one rolled-up
  verdict. `--deep` adds the slower `fleet_control_pane loop-audit`. Wrap it in
  `fak loop run --loop garden/default --` so the first invocation already lands in
  the ledger. *Proof: `make garden` exits 0 on a clean tree, the ledger shows one
  `garden/default` run, no working-tree mutation.*

- **Rung 1 — arm it on the real schedulers.** Add a `schedule:` workflow (clone
  `dogfood-coverage.yml`) so the bundle's payload is re-emitted daily in CI, and add
  the host tick — a `--mode garden` path on the existing
  `register_control_pane_tick.{sh,ps1}` installer — that fires `fak loop run --loop
  garden/default -- make garden`. *Proof: a scheduled run appears in the GitHub step
  summary and in `.fak/loops.jsonl` on a dogfood host.*

- **Rung 2 — governor policy defaults + kill switch.** Ship a default
  `garden/default` policy block (`min_interval_seconds`) as a tracked template and
  wire `FAK_GARDEN=off`. Document the brake in `docs/ROLLBACK.md` next to the
  scorecard ratchet entry. *Proof: `fak loop admit --loop garden/default` refuses
  with `CADENCE_FLOOR` inside the floor and `LOOP_PAUSED` when paused;
  `FAK_GARDEN=off` skips the tick.*

- **Rung 3 — auto-fix RSI with a witness gate.** Promote the mechanical, reversible
  passes (gofmt, index-sync, baseline re-pin, curate-cluster's gitignore step) to
  `auto_recover: true` in the catalog, each gated by `dos commit-audit` on the
  commit it produces — refuse to fold an unwitnessed "fixed it." Leave the
  judgment-heavy skills (quality-score's god-function splits, appeal-score's
  rewrites, memory-compact's deletions) as *propose-only*: the tick opens the work,
  a human or a fork commits it. *Proof: a planted gofmt regression is auto-fixed and
  the fix carries a `dos`-witnessed commit; a planted overclaim is only flagged, not
  silently rewritten.*

- **Rung 4 — one `fak garden` verb.** Fold the bundle into the kernel as a first-
  class verb (`cmd/fak`, additive leaf via `tools/new_leaf.py`), so the Go binary —
  not a Makefile and not Python — owns the gardening fold the way it owns
  `fak loop`. `fak garden status` / `fak garden tick` / `fak garden audit`. *Proof:
  the verb resolves on a clean `go install`, `architest` stays green, and the
  Makefile target becomes a thin shim over it.*

## The one risk to name

Auto-fix on a shared trunk is where this bites. The repo is a live multi-session
tree; an unattended gardening tick that commits must obey the same discipline a
human session does — commit by explicit path, never `git add -A`, defer to the
trunk guard, wait out a peer's `MERGE_HEAD`, never force-push (see
[[fak-shared-tree-high-churn-commit]]). That is precisely why Rung 3 gates every
auto-fix behind `dos commit-audit` and keeps the judgment-heavy passes propose-only:
the bundle's blast radius must stay inside mechanical, reversible, witnessed
changes until the witness path has earned trust. The brake (Rung 2) ships *before*
the auto-fix (Rung 3) for the same reason.

## Bottom line

We already paid for the machine. The scorecard half is default-on; the loop kernel
is built; the dogfood-coverage workflow is a working template for "gate + cadence."
The remaining work is convergence and arming, not invention: one bundle manifest,
one verb, one scheduled tick on the kernel we have, read-only first and auto-fix
last behind a witness gate. The end state is that "keep the repo clean" stops being
a thing a human remembers to do and becomes a workload the kernel carries — the most
honest possible dogfood of fak's own pitch as the loop kernel.
