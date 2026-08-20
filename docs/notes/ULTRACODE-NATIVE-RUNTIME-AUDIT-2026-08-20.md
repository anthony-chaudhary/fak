---
title: "Native Ultracode runtime audit: what works, what is observed, and what is not yet proven"
description: "Revision-bound audit of fak ultracode planning, Codex launch, status evidence, guard coverage, effect/witness/reconciliation gaps, and the issue path to a verified native runtime."
date: 2026-08-20
issue: 8286
---

# Native Ultracode runtime audit — 2026-08-20

## Verdict

`fak ultracode` is a working first-class front door for a deterministic orchestration
plan and a guarded, concurrent, read-only Codex inspection launch. It is **not yet a
verified coding-fleet runtime**: the launcher does not acquire the plan's declared
leases, launch a lead/reconciler, exchange declared effects, independently witness
them, or emit a reconciled terminal verdict. No paired evidence currently proves a
speed, token, cache, spend, or accepted-outcome advantage.

The trustworthy operator reading is therefore:

- **PROVEN:** command discovery, profile resolution, bounded plan shape, guarded Codex
  worker spawn, launch receipts, PID/log/turn progress, and explicit outcome
  uncertainty.
- **NOT PROVEN:** implementation effects, lease enforcement, worker-to-lead exchange,
  independent witness acceptance, reconciliation, useful-work completion, or fleet
  value.

Audit revisions: `cmd/fak@r3130+gd0a568f9` and
`internal/orchestration@r6+ga50b98cc`. The status truth fence shipped in
[`d0a568f9`](https://github.com/anthony-chaudhary/fak/commit/d0a568f9f983a86077577a439996aa3d62ed3122)
and closed [#8251](https://github.com/anthony-chaudhary/fak/issues/8251).

## Four different surfaces share the name

Treat these as related adapters, not interchangeable evidence:

| Surface | What it actually is | Evidence boundary |
| --- | --- | --- |
| `fak ultracode` | Top-level alias over the canonical `orchestration` resolver, limited Codex launcher, and status reader. | The subject of this audit. |
| `fak accounts launch --ultracode=on` | Claude-only launch posture that injects `--settings '{"ultracode":true}'` under `fak guard`. | Proves provider setting/argv construction, not the native orchestration runtime. |
| `fleet-wave` / `super-loop` | Repository operating workflows that launch and harvest guarded issue workers through dispatch/DOS machinery. | Mature operational precedent, but explicitly described as Ultracode-like rather than the `fak ultracode` runtime. |
| `qwen36codedemo` “fak UltraCode” | Browser demo branding over a pure-fak/Qwen coding-agent and fan-out surface. | Proves that demo's HTTP/model/cache path, not canonical orchestration plan execution. |

The older operational-mode explainer remains the concurrency-metric authority. The
new command did not add a second binary or kernel primitive; it added a first-class
product front door over orchestration code.

## End-to-end evidence matrix

| Layer | Current authority | Verdict | What the evidence proves | What it does not prove |
| --- | --- | --- | --- | --- |
| Discovery | `cmd/fak/ultracode.go`, root help, `docs/cli-reference.md` | **PROVEN** | `fak ultracode`, `status`, `--json`, `--launch`, and `--selfcheck` are reachable. | That launching produces correct work. |
| Plan resolution | `internal/orchestration.Resolve`; `TestUltracodeResolvesCanonicalFleetPlan`; exact-tip `--selfcheck` | **PROVEN** | Stable `fak-orchestration-plan/1`; four roles by default; 65,536-token cap; leases, independent witness, effect readback, and reconciliation are required in the resolved plan. | That any declared policy is realized at runtime. |
| Launch | `cmd/fak/orchestration_launch.go`; launch/receipt tests | **PROVEN, NARROW** | Non-lead roles start as `fak guard -- codex exec --json`; receipts bind session, run, task, route, PID, and log. A three-worker live launch was captured on 2026-08-18. | A lead is launched, tasks are disjoint, leases are acquired, or reports are gathered. |
| Worker behavior | `orchestrationWorkerPrompt` | **PROVEN, NARROW** | Workers are explicitly read-only inspectors and may return evidence-linked reports. | The advertised general “coding-agent fleet” implements or lands changes. |
| Progress/liveness | `cmd/fak/orchestration_status.go`; status tests; live JSON witness | **PROVEN** | Receipt selection, process liveness, log bytes/freshness, last event, and turn start/completion counts are joined read-only. | `turn.completed` means an accepted effect or successful workflow. |
| Security path | `orchestrationWorkerArgs`; per-worker guard audit path; argv tests | **PROVEN IN CONSTRUCTION** | Every launched Codex child is wrapped by `fak guard`, with a distinct audit-file path. | The status output does not join guard verdicts, and the committed live status witness contains no policy-effect audit. |
| Lease enforcement | Plan JSON plus launch source | **MISSING AT RUNTIME** | The plan says `taskmgr` leases are required. | The launcher never calls the lease/arbitration seam before spawn. [#5970](https://github.com/anthony-chaudhary/fak/issues/5970) and [#5971](https://github.com/anthony-chaudhary/fak/issues/5971) own this. |
| Effects and artifacts | Read-only worker logs only | **MISSING** | Reports may exist inside Codex JSONL output. | No typed effect/artifact contract, declared write set, acceptance record, or worker-to-lead channel exists. #5970/#5971 own this. |
| Independent witness | Plan declaration only | **MISSING AT RUNTIME** | The resolver requests independent witness and effect readback. | No launch/status record identifies a witness, verdict, corroborated effect, contradiction, or missing evidence. #5971 owns this. |
| Reconciliation/terminal outcome | Plan declaration; #8251 status truth fence | **MISSING AT RUNTIME; NOW HONESTLY OBSERVED** | Status now emits `outcome.verdict=unverified` with effects, witness, and reconciliation `not_observed`; human output calls `complete` a worker-execution state. | No reconciler runs, and no successful workflow verdict can be emitted. #5971 owns this. |
| Tokens, cache, spend, elapsed value | None in orchestration status | **MISSING** | Codex logs are retained for later analysis. | No normalized per-run usage or same-workload single-vs-fleet result exists. [#8168](https://github.com/anthony-chaudhary/fak/issues/8168) owns the paired proof. |
| Longitudinal health | Current status is one-run/local | **MISSING** | A selected or newest launch can be inspected. | Outcome counters and a health scorecard do not exist. [#7366](https://github.com/anthony-chaudhary/fak/issues/7366) and [#7367](https://github.com/anthony-chaudhary/fak/issues/7367) own them. |

## What the captured evidence says

The offline committed-tip probe returned:

```text
SELFCHECK PASS schema=fak-orchestration-plan/1 offline=true launched=0
```

The resolved JSON contained one lead plus three workers, required leases, independent
witness/effect readback, required reconciliation, and an `ultra` SOL route. This is a
strong resolver/schema witness and deliberately says `launched=0`.

The committed live witness
[`orchestration-status-live-2026-08-18.json`](../_witnesses/orchestration-status-live-2026-08-18.json)
shows a real three-worker launch: two logs had a completed turn and one worker was live
with an active turn. It proves spawn plus observation. It contains no effect, witness,
reconciliation, or accepted-outcome record, so it cannot prove useful-work completion.

The #8251 regression fixture now plants **two** completed worker logs and proves that
both JSON and human output remain outcome-unverified. Post-push verification ran the
committed `cmd/fak` status tests through WSL at `d0a568f9` and returned
`CLAIM_TEST_GREEN` (`fak validate` `ok:true`, tested package
`github.com/anthony-chaudhary/fak/cmd/fak`).

## Scorecards: useful, but not proof of this runtime

The claim-repro scorecard returned 100/100 on both the committed snapshot and live
tree: every row in `CLAIMS.md` and `BENCHMARK-AUTHORITY.md` was falsifiable. Native
Ultracode has no row in either corpus, so that perfect score **does not evaluate this
runtime**. A runtime or value claim should enter the ledger only with #5971 or #8168's
real effect/value artifact; the absence of a premature claim is currently preferable
to a plan-shaped `SHIPPED` claim.

The agent-readiness scorecard measured the broader live tree at 93.8/A with seven
friction-debt units. Those defects concern generic onboarding paths/configuration and
do not establish whether Ultracode executes or reconciles work.

## Reproduce the proven boundary

From the repository root:

```bash
fak ultracode --selfcheck
fak ultracode --task-text "fan out independent checks and reconcile them" --json
fak ultracode status --json
```

Read the commands in order:

1. `--selfcheck` proves stable offline plan serialization and launches nothing.
2. The JSON plan proves requested policy and route shape, not runtime enforcement.
3. `status` requires a prior launch receipt and reports worker execution separately
   from the now-explicit unverified workflow outcome.

A verified native coding fleet requires the remaining path in this order:

1. [#5970](https://github.com/anthony-chaudhary/fak/issues/5970): Codex adapter with
   pre-launch collision denial, declared artifacts, deterministic failure handling,
   and a live effect/witness receipt.
2. [#5971](https://github.com/anthony-chaudhary/fak/issues/5971): owned executor,
   supervisor, effect readback, independent witness, reconciliation, resume, and
   terminal verdict.
3. [#8168](https://github.com/anthony-chaudhary/fak/issues/8168): same-workload paired
   run for accepted outcome, wall time, billed/cache tokens, spend, and witness
   acceptance.
4. #7366/#7367: longitudinal outcome counts and health scoring after the terminal
   record exists.

Until those witnesses land, use native Ultracode for plan inspection and bounded
parallel read-only analysis—not as evidence that a self-reconciling implementation
fleet or a net-positive fleet multiple has shipped.
