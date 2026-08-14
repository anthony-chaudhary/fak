---
title: "Worktree ledger ownership — append-only telemetry belongs to the primary checkout"
description: "Inventory and ownership contract for cwd/root-relative durable writers under detached worker-worktree isolation (#3208)."
---

# Worktree ledger ownership

Date: 2026-08-13. Resolves [#3208](https://github.com/anthony-chaudhary/fak/issues/3208) under epic #3165.

## Decision

Append-only telemetry is control-plane state, not a worker's source delta. Every writer routed through `nightrunLedgerPath` therefore uses one **shared-root owner**: `FAK_LEDGER_ROOT` when the dispatcher supplies an absolute root, otherwise the primary checkout inferred from Git's absolute common directory. A process in `fak-worker-wt-*` must never create its own `.fak/nightrun` tree. The primary checkout resolves to itself, preserving existing behavior. If Git or the override is unavailable, the historical local `repoRoot()` fallback remains fail-open.

Land-exclusion was rejected as the primary rule: it prevents merge collisions but also discards worker observations. Shared-root ownership keeps those observations while removing them from the worktree delta. Existing append implementations retain their own locking/atomicity contracts; this decision changes ownership, not row serialization.

## Durable-writer inventory

The audit covered `cmd/fak` constants/calls containing `LedgerRel`, ledger paths, `.jsonl`, and durable append/write seams.

| Writer family | Representative seam | Ownership rule |
|---|---|---|
| Gateway usage | `serve.go`, `nightrun.go` → `gatewayusageledger.DefaultLedgerRel` | Shared root through `nightrunLedgerPath` |
| Cache value and savings | `run_model.go`, `serve.go`, `cachevalue_savings.go` → cache-value defaults | Shared root through `nightrunLedgerPath` |
| Harness resources | `harness_resources.go`, `fleet_res.go` → `harnessres.DefaultLedgerRel` | Shared root through `nightrunLedgerPath` |
| Fleet-status/history adapters | `fleet_metrics.go` paths recognized as nightrun telemetry | Shared root through `nightrunLedgerPath` |
| Guard compaction witness | `guard_compaction_witness.go` | Shared root through `nightrunLedgerPath` |
| Dispatch/DOS loop journals | `dispatch_progress.go`, `dispatch_tick_worker.go`, `.dos/metrics`, `.dispatch-runs` | Explicit operator/session `--root` or dispatch root; not an implicit nightrun writer, so ownership remains with that supplied control-plane root |
| Guard child transcript/tool journal | guard settings/session paths | Explicit session/config path; not rewritten or landed as source |
| Read-only audit/report paths | `audit_usage.go`, `dojo_lever_context_restore.go`, score readers | Caller-selected root; no durable writer ownership change |
| Debug/provider transcripts | provider home/project paths | External provider-owned state, outside the repository land delta |

No `docs/nightrun` writer remains on the current default constants; the active defaults live under `.fak/nightrun`. The contract applies to the path resolver rather than one directory spelling, so future default moves remain covered.

## Witness

`TestNightrunLedgerPathInsideWorktreeUsesPrimaryCheckout` creates a real primary Git checkout and detached linked worktree, changes cwd into the worker, resolves a ledger path, and proves both that the result is under the primary checkout and that no worker-local nightrun directory was created. `TestNightrunLedgerRootPrimaryCheckoutUnchanged` proves the primary-clone behavior is unchanged; the override test proves an absolute dispatcher-provided shared root wins.
