# Maintenance-loop tombstones — 2026-08-12

Status: **active retirement registry** for the recurring and one-shot tasks disabled by the
2026-08-11/12 maintenance audit. A disabled Scheduled Task is not self-explanatory, so this
registry records why it was stopped, what (if anything) supersedes it, and the evidence that
must exist before an operator may reactivate it.

## Operating rule

- A row is a tombstone, not a suggestion to delete the Scheduled Task. Keep the disabled
  definition long enough to preserve its payload and history.
- **Do not reactivate merely because the payload exists or Task Scheduler reports a prior 0.**
  Reactivation requires every condition in the row, a bounded canary, and an operator decision.
- If a replacement is named, reactivate the old task only if the replacement is deliberately
  retired and the old task has independently regained a measured advantage.
- One-shot rows are terminal: create a new, newly bounded task for a new campaign rather than
  re-enabling a task whose deadline or objective has already elapsed.
- GitHub issues are the change records. This note is the fleet-wide inventory and remains valid
  even if a task definition is later removed from one workstation.

## Tombstones

| Disabled task | Why it was disabled | Superseded by / current path | Reactivation gate | Evidence |
|---|---|---|---|---|
| `FakStallscanWatch` | Duplicate host sampler; its ledger contained thousands of malformed/scalar lines and duplicated the retained monitor. | On-demand `fak stallscan --json`; no recurring sampler until one canonical, measured loop exists. | Strict JSONL for a 10-minute soak, bounded rotation, one canonical binary, action/result and before/after counters. | #6496 |
| `FakStallMonitor` | The retained loop then proved non-useful: 2,741/2,741 samples said `stall`, ~16.5 MiB accrued, relief remained fail-closed `ABSTAIN`, and mitigation effects were not ledgered. | On-demand `fak stallscan` plus `fak terminal-relief`; no acting replacement. | Meet every #6496 done condition and prove at least one independently witnessed effective mitigation without unsafe terminal/session loss. | #6496 |
| `FleetScoutLoop` | Recurring scout had no successful recorded yield and shared the false-green scheduler class with logvault. | Operator-invoked `/scout-loop` for a selected lead. | Persist source, decision, filed issue/ship conversion, and nonzero failure status; demonstrate useful conversion in a bounded canary. | #6497 |
| `FleetStaleWorkGarden` | A live tick blocked inside `fak garden tick --timeout 900` for about 17 minutes, exceeding useful scheduler cadence. | Witnessed, operator-invoked `fak garden tick`; five clean folds already exist in `.fak/loops.jsonl`. | Bound the inner and outer timeout coherently, prove cancellation, and complete a soak without overlap or stranded processes. | #6493 |
| `FakMetaSuperloopNight100` | Enabled task pointed at a deleted scratch payload; it could not perform its advertised campaign. | None; durable checked-in dispatch tooling only. | Replace scratch payload with a committed first-class verb, bounded campaign state, admission check, and witnessed-close ledger. | #6499 |
| `FakOvernightMixedProfiles100` | Enabled overnight task had lost its scratch payload and therefore had no reproducible runnable definition. | None; use governed dispatch only after target-0 quarantine ends. | Committed payload, expiry, capacity/admission SSOT, per-worker witnesses, and bounded canary. | #6500 |
| `FakBenchmarkFleetLoop` | 8 requests yielded 0 successful numeric measurements (5 failed, 2 waiting, 1 stale running) while the scheduler reported green. | Explicit benchmark runs with read-back witnesses. | At least one successful numeric witness, typed nonzero failure/degraded results, stale-session recovery, and a green status derived from witnesses. | #6503 |
| `FakLogvaultCapture` | No successful recorded runs; scheduler success did not establish captured evidence. | Direct source logs remain authoritative; no automatic successor. | Prove capture writes a readable artifact, propagate failure, bound retention, and complete a multi-tick canary. | #6497 |
| `FakLogvaultVerify` | Verification loop had no successful recorded verification and could falsely appear healthy. | Explicit artifact read-back when evidence is consumed. | Verify independently authored artifacts, persist pass/fail, return nonzero on unreadable/stale input, and complete a multi-tick canary. | #6497 |
| `FleetDOSDispatchWatchdog` | Acting target watchdog conflicted with target-0 quarantine and another watchdog used a different desired population. | `dos loop --target 0` is the current typed authority; no live dispatcher. | One desired-population SSOT, target-0 regression test, final admission canary, and resolution of #6492/#6495. | #6502, #6492, #6495 |
| `FleetSupervisorWatchdog` | Conflicting target-4/target-8 automation could repopulate workers during quarantine. | Same target-0 DOS authority as above. | Same gates as `FleetDOSDispatchWatchdog`, plus status must enumerate every spawn-capable path. | #6502, #6505 |
| `FleetResolveProgress` | 472 of 499 nominal successes closed nothing (94.6%) yet emitted `claimed_done`/`witnessed_done`; only 33 total closures. | Explicit issue reconciliation with independent Git/GitHub witnesses. | No-effect is a typed non-success, claims require independent effects, and a canary shows useful closure rate with no false done. | #6504 |
| `FleetOwnerSeatResume` | Could enable/start dispatch after a seat reset and bypass target 0. | Manual recovery under the shared desired-population/admission gate. | Recovery reads the same SSOT, target 0 blocks starts, and a regression test covers reset paths. | #6505 |
| `FleetResumeWatchdog` | Acting resume loop could bypass target-0 quarantine. | Manual recovery under the shared gate. | Same #6505 gates; status must expose the resume path and its last decision. | #6505 |
| `FleetStrandedRecovery` | Acting stranded-worker recovery could repopulate quarantined dispatch. | Manual, witnessed reconciliation that preserves worker effects. | Same #6505 gates, plus worktree/process ownership must be independently read back before resume or reap. | #6505, #6510 |
| `FleetIdeaScout` | Created stock rather than throughput: 117 open scout issues remained untriaged while another dry run planned more; sampled closures were no-action/redundant. | Triage and convert the existing stock; operator-invoked scout only. | Establish conversion/SLA gate, drain or classify existing stock, handle source failures, and show PR/ship-backed conversion in a canary. | #6506 |
| `FleetBenchPlanDoc` | Scheduled documentation refresh modified tracked `docs/bench-plan.md` in the shared checkout and stranded WIP. | Generate/read status on demand; committed docs change only through an owned lane. | Write outside tracked tree or use serialized explicit-path commit, prove no peer-dirty mutation, and add consumer-safe failure handling. | #6507 |
| `FleetDispatchStatusDoc` | Scheduled status-doc generation could strand tracked shared-tree edits and reported a view inconsistent with final launch admission. | Live typed status commands; target remains 0. | No tracked-tree mutation, final admission reflected in status, and `READY_TO_GROW` cannot appear under target 0/cooldown. | #6507, #6495 |
| `FakFleetJanitorHeadless` | Scheduler stayed green while the janitor ledger was stale for ~2 days and ended in repeated `ENUMERATION_FAILED`; GPU state was unknowable. | Explicit GCP inventory/cleanup until janitor health is witnessed. | Fresh heartbeat every tick, typed nonzero enumeration failure, action/read-back evidence, and a successful bounded GCP canary. | #2341 regression evidence |
| `FleetWatchdogWatchdogAudit` | Bounded log contained 47 RED and 51 AMBER findings, but wrapper hardcoded exit 0. | Direct audit read-back; no false-green schedule. | Exit status derives from verdict, RED/AMBER is durable and actionable, and a canary proves scheduler result follows audit result. | #6509 |
| `FleetPushLagPusher` | 232 ticks were 85.3% no-op, 10.8% failed, and only 3.9% pushed; repeated rejected pushes still looked healthy. | Explicit `fak commit ...`/`fak sweep ... --push` by the owning lane. | Event-driven lag detection, typed failures, ownership/admission safety, and measured net benefit over explicit push. | #6511 |
| `FleetRunawayReaper` | Duplicate unledgered process killer overlapped maintained process-guard authority, making actions unattributable. | `FleetProcResourceGuard` detector plus explicit `fak process-guard ... --enact`; orphan tails remain a disjoint reaper. | Declare a unique uncovered failure class, persist findings/actions/before-after, and prove no overlap with process guard. | #6513 |
| `FakLabReadbackRetry20260811` | Completed bounded one-shot; its retry window and objective are over. | Final artifacts/read-back from that campaign. | **Never reactivate.** Create a newly named, newly bounded read-back task for a new campaign. | #6501 |
| `FakNightrun14h20260810` | Completed 14-hour one-shot remained enabled after its window. | Archived nightrun evidence. | **Never reactivate.** Create a new task with a new deadline and ledger. | #6501 |
| `FakOvernightReconcile20260811` | Completed one-shot reconciliation remained enabled after its evidence window. | Archived reconciliation result. | **Never reactivate.** New campaign requires a newly named bounded task. | #6501 |
| `FakPriorityP1PostReset20260809` | Completed post-reset one-shot remained enabled after its explicit 12-hour intent. | Normal issue dispatch/reconciliation, currently quarantined. | **Never reactivate.** Create a new bounded task only after launch admission is repaired. | #6501 |
| `FleetGLM52CampaignStop` | Stop timer had fired and its GLM-5.2 campaign was over; leaving it enabled added dead scheduler state. | None; campaign is closed. | **Never reactivate.** A future campaign owns a new stop task tied to its own campaign ID/deadline. | #6501 |
| `FakOvernightIsolatedNemotron` | Temporary one-minute refill maintained five unleased workers and recreated an unmanaged supervisor despite DOS target 0. | No replacement while quarantined; preserve existing effects and use DOS-governed launch only after repair. | Target-0 admission must be non-bypassable, all spawn paths visible, leases/witnesses mandatory, and #6492/#6495/#6505 resolved with a canary. | #6505 |

## Coverage boundary

This registry intentionally covers the **28 tasks disabled by the 2026-08-11/12 maintenance
audit**. Other disabled historical campaign/dispatch tasks on this workstation predate this audit
and are not silently claimed as reviewed here. They need the same schema before anyone treats
them as deliberate tombstones rather than unknown disabled state.

`\JobSearch\LightweightPushGardener` is not in the table because the audit could not disable it:
its ACL returned access denied. It remains an open action under #6512; once disabled, add its
actual transition timestamp and tombstone (duplicate unledgered pusher; explicit owned-lane push
supersedes it; reactivation requires a unique measured advantage and durable outcomes).

## Durability follow-up

Issue #6516 tracks a first-class task-tombstone verb and audit so future disables cannot lose reason, supersession, evidence, or reactivation policy when a task ACL refuses description mutation.

