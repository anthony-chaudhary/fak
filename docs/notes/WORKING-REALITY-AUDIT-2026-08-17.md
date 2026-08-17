# Working-reality audit — 2026-08-17

## Verdict

FAK is shipping real improvements, but two of its own status surfaces currently reward activity and static wiring more strongly than completed, witnessed operation. That makes the system look more finished than the evidence supports and sends operators toward “keep shipping” when the more important action is to retire inventory.

This is not a claim that nothing works. Lane leases are a strong counterexample: this audit used `dos arbitrate` and a journaled exclusive lease to select one disjoint path, then released it. The control was usable without repository mutation or manual collision reasoning. The miss is that FAK does not yet apply the same evidence standard to maturity and overall progress.

## Captured evidence

All observations below came from the live shared checkout on 2026-08-17. Counts are snapshots, not owned-work claims.

| Surface | Observed result | What the evidence actually proves | Gap |
|---|---|---|---|
| `fak progress --json` | `PROGRESS`; 3 commits; 972 WIP files; 1,520 open issues; reason `delivered-work evidence exists`; next `keep shipping` | Commits landed during the ten-minute window while a large inventory existed | Commit activity is sufficient for the positive verdict; the command does not establish that total inventory is shrinking |
| `fak maturity next --json` | 550 capabilities: 398 `dogfooded`, 69 `default`, 76 `tested`, 1 `prototyped`, 6 `proposed`; score 74/C | The scanner found production import reachability, tests, docs mentions, and benchmark-shaped source | `internal/maturity` sets `Dogfooded` from a production import. Static integration is not runtime use or a successful outcome |
| GitHub open inventory | 1,520 open: 28 P0, 200 P1, 233 P2, 1,059 without a priority label; 183 `gen/now` | The backlog has substantial active and unclassified inventory | A positive ten-minute progress verdict is not a portfolio-convergence claim |
| `fak tree-doctor --json` | no hygiene action; six untracked Go files classified live/resident/abandoned | The tree doctor distinguishes hygiene from durable WIP and does not delete it | Correct behavior; a smaller status count must not be fabricated by cleanup |
| DOS lane arbitration | disjoint path admitted, exclusive lease journaled, later released | Collision-safe concurrent work is operational on this path | Strong working baseline worth copying: explicit scope, external state, refusal semantics, lifecycle closure |
| Fleet bottleneck helper | `health: down`, no headline/counts/top rows | The configured helper could not produce a current system bottleneck map | The standing bottleneck loop currently has no useful system-level signal; do not interpret empty rows as “no bottleneck” |

Module witnesses at the captured tip: `cmd/fak r2867+g0acddf03ab`; `internal/maturity r7+ge309417393`.

## Highest-leverage inversions

1. **Imports count as dogfooding.** `internal/maturity/maturity.go` builds a set of imported internal packages from non-test Go files outside the package and sets `Dogfooded` when the package import appears. This is valuable *integrated* evidence, but it does not prove invocation, sustained use, or outcome. Because 398/550 capabilities occupy this rung, the semantic mismatch dominates the maturity picture. Issue #7116 makes runtime evidence mandatory while retaining integration as a separate fact.
2. **Commits count as convergence.** `fak progress` returns `PROGRESS` as soon as delivered-work evidence exists. During this audit WIP grew from the earlier 964-file snapshot to 972 and open issues from 1,516 to 1,520, yet the recommendation remained `keep shipping`. The command is an activity detector, not yet a finishability control. Issue #7117 adds opening/closing inventory deltas and convergence verdicts.
3. **Backlog scale outruns prioritization.** 1,059 of 1,520 open issues have no priority label, while 183 are already `gen/now`. More issue creation or worker fan-out is not the first response; the next loop must reduce and classify inventory, with explicit closure evidence.
4. **The bottleneck observer fails empty.** The configured read-only helper returned `health: down` with no ranked rows. Its failure is visible, which is better than fabricated data, but there is no fallback current system diagnosis. This audit therefore treats the system bottleneck layer as unavailable rather than clean.

## Shift-left operating loop

Run this loop at the start and end of WIP-clearing waves:

1. **Capture:** record `fak progress --json`, `fak maturity next --json`, `fak sweep --json`, `fak tree-doctor --json`, and issue-priority totals. Preserve unavailable/error states.
2. **Separate evidence classes:** label each signal `declared`, `statically integrated`, `test-witnessed`, `runtime-witnessed`, `default-on`, or `outcome-measured`. Never promote across classes by implication.
3. **Measure flow balance:** compare opening and closing WIP magnitude/age, open issue count, priority debt, and closures. Commits are delivered-work evidence, not convergence evidence.
4. **Pick the dominant miss:** fix the status/control surface that is misleading the next decision before launching more implementation. A false-green control compounds every subsequent action.
5. **Finish one coherent unit:** acquire a disjoint lane, reproduce first, implement, validate only owned paths, commit once, push, and independently read back the witness.
6. **Retire or file residuals:** every concrete leftover becomes a deduplicated issue with a done condition; avoid prose-only deferral.
7. **Re-capture:** require improvement in the relevant evidence class. If inventory grew or a witness remains unavailable, report `DIVERGING` or `UNKNOWN`, not progress.

## Immediate queue

- **#7116 (P1, gen/now):** make maturity dogfooding require runtime evidence, not imports.
- **#7117 (P1, gen/now):** make `fak progress` measure convergence, not commit activity.
- **#7098:** shipped during this audit window and added WIP magnitude/age signals; wire those signals into #7117 rather than stopping at display.
- **#7094 / #7095 / #7096:** the benefit-harm admission and default audit cluster is directionally aligned, but should consume the evidence classes above rather than treating declared records as operational proof.

## Completion boundary

This note establishes the recurring audit frame and captures concrete false-positive controls. It does not claim the repository is now converging. The goal remains active until #7116 and #7117 are shipped and the live re-capture demonstrates that imports no longer masquerade as dogfooding and activity no longer masquerades as convergence.
