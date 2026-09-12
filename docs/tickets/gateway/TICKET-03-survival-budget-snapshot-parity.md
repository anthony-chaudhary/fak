<!-- fak-gateway-key: survival-budget-snapshot-parity-3965 -->

# fix(gateway): pin the live configured survival budget for each request

Ref #3965. Status: open follow-up; no production fix or reload witness completed.

GitHub tracker: [#3965](https://github.com/anthony-chaudhary/fak/issues/3965).
Source-evidence update: [acceptance extension](https://github.com/anthony-chaudhary/fak/issues/3965#issuecomment-5645556493).

This repository ticket is the durable companion to that update. It is carried as
follow-up documentation in the #12583 proof-receipt patch; it does not close #3965.

## Acceptance extension: survival planning still reads the launch budget

Evidence status: **source-verified; reload regression witness NOT RUN**. This extends existing open #3965 rather than creating a duplicate. Discovered while independently testing #12583; no production fix is included in that proof-receipt change.

### Problem and value
For operators changing compaction budgets on a running gateway, the next request must use one pinned configured budget throughout compaction, survival planning, and readback. Today the published scalar config can change while survival planning still uses the launch-copied budget. A restart with the new launch budget avoids that disagreement but loses the live session comparison. Carrying one configured budget through the existing request path permits coherent restart-free tuning.

Priority hierarchy: all-in-one serving/harness/memory; centrality: Core.
P1 context: advances coherent bounded context decisions. P2 net value: removes the need to restart for this setting; no measured latency or token-savings claim. P3 adaptation: closes a source mismatch in the existing hot-swap mechanism. P4 operations: behavior and readback must identify the same configuration generation.

### Exact source evidence
Source base: `2d9d99f29d097f75dfdd4e30e3a42d5dc41aa84a` (isolated inspection worktree).

- [messages.go:747](https://github.com/anthony-chaudhary/fak/blob/2d9d99f29d097f75dfdd4e30e3a42d5dc41aa84a/internal/gateway/messages.go#L747): `compactAnthropicRawWithReason` loads `ScalarConfig()` and takes the live `CompactHistoryBudget`.
- [control.go:139](https://github.com/anthony-chaudhary/fak/blob/2d9d99f29d097f75dfdd4e30e3a42d5dc41aa84a/internal/gateway/control.go#L139): `PatchScalarConfig` updates the immutable snapshot and publishes it at line 157.
- [pinsurvival.go:349](https://github.com/anthony-chaudhary/fak/blob/2d9d99f29d097f75dfdd4e30e3a42d5dc41aa84a/internal/gateway/pinsurvival.go#L349): `ctxplan.PlanEviction(pages, s.compactHistoryBudget)` uses the launch field.
- [pinsurvival.go:366](https://github.com/anthony-chaudhary/fak/blob/2d9d99f29d097f75dfdd4e30e3a42d5dc41aa84a/internal/gateway/pinsurvival.go#L366): the retention-annotated application also receives `s.compactHistoryBudget`.
- [pinsurvival.go:319](https://github.com/anthony-chaudhary/fak/blob/2d9d99f29d097f75dfdd4e30e3a42d5dc41aa84a/internal/gateway/pinsurvival.go#L319) explains a required distinction: the survival budget is the **configured** budget, not the early-firing ramp's potentially smaller `opts.Budget`. Replacing the stale read with `opts.Budget` would violate that existing invariant.
- Corroborating readback scope already owned by #3965: `ctxvalue.go:436,522,594,610` still reads the launch field.

### Bounded implementation packet
Lane: gateway. Initial code seam: `messages.go` and `pinsurvival.go`; one focused regression file. Pin the configured scalar budget at the request decision, pass it explicitly through survival planning/application, and preserve the separately derived firing budget. Reuse #3965's existing readback/generation scope rather than adding a second setter or changing runtime defaults. Do not mutate launch fields concurrently or reload the atomic snapshot inside each helper.

### Required witness (planned, unrun)
Proposed focused test: `TestContextBudgetReloadSurvivalConsistency`.
Construct a gateway with launch budget zero; send a valid live control update to a positive budget that can retain the pinned floor. Drive a real Anthropic request through the gateway to a loopback upstream and capture the forwarded bytes. Assert that the next request compacts using the live configured budget, preserves pinned/prefix bytes, and reports coherent generation/readback. Include the retention-annotated branch and preserve the configured-budget-versus-ramp distinction. An overlapping update must not mix generations. Keep last-good invalid-update assertions from #3965.

Future command, only after the test exists:
`go test -race ./internal/gateway -run '^TestContextBudgetReloadSurvivalConsistency$' -count=1 -v`

No reload test was authored or executed for this note; no reload operating envelope or runtime fix is claimed. Existing #3965 operating-envelope numbers are not revalidated by this source inspection. This is separate from the held-SSE fixture's repetition guard and does not weaken #12583 proof assertions.

### Typed coordination and closure
Coordinates with: #12583 (discovery only; proof receipts can ship independently).
Coordinates with: #10867 (closed Tier-0 scalar-control implementation).
No new start blocker. #3965 closure should require the focused real-path reload witness and independent review of the configured-budget/firing-budget distinction.


## Validation baseline follow-up — #11983

Status: OPEN; independently reproduced during #12583 validation. This subsection tracks existing architecture-suite debt separately from the #3965 budget defect above. GitHub evidence: https://github.com/anthony-chaudhary/fak/issues/11983#issuecomment-5647456000. DOS prose is additionally tracked by #12671; harnesskit fixture repair by #12672.

On untouched published parent `514c26818bb00d1390751ac1de15ae1751c768ea`, the isolated `go test -p=1 -count=1 -timeout=180s` architecture cohort reproduced all nine failures:

- `TestDOSCLIProseUsesManWedgeAndKeepsMCPIdentifier`
- `TestThirdPartyEffectsRegisterCoversTree`
- `TestEffectRegisterMutationsAreCaught`
- `TestHarnessKitLockV2Export`
- `TestHarnesskitExternalImportBoundary`
- `TestHarnesskitUpgradeContractFromCleanModule`
- `TestSBOMMatchesGoMod`
- `TestSBOMDriftGateCatchesMutations`
- `TestNoStaleZeroDependencyClaim`

The result was FAIL, exit 1, 3.954s. Captured log: `/root/.local/state/fak-land/codex-top10/12583-parent-architest-cohort.log`; SHA256 `45482f6b4c3f6ece3eac90c09b7fa5d63f13caf1d66987027d521aa80a425219`. The same cohort fails on the proof candidate; no green full-architecture-suite claim is made. Inventory mismatches include x/sys v0.46.0 in the SBOM versus v0.48.0 in go.mod, missing third-party effects, stale DOS prose, external-module fixtures needing tidy, and a companion-document phrase classified as a stale dependency-count claim.

Acceptance for #11983: reconcile the actual module/tree inventory and precise documentation wording, retain the existing guards, and demonstrate green named regressions followed by the full `internal/architest` package. These existing failures do not justify weakening a gate. The proof patch's own test-isolation defects are repaired and verified within #12583, not deferred to this baseline follow-up.


## Readiness test synchronization — #12910

The full gateway race run exposed concurrent test-only reads and writes of the readiness log accumulator. An isolated run on untouched parent `514c26818bb00d1390751ac1de15ae1751c768ea` reproduced `TestGatewayReadyLogging` failing under the race detector (exit 1, cohort 0.933s). Parent log: `/root/.local/state/fak-land/codex-top10/12583-parent-gateway-race-cohort.log`; SHA256 `97c27dd6ba3c95e1ff141b11dc067564b47364d7bf9be0b11c8bbf4fcc1387f5`.

The independent fixture correction protects log append/snapshot with a mutex and captures the listener address before starting Serve. It preserves the readiness assertions, three-second deadline, cancellation and join. The corrected exact test passed under the race detector (1.071s); log `/root/.local/state/fak-land/codex-top10/proof12583-fixture/gateway-ready-logging-race.log`. The test-hygiene gate also passed (0.053s). This changes test synchronization, not production serving behavior. Full gateway race and strict validation remain required before the parent task closes.
