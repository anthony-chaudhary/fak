# Subsystem Checks and Boundary Benchmarks

This file is the map for "what subsystem is working?" checks. These are not
product-outcome claims. A passing row means the named boundary, adapter, cache,
or invariant is alive and mechanically checked; it does not by itself prove
production readiness, task quality, or the Phase 0 / Phase 1 product gates.

## Fast Run Sets

Use these when you want subsystem signal without running every benchmark artifact:

```powershell
cd fak

# Architecture, policy, syscall, and security boundaries.
go test ./internal/architest ./internal/abi ./internal/adjudicator ./internal/policy ./internal/ctxmmu ./internal/normgate ./internal/ifc ./internal/provenance ./internal/plancfi ./internal/witness ./internal/bench ./internal/metrics

# Wire, persistence, cache, and fleet-benchmark controls.
go test ./internal/gateway ./internal/recall ./internal/cdb ./internal/radixkv ./internal/kvmmu ./internal/vdso ./internal/turnbench

# Model/runtime correctness boundaries.
go test ./internal/model ./internal/compute ./internal/modelengine

# Python gate/tooling logic.
python -m pytest tools
```

Use the full repo gate when you need release confidence:

```powershell
cd fak
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ci.ps1
```

## Recording and Regression Protection

For audit-grade evidence, run the catalog through the JSON/Markdown recorder
instead of copying terminal output by hand:

```powershell
python tools\subsystem_check_audit.py --profile smoke --out-json fak\experiments\subsystem-checks\latest-smoke.json --out-md fak\experiments\subsystem-checks\latest-smoke.md
```

The recorder stores the exact command, working directory, exit code, elapsed
time, output hash, stdout/stderr tails, app version, git commit, and dirty-tree
digest. Use `--write-baseline fak\experiments\subsystem-checks\baseline-smoke.json`
only after a passing run; future runs can compare with:

```powershell
python tools\subsystem_check_audit.py --profile smoke --baseline fak\experiments\subsystem-checks\baseline-smoke.json --out-json fak\experiments\subsystem-checks\latest-smoke.json --out-md fak\experiments\subsystem-checks\latest-smoke.md
```

Regression rules:

- A baseline passing check that now fails is a hard failure.
- A baseline check that disappears is a hard failure.
- Command drift under the same check id is a hard failure; update the baseline
  intentionally when the boundary contract changes.
- Duration regression is a warning by default because local timing is noisy; use
  `--fail-on-duration-regression` and `--duration-min-slack-ms 0` for stable CI
  hardware or dedicated benchmark lanes.
- Keep small checked-in baselines separate from timestamped run archives. The
  baseline is the contract; the run archive is the evidence for a specific
  machine, git state, and command output.

## Boundary Ledger

| Subsystem | Check command | Load-bearing tests or artifacts | What it proves | What it does not prove |
|---|---|---|---|---|
| Architecture / request-path hygiene | `go test ./internal/architest` | `TestNoUpwardImports`, `TestHotPathHasNoExec`, `TestRequestPathLeavesRegistered`, `TestSingleOpenAIChatClient`, `TestRequestPathInterpreterFree`, `TestOracleSeamStaysOffPath` | The internal package graph remains layered; live request-path packages do not import `os/exec`; self-registering leaves are wired; the OpenAI client and Python oracle seams stay in their assigned places. | Runtime correctness of a subsystem. It is a structural guard, not a behavior benchmark. |
| ABI / closed vocabulary | `go test ./internal/abi` | `TestABIGoldenFreeze`, `TestClosedReasonVocabulary`, `TestFoldRankOrdering` | The syscall ABI is additive-only and refusal reasons remain a closed set with stable rank ordering. | That any particular policy is safe or sufficient. |
| Capability adjudication floor | `go test ./internal/adjudicator ./internal/policy` | `TestEmptyPolicyDefaultDeny`, `TestSelfModifyDeniedWithBoundedWitness`, `TestReasonsAreInClosedVocab`, `TestDevAgentDeniesGitMutations`, `TestAdjudicateP50UnderOneMillisecond`, policy round-trip tests | Unknown tools default-deny; self-modify returns a bounded witness; dev-agent dangerous operations are blocked; policy manifests load fail-loud. | That allow-listed dangerous tools are argument-safe, or that a model behaves well. |
| Syscall boundary tax | `go test ./internal/bench ./internal/metrics`; optional `go run ./cmd/fak bench --suite tau2-smoke --out report.json` | `TestRunArm_VDSOAblationChangesPath`, `TestRun_BothArmsPopulated_NoSpawn`, `TestComputeGate`, `report.json`, `baseline.json` | The in-process decide path is resident and not paying a per-call spawned-hook boundary; the vDSO on/off ablation really changes the path. | Production readiness, model quality, serving throughput, the Phase 0 45x gate, or a win over a long-lived sidecar. |
| Result admission / context-MMU | `go test ./internal/ctxmmu ./internal/normgate` | `TestAdmitSecretQuarantines`, `TestAdmitPoisonFixture`, `TestAdmitOversizeBenignTransforms`, `TestPageInGatedByClear`, `TestQuarantinesObfuscatedInjection`, `TestParaphraseEvadesByDesign` | Poison/secret/oversize result handling works at the admission boundary; `Clear()` gates page-in; the normalized detector improves catch rate while preserving the documented semantic residual. | That detection is complete. The detector remains non-load-bearing. |
| Information-flow / provenance / plan-CFI / witness | `go test ./internal/ifc ./internal/provenance ./internal/plancfi ./internal/witness` | `TestForgedSelfTrustCannotEvadeTaint`, `TestVDSOHitDoesNotLaunderTaint`, `TestModelCannotAuthorTrust`, `TestDeviationEscalates`, `TestCleanClaim`, `TestRealGitAncestor` | Trust and taint are kernel-authored; vDSO hits do not launder taint; plan deviations escalate; claims can be checked against git evidence. | That every real-world sink or plan has been modeled. |
| Gateway / wire boundary | `go test ./internal/gateway` | `TestServedResultArmsResultSideStack`, `TestChatProxyOpenAICompatibleToolResultsAreQuarantinedPreSend`, `TestHTTPSyscallDenyIsValueNot5xx`, `TestMCPStdioRoundtrip`, `TestHTTPMetricsEndpointExposesGatewayAndKernelCounters`, policy reload/reset tests | HTTP/MCP/OpenAI-compatible adapters route through the same kernel mechanisms; served result admission is armed; denials are values, not transport crashes; metrics and reload/reset surfaces work. | That an arbitrary external provider is live, authenticated, cheap, or semantically equivalent. |
| Durable recall / process boundary | `go test ./internal/recall ./internal/cdb` | `TestQuarantineSurvivesTheSessionBoundary`, `TestClearIsNecessaryButNotSufficient`, `TestTightenedReScreenCatchesObfuscatedPoisonOnReload`, `TestCorruptCASFailsClosed`, `TestExamineSealedIsRefusedAcrossTheBoundary`, `TestWorkingSetIsASmallResidentSlice` | Quarantine survives persist/reload; page-in re-screens content; corrupt CAS fails closed; the debugger can inspect without leaking sealed bytes. | That long-term memory improves task quality or recall precision. |
| KV quarantine / cache reuse | `go test ./internal/kvmmu ./internal/radixkv ./internal/cachemeta ./internal/vdso` | `TestWriteTimeEvictEqualsNeverSaw`, `TestEvictionIsContentDrivenNotPositional`, `TestReuseThroughSplitMatchesRecompute`, `TestNodeCacheEntryDescribesTokenPrefix`, vDSO soundness/invalidation tests | Poison-driven KV eviction equals never having seen the result; radix reuse through a split matches recompute; cache entries carry token/model/security metadata; writes invalidate reads. | That a live external serving engine shares KV zero-copy. That remains a separate product gate. |
| Model kernel / loader correctness | `go test ./internal/model` | `TestForwardMatchesHFOracle`, `TestGreedyMatchesHFOracle`, `TestKVPrefixReuseMatchesRecompute`, `TestBatchFromPrefixMatchesIndependentPrefill`, `TestQuantMatchesF32Logits`, `TestQuantTeacherForcedAgreement`, safetensors streaming/quant tests | The Go model path matches its oracles, prefix reuse is correct, quantized paths meet their own quality gates, and loaders do not silently change semantics. | That the model is capable enough for production tasks. |
| Compute HAL / device abstraction | `go test ./internal/compute ./internal/model -run "HAL|Device|Evict|Clone|Q8Dispatch"`; optional hardware lane `powershell -File internal\compute\build_vulkan.ps1 test` | `TestHALSessionMatchesLegacyCPUReference`, `TestDeviceTensorHasNoHostPointer`, `TestCorrectnessClassEnforcement`, `TestRegistryPickAndDefault`, Vulkan/CUDA tagged tests where hardware exists | The backend seam can drive the model loop and enforce correctness classes; reference HAL is byte-identical to legacy CPU; hardware-specific lanes can be tested separately. | That a non-reference GPU backend is production-ready, or that the Phase 1 7-9B rung is satisfied. |
| Turn-tax mechanism controls | `go test ./internal/turnbench -run "Run_|HappyPath|VDSO|Safety"` | `TestRun_AirlineClassesAreLiveKernelEvents`, `TestRun_VDSOAblationIsARealPathSwap`, `TestRun_SafetyFloorIsSeparateFromTurnTax`, `TestRun_HappyPathSavesNothing` | Each turn-tax class is a live kernel event; vDSO is a real path swap; safety events are separated from turn savings; clean happy path saves zero. | General task-quality improvement or arbitrary model-turn reduction. |
| Fleet / fan-out anti-inflation controls | `go test ./internal/turnbench -run "Fleet_|Fanout"` | `TestFleet_NoShareHasZeroCrossUplift`, `TestFleet_SingleAgentHasNoCrossUplift`, `TestFleet_FinerEraserPushesCrossoverOut`, `TestFanoutNoShareZeroUplift`, `TestFanoutSingleAgentNoUplift`, `TestFanoutPrefixGeometry` | Cross-agent savings vanish under no-share/single-agent controls; scoped invalidation changes the write-rate crossover; fan-out prefix geometry is explicit. | That every production fleet workload is read-heavy enough to benefit. |
| Phase 1 capability gate logic | `go test ./internal/turnbench -run Phase1CapabilityGate`; CLI: `go run ./cmd/paritybench ... --require-phase1` | `TestPhase1CapabilityGateRequiresLiveLocalGPU7BParity`, `TestPhase1CapabilityGateFailsWithoutGPU7B`, `TestPhase1CapabilityGateRejectsCPU7BAsNonCPUEvidence` | The gate refuses stale or wrong-class evidence and requires live local-GPU 7-9B parity evidence. | That such hardware evidence exists today. |
| API-host bridge tooling | `python -m pytest tools/api_host_*_test.py` | acceptance, roster, readiness, live inventory, qualification, bridge matrix/proof/verify-all, retry packet tests | Candidate-host and bridge artifacts fail closed on missing auth, unsupported wire, stale or malformed artifacts, and declared requirement drift. | That a specific live provider is available or paid for. |
<!-- DGX / endpoint-load artifact-integrity check excluded from the public copy (operator-private lab infra); its tools/dgx_*_test.py targets are private-side. -->
| Release / process safety | `python -m pytest tools/release_lock_test.py tools/safe_ff_sync_test.py`; optional `python -m pytest tools/fleet_*_test.py` | release lock, safe fast-forward sync, fleet account/bottleneck/control tests | Operator workflows fail closed on unsafe release ownership, unsafe sync, malformed fleet health inputs, and bottleneck classification drift. | Product performance or model quality. |

## How To Read These Checks

- A subsystem check should have a clear boundary: import graph, policy load,
  HTTP adapter, page-in, KV eviction, model oracle, or artifact gate.
- Prefer controls that can go to zero: no-share fleet, single-agent fan-out, happy
  path turn-tax, vDSO off, missing GPU rung, malformed artifact.
- Keep these checks out of the product headline. They tell us which layer is
  alive; the production gates are still the clean-node Phase 0 reproduction, the
  Phase 1 capability/backend rung, and any live workload partner validation.
