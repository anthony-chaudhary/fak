# fix(serve): select the registered native Vulkan backend automatically

<!-- fak-serve-key: halo-auto-vulkan-backend -->

GitHub issue: [#12722](https://github.com/anthony-chaudhary/fak/issues/12722).

```routing
lane: cmd
paths: ["cmd/fak/serve_model_load.go", "cmd/fak/serve_backend_auto_test.go", "cmd/fak/serve_memory_admission_test.go", "cmd/fak/serve_memory_admission.go", "cmd/fak/serve_test.go", "cmd/fak/up.go", "cmd/fak/up_backend_auto_test.go", "cmd/fak/guard.go", "cmd/fak/scout_native.go", "cmd/fak/serve.go", "cmd/fak/run_model.go", "docs/model-engine-env.md", "docs/fak/server-config.md", "docs/fak/advanced-topics.md", "docs/tickets/halo-backend-auto/TICKET-01.md"]
expected_steps: 4
```

## Parent context

Parent: #11572. Coordinates with #3420. This is a software startup leaf of the
useful-local-agent milestone; physical performance promotion remains #11963.

## Why now

Native Vulkan prefill and retained context are implemented, but normal local
startup can bypass them by leaving the backend unset. Connecting the existing
execution path is necessary before adding more kernel optimizations.

## Current state

At public revision `c8c7150621b3eeaeadc976ee7d4e68fdf5a8cc27`,
`cmd/fak/serve_model_load.go:149-153` returns a nil backend when the selector is
empty. `cmd/fak/up.go:434` passes that empty selector through the ordinary local
startup path; `cmd/fak/serve_stages.go:257-272` does the same for an omitted
`--backend`. As a result, a usable Vulkan backend may be registered but never
selected for native model loading and execution. Metal already has automatic
selection in `cmd/fak/serve_model_load.go:183-209`.

`internal/compute/vulkan.go:136-174` registers Vulkan only after shader
configuration and device initialization succeed. Selection must consume that
existing readiness decision. A build tag alone is insufficient evidence.

`cmd/fak/serve.go:700-709` also checks only the raw Vulkan flag before
`cmd/fak/serve_memory_admission.go:287-295` acquires a residency lease. Automatic
and environment selection must pass through that same admission check.

## Problem frame

- Centrality: Core.
- P1 Context: advanced - ordinary local startup reaches the available native GPU.
- P2 Net value: advanced - avoids an accidental CPU execution path and manual backend selection.
- P3 Adaptation: preserved - explicit backend choices and existing Metal behavior remain supported.
- P4 Operations: advanced - unavailable explicit choices fail visibly; native receipts retain actual backend identity.

For local native-model users / Problem: a registered GPU is ignored by default /
Today: callers must name Vulkan manually / Better because: the normal entry
point chooses an initialized backend / Witness: independent resolver tests and
existing startup/backend tests. Throughput improvement remains unmeasured.

## Working spine

1. Reproduce omitted-selector behavior with an independent available-device fixture.
2. Resolve CLI, environment, and automatic Vulkan selection through one shared policy.
3. Verify CPU and Metal behavior and explicit unavailable-name diagnostics.
4. Land the resolver, independent tests, and this ticket with the test receipt.

## Core through-line

`fak up` or `fak serve --gguf` -> shared selector -> initialized native Vulkan
backend -> existing model loader and planner -> existing native execution receipt.

Use explicit CLI selection before `FAK_BACKEND`. Automatic selection on Linux
and Windows considers the initialized Vulkan backend; it does not guess from
arbitrary registered accelerators. Preserve the native CPU floor when Vulkan is
unavailable and preserve automatic Metal on supported Apple hosts. Explicit
unavailable choices must fail rather than change execution engines.

## Gold-plating boundary

No new GPU backend, shader compilation workflow, model download, speculative
decoding policy, deployment, hardware tuning, or benchmark superiority claim.
The shared run/guard startup callers retain GPU residency for their command lifetime; the reusable scout keeps its existing CPU-first policy. The backend flag help and environment reference change with the selection
contract. The private appliance service is outside this work unit.

## Done condition

- [x] [SW-VERIFIED] Automatic selection reaches registered Vulkan on Linux/Windows.
- [x] [SW-VERIFIED] Explicit CLI selection overrides environment selection.
- [x] [SW-VERIFIED] An unavailable explicit selection fails with the selected name.
- [x] [SW-VERIFIED] CPU override, missing-Vulkan fallback, and Metal selection remain coherent.
- [x] [SW-VERIFIED] Automatic and environment-selected Vulkan retain residency admission; proxy mode remains excluded.
- [x] [SW-VERIFIED] Turnkey startup and local run/guard retain the GPU lease through model lifetime and release on errors/close.
- [x] [SW-VERIFIED] The reusable scout preserves its CPU-first policy until it has a device lifetime contract.
- [x] [SW-VERIFIED] Independent focused tests and existing backend tests pass.

Hardware throughput and complete model/device qualification are outside this
software wiring leaf and remain unmeasured. The broader native performance
campaign is tracked by #11572 and #11963.

## Witness

`go test ./cmd/fak -run 'TestServeBackendAuto' -count=1` must execute the independent behavioral regression and pass after the change.

### Verifiable Witness

Pre-change source observation:

```text
git show c8c7150621b3eeaeadc976ee7d4e68fdf5a8cc27:cmd/fak/serve_model_load.go
```

Observed: `resolveServeChatBackend` returns `nil, nil` for an empty selector
before consulting the registered backends.

Behavioral regression command:

```text
go test ./cmd/fak -run 'TestServeBackendAuto' -count=1
```

### File:Line Seam

`cmd/fak/serve_model_load.go:149-209` (backend resolver, diagnostics, Metal selection).

### Blast Radius & Affected Lanes

The public `cmd/fak` backend-selection path changes. Compute kernels, model
interfaces, private commercial policy, and factory coordination remain outside
the write surface.

## Definition of done

Land the shared resolver and its independent behavioral regression with green
focused tests, explicit selected-backend diagnostics, and no changes to the
native model/backend interfaces. Model execution and hardware speed are qualified
separately by the existing performance campaign.

### Quarantined Fallback Mechanism

Use the existing native CPU floor when automatic Vulkan selection has no usable
registration. Test registries model availability without invoking hardware. No
quarantine is necessary for unrelated lanes and no external engine is a fallback.

## Likely files

- `cmd/fak/serve_model_load.go`
- `cmd/fak/serve_backend_auto_test.go`
- `cmd/fak/serve_memory_admission_test.go` (effective selection and lease lifecycle)
- `cmd/fak/serve_memory_admission.go` (admission contract documentation)
- `cmd/fak/serve_test.go` (isolate selector environment)
- `cmd/fak/up.go` (hold GPU residency for the turnkey server lifetime)
- `cmd/fak/up_backend_auto_test.go` (independent turnkey admission regression)
- `cmd/fak/serve.go` (effective Vulkan residency admission and backend flag help)
- `cmd/fak/run_model.go` (backend flag help and one-shot/REPL residency lifetime)
- `cmd/fak/guard.go` (local model residency across the guarded command)
- `cmd/fak/scout_native.go` (preserve the documented CPU-first classifier)
- `docs/model-engine-env.md` (environment selection contract)
- `docs/fak/server-config.md` (matching environment reference)
- `docs/fak/advanced-topics.md` (matching environment reference)
- `docs/tickets/halo-backend-auto/TICKET-01.md`

## Lane

Public `cmd/fak`; exclusive file scope under `halo-backend-auto`.

## Expected steps

4: reproduce, implement shared selection, verify independent tests, land.

## Acceptance gate

The independent `TestServeBackendAuto` regression and existing backend-selection
tests pass with explicit CPU, Vulkan availability, environment precedence, and
Metal compatibility covered. The changed CLI package compiles without a GPU.

## Closure binding

Close only after the scoped resolver and regression are committed with the
passing software witness and a `(fak cmd)` trailer. Do not claim hardware
performance or broader milestone completion from this leaf.

## Software verification record

On 2026-09-10, the independent focused Linux regression passed in 5.509 seconds:

```text
GOWORK=off go test ./cmd/fak -run 'Test(ServeBackendAuto|UpBackendAuto|IsServeVulkan|LoadServeModelWithVulkanLease|ResolveServeMetal)' -count=1
```

The native Windows selector/admission regression also passed (35.503 seconds):

```text
GOWORK=off go test ./cmd/fak -run '^TestServeBackendAuto' -count=1
```

The tests register a CPU-backed sentinel in isolated child processes and load a
small synthetic GGUF for startup/lifecycle checks. They prove software selection,
admission ordering, error cleanup, and lease retention until active requests end;
they do not measure a GPU or validate a production model. Separate implementation
and test-author contexts produced and reviewed the change.
The same independent tests compiled against all seven original production files
from `c8c7150621b3eeaeadc976ee7d4e68fdf5a8cc27` using a Go overlay and failed
behaviorally (exit 1, 7.236 seconds). Failures include ignored automatic/environment
selection and omitted serve/turnkey residency admission. This is a real parent
regression, not a compilation failure.

After #12680 landed at `f0af47ecd0a56eb5d5e769fdcc6f6e4eac124a05`,
the backend patch was integrated onto that base without replacing its widened
planner interface or structured tool-call wire. The combined backend, residency,
Metal, and three turnkey tool-loop regressions passed on Linux (3.441 seconds).
The integrated lifecycle and tool-loop race run also passed (12.676 seconds):

```text
go test ./cmd/fak -run '^(TestServeBackendAuto.*|TestUpBackendAuto.*|TestIsServeVulkan|TestLoadServeModelWithVulkanLeaseUsesEffectiveBackendAndGGUF|TestResolveServeMetal|TestServeMetalFlagDefaultsFalse|TestTurnkeyServer(RejectsDroppedToolCall|ToolCallRoundTrip|StreamsStructuredToolCallWithoutExecutingIt))$' -count=1
go test -race ./cmd/fak -run '^(TestUpBackendAuto.*|TestTurnkeyServer(RejectsDroppedToolCall|ToolCallRoundTrip|StreamsStructuredToolCallWithoutExecutingIt))$' -count=1
```

Both integrated Linux commands set the managed worktree's explicit `GIT_DIR`
and `GIT_WORK_TREE`; no physical device or ambient backend registration was used.

The full CLI package diagnostic was **not green**: a 90-second global run reported
`TestMemoryPromptAssemblyVerifiedFreshOnly`, `TestArchCheckMineClean`, and
`TestRunAttestJSONShape` failures before the aggregate timeout. The active benchmark
test at timeout had only recently started; this is not evidence of an isolated
benchmark hang. The memory verifier failure was traced to missing managed-worktree
Git metadata in the WSL environment and passed in 0.684 seconds when `GIT_DIR` and
`GIT_WORK_TREE` were supplied. The aggregate diagnostic does not classify the
remaining output as source defects in this change.
Scoped RED/GREEN evidence, normal prospective build checks, and ordinary commit
hooks are retained; no unsafe symptom-skip flag or test-selection environment
workaround is used to represent the full package as passing.
