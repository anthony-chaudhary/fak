<!-- fak-agent-key: qwen38-vulkan-mtp-planner-gateway-v1 -->
<!-- target-repo: fak -->
<!-- fak-public-issue: anthony-chaudhary/fak#12721 -->

# feat(agent): activate resident Vulkan Qwen3.8 MTP in the production planner

```routing
lane: agent
paths: ["internal/agent/inkernel_planner.go", "internal/agent/chat.go", "internal/agent/inkernel_speculative_vulkan_mtp_test.go", "internal/gateway/gateway.go", "internal/gateway/chat_completions.go", "internal/gateway/http.go", "internal/gateway/stream_proxy.go", "internal/gateway/vulkan_mtp_test.go"]
expected_steps: 8
repo: fak
public_issue: anthony-chaudhary/fak#12721
```

## Parent context

Parent: #12184. Coordinates with #12720, which owns the model/compute API for
raw pre-final-normalization target hidden rows through resident Vulkan target
verification. Promotion requires #12720. Also coordinates with #12538, #12687,
and #12714.

## Current state

At public source `c8c7150621b3eeaeadc976ee7d4e68fdf5a8cc27`, commit
`7b115e5c38dac1824f061523eac04b3356062949` supplies an explicit backend-aware
resident Vulkan MTP draft session and commit
`98c0b8a51389f175959b79af85cfabd8795dffe6` supplies resident Vulkan target
verification with partial-prefix recurrent repair. Those halves are not joined
on production traffic.

`internal/gateway/gateway.go:803-815` selects Metal MTP through
`EnableMetalMTP` or Vulkan prompt n-gram speculation through
`EnableSpeculativeDecoding`. `FAK_SPECULATIVE=mtp` on a Vulkan Qwen3.8 planner
therefore retains MTP tensors at load time but does not construct or select a
Vulkan MTP proposer. `internal/agent/inkernel_planner.go:674-681` accepts a
long-lived proposal generator, while the backend-bound MTP draft session must
be created against each request's live target `Session` before a cold target's
prompt prefill arms and captures its raw hidden rows. A restored target may be
reused only when it already carries complete target-hidden history; otherwise
the request must cold-reset, construct, then prefill. The draft must be closed
on completion, cancellation, or downgrade.

Centrality: Core. Portfolio tier: 1 (fak all in one) and tier 2 (serving).
Priority: P0.

- P1 Context: advanced; makes the shipped native MTP path reachable from `fak serve`.
- P2 Net value: advanced; lets accepted drafts use the resident one-panel target path.
- P3 Adaptation: preserved; exact greedy target verification remains authoritative.
- P4 Operations: advanced; exposes selected proposer, downgrade reason, acceptance,
  replay, and lifecycle counters from the request that actually ran.

## Core through-line

`FAK_SPECULATIVE=mtp` on an admitted Qwen3.8 Vulkan model -> restore a reusable
target snapshot when available -> bind one request-owned MTP draft if its raw
hidden history is complete, otherwise cold-reset the target -> construct the
draft before prompt prefill so capture is armed -> prefill -> adapt it with
`MTPProposalGenerator` -> execute the existing greedy resident-device verifier
for K=1..4 -> emit accepted/correction tokens -> close the draft exactly once ->
report the actual route and counters.

## Working spine

Gateway MTP admission -> request-bound planner draft factory -> backend-aware
MTP proposal generator -> existing Vulkan target transaction -> exact emitted
tokens and observed counters -> deterministic lifecycle receipt.

## Why this is next

The resident draft, target panel, and zero-full-replay partial commit have
shipped, but no production constructor connects them. Until this leaf lands,
the large MTP work is reachable only from direct model tests and one-shot helper
calls; Halo product traffic can select only n-gram speculation.

## Gold-plating boundary

No new shader, target-verification algorithm, recurrent repair, model loader,
HTTP schema, adaptive draft-depth policy, non-greedy sampling, multi-request
speculative batching, or default-on promotion. Do not change Metal MTP behavior.
Do not claim a Halo speedup; physical qualification remains under #12184 and
the existing hardware witness tickets.

## Done condition

An explicitly selected supported Qwen3.8 Vulkan request uses the retained MTP
draft head and existing resident-device target verifier for bounded K=1..4,
releases the request-bound draft on every owned exit, and reports the actual MTP
route. This integration consumes the K/cut token and state-parity contract
validated by #12720. Unsupported admission envelopes remain on ordinary target
decode without selecting or claiming Vulkan MTP. For an admitted request,
constructor or proposal failure emits a typed request-local downgrade before
target speculative mutation; a resident-verifier downgrade completes that round
through the exact generic verifier, then stops subsequent drafting.

## Scoped Acceptance Criteria

- [x] [SW-VERIFIED] The planner owns a request-scoped Vulkan MTP factory/config.
  It binds a restored target only with complete raw-hidden history; otherwise it
  cold-resets, constructs the draft before prompt prefill, then prefills so raw
  target-hidden capture is armed. Mutable draft state is never shared across
  requests or tenants.
- [x] [SW-VERIFIED] `FAK_SPECULATIVE=mtp` selects the Vulkan MTP proposer only
  for an admitted Qwen3.8 hybrid model, retained MTP tensors, Q4_K target, Vulkan
  backend, K=1..4 target panel, and the target-hidden capability dependency.
- [x] [SW-VERIFIED] The proposer reaches the existing resident-device verifier
  and preserves its #12720-validated K=1..4 acceptance, correction, rollback,
  and complete target-state contract.
- [x] [SW-VERIFIED] Constructor refusal, proposal failure, cancellation, and
  normal completion release the request-owned draft; restored snapshots without
  complete raw-hidden history take the cold-reset/construct/prefill path.
- [x] [SW-VERIFIED] Unsupported backend, format, model, missing retained head, or
  unavailable target-hidden capability is not admitted and makes no Vulkan MTP
  execution claim; invalid depth is rejected as configuration. For an admitted
  request, constructor/proposal failure or resident-verifier downgrade emits a
  typed request-local reason and continues ordinary native decode. A verifier
  downgrade stops later drafting after one exact generic verification round.
- [x] [SW-VERIFIED] Metal MTP and explicit `FAK_SPECULATIVE=ngram` retain their
  existing selector semantics. Conflicting primary/alternate settings do not
  admit the new Vulkan MTP route.
- [x] [SW-VERIFIED] Request observability identifies proposer `mtp`, backend
  `vulkan`, requested/effective depth, proposals, accepted/rejected tokens,
  target operations, full replay steps, recurrent-only repairs, fallback reason,
  cancellation, and elapsed time from observed execution.
- [x] [SW-VERIFIED] `internal/agent/chat.go` carries request-local actual route
  and downgrade/completion data through buffered JSON and the live
  `internal/gateway/stream_proxy.go` SSE path. Headers or trailers come from that
  request only; concurrent requests cannot observe or overwrite another result.
- [ ] [HW-WITNESSED] Physical Halo end-to-end A/B remains unchecked here and is
  qualified separately with N>=5 matched samples, exact token parity, zero hidden
  fallback, complete source/artifact/system provenance, P50/P90, and confidence
  intervals before any performance or default-on claim.

## Definition of done

- [x] Production gateway construction reaches request-scoped resident Vulkan MTP.
- [x] Gateway-selected execution preserves the #12720 K/cut parity contract.
- [x] Constructor/proposal downgrade, cancellation, and completion release the draft.
- [x] Focused agent and gateway witnesses plus boundary and leak checks pass.

## Witness

Source repro before implementation:

```bash
rg -n "shouldEnableMetalMTP|shouldEnableNGramSpeculative|EnableMetalMTP|EnableSpeculativeDecoding" internal/gateway internal/agent/inkernel_planner.go
```

Deterministic regression:

```bash
go test ./internal/agent -run '^TestInKernelPlannerVulkanMTP(RequestLifecycle|GreedyParity|RestoredUncapturedPrefixReprefillsCold|ResidentVerifierDowngradesOnce|ExecutionReceiptAccountsResidentTransaction|Cancellation|CompleteStreamPreservesErrorReceipt)$' -count=1 -v
go test ./internal/gateway -run '^TestGateway(SelectsVulkanMTPOnlyForAdmittedEnvelope|VulkanMTPConcurrentReceiptsRemainRequestLocal)$' -count=1 -v
```

The agent test uses an independently instrumented backend/factory to observe
constructor, proposal, resident target-operation, close, cold re-prefill,
single runtime-verifier downgrade, token parity, cancellation, and error-receipt
propagation. Its controlled resident receipt accepts one token, rejects one,
rolls back one, records one target operation and one full replay, and records
zero recurrent repair. The gateway tests cover buffered JSON headers and live
SSE trailers, including concurrent JSON success and SSE downgrade requests on
the same `Server` without cross-request metadata. A configuration predicate
alone is not a production-path witness.

## Software validation record

Validated from dependency source `6e1c5eb7c4c998211ba6764b7314e3eff0682594`
plus this staged phase-2 diff. Implementation ran in the `mtp_code_scout`
context; the independent `hardware_route` context authored and ran the focused
and full-package witnesses:

```bash
go test ./internal/agent -run '^TestInKernelPlannerVulkanMTP' -count=1 -v
go test ./internal/gateway -run '^TestGateway(SelectsVulkanMTPOnlyForAdmittedEnvelope|VulkanMTPConcurrentReceiptsRemainRequestLocal)$' -count=1 -v
go test ./internal/agent ./internal/gateway -count=1
```

The focused run passed every software MTP group; the opt-in
`TestInKernelPlannerVulkanMTPRealArtifact` skipped because no artifact was
provided. Full packages passed in 103.803s (`internal/agent`) and 65.817s
(`internal/gateway`). Logs: `.gotmp-linux/mtp-focused-no-overlay.log` and
`.gotmp-linux/mtp-full-no-overlay.log`. The eight-Go-file boundary check reported
zero violations, and the staged public leak audit and ticket validation passed.

This record makes no Halo performance claim. It also does not claim a standard
tagged Vulkan artifact build: the current Linux link requires explicit `-lm`
because a newly shared shim references `sincosf`; the normal linker declaration
is being handled separately. Hardware qualification remains unchecked.

Optional physical correctness witness after the software gates pass:

```bash
FAK_VULKAN_MTP_PLANNER_GGUF=/path/to/admitted-qwen38-q4_k.gguf FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/agent -run '^TestInKernelPlannerVulkanMTPRealArtifact$' -count=1 -v
```

This optional test checks real-artifact route and token correctness only. It is
not a Halo performance measurement or evidence for a speed/default-on claim.

## Verifiable Witness

Run the two deterministic focused commands above from a clean source checkout.
They witness the phase-2 lifecycle, multi-token greedy parity, controlled
resident receipt accounting, admission, concurrent request-local HTTP metadata,
SSE trailers, cold re-prefill, and one-way generic verifier downgrade. Reuse
#12720's model/compute tests as the authority for complete target state at K=1..4
and every rejection cut. The optional artifact test remains a separate physical
correctness gate.

## Done condition / witness

The gateway witness constructs an `InKernelPlanner` configured for request-bound
Vulkan MTP on the admitted envelope. Separate agent lifecycle witnesses observe
the retained-head proposal, route it through the existing device verifier,
preserve #12720's parity contract, and close the request-owned draft. Focused
phase-2 tests cover request-local metadata on both buffered JSON and live SSE;
physical speed remains a separate parent-campaign qualification.

## File:Line Seams

- `internal/agent/inkernel_planner.go:674-681` (persistent proposal-generator configuration)
- `internal/agent/inkernel_planner.go:973-1055` (request speculative loop and verifier)
- `internal/gateway/gateway.go:803-815` (Metal MTP and Vulkan n-gram activation only)
- `internal/gateway/chat_completions.go:138-168` (backend-specific mode admission)
- `internal/gateway/stream_proxy.go` (live SSE completion and trailer propagation)

## Blast Radius & Affected Lanes

- Affected: `internal/agent` request lifecycle and `internal/gateway` native
  planner admission.
- Unaffected: `internal/model` and `internal/compute` mechanisms owned by the
  prerequisite leaf, Metal MTP, loaders, generic KV eviction, external-provider
  planners, HTTP payload schemas, installer, and private commercial policy.

## Quarantined Fallback Mechanism

Vulkan MTP remains explicit opt-in. A failed admission capability check leaves
the request on ordinary fak-native target decode without claiming Vulkan MTP.
After admission, constructor or proposal failure returns a typed route reason
before target speculative mutation. A resident-verifier downgrade reports its
typed reason, completes the current proposal through the exact generic verifier,
and disables later drafting. No fallback silently selects n-gram, Metal, a CPU
draft, or an external runtime.

## Likely files

- `internal/agent/inkernel_planner.go`
- `internal/agent/chat.go`
- `internal/agent/inkernel_speculative_vulkan_mtp_test.go`
- `internal/gateway/gateway.go`
- `internal/gateway/chat_completions.go`
- `internal/gateway/http.go`
- `internal/gateway/stream_proxy.go`
- `internal/gateway/vulkan_mtp_test.go`

## Lane

Public core runtime phase 2: `internal/agent` plus `internal/gateway`. Develop
against the agreed #12720 typed capability in an isolated tree; integrate and
test the production join before landing.

## Expected steps

8

## Acceptance gate

The focused agent and gateway witnesses pass, the production join preserves the
#12720 parity contract, owned lifecycle exits release request state, both JSON
and SSE report actual execution, and public boundary/leak checks pass. Physical
performance evidence is deliberately separate.

## Closure binding

Resolve with an explicit-path public commit referencing this issue, parent
#12184, the model/compute prerequisite issue, and the independent test context.

## Work estimate

Estimate: 5 points.

## Overall completion contribution

Contribution: 5/8 points toward #12184's production integration objective.

## Completion standard

development
