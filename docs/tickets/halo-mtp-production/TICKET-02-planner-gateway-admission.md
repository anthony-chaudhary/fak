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
- [x] [HW-WITNESSED] Physical Halo end-to-end A/B, qualified with N>=5 matched
  samples, exact token parity, zero hidden fallback, complete source/artifact/system
  provenance, P50/P90, and confidence intervals. strix1 at `7a71f63db5`, 6 matched
  pairs, identical greedy text: MTP decode is 4.495 -> 3.267 tok/s (-27.5%,
  p<0.001) at 40.5% draft acceptance. The result is negative, so it makes no
  performance or default-on claim; Vulkan MTP stays opt-in (see "Hardware A/B
  record").
- [x] [SW-VERIFIED] A resident Q4_K_M retained head (matrices in the quant stores,
  only F32 norms in the manifest) is admitted by `Qwen35MTPMode` and the forward
  tensor validator through its validated resident layout; an incomplete resident
  head still refuses with `*Qwen35MTPIncompleteError`.

## Hardware A/B record (2026-10-22, strix1)

`fak-strix bench abba` on strix1 at public `59ccab2e04b7`, private loopback
`fak serve --engine inkernel --backend vulkan`, greedy, B=`FAK_SPECULATIVE=off`
(the default serve retains no MTP head) vs C1=`FAK_SPECULATIVE=mtp`:

- `Qwen3.8-27B-UD-Q2_K_XL`: C1 refuses at load, 0/7 runs:
  `resident-Q4K weight admission: gguf: Qwen MTP tensor mtp.layers.0.self_attn.k_proj.weight has type Q8_0, want Q4_K`.
  That artifact is outside the resident Vulkan envelope (B: 13.35 tok/s, CV 0.5%).
- `Qwen3.8-27B-Q4_K_M`, 3 ABBA rounds: B 4.40 vs C1 4.47 tok/s, median delta
  -0.19% [-1.42, +14.12], p=0.45, WITHIN_NOISE, greedy text identical. Every C1
  ledger row reads `engine.path=serial` with no speculative rounds. This is a
  null A/B: MTP never ran, so it is not an MTP measurement.

Root cause, from an on-node probe of the gateway predicate: `Qwen35MTPMode`
checked only the F32 manifest, which holds the resident head's norms but not
its eight Q4_K/Q6_K/Q8_0 matrices. It returned `*Qwen35MTPIncompleteError`, so
`shouldEnableVulkanMTP` was false and nothing reported a route or downgrade.
`validateQwen35MTPForwardTensors` had the same manifest-only check, so the
draft constructor would refuse next. The fix routes both through the validated
resident layout (`qwen38MTPTensorLayout`, Format Q4_K).

A diagnostic strix1 build with the fix (modified source, not a qualified
receipt) reports `X-Fak-Speculative: mtp-vulkan`, 8 rounds, 32 drafted and 11
accepted tokens (34%), then `acceptance-floor-fallback`. Decode is 2.0-2.1 tok/s
against 4.2 tok/s serial on the same 64-token greedy request, with identical
text, and `fak_compute_host_fallback_total` stays 0 for `mtp_draft_host_forward`
and `spec_verify_attention`, so the device-only host-stage guard was not hit.

Qualified rerun at the fix, `abba-20261022T104553Z`:

- Setup: stage `abba-7a71f63db5c8`, binary sha256 `02b03ef4...8f1239`;
  Qwen3.8-27B-Q4_K_M, GGUF sha256 `7e78da5d...6fe169`; 8192 context, greedy,
  128 tokens. The node's own serve was parked for the window, then restored.
- Pairs and parity: 3 mirrored ABBA rounds give 6 matched pairs per arm, plus
  one profile-on run each. Greedy text is identical on every measured prompt.
- Route: all 18 C1 ledger rows read `engine.path=speculative` and all 18 B rows
  read `serial`. Draft acceptance is 282/696 = 40.5% (174 rounds, K=4).
- Fallback: the only downgrade is the reported `acceptance-floor-fallback`.
  Each 128-token request runs 8 rounds, then decodes serially. No other fallback
  occurs.
- Client decode, mean +- sd: B 4.495 +- 0.027, C1 3.267 +- 0.017 tok/s.
  P50/P90: B 4.491/4.528, C1 3.264/3.293.
- Delta: median -27.54% [-27.69, -26.71], 95% CI -27.32 +- 0.45%, p<0.001,
  verdict SLOWER.
- Engine decode: 4.469 -> 2.987 tok/s. TTFT: 4875 -> 8016 ms.

The 35-token warm turn speculates throughout: 13 rounds, 52 drafted, 22
accepted. Each K=4 resident verify round costs about 2.0 s, roughly 8.7 serial
0.23 s steps, and emits about 2.7 tokens. Five tokens per round would still run
at about 0.57x serial.

In the profile-on run, GPU time goes from 213.4 to 270.5 ms/token. The rows
added by speculation are the verify panel, `q4k_matmul` 1088x1 at 24.0
ms/token, and the all-position head, `q6k_matmul` 15520x1 at 12.3 ms/token.
The verify transaction is the lever, not the drafter.

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
