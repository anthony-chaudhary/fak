<!-- fak-model-key: qwen38-vulkan-resident-mtp-draft-v1 -->
<!-- fak-public-issue: anthony-chaudhary/fak#12538 -->

# Execute the retained Qwen3.8 MTP draft head on native Vulkan

```routing
lane: model
paths: ["internal/model/qwen35_mtp_forward.go", "internal/model/qwen35_mtp_draft.go", "internal/model/qwen35_mtp_vulkan.go", "internal/model/qwen35_mtp_forward_vulkan_test.go", "internal/model/qwen35_mtp_draft_ab_vulkan_test.go", "internal/compute/qwen35_mtp_contract.go", "internal/compute/vulkan_qwen35_mtp.go", "tools/concept_disambiguation_scorecard.data/rows-halo-mtp-vulkan.json"]
expected_steps: 8
```

## Parent context

Parent campaign: #11572. Coordinates with: #12184, #12687, #12714.
Promotion requires #12098 for cross-engine comparison claims only.
The retained MTP loader deliberately preserves its value/down tensors as exact
Q6 independently of the ordinary target's Q8 path. Coordinate with the open
native Q6 compute leaf #12403; do not widen this ticket into that implementation.

## Why this is next

Pickup is available now because resident drafting has a disjoint executable
slice and target-side Vulkan K<=4 speculative verification is already present.

## Current state

At public source `19d160b9fdac5203c518a713f09ceda93f5a5b68`, target-side
Vulkan K<=4 speculative verification is implemented. The retained one-layer
MTP draft still constructs a CPU session on non-Darwin platforms in
`internal/model/qwen35_mtp_forward.go:142-163`. Its forward path executes
`blockStep` and the shared head on that session at lines 192-214 and 235-242.
`internal/model/qwen35_mtp_draft.go:52-66` drops the target backend when
constructing the draft; its feedback entry at lines 457-482 duplicates the
CPU execution. The existing API accepts and returns host slices.

The retained MTP value/down tensors remain Q6 by loader contract even when the
ordinary target selects Q8. The resident adapter must preserve that exact format
and return a typed pre-execution refusal until #12403 supplies the native Q6
operation it needs.

The previous issue comment records a software-only implementation checkpoint
that was not landed. It does not establish the current capability or a
hardware speedup. This revision makes the required integration and transfer
accounting explicit instead of retaining an infeasible three-file boundary.

Centrality: Core. Portfolio tier: 2 (serving). Priority: P0.

- P1 Context: advanced - closes the resident draft gap for native Qwen3.8.
- P2 Net value: advanced - removes CPU matrix work between GPU target rounds.
- P3 Adaptation: preserved - retains exact checkpoint tensors and target verification.
- P4 Operations: advanced - typed refusal and measured transfer/time receipts.

## For / Problem / Today / Better because

For native Vulkan users with a retained one-layer Qwen3.8 MTP checkpoint.
Draft computation currently returns to the CPU although the target is resident.
Using the existing Vulkan operations for the complete retained draft head
removes this host computation and intermediate activation traffic. Exact target
verification remains the generated-output correctness boundary.

## Core through-line

Prior hidden and current embedding -> Vulkan normalization and retained
`mtp.fc` -> exact retained full-attention decoder with independent draft KV ->
`mtp.norm` and shared target head -> measured feedback/logit readback ->
existing greedy draft API -> durable software and physical receipt.

Expose explicit backend-aware forward and draft-session constructors. The
existing CPU constructor remains compatible. The draft-session constructor
preserves the existing F32 target hidden-history admission; the full Vulkan
target's raw pre-final-normalization hidden history is a separate prerequisite
for production self-speculation. Reuse existing Vulkan operations and
immutable checkpoint storage. The draft owns its KV state and keeps target
weights alive without closing the target backend. Unsupported Vulkan
capabilities or formats return typed errors before execution; they never
silently run a CPU draft.

## Gold-plating boundary

No new shader family, command graph, target-verification algorithm, partial
commit implementation, speculative acceptance policy, default-on serving,
external inference engine, or public HTTP schema. #12714 separately owns
partial target commit. End-to-end production promotion stays gated on its
own exact-output and net-benefit witness.

## Working spine

Existing MTP draft entry -> explicit Vulkan capability -> retained draft tensors
and independent KV -> native operations -> checked output and measured receipt.

## Done condition

- [ ] [SW-VERIFIED] Existing backend-aware MTP entry executes normalization,
  fusion projection, retained draft layer, final norm and shared head on Vulkan.
- [ ] [SW-VERIFIED] CPU-reference logits satisfy the established numerical
  tolerance and exact argmax on deterministic fixtures over successive positions.
- [ ] [SW-VERIFIED] Draft KV is independent, target state remains unchanged,
  and closing the draft releases its held target lifetime exactly once.
- [ ] [SW-VERIFIED] Unsupported capability/format is a typed pre-execution
  refusal with no hidden CPU path or target-layer substitution.
- [ ] [SW-VERIFIED] Receipts distinguish setup uploads, required host API
  boundary uploads/readbacks, and intermediate activation transfers. The
  latter are zero on the supported path; total transfers are not mislabeled zero.
- [ ] [SW-VERIFIED] Focused MTP regressions, affected-package checks and
  public/private boundary checks pass.
- [ ] [HW-WITNESSED] Same-artifact physical CPU/Vulkan draft A/B includes one
  warmup and at least five measured repetitions per arm, complete source,
  artifact, device, compiler, driver, clock/power and thermal provenance, raw
  samples, P50/P90 and confidence intervals, plus exact argmax parity.
- [ ] [HW-WITNESSED] Vulkan median draft-head latency is at least 5% below
  the matched CPU arm with `engine=fak-native`, `backend=vulkan` and zero
  fallback. Otherwise preserve the rejection receipt and hold promotion.

## Witness

Source repro (the current CPU draft binding and duplicated feedback path):

```text
rg -n "non-Darwin|NewQwen35MTPForward|blockStep|ProjectHead" internal/model/qwen35_mtp_forward.go internal/model/qwen35_mtp_draft.go
```

Deterministic device witness:

```text
CGO_ENABLED=1 FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestQwen35MTPForwardUsesResidentVulkanDraftAndHead$' -count=1 -v
```

The command above uses POSIX environment syntax; set the same environment
variables explicitly when running from PowerShell. The independent test must
prove actual execution, not a zero-test success. `CGO_ENABLED=0` excludes the
Vulkan implementation and its tests even when `-tags vulkan` is supplied.
The artifact-gated companion A/B test records measured draft-head time;
software or injected backends never earn physical performance credit.

## Done condition / witness

The scoped forward path executes on Vulkan, matches the native reference,
preserves target state and records observed transfers. The commands above
and the source-bound physical A/B provide independent acceptance evidence.

## Verifiable Witness

Compare the exact retained draft tensors with the existing native CPU oracle,
observe real Vulkan dispatch and transfer counts, and inspect target/draft state
before and after two draft positions. Bind physical results to a clean source
commit and the exact artifact used by both arms. No historical throughput
value substitutes for a missing measurement.

## Blast radius and affected lanes

Only `internal/model` and `internal/compute` implement this capability. Existing
CPU/Metal construction remains compatible. Other backends, ordinary target
decode, target partial-commit work, gateway policy and installer code are outside
the write set.

## Quarantined fallback mechanism

The Vulkan-specific constructor fails closed on unsupported capability, tensor
layout or buffer limits. The original CPU constructor remains an explicit
reference/control. The feature is opt-in; no hardware evidence means no default
promotion and no claimed speedup.

## Likely files

- `internal/model/qwen35_mtp_forward.go:142` (draft construction and execution)
- `internal/model/qwen35_mtp_draft.go:52` (backend and feedback binding)
- `internal/model/qwen35_mtp_vulkan.go` (resident model adapter)
- `internal/model/qwen35_mtp_forward_vulkan_test.go` (independent witness)
- `internal/model/qwen35_mtp_draft_ab_vulkan_test.go` (artifact-bound comparison)
- `internal/compute/qwen35_mtp_contract.go` (typed capability and counters)
- `internal/compute/vulkan_qwen35_mtp.go` (existing-operation capability binding)
- `tools/concept_disambiguation_scorecard.data/rows-halo-mtp-vulkan.json`
  (definitions for the reused attention operation and new capability/refusal)

## Lane

model, with a disjoint compute capability leaf.

## Expected steps

8: arbitrate; bind backend; implement resident draft; count actual transfers;
author independent oracle; run software checks; measure physical A/B; land.

## Acceptance gate

Software correctness and state isolation gate landing. Matched physical evidence
gates the performance claim. This leaf makes no whole-decode speedup claim.

## Definition of done

The reachable native Vulkan draft path passes the independent execution and
state-isolation witness, its physical A/B satisfies the declared performance
gate, and the source plus exact receipts are committed with #12538 tracking.

## Closure binding

Resolving commits reference #12538 and this ticket. Keep the issue open until
the declared physical gate is met or a measured rejection is recorded with the
next implementation step explicitly tracked.

## Work estimate

Estimate: 5 points. Contribution: one resident draft-head leaf of #12184.

## Completion standard

development plus physical qualification.
