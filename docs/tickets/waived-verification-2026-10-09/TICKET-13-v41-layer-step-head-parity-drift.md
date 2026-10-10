# TICKET-13: V4.1 layer-step, grouped-session and head parity drifted after BF16-boundary changes

## Current state

At fak `4cbddeafffb`, three parity tests drift well past tolerance:

- `TestV41LayerStepFullGeometry` (`v41_layer_step_full_test.go:216/269/298`): step vs. forwardV41 deltas of 7.6e-4, 9.8e-4 and 3.9e-3, against a tolerance of 1e-6. The sub-cases are prefix 7, patched-norm parity, and perturbed-mHC parity.
- `TestV41FullGroupedOwnedSessionContinuation` (`v41_grouped_output_session_test.go:753`): grouped parity max diff 2.8e-3, tolerance 1e-4.
- `TestV41HeadProjectionFullNormAndScaleSemantics/ignore-generic-layernorm-bias` (`v41_head_projection_test.go:471`): grouped parity max diff 8.1e-3, tolerance 1e-4.

The deltas look like BF16 rounding (for example 0.0009765625 = 2^-10). The likely cause is that waived commits added BF16 boundaries to one side of each comparison but not the other: 57cbe301f63 (latent norm), d3682639f3c (index head weight), 76c848b446f (compressed reader), 49124439fba (compressed latent copyback). Bisect to confirm.

Parent: #13764.

## Working spine

1. Bisect each test.
2. Decide from the published reference model where BF16 boundaries belong, then apply them to both the step path and the forward path (or to both the grouped and the reference path).
3. Never loosen a tolerance to absorb the drift.

## Witness

`go test -count=1 -timeout 60m -run 'V41LayerStepFullGeometry|V41FullGroupedOwnedSessionContinuation|V41HeadProjectionFullNormAndScaleSemantics' ./internal/model/` exit 0.

## Done condition

- [ ] All three tests green at their original tolerances; the fix cites the reference and the culprit sha.
