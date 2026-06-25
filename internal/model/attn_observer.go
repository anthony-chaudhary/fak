package model

// attn_observer.go — issue #852, rung 1 of the attention-witness epic (#851).
//
// The post-softmax attention weights are already materialized in the per-worker score
// scratch at the softmax seam, then discarded once the value accumulation reads them.
// This file emits them — and ONLY emits them — behind a default-off observer so the
// witnessed attention mass becomes available to the higher rungs (span attribution
// #853, the rolling accumulator #855, attention-informed eviction #856) without
// perturbing the forward pass.
//
// INVARIANTS (the rung-8 byte-identical foundation depends on these):
//   - nil observer == today's behavior: zero extra allocation, zero extra work in the
//     hot loop (every emission site is guarded behind `if obs != nil`).
//   - emission only: the softmax and the value-accumulation math are never touched.
//     The observer receives a COPY of the post-softmax weights, so even a misbehaving
//     observer cannot mutate the scratch the value loop reads.
//   - the scratch is worker-local at the seam, so the synchronous copy-out is safe to
//     do inside the parallel worker without a lock.

// AttnObserver receives the post-softmax attention distribution for one (layer, query
// position, head) row. weights[i] is the normalized weight the query placed on key
// position keyPositions[i]; the row sums to ~1.0 (the post-softmax invariant). The two
// slices are freshly allocated per call and owned by the observer — the caller never
// reads them again, so the observer may retain them.
type AttnObserver func(layer, queryPos, head int, keyPositions []int, weights []float32)

// SetAttnObserver installs (or clears, with nil) the attention-mass witness on this
// model. Default is nil — the unobserved forward pass is byte-identical to a model that
// never had this method called. Not safe to change concurrently with a forward pass;
// set it before the pass and clear it after.
func (m *Model) SetAttnObserver(obs AttnObserver) { m.attnObs = obs }

// AttnObserverSet reports whether an attention observer is currently installed. The hot
// paths guard on the local `obs != nil` they were threaded; this is for callers/tests.
func (m *Model) AttnObserverSet() bool { return m.attnObs != nil }

// emitAttnRow copies the post-softmax weights for the window [j0, j0+len(weights)) and
// hands them to obs. keyPositions[i] = j0+i is the ACTUAL absolute key position for the
// contiguous-cache paths (decode / prefill panel), where pos[j]==j. The caller MUST
// have already checked obs != nil — this helper does the per-row allocation, so calling
// it unconditionally would defeat the zero-alloc-when-off invariant.
func emitAttnRow(obs AttnObserver, layer, queryPos, head, j0 int, weights []float32) {
	kp := make([]int, len(weights))
	for i := range kp {
		kp[i] = j0 + i
	}
	w := make([]float32, len(weights))
	copy(w, weights)
	obs(layer, queryPos, head, kp, w)
}
