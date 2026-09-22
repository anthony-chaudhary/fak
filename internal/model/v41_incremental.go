package model

// v41_incremental.go — the fak#13311 SHADOW full-stack single-token V4.1
// composition (halo-ds41-100-30 packet 11).
//
// Session.Step (stepV41, v41_forward.go) routes every decode token through
// forwardV41, which folds the token into the session history and RECOMPUTES the
// whole history: embedding, mHC streams, per-layer projections and the
// attention contraction over every prefix position, then the head over every
// position. That is exactly what a prefill needs and exactly the cost the
// 100/30 plan's "Session decode recomputes all history" finding names.
//
// forwardV41Step is the one-position counterpart. It composes the ALREADY-LANDED
// per-layer seam (v41LayerStep, v41_layer_step.go, leaf 07/#13306) across the
// configured layer stack for exactly ONE token: embed the token, carry the four
// persistent mHC streams through the layer sequence, then run the head on the
// final row. It introduces NO new arithmetic — every per-layer function is the
// same one v41Layer calls with seq == 1, reached through v41LayerStep.
//
// Scope (fak#13311): this is a SHADOW/test-only composition. The production
// Session.Step route is deliberately NOT switched here; leaf 10 owns activation.
//
// Transactional contract (never a silent partial commit):
//   - the session history append and every per-layer ring advance are STAGED as
//     per-token deltas; only the small scalars needed to undo an append are
//     snapshotted, never a prefix-sized copy of the retained state;
//   - commit happens only after EVERY layer AND the head succeed;
//   - any failure rolls back by truncating the staged history append and
//     restoring each layer's ring cursor to its staged pre-value, so a failed
//     step behaves as if it never occurred and a retry sees the same prefix.
//
// Fail-closed: a nil state, a non-positive id range, a malformed geometry, or a
// layer whose role the step seam refuses (compressed/source/reader) surfaces the
// typed ErrV41ForwardStage and mutates nothing.

import "fmt"

// v41StepTxnStage records the minimal undo state for one layer's ring advance:
// the ring cursor before the step. The ring is a fixed-size circular buffer, so
// restoring this scalar is a complete rollback of the logical append — no
// prefix-sized buffer copy is ever taken.
type v41StepTxnStage struct {
	layer           int
	state           *V41AttentionState
	prevWindowPos   int
	prevCompressRow int
}

// forwardV41Step advances the session by exactly ONE token through the full
// configured layer stack, as a SHADOW composition. It returns the final hidden
// row and the head logits for that one position.
//
// It never touches the production route: a caller (a parity test, or a future
// activation leaf) owns the decision to prefer it. st carries the session-owned
// retained per-layer temporal state (seeded by a prior full forward) and the
// optional device gate/up callback; it is updated on success and left unchanged
// on any failure.
func (m *Model) forwardV41Step(id int, st *v41ForwardState) (hidden []float32, logits []float32, err error) {
	if err := m.v41ForwardAdmitted(); err != nil {
		return nil, nil, err
	}
	if st == nil {
		return nil, nil, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: forwardV41Step requires session state", ErrV41ForwardStage))
	}
	cfg := m.Cfg
	if id < 0 || id >= cfg.VocabSize {
		return nil, nil, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: token id %d out of range [0,%d)", ErrV41ForwardStage, id, cfg.VocabSize))
	}
	if _, err := v41ForwardGeometry(cfg); err != nil {
		return nil, nil, err
	}
	if len(st.layers) != cfg.NumLayers {
		return nil, nil, v41StageErr(v41StageAttention, -1,
			fmt.Errorf("%w: session state carries %d layer states, want %d", ErrV41ForwardStage, len(st.layers), cfg.NumLayers))
	}

	H := cfg.HiddenSize
	pos := len(st.history)

	// ---- stage the history append (undo state is its prior length) ----
	prevHistory := len(st.history)
	st.history = append(st.history, id)

	// ---- stage per-layer ring cursors ----
	stages := make([]v41StepTxnStage, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		ls := st.layers[l]
		if ls == nil {
			st.history = st.history[:prevHistory]
			return nil, nil, v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: layer %d has no retained decode state", ErrV41ForwardStage, l))
		}
		stages[l] = v41StepTxnStage{layer: l, state: ls, prevWindowPos: ls.nextWindowPos, prevCompressRow: ls.nextCompressRow}
	}

	rollback := func() {
		st.history = st.history[:prevHistory]
		for i := range stages {
			stages[i].state.nextWindowPos = stages[i].prevWindowPos
			stages[i].state.nextCompressRow = stages[i].prevCompressRow
		}
	}

	// ---- embedding for the one position ----
	embed := m.tensor("model.embed_tokens.weight")
	if len(embed) < cfg.VocabSize*H {
		rollback()
		return nil, nil, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: embedding table has %d values, want %d", ErrV41ForwardStage, len(embed), cfg.VocabSize*H))
	}
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)
	scaleEmbedInPlace(x, cfg)
	// The four persistent mHC streams a full forward initializes: stream 0 is the
	// live hidden, streams 1..3 the zero residuals.
	streams := [][]float32{x, make([]float32, H), make([]float32, H), make([]float32, H)}

	// ---- one position through every configured layer ----
	scratch := &v41ProjScratch{}
	for l := 0; l < cfg.NumLayers; l++ {
		if err := m.v41LayerStep(l, x, streams, pos, st.layers[l], scratch); err != nil {
			rollback()
			return nil, nil, err
		}
	}

	// ---- head on the final row (the last commit gate) ----
	out, err := m.v41Head(x)
	if err != nil {
		rollback()
		return nil, nil, err
	}

	return append([]float32(nil), x...), out, nil
}
