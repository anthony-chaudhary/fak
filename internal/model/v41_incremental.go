package model

// v41_incremental.go — the fak#13311 shadow incremental one-token composition
// for the V4.1 native decode path (halo-ds41-100-30 packet ...).
//
// It composes the ALREADY-LANDED single-position plain-layer seam
// (v41LayerStep, fak#13306) into the ticket's through-line:
//
//	"embed one token, carry the four mHC streams through the configured layer
//	 sequence, then run the existing head."
//
// forwardV41 (v41_forward.go) is a FULL-SEQUENCE assembly: it reprojects and
// re-contracts the entire history to emit one token. forwardV41Step consumes
// exactly ONE new token -- the position it advances is pos = len(st.history) --
// and never recomputes old history. It is a SHADOW/TEST-ONLY composition: it
// does NOT touch stepV41, Session.Step, or forwardV41's behavior, and it makes
// no production activation, benchmark success claim, sampling change, or hidden
// old-history recomputation.
//
// Commit discipline (the ticket's "roll back by truncation/restoration, never by
// cloning prefix-sized state"):
//   - The token id is appended to st.history only AFTER the full layer sequence
//     AND the head succeed.
//   - Each layer's per-step scalars (V41AttentionState.nextWindowPos and
//     nextCompressRow) are staged before that layer runs and restored on a fault
//     in any layer, so every already-stepped layer's cursors are rolled back in
//     O(1) -- no prefix-sized clone.
//   - A layer's Step ALSO writes one row into the fixed windowSize-row ring, at
//     the slot nextWindowPos%windowSize. Restoring only the cursor would leave
//     that overwritten slot holding the failed step's KV row -- visible through
//     retainedTailRows() once the ring has wrapped -- so the composition stages
//     the ONE row at the slot the step would overwrite (its contents, not the
//     slice header) before the layer runs and copies it back on rollback. That
//     is one O(HiddenSize) row per stepped layer (O(NumLayers) rows total), never
//     a copy of the ring or the session history.
//   - The commit is a truncation/restoration of those per-layer scalars and the
//     at-most-one-row ring slot each layer overwrote, never a copy of the
//     retained ring or the session history.
//
// A layer whose role is compressed/reader is refused by v41LayerStep itself with
// the typed ErrV41ForwardStage; that refusal is surfaced, never bypassed (making
// compressed layers step is a later leaf).

// v41StepStats is the observable per-step counter block the shadow composition
// returns. It carries ONLY single-row counters -- never prefix-sized state -- so
// a test can prove (a) exactly NumLayers single-position layer calls happen per
// decode step and (b) a fault in the last layer rolls the whole step back.
//
// It is owned by the CALLER (returned by value from forwardV41Step), not stored
// on Model: the ticket forbids mutable globals on Model, and a returned struct
// keeps state ownership with the step's caller.
type v41StepStats struct {
	// LayerCalls counts the single-position v41LayerStep invocations made by
	// this step. On success it equals cfg.NumLayers.
	LayerCalls int
	// LayersRolledBack counts layers whose staged nextWindowPos, nextCompressRow
	// and overwritten ring row were restored after a fault. On success it is 0;
	// on a fault in layer l it is l+1 (layer l's own failed arithmetic plus
	// every earlier layer).
	LayersRolledBack int
	// Committed reports whether the step append and step-scoped publication were
	// published (history append). False with a nil error cannot happen; a false
	// with a non-nil error marks a fully rolled-back fault.
	Committed bool
}

// forwardV41Step advances the shadow V4.1 decode path by exactly one token.
//
// id is the new token (in [0, cfg.VocabSize)); its position is pos =
// len(st.history). st must already carry a seeded, append-ready retained state
// for EVERY layer -- a shadow step cannot seed a prefix, so a nil per-layer
// state fails closed. scratch is the optional reused projection scratch; nil is
// replaced with a fresh empty one.
//
// It embeds the token, carries the four mHC streams (stream 0 == x) through
// every configured layer via v41LayerStep, runs the existing v41Head, and only
// then commits: append id to st.history and publish the step-scoped layer state
// (st.attn = nil, matching forwardV41's cross-layer publication release).
//
// On any layer or head fault it returns the typed ErrV41ForwardStage with
// already-stepped layers rolled back by restoring their staged cursors
// (nextWindowPos, nextCompressRow) and the one ring row each overwrote, and the
// uncommitted history left untouched. It returns one logits row on success.
func (m *Model) forwardV41Step(id int, st *v41ForwardState, scratch *v41ProjScratch) ([]float32, v41StepStats, error) {
	return m.forwardV41StepCore(id, st, scratch, false, true)
}

// forwardV41StepAdmitted is for Session.Step after its full-model check in the
// same owned dispatch. It never carries admission across calls; live token,
// state, cursor, selected-operation and rollback checks stay in the transaction.
func (m *Model) forwardV41StepAdmitted(id int, st *v41ForwardState, scratch *v41ProjScratch) ([]float32, v41StepStats, error) {
	return m.forwardV41StepCore(id, st, scratch, true, true)
}

func (m *Model) forwardV41StepCore(id int, st *v41ForwardState, scratch *v41ProjScratch, admitted, wantLogits bool) (logits []float32, stats v41StepStats, err error) {
	var tx *v41StepTransaction
	if admitted {
		tx, err = m.beginV41StepAdmitted(id, st, scratch)
	} else {
		tx, err = m.beginV41Step(id, st, scratch)
	}
	if err != nil {
		return nil, tx.stats, err
	}
	defer tx.abort()
	for tx.nextLayer < tx.layers {
		if err := tx.advanceLayer(); err != nil {
			return nil, tx.stats, err
		}
	}
	if !wantLogits {
		err = tx.finalizeNoLogits()
		return nil, tx.stats, err
	}
	logits, err = tx.finalize()
	return logits, tx.stats, err
}

type v41StepPublicationUndo struct {
	rows   map[v41AttentionPublicationKey][]float32
	ends   map[int]int
	key    v41AttentionPublicationKey
	row    []float32
	end    int
	rowSet bool
	endSet bool
}

func stageV41StepPublication(rows map[v41AttentionPublicationKey][]float32, ends map[int]int, layer int) v41StepPublicationUndo {
	end, endSet := ends[layer]
	key := v41AttentionPublicationKey{sourceLayer: layer, start: end, end: end + 1}
	row, rowSet := rows[key]
	return v41StepPublicationUndo{rows: rows, ends: ends, key: key, row: row, end: end, rowSet: rowSet, endSet: endSet}
}

func (s v41StepPublicationUndo) restore() {
	if s.rowSet {
		s.rows[s.key] = s.row
	} else {
		delete(s.rows, s.key)
	}
	if s.endSet {
		s.ends[s.key.sourceLayer] = s.end
	} else {
		delete(s.ends, s.key.sourceLayer)
	}
}

// v41StepRegistryUndo retains only the prospective publication entry for each
// source layer. publish allocates new rows; it never mutates retained rows.
// PublishTopK and PublishCandidates also replace owned slices, and their public
// readers copy, so old headers remain immutable while the step runs. External
// prefix/session snapshots still require the deep copies in kv_prefix_snapshot.go.
// This bounds rollback staging only: attention readers may still copy history.
type v41StepRegistryUndo struct {
	state          *V41AttentionState
	kv, index      []v41StepPublicationUndo
	topk           [][]int32
	topkRatio      int
	topkSet        bool
	candidates     []bool
	candidateRatio int
	candidatesSet  bool
}

func stageV41StepRegistry(state *V41AttentionState, layers int) v41StepRegistryUndo {
	u := v41StepRegistryUndo{
		state: state,
		kv:    make([]v41StepPublicationUndo, layers),
		index: make([]v41StepPublicationUndo, layers),
		topk:  state.topk, topkRatio: state.topkRatio, topkSet: state.topkSet,
		candidates: state.candidates, candidateRatio: state.candidateRatio, candidatesSet: state.candidatesSet,
	}
	// A live single-token layer appends at most one row to each of its own
	// source stores. No step resets the maps or mutates another source's rows.
	for layer := 0; layer < layers; layer++ {
		u.kv[layer] = stageV41StepPublication(state.kvPublications, state.kvPublishedEnd, layer)
		u.index[layer] = stageV41StepPublication(state.indexPublications, state.indexPublishedEnd, layer)
	}
	return u
}

func (u v41StepRegistryUndo) restore() {
	for layer := len(u.kv) - 1; layer >= 0; layer-- {
		u.kv[layer].restore()
		u.index[layer].restore()
	}
	u.state.topk, u.state.topkRatio, u.state.topkSet = u.topk, u.topkRatio, u.topkSet
	u.state.candidates, u.state.candidateRatio, u.state.candidatesSet = u.candidates, u.candidateRatio, u.candidatesSet
}
