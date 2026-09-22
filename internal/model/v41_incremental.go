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

import "fmt"

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
func (m *Model) forwardV41Step(id int, st *v41ForwardState, scratch *v41ProjScratch) (logits []float32, stats v41StepStats, err error) {
	if st == nil {
		return nil, stats, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: forwardV41Step requires a seeded session state", ErrV41ForwardStage))
	}
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		return nil, stats, err
	}
	cfg := m.Cfg
	if id < 0 || id >= cfg.VocabSize {
		return nil, stats, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: token id %d out of range [0,%d)", ErrV41ForwardStage, id, cfg.VocabSize))
	}

	H := cfg.HiddenSize
	// This step consumes the NEXT token, so its absolute position is the current
	// committed history length. The per-layer retained states must already match.
	pos := len(st.history)

	// Embed the token into a fresh hidden row.
	embed := m.tensor("model.embed_tokens.weight")
	if len(embed) < cfg.VocabSize*H {
		return nil, stats, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: embedding table has %d values, want %d", ErrV41ForwardStage, len(embed), cfg.VocabSize*H))
	}
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)
	scaleEmbedInPlace(x, cfg)

	// The four persistent mHC streams: stream 0 is the live hidden row, streams
	// 1..3 are the persistent zero residuals a full forward initializes them to.
	streams := [][]float32{x, make([]float32, H), make([]float32, H), make([]float32, H)}

	// Fail closed BEFORE any mutation: every layer needs a seeded retained state.
	// A shadow step cannot seed a prefix -- the caller must seed.
	for l := 0; l < cfg.NumLayers; l++ {
		if st.layerState(l) == nil {
			return nil, stats, v41StageErr(v41StageLayer, l,
				fmt.Errorf("%w: layer %d has no seeded retained state; forwardV41Step cannot seed a prefix", ErrV41ForwardStage, l))
		}
	}

	// Stage, for each layer, its pre-step cursors AND the one ring row that
	// layer's Step may overwrite. The staged slices hold one entry per stepped
	// layer -- per-step scalars plus at most one O(HiddenSize) row each -- never
	// a prefix-sized clone. This is the entire rollback ledger.
	//
	// V41AttentionState.Step writes exactly one row into the fixed
	// windowSize-row ring, at slot nextWindowPos%windowSize, and increments
	// nextWindowPos/nextCompressRow. retainedTailRows() derives the visible
	// tail from nextWindowPos, so once the ring has wrapped (pos >= windowSize)
	// a later fault rolled back by cursor alone would still expose the failed
	// step's row at that slot. Staging the row's CONTENT (copied, so a later
	// in-place write cannot alias the backup) and restoring it on rollback makes
	// the retained ring byte-identical to its pre-step contents.
	//
	// At pos == 0 the layer commits via Prefill, not Step: Prefill requires an
	// empty state (nextWindowPos == 0) and writes sequentially from slot 0
	// without wrapping, so it cannot clobber a pre-existing row. Its cursor
	// restore is still staged below; the ring-row backup is nil then.
	stagedPos := make([]int, 0, cfg.NumLayers)
	stagedCompress := make([]int, 0, cfg.NumLayers)
	stagedRow := make([][]float32, 0, cfg.NumLayers)
	rollback := func() {
		for i := len(stagedPos) - 1; i >= 0; i-- {
			state := st.layerState(i)
			state.nextWindowPos = stagedPos[i]
			state.nextCompressRow = stagedCompress[i]
			if row := stagedRow[i]; row != nil && state.windowSize > 0 {
				slot := stagedPos[i] % state.windowSize
				if slot >= 0 && slot < len(state.window) && len(state.window[slot]) == len(row) {
					copy(state.window[slot], row)
				}
			}
		}
		stats.LayersRolledBack = len(stagedPos)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		state := st.layerState(l)
		stagedPos = append(stagedPos, state.nextWindowPos)
		stagedCompress = append(stagedCompress, state.nextCompressRow)
		// Capture the ring row Step will overwrite. pos != 0 (Step) writes
		// window[nextWindowPos%windowSize]; pos == 0 (Prefill) cannot clobber a
		// pre-existing row and gets a nil backup.
		var rowBackup []float32
		if pos != 0 && state.windowSize > 0 && len(state.window) > 0 {
			slot := state.nextWindowPos % state.windowSize
			if slot >= 0 && slot < len(state.window) {
				rowBackup = append([]float32(nil), state.window[slot]...)
			}
		}
		stagedRow = append(stagedRow, rowBackup)
		stats.LayerCalls++
		if lerr := m.v41LayerStep(l, x, streams, pos, state, scratch); lerr != nil {
			rollback()
			return nil, stats, lerr
		}
	}

	// Head runs BEFORE any commit; a head fault rolls back exactly like a layer
	// fault (truncate/restore, no clone).
	res, herr := m.v41Head(x)
	if herr != nil {
		rollback()
		return nil, stats, herr
	}

	// Commit: append the token and publish the step-scoped layer state. History
	// grows by exactly one; attn is released to match forwardV41.
	st.history = append(st.history, id)
	st.attn = nil
	stats.Committed = true
	return res, stats, nil
}
