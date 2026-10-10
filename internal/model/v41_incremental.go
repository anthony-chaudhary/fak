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

import (
	"errors"
	"fmt"
)

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
	scratch.expertGateUp, scratch.expertDown = st.expertGateUp, st.expertDown
	scratch.denseProjection = st.denseProjection
	scratch.groupedOutput = st.groupedOutput
	scratch.mhcProjection = st.mhcProjection
	scratch.mhcFFNProjection = st.mhcFFNProjection
	scratch.mhcCarry = nil
	scratch.queryNorm = st.queryNorm
	scratch.kvNorm = st.kvNorm
	scratch.ffnNorm = st.ffnNorm
	scratch.compressorNorm = st.compressorNorm
	scratch.indexKeyNorm = st.indexKeyNorm
	scratch.indexerScore = st.indexerScore
	scratch.indexScoreHealth = st.indexScoreHealth
	scratch.sharedActivation = st.sharedActivation
	scratch.tailRoPE = st.tailRoPE
	scratch.sharedAttention = st.sharedAttention
	defer func() {
		scratch.expertGateUp, scratch.expertDown = nil, nil
		scratch.denseProjection = nil
		scratch.groupedOutput = nil
		scratch.mhcProjection = nil
		scratch.mhcFFNProjection = nil
		scratch.mhcCarry = nil
		scratch.queryNorm = nil
		scratch.kvNorm = nil
		scratch.ffnNorm = nil
		scratch.compressorNorm = nil
		scratch.indexKeyNorm = nil
		scratch.indexerScore = nil
		scratch.indexScoreHealth = nil
		scratch.sharedActivation = nil
		scratch.tailRoPE = nil
		scratch.sharedAttention = nil
	}()
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
	x, embedErr := m.v41EmbeddingPanel([]int{id})
	if embedErr != nil {
		return nil, stats, embedErr
	}

	full, geometryErr := v41ForwardGeometry(cfg)
	if geometryErr != nil {
		return nil, stats, geometryErr
	}
	streams := [][]float32{x, make([]float32, H), make([]float32, H), make([]float32, H)}
	if full {
		streams, err = v41FullInitialStreams(x)
		if err != nil {
			return nil, stats, err
		}
		copy(x, streams[0])
		scratch.mhcCarry = newV41MHCCarry(1)
	}

	// Fail closed BEFORE any mutation: every layer needs a seeded retained state.
	// A shadow step cannot seed a prefix -- the caller must seed.
	for l := 0; l < cfg.NumLayers; l++ {
		if st.layerState(l) == nil {
			return nil, stats, v41StageErr(v41StageLayer, l,
				fmt.Errorf("%w: layer %d has no seeded retained state; forwardV41Step cannot seed a prefix", ErrV41ForwardStage, l))
		}
	}
	// The shared source registry a role layer publishes into / resolves from. It
	// is resolved lazily (a plain-layer session may never have built it) so a role
	// step always sees the same store the full forward used.
	previousRegistry := st.attn
	registry, err := st.attentionState(cfg.HeadDim, 8)
	if err != nil {
		return nil, stats, v41StageErr(v41StageAttention, -1, err)
	}

	// Stage, for each layer, its PRE-STEP state so a fault in any layer or the
	// head can be rolled back by IN-PLACE restoration. A plain layer's Step writes
	// one row into the fixed windowSize-row ring and advances
	// nextWindowPos/nextCompressRow; a role layer's source append ALSO advances the
	// compressor group (partialInputs/partialPositions) and publishes into the
	// shared registry. The staged ledger holds one entry per STEPPED layer -- the
	// cursor scalars, the one O(state) ring row the step may overwrite, and a copy
	// of the at-most-(ratio-1)-row partial group -- so it is O(NumLayers) bounded
	// rows, never a prefix-sized clone. Restoration writes back INTO the same state
	// objects (never replacing st.layers[i]), so a caller holding a layer-state
	// pointer observes the restored contents.
	stagedPos := make([]int, 0, cfg.NumLayers)
	stagedCompress := make([]int, 0, cfg.NumLayers)
	stagedRow := make([][]float32, 0, cfg.NumLayers)
	stagedInputs := make([][][]float32, 0, cfg.NumLayers)
	stagedInputPos := make([][]int, 0, cfg.NumLayers)
	stagedCopies := make([]int, 0, cfg.NumLayers)
	stagedWindowRows := make([]int, 0, cfg.NumLayers)
	stagedKV := make([]v41StepPublicationUndo, 0, cfg.NumLayers)
	stagedIndex := make([]v41StepPublicationUndo, 0, cfg.NumLayers)
	// The live registry is append-only per source during this call. Journal one
	// prospective KV/index entry per layer, never the retained prefix payloads.
	registryUndo := stageV41StepRegistry(registry, cfg.NumLayers)
	rollback := func() {
		for i := len(stagedPos) - 1; i >= 0; i-- {
			state := st.layerState(i)
			state.nextWindowPos = stagedPos[i]
			state.nextCompressRow = stagedCompress[i]
			state.partialInputs = stagedInputs[i]
			state.partialPositions = stagedInputPos[i]
			state.retainedCopies = stagedCopies[i]
			state.retainedWindowRows = stagedWindowRows[i]
			stagedKV[i].restore()
			stagedIndex[i].restore()
			if row := stagedRow[i]; row != nil && state.windowSize > 0 {
				slot := stagedPos[i] % state.windowSize
				if slot >= 0 && slot < len(state.window) && len(state.window[slot]) == len(row) {
					copy(state.window[slot], row)
				}
			}
		}
		registryUndo.restore()
		st.attn = previousRegistry
		stats.LayersRolledBack = len(stagedPos)
	}
	defer func() {
		if r := recover(); r != nil {
			rollback()
			panic(r)
		}
	}()

	// roleSession reports whether any configured layer resolves to a non-plain
	// role. A plain-only session keeps the historical post-step release of the
	// shared registry (st.attn = nil) byte-for-byte; a role session must RETAIN it
	// so a reader layer can resolve a source's published rows on the next Step.
	roleSession := m.v41RoleSchedule()
	var engramStage *v41EngramStage
	var engramRows []uint32
	var engramFull bool
	if cfg.DeepSeekV41 != nil {
		for _, layer := range cfg.DeepSeekV41.EngramLayerIDs {
			if layer < 0 || layer >= cfg.NumLayers {
				continue
			}
			engramStage = m.v41EngramStageFor()
			engramRows, err = m.v41IncrementalEngramRows(engramStage, st.history, id, layer)
			if err != nil {
				rollback()
				return nil, stats, err
			}
			engramFull, err = v41ForwardGeometry(cfg)
			if err != nil {
				rollback()
				return nil, stats, err
			}
			break
		}
	}

	for l := 0; l < cfg.NumLayers; l++ {
		state := st.layerState(l)
		stagedPos = append(stagedPos, state.nextWindowPos)
		stagedCompress = append(stagedCompress, state.nextCompressRow)
		stagedInputs = append(stagedInputs, cloneV41Rows(state.partialInputs))
		stagedInputPos = append(stagedInputPos, append([]int(nil), state.partialPositions...))
		stagedCopies = append(stagedCopies, state.retainedCopies)
		stagedWindowRows = append(stagedWindowRows, state.retainedWindowRows)
		stagedKV = append(stagedKV, stageV41StepPublication(state.kvPublications, state.kvPublishedEnd, l))
		stagedIndex = append(stagedIndex, stageV41StepPublication(state.indexPublications, state.indexPublishedEnd, l))
		// Capture the ring slot even at position zero: an internal empty-state
		// role step can commit its first window row before a later layer fails.
		var rowBackup []float32
		if state.windowSize > 0 && len(state.window) > 0 {
			slot := state.nextWindowPos % state.windowSize
			if slot >= 0 && slot < len(state.window) {
				rowBackup = append([]float32(nil), state.window[slot]...)
			}
		}
		stagedRow = append(stagedRow, rowBackup)
		if engramStage != nil {
			declared := false
			for _, layer := range cfg.DeepSeekV41.EngramLayerIDs {
				declared = declared || layer == l
			}
			if cacheIdx := engramStage.cacheIndex(l); declared && cacheIdx >= 0 {
				geometry, geometryErr := m.v41EngramInjectionGeometry(l, 1)
				geometry.project = st.engramProjection
				if geometryErr != nil {
					rollback()
					return nil, stats, geometryErr
				}
				base, ok := checkedMulInt(cacheIdx, geometry.cols)
				if !ok || base < 0 || base > len(engramRows) || geometry.cols > len(engramRows)-base {
					rollback()
					return nil, stats, v41StageErr(v41StageEngram, l,
						fmt.Errorf("%w: Engram current row identifiers do not cover layer %d", ErrV41NativeUnsupported, l))
				}
				opened := m.v41NowNanos()
				rows, injectionErr := m.v41EngramInjectPrepared(l, [][]float32{x}, [][][]float32{streams}, engramFull, 1,
					engramRows[base:base+geometry.cols], float32(cfg.RMSNormEps), geometry)
				injections := 0
				if injectionErr == nil {
					injections = 1
				}
				var nanos int64
				if opened != 0 {
					nanos = m.v41NowNanos() - opened
				}
				m.v41NoteIncrementalEngram(injections, rows, 0, nanos)
				if injectionErr != nil {
					rollback()
					var projection *V41ProjectionOperationError
					if errors.As(injectionErr, &projection) {
						panic(injectionErr)
					}
					return nil, stats, injectionErr
				}
			}
		}
		stats.LayerCalls++
		if lerr := m.v41LayerStepWithRegistry(l, x, streams, pos, state, registry, scratch); lerr != nil {
			rollback()
			var operation *V41ExpertOperationError
			var projection *V41ProjectionOperationError
			if errors.As(lerr, &operation) || errors.As(lerr, &projection) {
				panic(lerr)
			}
			return nil, stats, lerr
		}
	}

	// Head runs BEFORE any commit; a head fault rolls back exactly like a layer
	// fault (truncate/restore, no clone).
	headInput := x
	if full {
		headInput, err = v41MHCPreBF16(cfg.NumLayers-1, streams, scratch.mhcCarry.pre[0])
		if err != nil {
			rollback()
			return nil, stats, err
		}
	}
	res, herr := m.v41HeadWithFinalNorm(headInput, scratch.denseProjection, st.finalNorm)
	if herr != nil {
		rollback()
		var projection *V41ProjectionOperationError
		if errors.As(herr, &projection) {
			panic(herr)
		}
		return nil, stats, herr
	}

	// Commit: append the token and, for a plain-only session, release the
	// step-scoped registry to match forwardV41. A role session retains the registry
	// because it is the cross-step source publication store a reader resolves.
	st.history = append(st.history, id)
	if !roleSession {
		st.attn = nil
	}
	stats.Committed = true
	return res, stats, nil
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
