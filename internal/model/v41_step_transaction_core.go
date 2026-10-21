package model

import (
	"errors"
	"fmt"
)

func v41StepEntryAdmits(st *v41ForwardState, entry *v41StepRoundEntry, m *Model, id int) bool {
	if st.activeStepRound != nil && (entry == nil || entry.round != st.activeStepRound || entry.round.model != m || entry.round.state != st || entry.round.pending != entry || entry.ordinal != len(entry.round.entries)-1 || entry.token != id || entry.historyLen != len(st.history)) {
		return false
	}
	return entry == nil || (st.activeStepRound == entry.round && !entry.round.closed && !entry.round.poisoned)
}

func (scratch *v41ProjScratch) bindV41StepState(st *v41ForwardState) {
	scratch.expertGateUp, scratch.expertDown = st.expertGateUp, st.expertDown
	scratch.routedExpert = st.routedExpert
	scratch.routedLayer = st.routedLayer
	scratch.projectionRoute = st.projectionRoute
	scratch.groupedOutput = st.groupedOutput
	scratch.mhcProjection = st.mhcProjection
	scratch.mhcFFNProjection = st.mhcFFNProjection
	scratch.mhcCarry = nil
	scratch.attentionNormBinding = st.attentionNormBinding
	scratch.queryNorm = st.queryNorm
	scratch.queryPrefix = st.queryPrefix
	scratch.kvNorm = st.kvNorm
	scratch.ffnNorm = st.ffnNorm
	scratch.compressorNorm = st.compressorNorm
	scratch.indexKeyNorm = st.indexKeyNorm
	scratch.indexerScoreBinding = st.indexerScoreBinding
	scratch.indexScoreHealth = st.indexScoreHealth
	scratch.sharedActivation = st.sharedActivation
	scratch.tailRoPE = st.tailRoPE
	scratch.copySharedAttentionFrom(st)
	scratch.copyRoutedSumFrom(st)
	scratch.sharedAdd = st.sharedAdd
	scratch.gatherOps = st.gatherOps
}

func (scratch *v41ProjScratch) clearV41StepState() {
	scratch.activationTrace = nil
	scratch.expertGateUp, scratch.expertDown = nil, nil
	scratch.routedExpert = nil
	scratch.routedLayer = nil
	scratch.projectionRoute = nil
	scratch.groupedOutput = nil
	scratch.mhcProjection = nil
	scratch.mhcFFNProjection = nil
	scratch.mhcCarry = nil
	scratch.attentionNormBinding = nil
	scratch.queryNorm = nil
	scratch.queryPrefix = nil
	scratch.kvNorm = nil
	scratch.ffnNorm = nil
	scratch.compressorNorm = nil
	scratch.indexKeyNorm = nil
	scratch.indexerScoreBinding = nil
	scratch.indexScoreHealth = nil
	scratch.sharedActivation = nil
	scratch.tailRoPE = nil
	scratch.clearSharedAttention()
	scratch.setRoutedSum(nil)
	scratch.sharedAdd = nil
	scratch.gatherOps = nil
}

// v41StepInitialStreams returns resident even on failure so the transaction
// cleanup frees a panel it already allocated.
func (m *Model) v41StepInitialStreams(st *v41ForwardState, x []float32, full bool, scratch *v41ProjScratch) (streams [][]float32, resident *v41MHCResident, err error) {
	switch {
	case full && st.mhcDevice.resident():
		// Only shapes are read on the resident path; the panel owns all values.
		streams = [][]float32{x, x, x, x}
		if resident, err = st.mhcDevice.newResident([][]float32{x}); err != nil {
			return nil, resident, err
		}
		zero, zerr := resident.streamZero()
		if zerr != nil {
			return nil, resident, zerr
		}
		copy(x, zero[0])
		scratch.mhcCarry = newV41MHCCarry(1)
		scratch.mhcCarry.device = st.mhcDevice
		scratch.mhcCarry.resident = resident
	case full:
		streams, err = v41FullInitialStreamsOn(st.mhcDevice, x)
		if err != nil {
			return nil, nil, err
		}
		copy(x, streams[0])
		scratch.mhcCarry = newV41MHCCarry(1)
		scratch.mhcCarry.device = st.mhcDevice
	default:
		H := m.Cfg.HiddenSize
		streams = [][]float32{x, make([]float32, H), make([]float32, H), make([]float32, H)}
	}
	return streams, resident, nil
}

// v41StepJournal stages each stepped layer's pre-step state so a fault in any
// layer or the head restores it in place. A plain Step writes one ring row and
// advances the cursors; a role source append also advances the compressor
// group and publishes into the shared registry. Each entry holds the cursor
// scalars, the one ring row the step may overwrite and the retained
// partial-group headers, so the ledger is O(NumLayers) rows, never a
// prefix-sized clone. Restoration writes back into the same state objects, so a
// caller holding a layer-state pointer observes the restored contents.
type v41StepJournal struct {
	st               *v41ForwardState
	stats            *v41StepStats
	previousRegistry *V41AttentionState
	registry         v41StepRegistryUndo
	layers           []v41StepLayerUndo
	rolledBack       bool
}

// Keep per-layer undo data together so each token allocates one ledger rather
// than a backing array for each field. Ring rows still own their backups.
type v41StepLayerUndo struct {
	pos, compress int
	copies, rows  int
	ring          []float32
	inputs        [][]float32
	inputPos      []int
	kv, index     v41StepPublicationUndo
}

func newV41StepJournal(st *v41ForwardState, stats *v41StepStats, previousRegistry, registry *V41AttentionState, layers int) *v41StepJournal {
	return &v41StepJournal{
		st:               st,
		stats:            stats,
		previousRegistry: previousRegistry,
		// The live registry is append-only per source during this call. Journal
		// one prospective KV/index entry per layer, never the retained prefix.
		registry: stageV41StepRegistry(registry, layers),
		layers:   make([]v41StepLayerUndo, 0, layers),
	}
}

func (j *v41StepJournal) stage(l int, state *V41AttentionState) {
	// The compressor appends a fresh owned input or releases a completed group;
	// append only writes beyond the old lengths, even with spare capacity, so the
	// headers suffice for this exclusively owned transaction. External snapshots
	// still deep-copy partial groups in kv_prefix_snapshot.go.
	undo := v41StepLayerUndo{
		pos: state.nextWindowPos, compress: state.nextCompressRow,
		inputs: state.partialInputs, inputPos: state.partialPositions,
		copies: state.retainedCopies, rows: state.retainedWindowRows,
		kv:    stageV41StepPublication(state.kvPublications, state.kvPublishedEnd, l),
		index: stageV41StepPublication(state.indexPublications, state.indexPublishedEnd, l),
	}
	// Capture the ring slot even at position zero: an internal empty-state role
	// step can commit its first window row before a later layer fails.
	var rowBackup []float32
	if state.windowSize > 0 && len(state.window) > 0 {
		slot := state.nextWindowPos % state.windowSize
		if slot >= 0 && slot < len(state.window) {
			rowBackup = append([]float32(nil), state.window[slot]...)
		}
	}
	undo.ring = rowBackup
	j.layers = append(j.layers, undo)
}

func (j *v41StepJournal) rollback() {
	if j.rolledBack {
		return
	}
	for i := len(j.layers) - 1; i >= 0; i-- {
		undo := &j.layers[i]
		state := j.st.layerState(i)
		state.nextWindowPos = undo.pos
		state.nextCompressRow = undo.compress
		// Append may reuse the saved backing array. Drop failed rows beyond its
		// old length so rollback does not keep their allocations alive.
		if n := len(undo.inputs); n < len(state.partialInputs) {
			clear(state.partialInputs[n:])
		}
		state.partialInputs = undo.inputs
		state.partialPositions = undo.inputPos
		state.retainedCopies = undo.copies
		state.retainedWindowRows = undo.rows
		undo.kv.restore()
		undo.index.restore()
		if row := undo.ring; row != nil && state.windowSize > 0 {
			slot := undo.pos % state.windowSize
			if slot >= 0 && slot < len(state.window) && len(state.window[slot]) == len(row) {
				copy(state.window[slot], row)
			}
		}
	}
	j.registry.restore()
	j.st.attn = j.previousRegistry
	j.stats.LayersRolledBack = len(j.layers)
	j.rolledBack = true
}

// v41StepEngram is the incremental Engram row set for the step's first
// declared Engram layer; a nil stage means the checkpoint declares none.
type v41StepEngram struct {
	stage *v41EngramStage
	rows  []uint32
	full  bool
}

func (m *Model) v41StepEngramRows(st *v41ForwardState, id int) (v41StepEngram, error) {
	cfg := m.Cfg
	if cfg.DeepSeekV41 == nil {
		return v41StepEngram{}, nil
	}
	for _, layer := range cfg.DeepSeekV41.EngramLayerIDs {
		if layer < 0 || layer >= cfg.NumLayers {
			continue
		}
		e := v41StepEngram{stage: m.v41EngramStageFor()}
		var err error
		e.rows, err = m.v41IncrementalEngramRows(e.stage, st.history, id, layer)
		if err != nil {
			return e, err
		}
		e.full, err = v41ForwardGeometry(cfg)
		return e, err
	}
	return v41StepEngram{}, nil
}

// v41StepInjectEngram injects the current row into layer l when the layer is a
// declared Engram layer. A selected projection-operation failure panics after
// rollback so it is never replayed on host.
func (m *Model) v41StepInjectEngram(l int, e v41StepEngram, st *v41ForwardState, x []float32, streams [][]float32, resident *v41MHCResident, scratch *v41ProjScratch, rollback func()) error {
	if e.stage == nil {
		return nil
	}
	cfg := m.Cfg
	declared := false
	for _, layer := range cfg.DeepSeekV41.EngramLayerIDs {
		declared = declared || layer == l
	}
	cacheIdx := e.stage.cacheIndex(l)
	if !declared || cacheIdx < 0 {
		return nil
	}
	geometry, geometryErr := m.v41EngramInjectionGeometry(l, 1)
	geometry.project = st.engramProjection
	if geometryErr != nil {
		rollback()
		return geometryErr
	}
	base, ok := checkedMulInt(cacheIdx, geometry.cols)
	if !ok || base < 0 || base > len(e.rows) || geometry.cols > len(e.rows)-base {
		rollback()
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram current row identifiers do not cover layer %d", ErrV41NativeUnsupported, l))
	}
	opened := m.v41NowNanos()
	engramStreams := [][][]float32{streams}
	layerRows := e.rows[base : base+geometry.cols]
	var rows int
	var injectionErr error
	var gathered *v41EngramGatheredRows
	handled := false
	if resident != nil {
		if gate := scratch.gatherOps.engramGateFunc(); gate != nil {
			gathered, injectionErr = m.v41EngramGatherRows(l, 1, layerRows, float32(cfg.RMSNormEps), geometry)
			if injectionErr == nil {
				rows = len(gathered.rows)
				handled, injectionErr = gate(l, resident.streams, gathered, 1, geometry, float32(cfg.RMSNormEps))
			}
			handled = handled || injectionErr != nil
		}
	}
	if !handled {
		if resident != nil {
			// An absent or declining gate reads the panel back around this layer.
			host, readErr := resident.hostStreams(l)
			if readErr != nil {
				rollback()
				return readErr
			}
			copy(x, host[0][0])
			engramStreams = host
		}
		if gathered == nil {
			rows, injectionErr = m.v41EngramInjectPrepared(l, [][]float32{x}, engramStreams, e.full, 1,
				layerRows, float32(cfg.RMSNormEps), geometry)
		} else {
			rows, injectionErr = m.v41EngramInjectOwnedPrepared(l, [][]float32{x}, engramStreams, e.full, 1,
				layerRows, float32(cfg.RMSNormEps), geometry, gathered)
		}
		if injectionErr == nil && resident != nil {
			injectionErr = resident.setHostStreams(l, engramStreams)
		}
	}
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
		return injectionErr
	}
	return nil
}
