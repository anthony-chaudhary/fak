package model

import (
	"errors"
	"fmt"
)

// v41StepTransaction is a single-session, single-token continuation. The caller
// owns the session and its scratch exclusively until finalize or abort. Distinct
// transactions may be advanced in layer-major order by one device owner; this
// seam does not open a backend batch, permit concurrent forwards, or share rows.
// Captured locals retain the same bounded per-layer undo journal as serial Step.
type v41StepTransaction struct {
	stats             v41StepStats
	nextLayer, layers int
	closed            bool
	headPrepared      bool
	layer             func(int) (*v41LayerStepWork, error)
	pendingLayer      *v41LayerStepWork
	headInput         func() ([]float32, error)
	headProject       func([]float32) ([]float32, error)
	headFinish        func([]float32, func() error) ([]float32, error)
	rollback, cleanup func()
	afterLayer        func(int) error
	activationCapture *v41StepActivationCapture
}

// beginV41Step performs admission, embedding and journal setup in serial order.
// Even on failure it returns a closed transaction carrying the rollback stats.
func (m *Model) beginV41Step(id int, st *v41ForwardState, scratch *v41ProjScratch) (*v41StepTransaction, error) {
	return m.beginV41StepCore(id, st, scratch, false)
}

// beginV41StepAdmitted reuses full-model admission only within one owned cohort
// dispatch. The cohort already validated this exact immutable model; the token,
// retained state, cursor, and transaction checks still run for every lane.
// It must not be used as a successful-admission cache across calls.
func (m *Model) beginV41StepAdmitted(id int, st *v41ForwardState, scratch *v41ProjScratch) (*v41StepTransaction, error) {
	return m.beginV41StepCore(id, st, scratch, true)
}

func (m *Model) beginV41StepCore(id int, st *v41ForwardState, scratch *v41ProjScratch, admitted bool) (tx *v41StepTransaction, err error) {
	tx = &v41StepTransaction{}
	stats := &tx.stats
	var resident *v41MHCResident
	ready := false
	defer func() {
		if !ready {
			tx.abortAfterFailure()
		}
	}()
	if st == nil {
		return tx, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: forwardV41Step requires a seeded session state", ErrV41ForwardStage))
	}
	st.forwardCoverage = nil
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	tx.activationCapture = m.beginV41StepActivationCapture(id, st)
	if tx.activationCapture != nil {
		scratch.activationTrace = tx.activationCapture.trace
	}
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
	tx.cleanup = func() {
		defer func() {
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
		}()
		if resident != nil {
			resident.free()
		}
	}
	if !admitted {
		if err := m.v41ForwardAdmitted(); err != nil {
			return tx, err
		}
	}
	cfg := m.Cfg
	tx.layers = cfg.NumLayers
	if id < 0 || id >= cfg.VocabSize {
		return tx, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: token id %d out of range [0,%d)", ErrV41ForwardStage, id, cfg.VocabSize))
	}

	H := cfg.HiddenSize
	// This step consumes the NEXT token, so its absolute position is the current
	// committed history length. The per-layer retained states must already match.
	pos := len(st.history)
	if err := v41CandidatePoolCovers(cfg, pos+1); err != nil {
		return tx, err
	}

	// Embed the token into a fresh hidden row.
	x, embedErr := m.v41EmbeddingPanelOn([]int{id}, st.deviceGatherOps())
	if embedErr != nil {
		return tx, embedErr
	}

	full, geometryErr := v41ForwardGeometry(cfg)
	if geometryErr != nil {
		return tx, geometryErr
	}
	if !full {
		if err := st.mhcDevice.refuseReduced(); err != nil {
			return tx, err
		}
	}
	var streams [][]float32
	if full && st.mhcDevice.resident() {
		// Only shapes are read on the resident path; the panel owns all values.
		streams = [][]float32{x, x, x, x}
		if resident, err = st.mhcDevice.newResident([][]float32{x}); err != nil {
			return tx, err
		}
		zero, zerr := resident.streamZero()
		if zerr != nil {
			return tx, zerr
		}
		copy(x, zero[0])
		scratch.mhcCarry = newV41MHCCarry(1)
		scratch.mhcCarry.device = st.mhcDevice
		scratch.mhcCarry.resident = resident
	} else if full {
		streams, err = v41FullInitialStreamsOn(st.mhcDevice, x)
		if err != nil {
			return tx, err
		}
		copy(x, streams[0])
		scratch.mhcCarry = newV41MHCCarry(1)
		scratch.mhcCarry.device = st.mhcDevice
	} else {
		streams = [][]float32{x, make([]float32, H), make([]float32, H), make([]float32, H)}
	}
	if scratch.activationTrace != nil {
		tx.afterLayer = func(layer int) error {
			return scratch.activationTrace.recordStepStreams(layer, pos, streams, resident)
		}
	}

	// Fail closed BEFORE any mutation: every layer needs a seeded retained state.
	// A shadow step cannot seed a prefix -- the caller must seed.
	for l := 0; l < cfg.NumLayers; l++ {
		if st.layerState(l) == nil {
			return tx, v41StageErr(v41StageLayer, l,
				fmt.Errorf("%w: layer %d has no seeded retained state; forwardV41Step cannot seed a prefix", ErrV41ForwardStage, l))
		}
	}
	// The shared source registry a role layer publishes into / resolves from. It
	// is resolved lazily (a plain-layer session may never have built it) so a role
	// step always sees the same store the full forward used.
	previousRegistry := st.attn
	registry, err := st.attentionState(cfg.HeadDim, 8)
	if err != nil {
		return tx, v41StageErr(v41StageAttention, -1, err)
	}

	// Stage, for each layer, its PRE-STEP state so a fault in any layer or the
	// head can be rolled back by IN-PLACE restoration. A plain layer's Step writes
	// one row into the fixed windowSize-row ring and advances
	// nextWindowPos/nextCompressRow; a role layer's source append ALSO advances the
	// compressor group (partialInputs/partialPositions) and publishes into the
	// shared registry. The staged ledger holds one entry per STEPPED layer -- the
	// cursor scalars, the one O(state) ring row the step may overwrite, and the
	// retained partial-group headers -- so it is O(NumLayers) bounded
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
	rolledBack := false
	rollback := func() {
		if rolledBack {
			return
		}
		for i := len(stagedPos) - 1; i >= 0; i-- {
			state := st.layerState(i)
			state.nextWindowPos = stagedPos[i]
			state.nextCompressRow = stagedCompress[i]
			// Append may reuse the saved backing array. Drop failed rows beyond
			// its old length so rollback does not keep their allocations alive.
			if n := len(stagedInputs[i]); n < len(state.partialInputs) {
				clear(state.partialInputs[n:])
			}
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
		rolledBack = true
	}
	tx.rollback = rollback

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
				return tx, err
			}
			engramFull, err = v41ForwardGeometry(cfg)
			if err != nil {
				rollback()
				return tx, err
			}
			break
		}
	}

	tx.layer = func(l int) (*v41LayerStepWork, error) {
		state := st.layerState(l)
		stagedPos = append(stagedPos, state.nextWindowPos)
		stagedCompress = append(stagedCompress, state.nextCompressRow)
		// The compressor appends a fresh owned input or releases a completed
		// group. Projection reads retained inputs without changing their values,
		// and append only writes beyond the old lengths, even with spare capacity.
		// Keep these headers for this exclusively owned transaction; external
		// snapshots still deep-copy partial groups in kv_prefix_snapshot.go.
		stagedInputs = append(stagedInputs, state.partialInputs)
		stagedInputPos = append(stagedInputPos, state.partialPositions)
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
					return nil, geometryErr
				}
				base, ok := checkedMulInt(cacheIdx, geometry.cols)
				if !ok || base < 0 || base > len(engramRows) || geometry.cols > len(engramRows)-base {
					rollback()
					return nil, v41StageErr(v41StageEngram, l,
						fmt.Errorf("%w: Engram current row identifiers do not cover layer %d", ErrV41NativeUnsupported, l))
				}
				opened := m.v41NowNanos()
				engramStreams := [][][]float32{streams}
				layerRows := engramRows[base : base+geometry.cols]
				var rows int
				var injectionErr error
				handled := false
				if resident != nil {
					if gate := scratch.gatherOps.engramGateFunc(); gate != nil {
						rows, handled, injectionErr = m.v41EngramInjectResident(l, resident.streams, 1, layerRows, float32(cfg.RMSNormEps), geometry, gate)
						handled = handled || injectionErr != nil
					}
				}
				if !handled {
					if resident != nil {
						// The session has no Engram gate: read the panel back around this layer.
						host, readErr := resident.hostStreams(l)
						if readErr != nil {
							rollback()
							return nil, readErr
						}
						copy(x, host[0][0])
						engramStreams = host
					}
					rows, injectionErr = m.v41EngramInjectPrepared(l, [][]float32{x}, engramStreams, engramFull, 1,
						layerRows, float32(cfg.RMSNormEps), geometry)
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
					return nil, injectionErr
				}
			}
		}
		stats.LayerCalls++
		work, lerr := m.prepareV41LayerStepWithRegistry(l, x, streams, pos, state, registry, scratch)
		if lerr != nil {
			return nil, tx.abortLayerError(lerr)
		}
		return work, nil
	}
	tx.headInput = func() ([]float32, error) {

		// Head runs BEFORE any commit; a head fault rolls back exactly like a layer
		// fault (truncate/restore, no clone).
		headInput := x
		if full {
			if resident != nil {
				var rows [][]float32
				if rows, err = resident.hiddenRows(cfg.NumLayers - 1); err == nil {
					headInput = rows[0]
				}
			} else {
				headInput, err = v41MHCPreBF16On(scratch.mhcCarry.device, cfg.NumLayers-1, streams, scratch.mhcCarry.pre[0])
			}
			if err != nil {
				rollback()
				return nil, err
			}
		}
		return m.v41PrepareHeadInput(headInput, st.finalNorm, st.bf16Boundary)
	}
	tx.headProject = func(input []float32) ([]float32, error) {
		return m.v41ProjectHeadInput(input, scratch.denseProjection(), 1)
	}
	tx.headFinish = func(logits []float32, beforeCommit func() error) ([]float32, error) {
		if len(logits) != cfg.VocabSize {
			return nil, v41ProjectionOperationErr(-1, m.headName(), errV41ProjectionResult)
		}
		res, err := v41ScaleHeadLogits(logits, cfg, st.logitScale, st.logitHostRefusal())
		if err != nil {
			return nil, err
		}

		if beforeCommit != nil {
			if err := beforeCommit(); err != nil {
				return nil, err
			}
		}

		// Commit: append the token and, for a plain-only session, release the
		// step-scoped registry to match forwardV41. A role session retains the registry
		// because it is the cross-step source publication store a reader resolves.
		st.history = append(st.history, id)
		if !roleSession {
			st.attn = nil
		}
		stats.Committed = true
		st.forwardCoverage = newNativeV41ForwardCoverage(cfg, stats.LayerCalls, NativeForwardLogitsSourceV41Incremental)
		return res, nil
	}
	ready = true
	return tx, nil
}

// advanceLayer runs precisely the next layer. An error or panic aborts only
// this transaction; a selected device-operation panic is never replayed on host.
func (tx *v41StepTransaction) advanceLayer() error {
	if err := tx.prepareLayer(); err != nil {
		return err
	}
	return tx.finishLayer(nil)
}

// prepareLayer owns one lane's attention/router work and undo journal. It
// yields before expert contraction; no peer may borrow its scratch or state.
func (tx *v41StepTransaction) prepareLayer() (err error) {
	if tx.closed || tx.pendingLayer != nil || tx.nextLayer >= tx.layers {
		return fmt.Errorf("%w: incremental transaction has no next layer", ErrV41ForwardStage)
	}
	complete := false
	defer func() { tx.abortIncomplete(complete) }()
	err = func() error {
		tx.pendingLayer, err = tx.layer(tx.nextLayer)
		if err != nil {
			return tx.abortLayerError(err)
		}
		return nil
	}()
	complete = true
	return err
}

func (tx *v41StepTransaction) finishLayer(routed []float32) (err error) {
	if tx.closed || tx.pendingLayer == nil {
		return fmt.Errorf("%w: incremental transaction has no prepared layer", ErrV41ForwardStage)
	}
	complete := false
	defer func() { tx.abortIncomplete(complete) }()
	err = func() error {
		work := tx.pendingLayer
		if err = work.complete(routed); err != nil {
			return tx.abortLayerError(err)
		}
		if tx.afterLayer != nil {
			if err = tx.afterLayer(tx.nextLayer); err != nil {
				return tx.abortLayerError(err)
			}
		}
		tx.pendingLayer = nil
		tx.nextLayer++
		return nil
	}()
	complete = true
	return err
}

func (tx *v41StepTransaction) abortLayerError(err error) error {
	tx.abortAfterFailure()
	var operation *V41ExpertOperationError
	var projection *V41ProjectionOperationError
	if errors.As(err, &operation) || errors.As(err, &projection) {
		panic(err)
	}
	return err
}

// prepareHead leaves the token uncommitted and can run only once. The returned
// row owns its final-normalization/BF16 boundary and may join compatible peers.
func (tx *v41StepTransaction) prepareHead() (row []float32, err error) {
	if tx.closed || tx.pendingLayer != nil || tx.headPrepared || tx.nextLayer != tx.layers {
		return nil, fmt.Errorf("%w: incremental transaction is not ready for head", ErrV41ForwardStage)
	}
	complete := false
	defer func() { tx.abortIncomplete(complete) }()
	row, err = func() ([]float32, error) {
		row, err = tx.headInput()
		if err != nil {
			return nil, tx.abortHeadError(err)
		}
		tx.headPrepared = true
		return row, nil
	}()
	complete = true
	return row, err
}

// finishHead scales exactly this lane's projected row before committing its
// history. A peer's projection or cancellation cannot publish this token early.
func (tx *v41StepTransaction) finishHead(projected []float32, beforeCommit func() error) (logits []float32, err error) {
	if tx.closed || !tx.headPrepared {
		return nil, fmt.Errorf("%w: incremental transaction has no prepared head", ErrV41ForwardStage)
	}
	complete := false
	defer func() { tx.abortIncomplete(complete) }()
	logits, err = func() ([]float32, error) {
		logits, err = tx.headFinish(projected, beforeCommit)
		if err != nil {
			return nil, tx.abortHeadError(err)
		}
		tx.close()
		return logits, nil
	}()
	complete = true
	return logits, err
}

func (tx *v41StepTransaction) finalize() (logits []float32, err error) {
	complete := false
	defer func() { tx.abortIncomplete(complete) }()
	logits, err = func() ([]float32, error) {
		row, err := tx.prepareHead()
		if err != nil {
			return nil, err
		}
		logits, err = tx.headProject(row)
		if err != nil {
			return nil, tx.abortHeadError(err)
		}
		return tx.finishHead(logits, nil)
	}()
	complete = true
	return logits, err
}

func (tx *v41StepTransaction) abortHeadError(err error) error {
	tx.abortAfterFailure()
	var projection *V41ProjectionOperationError
	if errors.As(err, &projection) {
		panic(err)
	}
	return err
}

func (tx *v41StepTransaction) abortIncomplete(complete bool) {
	if !complete {
		tx.abortAfterFailure()
	}
}

// An already-selected error or panic owns the result. Abort still attempts its
// rollback and cleanup, but a secondary resource-release panic cannot replace it.
// With no primary failure callers use abort directly, so cleanup faults escape.
func (tx *v41StepTransaction) abortAfterFailure() {
	defer func() { _ = recover() }()
	tx.abort()
}

// abort is idempotent and never undoes an already committed token.
func (tx *v41StepTransaction) abort() {
	if tx.closed {
		return
	}
	// A suspended resident layer must release its local epoch before journal
	// rollback and transaction cleanup. Even a release panic cannot skip them.
	var cleanup any
	if work := tx.pendingLayer; work != nil {
		tx.pendingLayer = nil
		func() { defer func() { cleanup = recover() }(); work.close() }()
	}
	if !tx.stats.Committed && tx.rollback != nil {
		tx.rollback()
	}
	tx.close()
	if cleanup != nil {
		panic(cleanup)
	}
}

func (tx *v41StepTransaction) close() {
	tx.closed = true
	defer func() {
		if tx.activationCapture != nil {
			tx.activationCapture.finish(tx.stats.Committed)
			tx.activationCapture = nil
		}
	}()
	defer func() {
		tx.layer, tx.headInput, tx.headProject, tx.headFinish = nil, nil, nil, nil
		tx.rollback, tx.cleanup = nil, nil
		tx.afterLayer = nil
	}()
	if tx.cleanup != nil {
		tx.cleanup()
	}
}
