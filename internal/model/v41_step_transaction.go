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
	layer             func(int) error
	headInput         func() ([]float32, error)
	headProject       func([]float32) ([]float32, error)
	headFinish        func([]float32, func() error) ([]float32, error)
	rollback, cleanup func()
}

// beginV41Step performs admission, embedding and journal setup in serial order.
// Even on failure it returns a closed transaction carrying the rollback stats.
func (m *Model) beginV41Step(id int, st *v41ForwardState, scratch *v41ProjScratch) (tx *v41StepTransaction, err error) {
	tx = &v41StepTransaction{}
	stats := &tx.stats
	var resident *v41MHCResident
	ready := false
	defer func() {
		if ready {
			return
		}
		if primary := recover(); primary != nil {
			tx.abortAfterFailure()
			panic(primary)
		}
		if err != nil {
			tx.abortAfterFailure()
			return
		}
		tx.abort()
	}()
	if st == nil {
		return tx, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: forwardV41Step requires a seeded session state", ErrV41ForwardStage))
	}
	st.forwardCoverage = nil
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	scratch.expertGateUp, scratch.expertDown = st.expertGateUp, st.expertDown
	scratch.routedExpert = st.routedExpert
	scratch.routedLayer = st.routedLayer
	scratch.denseProjection = st.denseProjection
	scratch.groupedOutput = st.groupedOutput
	scratch.mhcProjection = st.mhcProjection
	scratch.mhcFFNProjection = st.mhcFFNProjection
	scratch.mhcCarry = nil
	scratch.attentionNorm = st.attentionNorm
	scratch.attentionBF16 = st.attentionBF16
	scratch.queryNorm = st.queryNorm
	scratch.queryPrefix = st.queryPrefix
	scratch.kvNorm = st.kvNorm
	scratch.ffnNorm = st.ffnNorm
	scratch.compressorNorm = st.compressorNorm
	scratch.indexKeyNorm = st.indexKeyNorm
	scratch.indexerScore = st.indexerScore
	scratch.indexScoreHealth = st.indexScoreHealth
	scratch.sharedActivation = st.sharedActivation
	scratch.tailRoPE = st.tailRoPE
	scratch.sharedAttention = st.sharedAttention
	scratch.routerTopK = st.routerTopK
	scratch.routedSum = st.routedSum
	scratch.sharedAdd = st.sharedAdd
	scratch.gatherOps = st.gatherOps
	tx.cleanup = func() {
		defer func() {
			scratch.expertGateUp, scratch.expertDown = nil, nil
			scratch.routedExpert = nil
			scratch.routedLayer = nil
			scratch.denseProjection = nil
			scratch.groupedOutput = nil
			scratch.mhcProjection = nil
			scratch.mhcFFNProjection = nil
			scratch.mhcCarry = nil
			scratch.attentionNorm = nil
			scratch.attentionBF16 = nil
			scratch.queryNorm = nil
			scratch.queryPrefix = nil
			scratch.kvNorm = nil
			scratch.ffnNorm = nil
			scratch.compressorNorm = nil
			scratch.indexKeyNorm = nil
			scratch.indexerScore = nil
			scratch.indexScoreHealth = nil
			scratch.sharedActivation = nil
			scratch.tailRoPE = nil
			scratch.sharedAttention = nil
			scratch.routerTopK = nil
			scratch.routedSum = nil
			scratch.sharedAdd = nil
			scratch.gatherOps = nil
		}()
		if resident != nil {
			resident.free()
		}
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		return tx, err
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
	streams := [][]float32{x, make([]float32, H), make([]float32, H), make([]float32, H)}
	if full && st.mhcDevice.resident() {
		// The panel is authoritative; streams stays a shape placeholder.
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
	rolledBack := false
	rollback := func() {
		if rolledBack {
			return
		}
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

	tx.layer = func(l int) error {
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
					return geometryErr
				}
				base, ok := checkedMulInt(cacheIdx, geometry.cols)
				if !ok || base < 0 || base > len(engramRows) || geometry.cols > len(engramRows)-base {
					rollback()
					return v41StageErr(v41StageEngram, l,
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
							return readErr
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
					return injectionErr
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
			return lerr
		}
		return nil
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
		return m.v41ProjectHeadInput(input, scratch.denseProjection, 1)
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
func (tx *v41StepTransaction) advanceLayer() (err error) {
	if tx.closed || tx.nextLayer >= tx.layers {
		return fmt.Errorf("%w: incremental transaction has no next layer", ErrV41ForwardStage)
	}
	defer tx.abortOnPanic()
	if err = tx.layer(tx.nextLayer); err != nil {
		tx.abortAfterFailure()
		return err
	}
	tx.nextLayer++
	return nil
}

// prepareHead leaves the token uncommitted and can run only once. The returned
// row owns its final-normalization/BF16 boundary and may join compatible peers.
func (tx *v41StepTransaction) prepareHead() (row []float32, err error) {
	if tx.closed || tx.headPrepared || tx.nextLayer != tx.layers {
		return nil, fmt.Errorf("%w: incremental transaction is not ready for head", ErrV41ForwardStage)
	}
	defer tx.abortOnPanic()
	row, err = tx.headInput()
	if err != nil {
		return nil, tx.abortHeadError(err)
	}
	tx.headPrepared = true
	return row, nil
}

// finishHead scales exactly this lane's projected row before committing its
// history. A peer's projection or cancellation cannot publish this token early.
func (tx *v41StepTransaction) finishHead(projected []float32, beforeCommit func() error) (logits []float32, err error) {
	if tx.closed || !tx.headPrepared {
		return nil, fmt.Errorf("%w: incremental transaction has no prepared head", ErrV41ForwardStage)
	}
	defer tx.abortOnPanic()
	logits, err = tx.headFinish(projected, beforeCommit)
	if err != nil {
		return nil, tx.abortHeadError(err)
	}
	tx.close()
	return logits, nil
}

func (tx *v41StepTransaction) finalize() (logits []float32, err error) {
	defer tx.abortOnPanic()
	row, err := tx.prepareHead()
	if err != nil {
		return nil, err
	}
	logits, err = tx.headProject(row)
	if err != nil {
		return nil, tx.abortHeadError(err)
	}
	return tx.finishHead(logits, nil)
}

func (tx *v41StepTransaction) abortHeadError(err error) error {
	tx.abortAfterFailure()
	var projection *V41ProjectionOperationError
	if errors.As(err, &projection) {
		panic(err)
	}
	return err
}

func (tx *v41StepTransaction) abortOnPanic() {
	if r := recover(); r != nil {
		tx.abortAfterFailure()
		panic(r)
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
	if !tx.stats.Committed && tx.rollback != nil {
		tx.rollback()
	}
	tx.close()
}

func (tx *v41StepTransaction) close() {
	tx.closed = true
	defer func() {
		tx.layer, tx.headInput, tx.headProject, tx.headFinish = nil, nil, nil, nil
		tx.rollback, tx.cleanup = nil, nil
	}()
	if tx.cleanup != nil {
		tx.cleanup()
	}
}
