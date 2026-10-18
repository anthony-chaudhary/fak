package model

import (
	"errors"
	"fmt"
)

var errV41StepRoundState = errors.New("model: invalid V4.1 step round state")

type v41StepRound struct {
	model                    *Model
	state                    *v41ForwardState
	layers                   []*V41AttentionState
	entries                  []*v41StepRoundEntry
	pending                  *v41StepRoundEntry
	baseHistoryLen, capacity int
	failed, closed, poisoned bool
}

type v41StepRoundEntry struct {
	round                        *v41StepRound
	tx                           *v41StepTransaction
	ordinal, token, historyLen   int
	coverage                     *NativeForwardCoverage
	undo                         func()
	committed, closed, cleanupOK bool
}

func (m *Model) beginV41StepRound(st *v41ForwardState, capacity int) (*v41StepRound, error) {
	if m == nil || st == nil || st.activeStepRound != nil || capacity < 1 || capacity > 32 ||
		len(st.history) == 0 || m.Cfg.NumLayers < 1 || v41ActivationTraceEnvPath() != "" {
		return nil, errV41StepRoundState
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		return nil, err
	}
	roles := m.v41AttentionRolesCached()
	layers := make([]*V41AttentionState, m.Cfg.NumLayers)
	seen := make(map[*V41AttentionState]bool, len(layers)+1)
	if st.attn != nil {
		seen[st.attn] = true
	}
	for layer := range layers {
		state := st.layerState(layer)
		plan, err := v41AttentionPlanFor(m.Cfg, layer, roles)
		if err != nil || state == nil || seen[state] || state.nextWindowPos != len(st.history) ||
			state.windowSize <= 0 || len(state.window) != state.windowSize ||
			(plan.Role != V41AttentionRolePerLayer && plan.Ratio <= 1 && plan.KVSourceLayer < 0) {
			return nil, errV41StepRoundState
		}
		seen[state] = true
		layers[layer] = state
	}
	r := &v41StepRound{model: m, state: st, layers: layers, capacity: capacity,
		baseHistoryLen: len(st.history), entries: make([]*v41StepRoundEntry, 0, capacity)}
	st.activeStepRound = r
	return r, nil
}

func (r *v41StepRound) ownsState() bool {
	if r == nil || r.closed || r.model == nil || r.state == nil || r.state.activeStepRound != r {
		return false
	}
	for layer, state := range r.layers {
		if r.state.layerState(layer) != state {
			return false
		}
	}
	return true
}

func (r *v41StepRound) committedCount() int {
	n := 0
	for _, entry := range r.entries {
		if entry.committed {
			n++
		}
	}
	return n
}

func (r *v41StepRound) atCommittedBoundary() bool {
	if len(r.state.history) != r.baseHistoryLen+r.committedCount() {
		return false
	}
	for _, state := range r.layers {
		if state.nextWindowPos != len(r.state.history) {
			return false
		}
	}
	return true
}

func (r *v41StepRound) beginStep(id int, scratch *v41ProjScratch) (*v41StepTransaction, error) {
	if !r.ownsState() || r.failed || r.poisoned || r.pending != nil ||
		len(r.entries) >= r.capacity || !r.atCommittedBoundary() ||
		id < 0 || id >= r.model.Cfg.VocabSize || v41ActivationTraceEnvPath() != "" ||
		(scratch != nil && scratch.activationTrace != nil) {
		return nil, errV41StepRoundState
	}
	entry := &v41StepRoundEntry{round: r, ordinal: len(r.entries), token: id,
		historyLen: len(r.state.history), coverage: r.state.forwardCoverage}
	r.entries = append(r.entries, entry)
	r.pending = entry
	return r.model.beginV41StepCore(id, r.state, scratch, false, entry)
}

func (r *v41StepRound) finish(keep int) error {
	if !r.ownsState() || r.pending != nil || keep < 0 || keep > r.committedCount() ||
		((r.failed || r.poisoned) && keep != 0) {
		return errV41StepRoundState
	}
	if !r.atCommittedBoundary() {
		r.poisoned = true
		return errV41StepRoundState
	}
	return r.drain(keep)
}

// A direct defer r.abort() preserves a non-nil active panic. A wrapping defer must
// preserve its own primary because recover does not cross that wrapper.
func (r *v41StepRound) abort() (err error) {
	primary := recover()
	if primary != nil {
		defer func() {
			_ = recover()
			panic(primary)
		}()
	}
	if r == nil {
		return errV41StepRoundState
	}
	if r.closed {
		if r.poisoned {
			return errV41StepRoundState
		}
		return nil
	}
	if !r.ownsState() {
		r.poisoned = true
		return errV41StepRoundState
	}
	return r.drain(0)
}

func (r *v41StepRound) drain(keep int) (err error) {
	st, entries, pending := r.state, r.entries, r.pending
	poisoned, incomplete := r.poisoned, false
	r.closed, r.poisoned = true, true
	defer func() {
		for _, entry := range entries {
			if !entry.closed || !entry.cleanupOK {
				poisoned = true
			}
			*entry = v41StepRoundEntry{}
		}
		r.entries, r.layers, r.pending = nil, nil, nil
		r.model, r.state = nil, nil
		if st.activeStepRound != r {
			poisoned = true
		}
		r.poisoned = poisoned
		if poisoned {
			err = errV41StepRoundState
		} else {
			st.activeStepRound = nil
		}
	}()
	attempt := func(fn func()) {
		complete := false
		defer func() {
			if !complete {
				incomplete, poisoned = true, true
			}
		}()
		if incomplete {
			func() {
				defer func() { _ = recover() }()
				fn()
			}()
		} else {
			fn()
		}
		complete = true
	}
	for i := keep; i < len(entries); i++ {
		entry := entries[i]
		defer attempt(func() {
			defer func() {
				if len(st.history) < entry.historyLen {
					poisoned = true
				} else {
					clear(st.history[entry.historyLen:])
					st.history = st.history[:entry.historyLen]
				}
				st.forwardCoverage = entry.coverage
			}()
			if entry.undo != nil {
				entry.undo()
			} else if entry.committed {
				poisoned = true
			}
		})
	}
	if pending != nil && pending.tx != nil {
		tx := pending.tx
		defer attempt(func() {
			if !tx.closed {
				tx.close()
			}
		})
		defer attempt(tx.abort)
	}
	return nil
}

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
	pendingAttention  *v41LayerAttentionSuspension
	headInput         func() ([]float32, error)
	headProject       func([]float32) ([]float32, error)
	headFinish        func([]float32, func() error) ([]float32, error)
	commit            func(bool)
	captureValidate   func() error
	rollback, cleanup func()
	afterLayer        func(int) error
	activationCapture *v41StepActivationCapture
	roundEntry        *v41StepRoundEntry
}

// beginV41Step performs admission, embedding and journal setup in serial order.
// Even on failure it returns a closed transaction carrying the rollback stats.
func (m *Model) beginV41Step(id int, st *v41ForwardState, scratch *v41ProjScratch) (*v41StepTransaction, error) {
	return m.beginV41StepCore(id, st, scratch, false, nil)
}

// beginV41StepAdmitted reuses full-model admission only within one owned cohort
// dispatch. The cohort already validated this exact immutable model; the token,
// retained state, cursor, and transaction checks still run for every lane.
// It must not be used as a successful-admission cache across calls.
func (m *Model) beginV41StepAdmitted(id int, st *v41ForwardState, scratch *v41ProjScratch) (*v41StepTransaction, error) {
	return m.beginV41StepCore(id, st, scratch, true, nil)
}

func (m *Model) beginV41StepCore(id int, st *v41ForwardState, scratch *v41ProjScratch, admitted bool, entry *v41StepRoundEntry) (tx *v41StepTransaction, err error) {
	tx = &v41StepTransaction{roundEntry: entry}
	if entry != nil {
		entry.tx = tx
	}
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
	if !v41StepEntryAdmits(st, entry, m, id) {
		return tx, errV41StepRoundState
	}
	st.forwardCoverage = nil
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	tx.activationCapture = m.beginV41StepActivationCapture(id, st)
	if tx.activationCapture != nil {
		scratch.activationTrace = tx.activationCapture.trace
	}
	scratch.bindV41StepState(st)
	tx.cleanup = func() {
		defer scratch.clearV41StepState()
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

	// This step consumes the NEXT token, so its absolute position is the current
	// committed history length. The per-layer retained states must already match.
	pos := len(st.history)
	if capture := st.dsparkCapture; capture != nil {
		tx.captureValidate = func() error { return capture.beforeCommit(pos) }
	}
	if err := v41CandidatePoolCovers(cfg, pos+1); err != nil {
		return tx, err
	}
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
	streams, resident, err = m.v41StepInitialStreams(st, x, full, scratch)
	if err != nil {
		return tx, err
	}
	if scratch.activationTrace != nil {
		tx.afterLayer = func(layer int) error {
			return scratch.activationTrace.recordStepStreams(layer, pos, streams, resident)
		}
	}

	// A shadow step cannot seed a prefix, so refuse before any mutation.
	for l := 0; l < cfg.NumLayers; l++ {
		if st.layerState(l) == nil {
			return tx, v41StageErr(v41StageLayer, l,
				fmt.Errorf("%w: layer %d has no seeded retained state; forwardV41Step cannot seed a prefix", ErrV41ForwardStage, l))
		}
	}
	// Resolved lazily (a plain-layer session may never have built it) so a role
	// step always sees the same store the full forward used.
	previousRegistry := st.attn
	registry, err := st.attentionState(cfg.HeadDim, 8)
	if err != nil {
		return tx, v41StageErr(v41StageAttention, -1, err)
	}
	journal := newV41StepJournal(st, stats, previousRegistry, registry, cfg.NumLayers)
	rollback := journal.rollback
	tx.rollback = rollback
	if entry != nil {
		entry.undo = rollback
	}

	// A plain-only session keeps the historical post-step release of the shared
	// registry; a role session must retain it so a reader layer can resolve a
	// source's published rows on the next Step.
	roleSession := m.v41RoleSchedule()
	engram, err := m.v41StepEngramRows(st, id)
	if err != nil {
		rollback()
		return tx, err
	}

	tx.layer = func(l int) (*v41LayerStepWork, error) {
		state := st.layerState(l)
		journal.stage(l, state)
		if err := m.v41StepInjectEngram(l, engram, st, x, streams, resident, scratch, rollback); err != nil {
			return nil, err
		}
		if st.dsparkCapture != nil {
			if err := st.dsparkCapture.captureStep(st, l, id, pos, streams, resident, full); err != nil {
				return nil, tx.abortLayerError(err)
			}
		}
		stats.LayerCalls++
		if tx.pendingAttention != nil {
			err := m.prepareV41LayerAttention(tx.pendingAttention, l, x, streams, pos, state, registry, scratch)
			return nil, err
		}
		work, lerr := m.prepareV41LayerStepWithRegistry(l, x, streams, pos, state, registry, scratch)
		if lerr != nil {
			return nil, tx.abortLayerError(lerr)
		}
		return work, nil
	}
	tx.headInput = func() ([]float32, error) {
		// A head fault rolls back exactly like a layer fault.
		headInput := x
		if full {
			var err error
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
		headState := st
		if scratch.activationTrace != nil {
			headState = nil
		}
		return m.v41PrepareHeadInputBound(headInput, st.headNorm, headState)
	}
	// Only the synchronous single-row projection can prepare a scaled row.
	// Shared-panel projection bypasses this callback and retains ordinary finish.
	var composedScale []float32
	tx.headProject = func(input []float32) ([]float32, error) {
		composedScale = nil
		if scratch.activationTrace == nil {
			raw, scaled, selected, err := m.v41TryHeadScaleLoan(st, scratch.projectionRoute, input, cfg)
			if selected {
				composedScale = scaled
				return raw, err
			}
		}
		return m.v41ProjectHeadInput(input, scratch.denseProjection(), 1)
	}
	tx.headFinish = func(logits []float32, beforeCommit func() error) ([]float32, error) {
		if len(logits) != cfg.VocabSize {
			return nil, v41ProjectionOperationErr(-1, m.headName(), errV41ProjectionResult)
		}
		res := composedScale
		composedScale = nil
		var err error
		if res == nil {
			res, err = v41ScaleHeadLogits(logits, cfg, st.logitScale(), st.logitHostRefusal())
		}
		if err != nil {
			return nil, err
		}
		if beforeCommit != nil {
			if err := beforeCommit(); err != nil {
				return nil, err
			}
		}
		if tx.captureValidate != nil {
			if err := tx.captureValidate(); err != nil {
				return nil, err
			}
		}
		tx.commit(true)
		return res, nil
	}
	tx.commit = func(withLogits bool) {
		st.history = append(st.history, id)
		if !roleSession {
			st.attn = nil
		}
		stats.Committed = true
		if entry != nil {
			entry.committed = true
		}
		if tx.activationCapture != nil {
			tx.activationCapture.stateOnlyCommit = !withLogits
		}
		st.forwardCoverage = nil
		if withLogits {
			st.forwardCoverage = newNativeV41ForwardCoverage(cfg, stats.LayerCalls, NativeForwardLogitsSourceV41Incremental)
		}
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
	if tx.closed || tx.pendingLayer != nil || tx.pendingAttention != nil || tx.nextLayer >= tx.layers {
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

// publishPendingKV commits the prepared layer's window row before its finish so
// the next position of the same session can be prepared at this layer.
func (tx *v41StepTransaction) publishPendingKV() (err error) {
	if tx.closed || tx.pendingLayer == nil {
		return fmt.Errorf("%w: incremental transaction has no prepared layer", ErrV41ForwardStage)
	}
	complete := false
	defer func() { tx.abortIncomplete(complete) }()
	if err = tx.pendingLayer.publishEarly(); err != nil {
		err = tx.abortLayerError(err)
	}
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

// finalizeNoLogits preserves head-input validation and the normal transaction
// commit/cleanup, but omits projection and scaling when no caller consumes logits.
// It cannot finish a partially advanced or already prepared transaction.
func (tx *v41StepTransaction) finalizeNoLogits() (err error) {
	complete := false
	defer func() { tx.abortIncomplete(complete) }()
	err = func() error {
		row, err := tx.prepareHead()
		if err != nil {
			return err
		}
		if err = v41ValidateDiscardedHeadInput(row); err != nil {
			return tx.abortHeadError(err)
		}
		if tx.captureValidate != nil {
			if err := tx.captureValidate(); err != nil {
				return tx.abortHeadError(err)
			}
		}
		tx.commit(false)
		tx.close()
		return nil
	}()
	complete = true
	return err
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
	if attention := tx.pendingAttention; attention != nil {
		tx.pendingAttention = nil
		func() { defer func() { cleanup = recover() }(); attention.close() }()
	}
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
	cleanupOK := false
	defer func() {
		if entry := tx.roundEntry; entry != nil {
			entry.closed, entry.cleanupOK, entry.tx = true, cleanupOK, nil
			if !entry.committed {
				entry.round.failed = true
			}
			if !cleanupOK {
				entry.round.failed, entry.round.poisoned = true, true
			}
			if entry.round.pending == entry {
				entry.round.pending = nil
			}
			tx.roundEntry = nil
		}
	}()
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
		tx.commit = nil
		tx.captureValidate = nil
	}()
	if tx.cleanup != nil {
		tx.cleanup()
	}
	cleanupOK = true
}

// v41LayerAttentionSuspension is caller-held storage for exactly one lane.
// The concrete continuations retain shared lifecycle ownership across copies;
// the transaction owns abandonment until successful resume transfers the work.
type v41LayerAttentionSuspension struct {
	plain  v41PlainAttentionContinuation
	role   v41RoleAttentionContinuation
	isRole bool
}

func (a *v41LayerAttentionSuspension) close() {
	if a == nil {
		return
	}
	if a.isRole {
		a.role.close()
	} else {
		a.plain.close()
	}
}
func (a *v41LayerAttentionSuspension) input() []float32 {
	if a.isRole {
		return a.role.input
	}
	return a.plain.input
}
func (a *v41LayerAttentionSuspension) project() ([]float32, error) {
	if a.isRole {
		return a.role.project(a.role.input)
	}
	return a.plain.project(a.plain.input)
}
func (a *v41LayerAttentionSuspension) resume(output []float32, validate bool) (*v41LayerStepWork, error) {
	if a.isRole {
		return a.role.resumeProjected(output, validate)
	}
	return a.plain.resumeProjected(output, validate)
}

func (m *Model) prepareV41LayerAttention(a *v41LayerAttentionSuspension, l int, x []float32, streams [][]float32, pos int, state, registry *V41AttentionState, scratch *v41ProjScratch) (err error) {
	cfg := m.Cfg
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	if err = v41CheckLayerStepRows(cfg, l, x, streams); err != nil {
		return err
	}
	plan, err := v41AttentionPlanFor(cfg, l, m.v41AttentionRolesCached())
	if err != nil {
		return err
	}
	if state == nil {
		return v41StageErr(v41StageAttention, l, fmt.Errorf("%w: layer %d step requires retained decode state", ErrV41ForwardStage, l))
	}
	a.isRole = plan.Role != V41AttentionRolePerLayer || plan.Ratio > 0
	if a.isRole {
		a.role, err = m.prepareV41RoleAttention(l, plan, x, streams, pos, state, registry, scratch)
		return err
	}
	if pos != state.nextWindowPos {
		return v41StageErr(v41StageAttention, l, fmt.Errorf("%w: layer %d step position %d is not the next retained position %d", ErrV41ForwardStage, l, pos, state.nextWindowPos))
	}
	a.plain, err = m.prepareV41PlainAttention(l, x, streams, pos, state, scratch)
	return err
}

func (tx *v41StepTransaction) prepareAttention(a *v41LayerAttentionSuspension) (err error) {
	if a == nil || tx.closed || tx.pendingLayer != nil || tx.pendingAttention != nil || tx.nextLayer >= tx.layers {
		return fmt.Errorf("%w: incremental transaction has no next attention", ErrV41ForwardStage)
	}
	*a = v41LayerAttentionSuspension{}
	tx.pendingAttention = a
	complete := false
	defer func() { tx.abortIncomplete(complete) }()
	work, err := tx.layer(tx.nextLayer)
	if err != nil {
		return tx.abortLayerError(err)
	}
	if work != nil {
		work.close()
		return tx.abortLayerError(fmt.Errorf("%w: attention suspension returned completed work", ErrV41ForwardStage))
	}
	complete = true
	return nil
}

func (tx *v41StepTransaction) resumeAttention(output []float32, grouped bool) (err error) {
	if tx.closed || tx.pendingAttention == nil || tx.pendingLayer != nil {
		return fmt.Errorf("%w: incremental transaction has no suspended attention", ErrV41ForwardStage)
	}
	complete := false
	defer func() { tx.abortIncomplete(complete) }()
	a := tx.pendingAttention
	if !grouped {
		output, err = a.project()
		if err != nil {
			return tx.abortLayerError(v41StageErr(v41StageAttention, tx.nextLayer, err))
		}
	}
	tx.pendingLayer, err = a.resume(output, grouped)
	if err != nil {
		return tx.abortLayerError(err)
	}
	tx.pendingAttention = nil
	complete = true
	return nil
}

func (a *v41LayerAttentionSuspension) plainScratch() *v41ProjScratch {
	if a == nil {
		return nil
	}
	if a.isRole {
		return a.role.scratch
	}
	if a.plain.st == nil {
		return nil
	}
	return a.plain.st.scratch
}
