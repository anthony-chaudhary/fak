package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Qwen38DFlash2Request owns a fresh target Session and its five-layer sidecar
// until Close. Successful construction transfers the Session: callers must not
// Step, Close, restore, or otherwise use that Session afterward. Public request
// operations are serialized; independent requests share no mutable draft state.
//
// Bootstrap obtains genuine teacher-forced logits and residuals. Prepare only
// exposes exactly verified token IDs; Commit selects the prefix BEFORE callers
// publish text or invoke observers. After uncertain stream delivery, Close the
// request instead of reusing its generated prefix. This diagnostic host-side
// draft path does not itself admit artifacts, enable serving, or claim speedup.
type Qwen38DFlash2Request struct {
	mu                   sync.Mutex
	options              Qwen38DFlash2RequestOptions
	ref                  *dflash2Reference
	logits               []float32
	bootstrapped         bool
	pending              *Qwen38DFlash2Round
	closed               bool
	closeErr             error
	cleanup              []*dflash2CleanupObligation
	quarantinedTarget    *Session
	quarantinedReference *dflash2Reference
	quarantinedFence     compute.Tensor
}

// NewQwen38DFlash2Request opens the pinned stock sidecar for a fresh, compatible
// target. On failure the caller retains target ownership. On success Close is
// required even if Bootstrap fails. No tokenizer or template identity is inferred
// from shape compatibility; serving admission must bind those separately.
func NewQwen38DFlash2Request(ctx context.Context, target *Session, sidecarPath string) (*Qwen38DFlash2Request, error) {
	if ctx == nil {
		return nil, errors.New("dflash2: request context required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := dflash2FreshTarget(target); err != nil {
		return nil, err
	}
	f, err := newQwen38DFlash2Reference(ctx, target, sidecarPath, 0)
	if err != nil {
		return nil, err
	}
	q, err := newDFlash2Request(f)
	if err != nil {
		return nil, errors.Join(err, f.Close())
	}
	return q, nil
}

func dflash2FreshTarget(target *Session) error {
	if err := validateQwen35TargetFeatureSession(target); err != nil {
		return err
	}
	// This owner admits dense Qwen sessions only. Unrelated auxiliary owners
	// need their own disposal contracts rather than inheriting this ledger.
	if target.M.Cfg.NumExperts != 0 || target.sharedRing != nil || target.expertRing != nil || target.v4ExpertOwner != nil || target.denseDecode != nil {
		return errors.New("dflash2: fresh dense Qwen ownership required")
	}
	position, _, err := qwen35TargetFeaturePrefix(target)
	if err != nil || position != 0 || target.qwen35TargetFeatureTxn != nil || target.qwen35TargetLayerCapture != nil {
		return errors.New("dflash2: request requires an unused exclusive target")
	}
	return nil
}

// The private seam uses the same request implementation with tiny real weights.
func newDFlash2Request(f *dflash2Reference) (*Qwen38DFlash2Request, error) {
	if f == nil || f.closed || f.nextPosition != 0 || len(f.prefixTokens) != 0 {
		return nil, errors.New("dflash2: fresh reference required")
	}
	if err := dflash2FreshTarget(f.target); err != nil {
		return nil, err
	}
	if err := f.proposalReady(context.Background()); err != nil {
		return nil, err
	}
	return &Qwen38DFlash2Request{ref: f}, nil
}

func (q *Qwen38DFlash2Request) usable(ctx context.Context) error {
	if q.closed || q.ref == nil || ctx == nil {
		return errors.New("dflash2: live request and context required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return validateQwen35TargetFeatureSession(q.ref.target)
}

// A complete target snapshot includes attention KV, recurrent/conv state,
// lineage, and final-hidden history. Draft cache slices are immutable between
// publications, so retaining their descriptors preserves the exact old tail.
type dflash2RequestCheckpoint struct {
	snapshot *PrefixSnapshot
	tx       *qwen35TargetFeatureTransaction
	cache    [5]dflash2ContextKV
	prefix   []uint32
	position int
	step     int
	warm     bool
	logits   []float32
}

func (q *Qwen38DFlash2Request) checkpoint() (*dflash2RequestCheckpoint, error) {
	f := q.ref
	p, err := f.target.PrefixSnapshot()
	if err != nil {
		return nil, err
	}
	return &dflash2RequestCheckpoint{snapshot: p, cache: f.cache, prefix: f.prefixTokens, position: f.nextPosition,
		step: f.target.halStep, warm: f.target.halLogitsWarm, logits: q.logits}, nil
}

func dflash2Interrupted(value any) error {
	if err, ok := value.(error); ok {
		return fmt.Errorf("dflash2: interrupted request operation: %w", err)
	}
	return fmt.Errorf("dflash2: interrupted request operation: %v", value)
}

// Cleanup failures become errors without replacing the original operation's
// cause. The completion bit also catches legacy panic(nil). Goexit still unwinds
// its goroutine; enclosing completion fences retire any incomplete owner.
func dflash2Cleanup(fn func() error) (err error) {
	complete := false
	defer func() {
		if !complete {
			err = errors.Join(err, dflash2Interrupted(recover()))
		}
	}()
	err = fn()
	complete = true
	return err
}

func (q *Qwen38DFlash2Request) rollback(cp *dflash2RequestCheckpoint) (err error) {
	complete := false
	defer func() {
		if !complete || err != nil {
			err = errors.Join(err, q.closeLocked())
		}
	}()
	if cp == nil || cp.snapshot == nil || q.closed {
		return errors.New("dflash2: rollback ownership unavailable")
	}
	err = func() (err error) {
		// Register release before any backend callback can interrupt restoration.
		p := cp.snapshot
		cp.snapshot = nil
		defer func() { err = errors.Join(err, dflash2Cleanup(func() error { p.Close(); return nil })) }()
		if cp.tx != nil {
			tx := cp.tx
			cp.tx = nil
			err = dflash2Cleanup(tx.Abort)
		}
		err = errors.Join(err, dflash2Cleanup(func() error { return p.Restore(q.ref.target) }))
		if err != nil {
			return err
		}
		f := q.ref
		f.target.halStep, f.target.halLogitsWarm = cp.step, cp.warm
		f.cache, f.prefixTokens, f.nextPosition = cp.cache, cp.prefix, cp.position
		q.logits = cp.logits
		return validateQwen35TargetFeatureSession(f.target)
	}()
	if err != nil {
		return err
	}
	// Include snapshot cleanup in the restoration completion boundary.
	complete = true
	return nil
}

// finishCheckpoint releases the old state before a new boundary can escape. A
// failed release cannot be rolled back through an already-consumed snapshot;
// retire the request instead. No output is publishable after that error.
func (q *Qwen38DFlash2Request) finishCheckpoint(cp *dflash2RequestCheckpoint) error {
	p := cp.snapshot
	cp.snapshot = nil
	err := dflash2Cleanup(func() error { p.Close(); return nil })
	if err != nil {
		return errors.Join(err, q.closeLocked())
	}
	return nil
}

// Bootstrap teacher-forces a nonempty prompt on a fresh target. Only eight tap
// rows are staged at a time; the draft retains at most window-1 committed rows.
// The first token is a real target Step, never a fabricated logits boundary.
// A normal failure restores the empty complete state so a caller may retry;
// failed restoration disposes both owners. No partial prompt becomes ready.
func (q *Qwen38DFlash2Request) Bootstrap(ctx context.Context, prompt []int) (err error) {
	if q == nil {
		return errors.New("dflash2: nil request")
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if err = q.usable(ctx); err != nil {
		return err
	}
	if q.bootstrapped || q.pending != nil || len(prompt) == 0 {
		return errors.New("dflash2: bootstrap requires one nonempty fresh prompt")
	}
	if err = dflash2FreshTarget(q.ref.target); err != nil {
		return err
	}
	for _, token := range prompt {
		if token < 0 || token >= q.ref.g.vocab {
			return errors.New("dflash2: prompt token outside target vocabulary")
		}
	}
	var cp *dflash2RequestCheckpoint
	complete := false
	defer func() {
		if !complete {
			if value := recover(); value != nil || err == nil {
				err = errors.Join(err, dflash2Interrupted(value))
			}
			err = errors.Join(err, q.rollback(cp))
		}
	}()
	cp, err = q.checkpoint()
	if err != nil {
		return err
	}
	var logits []float32
	for start := 0; start < len(prompt); start += qwen35TargetFeatureMaxTokens {
		end := min(start+qwen35TargetFeatureMaxTokens, len(prompt))
		var rows []qwen35TargetFeatureRow
		logits, rows, err = q.teacherForce(ctx, prompt[start:end])
		if err != nil {
			return err
		}
		if err = q.ref.AppendCommitted(ctx, rows); err != nil {
			return err
		}
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if err = q.ref.proposalReady(ctx); err != nil {
		return err
	}
	if err = q.finishCheckpoint(cp); err != nil {
		return err
	}
	if err = ctx.Err(); err != nil {
		return errors.Join(err, q.closeLocked())
	}
	q.logits, q.bootstrapped = logits, true
	complete = true
	return nil
}

func (q *Qwen38DFlash2Request) teacherForce(ctx context.Context, tokens []int) (logits []float32, rows []qwen35TargetFeatureRow, err error) {
	target := q.ref.target
	scope := target.beginBackendExecution()
	complete := false
	defer scope.finishCompleted(&complete)
	for _, token := range tokens {
		if err = ctx.Err(); err != nil {
			return nil, nil, err
		}
		position, _, prefixErr := qwen35TargetFeaturePrefix(target)
		if prefixErr != nil {
			return nil, nil, prefixErr
		}
		capture := &qwen35TargetLayerCapture{row: qwen35TargetFeatureRow{
			Position: position, Token: token, Layers: append([]int(nil), q.ref.g.taps...),
		}}
		logits = func() []float32 {
			target.qwen35TargetLayerCapture = capture
			defer func() { target.qwen35TargetLayerCapture = nil }()
			return append([]float32(nil), target.stepOwned(token)...)
		}()
		next, lineage, prefixErr := qwen35TargetFeaturePrefix(target)
		if prefixErr != nil || next != position+1 || lineage.ids[position] != uint32(token) || len(capture.row.Hidden) != len(q.ref.g.taps) || len(logits) != q.ref.g.vocab {
			return nil, nil, errors.New("dflash2: incomplete teacher-forced boundary")
		}
		for _, hidden := range capture.row.Hidden {
			if len(hidden) != q.ref.g.hidden {
				return nil, nil, errors.New("dflash2: incomplete teacher-forced tap")
			}
		}
		for _, value := range logits {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, nil, errors.New("dflash2: nonfinite teacher-forced logits")
			}
		}
		rows = append(rows, capture.row)
	}
	if err = ctx.Err(); err != nil {
		return nil, nil, err
	}
	complete = true
	return logits, rows, nil
}

// Qwen38DFlash2Round holds a privately verified block. Tokens returns an owned
// copy for pure stop/EOS/budget inspection. Commit accepts a prefix of that copy;
// Abort rejects the whole round. Neither method emits or calls a user callback.
type Qwen38DFlash2Round struct {
	owner  *Qwen38DFlash2Request
	cp     *dflash2RequestCheckpoint
	tokens []int
	done   bool
}

// Prepare stages one exact-greedy block of at most eight tokens. The anchor is
// selected from privately owned current logits; callers cannot supply a false
// boundary. Until Commit or Abort, another Prepare/Bootstrap is refused.
func (q *Qwen38DFlash2Request) Prepare(ctx context.Context, maxTokens int) (round *Qwen38DFlash2Round, err error) {
	return q.prepare(ctx, maxTokens, nil)
}

func (q *Qwen38DFlash2Request) prepare(ctx context.Context, maxTokens int, timing *DFlash2PrepareTiming) (round *Qwen38DFlash2Round, err error) {
	if q == nil {
		return nil, errors.New("dflash2: nil request")
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if err = q.usable(ctx); err != nil {
		return nil, err
	}
	if !q.bootstrapped || q.pending != nil || maxTokens < 1 || maxTokens > 8 || len(q.logits) != q.ref.g.vocab {
		return nil, errors.New("dflash2: ready request and one-to-eight token budget required")
	}
	width, widthErr := q.options.panelWidth()
	if widthErr != nil {
		return nil, widthErr
	}
	maxTokens, err = q.options.blockBudget(maxTokens)
	if err != nil {
		return nil, err
	}
	if err = q.ref.proposalReady(ctx); err != nil {
		return nil, err
	}
	var cp *dflash2RequestCheckpoint
	complete := false
	defer func() {
		if !complete {
			if value := recover(); value != nil || err == nil {
				err = errors.Join(err, dflash2Interrupted(value))
			}
			err = errors.Join(err, q.rollback(cp))
			round = nil
		}
	}()
	cp, err = q.checkpoint()
	if err != nil {
		return nil, err
	}
	anchor := argmaxF32(q.logits)
	proposal := []int{anchor}
	if maxTokens > 1 && !q.ref.target.M.Cfg.IsEOS(anchor) {
		proposalStart := timing.start()
		draft, draftErr := q.ref.Propose(ctx, anchor, maxTokens-1)
		if draftErr != nil {
			return nil, draftErr
		}
		timing.finishProposal(proposalStart, len(draft))
		proposal = append(proposal, draft...)
	}
	cp.tx, err = q.beginVerification()
	if err != nil {
		return nil, err
	}
	verifyStart := timing.start()
	verified, err := cp.tx.Verify(ctx, proposal)
	if err != nil {
		return nil, err
	}
	timing.finishVerify(verifyStart)
	timing.recordVerification(cp.tx.tx.VerificationReceipt(), len(proposal), width)
	accepted, current := 0, q.logits
	for i, token := range proposal {
		if token != argmaxF32(current) {
			break
		}
		accepted++
		current = verified[i]
		if q.ref.target.M.Cfg.IsEOS(token) {
			break
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	round = &Qwen38DFlash2Round{owner: q, cp: cp, tokens: append([]int(nil), proposal[:accepted]...)}
	q.pending = round
	complete = true
	return round, nil
}

// Tokens returns no mutable reference to the verified proposal. It is empty
// after the round ends or its request closes.
func (r *Qwen38DFlash2Round) Tokens() []int {
	if r == nil || r.owner == nil {
		return nil
	}
	q := r.owner
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed || r.done || q.pending != r {
		return nil
	}
	return append([]int(nil), r.tokens...)
}

// Commit atomically commits the chosen prefix to target and draft before
// returning token IDs for emission. Choosing zero is a valid stop-before-anchor
// decision. Invalid counts, cancellation, and late draft failures abort the
// round. A caller must never publish the prepared copy before this succeeds.
func (r *Qwen38DFlash2Round) Commit(ctx context.Context, accepted int) (tokens []int, err error) {
	return r.commit(ctx, accepted, nil)
}

func (r *Qwen38DFlash2Round) commit(ctx context.Context, accepted int, observation *DFlash2CommitObservation) (tokens []int, err error) {
	if r == nil || r.owner == nil {
		return nil, errors.New("dflash2: nil prepared round")
	}
	q := r.owner
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed || r.done || q.pending != r {
		return nil, errors.New("dflash2: prepared round is no longer active")
	}
	// Capture the transaction before terminal cleanup clears r.cp. This observer
	// runs after cleanup and never changes the ordinary Commit path or clocks.
	if observation != nil {
		tx := r.cp.tx.tx
		defer func() {
			observation.Verification = tx.VerificationReceipt()
			// This route has no scheduler shape; do not export an aliased pointer.
			observation.Verification.Shape = nil
			observation.ReceiptAvailable = true
		}()
	}
	complete := false
	defer func() {
		if !complete {
			if value := recover(); value != nil || err == nil {
				err = errors.Join(err, dflash2Interrupted(value))
			}
			err = errors.Join(err, q.rollback(r.cp))
			tokens = nil
		}
		q.pending, r.cp, r.tokens, r.done = nil, nil, nil, true
	}()
	if err = q.usable(ctx); err != nil {
		return nil, err
	}
	if accepted < 0 || accepted > len(r.tokens) {
		return nil, errors.New("dflash2: commit count exceeds verified prefix")
	}
	logits, rows, err := r.cp.tx.Commit(ctx, accepted)
	if err != nil {
		return nil, err
	}
	if accepted > 0 {
		if err = q.ref.AppendCommitted(ctx, rows); err != nil {
			return nil, err
		}
	}
	if err = q.ref.proposalReady(ctx); err != nil {
		return nil, err
	}
	tokens = append([]int(nil), r.tokens[:accepted]...)
	logits = append([]float32(nil), logits...)
	if err = q.finishCheckpoint(r.cp); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, errors.Join(err, q.closeLocked())
	}
	q.logits = logits
	complete = true
	return tokens, nil
}

// Abort restores the exact pre-round target and draft boundary. Repeated Abort
// after a terminal round is harmless; Commit on that round remains invalid.
func (r *Qwen38DFlash2Round) Abort() error {
	if r == nil || r.owner == nil {
		return nil
	}
	q := r.owner
	q.mu.Lock()
	defer q.mu.Unlock()
	if r.done || q.closed {
		return nil
	}
	if q.pending != r {
		return errors.New("dflash2: prepared round owner mismatch")
	}
	defer func() { q.pending, r.cp, r.tokens, r.done = nil, nil, nil, true }()
	return q.rollback(r.cp)
}

// Close serializes with in-flight operations, aborts any pending round, and
// disposes both owners. A request cannot resume after stream delivery failure.
func (q *Qwen38DFlash2Request) Close() error {
	if q == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.closeLocked()
}

// Each obligation remains reachable after an interrupted callback. Attempted
// releases are never retried: a panic may have happened after the physical free.
// An unfenced target and fence tensor stay explicitly quarantined with their
// model hold. The caller must retain a failed request for backend recovery.
type dflash2CleanupObligation struct {
	action               func() error
	attempted, completed bool
	err                  error
}

func (o *dflash2CleanupObligation) run() {
	if o.attempted {
		return
	}
	o.attempted = true
	o.err = errors.New("dflash2: cleanup callback did not complete")
	err := o.action()
	o.err, o.completed = err, true
	if err == nil {
		o.action = nil
	}
}

func (q *Qwen38DFlash2Request) cleanupAction(action func() error) func() {
	o := &dflash2CleanupObligation{action: action}
	q.cleanup = append(q.cleanup, o)
	return o.run
}

func (q *Qwen38DFlash2Request) closeLocked() (err error) {
	if q.closed {
		return q.closeErr
	}
	q.closed, q.bootstrapped = true, false
	q.closeErr = errors.New("dflash2: request cleanup did not complete; retain owner for recovery")
	f, pending := q.ref, q.pending
	q.ref, q.pending, q.logits = nil, nil, nil
	if f != nil {
		q.quarantinedTarget, q.quarantinedReference = f.target, f
	}
	returned := false
	defer func() {
		if !returned {
			err = errors.Join(q.closeErr, err)
		}
		for _, obligation := range q.cleanup {
			err = errors.Join(err, obligation.err)
		}
		q.closeErr = err
	}()
	var actions []func()
	if pending != nil {
		cp := pending.cp
		pending.cp, pending.tokens, pending.done = nil, nil, true
		if cp != nil {
			if cp.tx != nil {
				actions = append(actions, q.cleanupAction(cp.tx.Abort))
			}
			if cp.snapshot != nil {
				p := cp.snapshot
				cp.snapshot = nil
				actions = append(actions, q.cleanupAction(func() error { p.Close(); return nil }))
			}
		}
	}
	if f != nil {
		actions = append(actions, q.cleanupAction(q.disposeTarget))
	}
	err = dflash2Cleanup(func() error { runBackendSiblingCleanup(actions...); return nil })
	returned = true
	return err
}

// disposeTarget is request-specific. Ordinary Session.Close is unchanged. Read
// is the backend's whole-queue fence; FlushBatch alone is insufficient. A failed
// fence leaves the target, reference, fence tensor, and remaining model pins
// in the failure ledger, including when the target already failed closed.
func (q *Qwen38DFlash2Request) disposeTarget() error {
	target := q.quarantinedTarget
	if target == nil {
		return nil
	}
	func() {
		scope := target.beginBackendExecution()
		complete := false
		defer scope.finishCompleted(&complete)
		be := target.Backend
		if be != nil {
			q.quarantinedFence = target.uploadHostF32([]int{1}, []float32{0}, compute.MemoryActivation, "dflash2-close-fence")
			if batch, ok := be.(batchBackend); ok {
				batch.FlushBatch()
			}
			row := be.Read(q.quarantinedFence)
			if len(row) != 1 || row[0] != 0 {
				panic("dflash2: target completion fence failed; owner quarantined")
			}
		}
		// Preserve the reference pin through the fence: selected target failure
		// may already have released its own pin. Retire both after owner unlock.
		var retire []func()
		if reference := q.quarantinedReference; reference != nil {
			retire = append(retire, q.cleanupAction(func() error {
				err := reference.Close()
				if err == nil {
					q.quarantinedReference = nil
				}
				return err
			}))
		}
		if target.modelWeightsHeld {
			target.modelWeightsHeld = false
			retire = append(retire, q.cleanupAction(func() error { target.M.releaseWeightSession(); return nil }))
		}
		// A single registered batch attempts every retirement even during Goexit.
		defer scope.afterRelease(func() { runBackendSiblingCleanup(retire...) })
		var actions []func()
		if recurrent := target.qwen35HAL; recurrent != nil {
			target.qwen35HAL = nil
			actions = append(actions, q.cleanupAction(func() error { recurrent.free(be); return nil }))
		}
		if logits := target.greedyLogits; logits.Buf() != nil {
			target.greedyLogits = compute.Tensor{}
			actions = append(actions, q.cleanupAction(func() error { be.Free(logits); return nil }))
		}
		for name, weight := range target.halW {
			if _, borrowed := target.borrowedHALW[name]; !borrowed {
				actions = append(actions, q.cleanupAction(func() error { be.Free(weight); return nil }))
			}
		}
		target.halW, target.borrowedHALW = nil, nil
		if kv := target.halKV; kv != nil {
			target.halKV = nil
			actions = append(actions, q.cleanupAction(func() error { kv.Free(); return nil }))
		}
		if fence := q.quarantinedFence; fence.Buf() != nil {
			actions = append(actions, q.cleanupAction(func() error {
				be.Free(fence)
				q.quarantinedFence = compute.Tensor{}
				return nil
			}))
		}
		actions = append(actions, q.cleanupAction(func() error { target.closeOwned(scope); return nil }))
		actions = append(actions, q.cleanupAction(func() error { target.releaseHostPathBypass(); return nil }))
		runBackendSiblingCleanup(actions...)
		complete = true
	}()
	// Include deferred release/retirement in the normal-completion boundary.
	q.quarantinedTarget = nil
	return nil
}
