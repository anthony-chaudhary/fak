package modelengine

import (
	"context"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

// promoteWaitingLocked moves waiting lanes into the running set in deadline order
// (FIFO without deadlines), up to
// maxRun and, when set, within maxTokensPerStep (vLLM v1 scheduler token_budget,
// vllm/v1/core/sched/scheduler.py:569@975dca5, Apache-2.0). An empty running set always
// admits its first live candidate so an oversized first chunk cannot stall the loop.
func (s *NativeScheduler) promoteWaitingLocked(maxRun int) (queueWaits []time.Duration) {
	for _, ln := range s.waiting {
		if ln.materializing {
			return
		}
	}
	deadlineOrdered := s.orderWaitingDeadlinesLocked()
	budget := 0
	if s.maxTokensPerStep > 0 {
		budget = s.maxTokensPerStep
		for _, ln := range s.lanes {
			budget -= laneStepTokens(ln)
		}
	}
	kept := s.waiting[:0]
	capacityBlocked := false
	for _, ln := range s.waiting {
		if ln.ctx.Err() != nil {
			ln.finish(nil, ln.ctx.Err())
			continue
		}
		if maxRun > 0 && s.preemptionEnabledLocked() {
			// EDF can put a deferred lane ahead of the waiting session whose
			// reservation blocks it. Let that owner pass through the same capacity
			// checks and drain; deferred waiters must not overtake the blocked one.
			if capacityBlocked && (!deadlineOrdered || ln.sess == nil) {
				kept = append(kept, ln)
				continue
			}
			switch s.canMaterializeLaneLocked(ln) {
			case nativeAllocationNever:
				ln.finish(nil, ErrNativePromptExceedsKVCapacity)
				continue
			case nativeAllocationLater:
				capacityBlocked = true
				kept = append(kept, ln)
				continue
			case nativeAllocationOK:
				// ready now: fall through to the remaining capacity checks below
			}
		}
		if maxRun > 0 && len(s.lanes) >= maxRun {
			kept = append(kept, ln)
			continue
		}
		cost := laneStepTokens(ln)
		pad := 0
		if s.specPadK > 0 && cost == 1 && s.specStepCarriesLocked() {
			// pad_spec_decode admission: reserve K placeholder draft slots so the
			// newcomer's first decode step rides the same fixed-shape panel as the
			// resident 1+K lanes instead of forcing the step off the captured bucket.
			pad = s.specPadK
			cost += pad
		}
		if s.maxTokensPerStep > 0 {
			if cost > budget && len(s.lanes) > 0 {
				// Skip this lane, not break the pass: a padded row that does not fit
				// stays in waiting while later lanes that do fit are still admitted.
				kept = append(kept, ln)
				continue
			}
			budget -= cost
		}
		s.lanes = append(s.lanes, ln)
		if pad > 0 {
			ln.specAdmissionPad = pad
			s.specAdmissionPads++
			s.specAdmissionPadSlots += uint64(pad)
		}
		// First scheduling only, following vLLM v1 metrics/stats.py at
		// 0c63ce9f0e0b5b2782f0f30cb491069dac158bb2 (Apache-2.0).
		// This is a Go concept adaptation, not copied upstream code. A later
		// materialization requeue or preemption does not restart the interval.
		if ln.queueWaitPending {
			ln.queueWaitPending = false
			if s.recorder != nil {
				d := s.now().Sub(ln.queueEnteredAt)
				// time.Sub saturates instead of wrapping; omit an unrepresentable
				// interval or a regressing injected clock rather than invent zero.
				if d >= 0 && d < time.Duration(1<<63-1) {
					queueWaits = append(queueWaits, d)
				}
			}
		}
	}
	clear(s.waiting[len(kept):])
	s.waiting = kept
	return queueWaits
}

// run is the single scheduler loop. Each iteration recomputes the running set: it
// compacts retired lanes out, then promotes waiting lanes into the freed slots (EDF,
// or FIFO without deadlines,
// up to maxRunning) — so the per-step batch geometry tracks admissions and completions
// between every decode step. All lane mutation (terminal/logits/gen) happens HERE, on
// this one goroutine, so those fields need no lock; only the shared waiting/lanes
// slices (appended by Admit) and the closed flag are mutex-guarded.
func (s *NativeScheduler) run() {
	defer close(s.stopped)
	for {
		s.mu.Lock()
		donorOwnsWork := s.drainDonation && s.blockedDrains > 0
		closed := s.closed
		idle := len(s.lanes) == 0 && len(s.waiting) == 0 && len(s.preempted) == 0
		s.mu.Unlock()
		if donorOwnsWork {
			if closed && idle {
				return
			}
			<-s.wake
			continue
		}

		s.executor.Lock()
		_, idle, closed = s.runIteration(false)
		s.executor.Unlock()
		if idle {
			if closed {
				return
			}
			<-s.wake
		}
	}
}

func (s *NativeScheduler) beginBlockedDrain() bool {
	s.mu.Lock()
	if !s.drainDonation || s.closed {
		s.mu.Unlock()
		return false
	}
	s.blockedDrains++
	s.mu.Unlock()
	s.signal()
	return true
}

func (s *NativeScheduler) endBlockedDrain() {
	s.mu.Lock()
	s.blockedDrains--
	s.mu.Unlock()
	s.signal()
}

func (s *NativeScheduler) drainWithDonation(tokens <-chan abi.EngineToken) {
	for {
		select {
		case _, ok := <-tokens:
			if !ok {
				return
			}
			continue
		default:
		}

		if s.executor.TryLock() {
			didWork, _, _ := s.runIteration(true)
			s.executor.Unlock()
			if didWork {
				continue
			}
		}

		s.mu.Lock()
		s.blockedDrains--
		s.mu.Unlock()
		s.signal()

		_, ok := <-tokens

		s.mu.Lock()
		s.blockedDrains++
		s.mu.Unlock()

		if !ok {
			return
		}
	}
}

// runIteration advances one native scheduler boundary. executor must be held.
// donated limits the solo fast path to one token so the donating caller can return
// to draining its bounded token channel before executing another iteration.
func (s *NativeScheduler) runIteration(donated bool) (didWork, idle, closed bool) {
	s.mu.Lock()
	// 1. Drop finished/cancelled lanes from the running set, freeing their slots.
	live := s.lanes[:0]
	for _, ln := range s.lanes {
		if !ln.terminal {
			live = append(live, ln)
		}
	}
	clear(s.lanes[len(live):])
	s.lanes = live
	maxRun := s.effectiveMaxRunningLocked()
	// 2. Retire cancelled preempted lanes, then readmit older preempted lanes before
	// promoting fresh waiting work so a preempted victim cannot starve behind arrivals.
	s.dropCanceledPreemptedLocked()
	// An Admit may be materializing a reserved waiting lane without the executor.
	// Do not restore a preempted lane until that reservation has either published
	// its session or disappeared; readmit's resident-block count cannot see it yet.
	waitingReservation := false
	if maxRun > 0 && s.preemptionEnabledLocked() && s.preemption.Mode == NativePreemptRecompute {
		for _, ln := range s.waiting {
			if ln.materializing || ln.sess != nil {
				waitingReservation = true
				break
			}
		}
	}
	if !waitingReservation {
		s.readmitPreemptedLocked()
	}
	// 3. Promote waiting lanes by deadline (FIFO without deadlines), up to maxRunning. A lane
	// cancelled while it was still waiting is retired here rather than promoted.
	waitingMaterialization := false
	for _, ln := range s.waiting {
		if ln.materializing {
			waitingMaterialization = true
			break
		}
	}
	if s.promotionPicker != nil && len(s.waiting) > 1 && !waitingMaterialization {
		slots := len(s.waiting)
		if maxRun > 0 {
			slots = maxRun - len(s.lanes)
		}
		c := make([]PromotionCandidate, len(s.waiting))
		for i, ln := range s.waiting {
			c[i] = PromotionCandidate{Index: i, Hint: ln.hint}
		}
		order := s.promotionPicker(c, slots, s.sessionAffinity)
		reordered := make([]*schedLane, 0, len(s.waiting))
		used := map[int]bool{}
		for _, i := range order {
			if i >= 0 && i < len(s.waiting) && !used[i] {
				reordered = append(reordered, s.waiting[i])
				used[i] = true
			}
		}
		for i, ln := range s.waiting {
			if !used[i] {
				reordered = append(reordered, ln)
			}
		}
		s.waiting = reordered
	}
	queueWaits := s.promoteWaitingLocked(maxRun)
	// Every return below releases mu. Record after this iteration so recorder
	// callbacks never join the scheduler's mutex ordering, including the early
	// materialization/error returns. Durations were fixed at promotion.
	defer func() {
		for _, d := range queueWaits {
			s.recorder.ObservePhase(enginestep.PhaseSchedulerWait, d)
		}
	}()
	var pendingMaterialization *schedLane
	for _, ln := range s.lanes {
		if !ln.terminal && ln.sess == nil {
			pendingMaterialization = ln
			break
		}
	}
	if pendingMaterialization != nil {
		if maxRun > 0 && s.preemptionEnabledLocked() {
			switch s.canMaterializeLaneLocked(pendingMaterialization) {
			case nativeAllocationNever:
				pendingMaterialization.finish(nil, ErrNativePromptExceedsKVCapacity)
				closed = s.closed
				s.mu.Unlock()
				return true, false, closed
			case nativeAllocationOK:
				// materializable now: proceed to lane initialization below
			case nativeAllocationLater:
				for i, ln := range s.lanes {
					if ln == pendingMaterialization {
						s.lanes = append(s.lanes[:i], s.lanes[i+1:]...)
						break
					}
				}
				s.waiting = append([]*schedLane{pendingMaterialization}, s.waiting...)
				closed = s.closed
				s.mu.Unlock()
				return true, false, closed
			}
		}
		closed = s.closed
		s.mu.Unlock()
		if err := pendingMaterialization.ctx.Err(); err != nil {
			s.mu.Lock()
			pendingMaterialization.finish(nil, err)
			closed = s.closed
			s.mu.Unlock()
			return true, false, closed
		}
		candidate := *pendingMaterialization
		initializationErr := s.initializeLaneExecution(&candidate)
		s.mu.Lock()
		switch {
		case initializationErr != nil:
			pendingMaterialization.finish(nil, initializationErr)
		case s.closed:
			s.closeLaneSession(candidate.sess)
			pendingMaterialization.finish(nil, errSchedClosed)
		case pendingMaterialization.ctx.Err() != nil:
			s.closeLaneSession(candidate.sess)
			pendingMaterialization.finish(nil, pendingMaterialization.ctx.Err())
		default:
			pendingMaterialization.sess = candidate.sess
			pendingMaterialization.logits = candidate.logits
			pendingMaterialization.state = candidate.state
			pendingMaterialization.promptCursor = candidate.promptCursor
			pendingMaterialization.promptLen = candidate.promptLen
			pendingMaterialization.prefillChunkTokens = candidate.prefillChunkTokens
			pendingMaterialization.packedPrefill = candidate.packedPrefill
		}
		closed = s.closed
		s.mu.Unlock()
		return true, false, closed
	}
	if used := s.usedKVBlocksLocked(); used > s.maxObservedKVBlocks {
		s.maxObservedKVBlocks = used
	}
	if promoted := len(s.lanes); promoted > s.maxObservedRunning {
		s.maxObservedRunning = promoted
	}
	// Keep enforcing the live KV-block budget while protecting partially-prefilled
	// lanes, whose prompt cursor and recurrent state cannot yet use decode's
	// swap/recompute restore contract.
	s.enforcePreemptionPreservingPrefillLocked()
	running := len(s.lanes)
	var solo *schedLane
	var prefill *schedLane
	var active []*schedLane
	for _, ln := range s.lanes {
		if ln.state == schedLanePrefilling {
			prefill = ln
			break
		}
	}
	if prefill == nil {
		if running == 1 && len(s.waiting) == 0 && len(s.preempted) == 0 {
			solo = s.lanes[0]
		} else if running > 0 {
			active = make([]*schedLane, running)
			copy(active, s.lanes)
		}
	} else {
		active = make([]*schedLane, 0, running-1)
		for _, ln := range s.lanes {
			if ln.state == schedLaneDecode {
				active = append(active, ln)
			}
		}
	}
	// Prefill cadence gate (vLLM local prefill interval, scheduler.py:748-769):
	// defer this iteration's prefill when fewer than prefillInterval executed
	// steps have passed since the last prefill-carrying step and at least one
	// running lane is not a prefill chunk. A capacity-bound engine stops
	// throttling so a backed-up waiting queue cannot grow without bound. The
	// captured cursor/state let the stamp below tell a step that carried
	// prefill from one that yielded it unchanged.
	var prefillStartCursor int
	var prefillStartState schedLaneState
	if prefill != nil {
		prefillStartCursor = prefill.promptCursor
		prefillStartState = prefill.state
	}
	deferPrefill := prefill != nil && s.prefillInterval > 1 && !s.prefillCapacityBound &&
		(s.executedSteps+1-s.lastPrefillStep) < s.prefillInterval && len(active) > 0

	// Skip-drafts verdict inputs (vLLM skip_drafts adaptation): the cursors
	// the prefilling lanes hold before this iteration's prefill execution, so
	// the post-execution verdict counts only lanes whose chunk advanced.
	prefillCursorBefore := -1
	var packedCandidates []*schedLane
	var packedCandidateCursors []int
	if prefill != nil {
		if prefill.packedPrefill {
			for _, ln := range s.lanes {
				if ln.packedPrefill && ln.state == schedLanePrefilling {
					packedCandidates = append(packedCandidates, ln)
					packedCandidateCursors = append(packedCandidateCursors, ln.promptCursor)
				}
			}
		} else {
			prefillCursorBefore = prefill.promptCursor
		}
	}
	closed = s.closed
	waitingDepth := len(s.waiting)
	idle = len(s.lanes) == 0 && len(s.waiting) == 0 && len(s.preempted) == 0
	// Cross-request fairness cap (vLLM long_prefill_token_threshold): while
	// other requests are queued, a lone long prefill advances at most the
	// armed threshold this step. Packed prefill keeps its own packing
	// economics and stays uncapped.
	longPrefillLimit := 0
	if prefill != nil && !prefill.packedPrefill {
		longPrefillLimit = s.longPrefillStepCapLocked()
	}
	s.mu.Unlock()

	s.recorder.SetQueueDepth(waitingDepth)

	if idle {
		return false, idle, closed
	}
	s.iteration++
	iteration := s.iteration
	mixedExecuted := false
	if prefill != nil && !deferPrefill {
		prefillStarted := s.now()
		var packedReady []*schedLane
		var packedGen ForwardGen
		if prefill.packedPrefill {
			var mixed []*schedLane
			packedGen = s.fwd.Load()
			packedReady, mixed = s.advanceMixedPackedPrefill(iteration, active, packedGen)
			mixedExecuted = len(mixed) > 0
			remaining := active[:0]
			for _, ln := range active {
				executed := false
				for _, done := range mixed {
					if ln == done {
						executed = true
						break
					}
				}
				if !executed {
					remaining = append(remaining, ln)
				}
			}
			active = remaining
			if len(mixed) == 0 {
				s.notePackedPrefillSkipDrafts(iteration, packedCandidates, packedCandidateCursors)
			}
		} else {
			origChunk, clampedChunk := s.clampLongPrefillStep(prefill, longPrefillLimit)
			s.advanceQwenPrefill(prefill, iteration)
			s.restoreLongPrefillStep(prefill, origChunk, clampedChunk)
			s.noteQwenPrefillSkipDrafts(iteration, prefill, prefillCursorBefore)
		}
		// PhasePrefill only: ObservePrefillChunk counts prompt tokens forwarded by
		// prefill, and advanceQwenPrefill may resolve a KV prefix hit that skips
		// tokens before the real chunk with no separate signal here. Reporting the
		// cursor delta would bill cache hits as prefill; the prefix-match family is
		// the honest owner of those tokens.
		s.recorder.ObservePhase(enginestep.PhasePrefill, s.now().Sub(prefillStarted))
		if prefill.packedPrefill {
			active = append(active, packedReady...)
		} else if !prefill.terminal && prefill.state == schedLaneDecode {
			active = append(active, prefill)
		}
		if len(active) > 0 {
			s.recorder.ObserveCohort(len(active))
			if prefill.packedPrefill {
				s.stepOnceGeneration(active, iteration, packedGen)
			} else {
				s.stepOnce(active, iteration)
			}
		}
	} else if solo != nil {
		s.recorder.ObserveCohort(1)
		if donated {
			s.stepOnce([]*schedLane{solo}, iteration)
		} else {
			s.stepSolo(solo, iteration)
		}
	} else {
		s.recorder.ObserveCohort(len(active))
		s.stepOnce(active, iteration)
	}
	// Cadence stamping (vLLM scheduler.py:1562-1577): count only steps that
	// actually scheduled tokens, restart the interval from any step that
	// carried prefill (new admission or running chunk), and record the capacity
	// escape on every release step so a backed-up engine stops throttling.
	s.mu.Lock()
	carried := prefill != nil && !deferPrefill &&
		(prefill.promptCursor != prefillStartCursor || prefill.state != prefillStartState)
	if solo != nil || len(active) > 0 || mixedExecuted || carried {
		s.executedSteps++
	}
	if carried {
		s.lastPrefillStep = s.executedSteps
	}
	if !deferPrefill {
		s.prefillCapacityBound = len(s.waiting) > prefillDrainPerInterval
	}
	s.mu.Unlock()
	if donated {
		s.mu.Lock()
		s.donatedIterations++
		s.mu.Unlock()
	}
	return true, false, closed
}

// nativeDeferredToken separates sampling from externally visible publication.
// A token sampled from logits produced by forward N is retained until the next
// scheduler boundary owns its forward, then retired exactly once.
type nativeDeferredToken struct {
	id        int
	done      bool
	accounted bool
}

func (ln *schedLane) stageDeferredToken() bool {
	s := ln.sched
	s.mu.Lock()
	defer s.mu.Unlock()
	if ln.terminal || ln.deferredToken != nil {
		return !ln.terminal
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		return false
	}
	if ln.state != schedLaneDecode || len(ln.logits) == 0 || ln.sess == nil || ln.sess == ln.statsSess {
		ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
		return false
	}
	// pad_spec_decode placeholder resolution: the lane's first decode forward
	// consumes the admission-reserved draft slots; rejected placeholders publish
	// nothing (upstream resolves them to -1 in update_draft_token_ids_in_output).
	if ln.specAdmissionPad > 0 {
		ln.specAdmissionPad = 0
	}
	next := argmax(ln.logits)
	budget := genTokens
	if ln.genBudget > 0 {
		budget = ln.genBudget
	}
	ln.deferredToken = &nativeDeferredToken{
		id:   next,
		done: ln.sess.M.Cfg.IsEOS(next) || ln.emitted+1 >= budget,
	}
	return true
}

func (ln *schedLane) deferredForForward() (int, bool) {
	s := ln.sched
	s.mu.Lock()
	defer s.mu.Unlock()
	if ln.terminal || ln.deferredToken == nil || ln.deferredToken.done {
		return 0, false
	}
	return ln.deferredToken.id, true
}

// retireDeferredToken is the synchronize/postprocess/release boundary. It is
// the only operation that publishes a staged token or advances generated state.
func (ln *schedLane) retireDeferredToken(iteration uint64) bool {
	s := ln.sched
	var pending nativeDeferredToken
	ready := func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		if ln.terminal || ln.deferredToken == nil {
			return false
		}
		if err := ln.ctx.Err(); err != nil {
			ln.finish(nil, err)
			return false
		}
		pending = *ln.deferredToken
		ln.deferredToken = nil
		return true
	}()
	if !ready {
		return false
	}

	var publishStarted time.Time
	if s.recorder != nil {
		publishStarted = s.now()
	}
	publishCanceled := false
	select {
	case ln.tokens <- abi.EngineToken{ID: pending.id}:
	case <-ln.ctx.Done():
		publishCanceled = true
	}
	// This is the completed select's wall time, including already-ready sends
	// and cancellation winners, not model compute or pure blocked time.
	if s.recorder != nil {
		elapsed := s.now().Sub(publishStarted)
		if elapsed >= 0 && elapsed < time.Duration(1<<63-1) {
			s.recorder.ObservePhase(enginestep.PhaseTokenPublish, elapsed)
		}
	}
	if publishCanceled {
		func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			ln.finish(nil, ln.ctx.Err())
		}()
		return false
	}

	var state schedLaneState
	var res *abi.Result
	func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		ln.gen = append(ln.gen, pending.id)
		if !pending.accounted {
			ln.emitted++
		}
		state = ln.state
		if pending.done {
			modelName := s.modelName
			if modelName == "" {
				modelName = Default.ModelName()
			}
			res = assembleSyscallResult(ln.putCtx, ln.tool, EngineID, modelName, ln.promptLen, append([]int(nil), ln.gen...), ln.tok)
		}
	}()

	s.observeEvent(nativeSchedulerEvent{Iteration: iteration, Kind: nativeSchedulerEventDecode, Lane: ln, State: state, Token: pending.id})
	if pending.done {
		func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			ln.finish(res, nil)
		}()
		return false
	}
	return true
}

// stepSolo advances one lane without rebuilding the scheduler batch between every token.
// It returns to run() whenever another Admit/Close signal arrives, preserving in-flight
// batch addition while keeping uncontended B=1 latency off the shared-batch bookkeeping path.
func (s *NativeScheduler) stepSolo(ln *schedLane, iteration uint64) {
	if ln.deferredToken == nil {
		if !ln.stageDeferredToken() {
			return
		}
	}
	for {
		next, forward := ln.deferredForForward()
		if !forward {
			ln.retireDeferredToken(iteration)
			return
		}
		sess := s.takeDecodeSession(ln)
		if sess == nil {
			return
		}
		s.beforeModelExecution(nativeSchedulerEventDecode, ln)
		decodeStarted := s.now()
		// Load the live forward generation once, BEFORE the forward call, so
		// this whole step runs on one generation (in-flight continuity).
		gen := s.fwd.Load()
		logits := RunWithOp(s.coupler, OpDecode, func() []float32 {
			return gen.Impl.Step(sess, next)
		})
		elapsed := s.now().Sub(decodeStarted)
		s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, elapsed)
		s.recorder.ObserveDecodeStep(enginestep.PathSerial, 1, elapsed)
		if !s.publishDecodeSession(ln, sess, logits) {
			return
		}
		if !ln.retireDeferredToken(iteration) || !ln.stageDeferredToken() {
			return
		}
		s.noteStepSkipDrafts(iteration, false, []*schedLane{ln})
		select {
		case <-s.wake:
			return
		default:
		}
	}
}

// stepOnce stages a token per active lane, advances it through the selected model
// route (bounded groups for eligible independent Qwen cohorts),
// then publishes that prior token and stages its successor. A terminal staged token
// is retired before the batch so it never enters StepBatch's id panel.
//
// Batch composition is ordered by the stable request UID (schedLane.seqNo), never by
// mutable scheduler slice position: promotion appends to the running set, preemption
// removes from the middle, and readmission appends an OLDER victim behind a lane
// admitted after it, so slice order is not a function of the UID. If two ranks of a
// tensor-parallel deployment observed different promotion or preemption history, they
// would assign different requests to the same batch slot, MoE expert and route
// selection would diverge across ranks, and a batched decode step would disagree with
// itself. Ordering by the immutable admission identity makes every rank derive the same
// slot assignment from the same running set, so churn cannot silently change it.
//
// Scope and honesty. seqNo is the scheduler-local monotonic admission counter, so this
// invariant is "equal admission order implies equal batch order" — it removes churn as a
// source of divergence and makes the assignment reproducible, but it does not by itself
// join ranks that admit in different orders (there is no distributed multi-rank
// admission plane in this scheduler today). Port of the stable-UID decode ordering
// convention from sgl-project/mini-sglang at pinned revision
// 9a91cfafe754aa85daee49998176275667eb58f2 (MIT), referenced at decode.py:32-35; the
// upstream file is not vendored. Slot position only, and a correctness fix rather than a
// speed one: the per-request token stream, KV accounting and the existing seqNo
// preemption ordering are untouched, no latency, throughput or token-rate gain is
// claimed, and StepBatch already guarantees per-user logits are bit-identical to a
// serial Step, so ordering cannot change any lane's output.
func (s *NativeScheduler) stepOnce(active []*schedLane, iteration uint64) {
	s.stepOnceGeneration(active, iteration, s.fwd.Load())
}

// stepOnceGeneration shares the iteration's forward identity with a preceding
// packed mixed panel, including no-effects declines and overflow decode rows.
func (s *NativeScheduler) stepOnceGeneration(active []*schedLane, iteration uint64, gen ForwardGen) {
	sort.SliceStable(active, func(i, j int) bool { return active[i].seqNo < active[j].seqNo })
	cont := make([]*schedLane, 0, len(active))
	ids := make([]int, 0, len(active))
	for _, ln := range active {
		if ln.deferredToken == nil {
			if !ln.stageDeferredToken() {
				continue
			}
		}
		next, forward := ln.deferredForForward()
		if !forward {
			ln.retireDeferredToken(iteration)
			continue
		}
		cont = append(cont, ln)
		ids = append(ids, next)
	}
	if len(cont) == 0 {
		return
	}
	limit := s.decodeCohortLimit(cont)
	for start := 0; start < len(cont); {
		width := len(cont) - start
		if limit > 0 && width > limit {
			width = limit
			if len(cont)-start-width == 1 && width > 2 {
				width-- // 9 -> 7+2; 17 -> 8+7+2, never an avoidable singleton.
			}
		}
		end := start + width
		s.stepDecodeCohort(cont[start:end], ids[start:end], iteration, gen)
		start = end
	}
	s.noteStepSkipDrafts(iteration, false, active)
}

// decodeCohortLimit observes the complete offered owner set while the serialized
// executor owns mutation. Do not borrow future groups just to inspect them.
func (s *NativeScheduler) decodeCohortLimit(lanes []*schedLane) int {
	if len(lanes) <= 8 {
		return 0 // The established call already fits; avoid preflight on this path.
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	seqs := make([]*model.Session, 0, len(lanes))
	outboxes := make(map[chan abi.EngineToken]bool, len(lanes))
	for _, ln := range lanes {
		if !s.runningDecodeLaneLocked(ln) || ln.sess == nil ||
			ln.sess == ln.statsSess || ln.inflightSess != nil {
			return 0
		}
		// Each subgroup publishes before the next forwards. The executor is
		// the sole sender, so a distinct buffered outbox with one free slot
		// reserves nonblocking publication. Consumers only free capacity.
		// Otherwise preserve legacy whole-cohort forwarding before blocking.
		if outboxes[ln.tokens] || len(ln.tokens) >= cap(ln.tokens) {
			return 0
		}
		outboxes[ln.tokens] = true
		seqs = append(seqs, ln.sess)
	}
	return (&model.BatchSession{M: s.m, Seqs: seqs}).Qwen35IndependentDecodeCohortLimit()
}

// retireDecodeBorrow is also a defer boundary: a model, observer or publication
// panic/Goexit must not leave a real session stranded behind a statistics shell.
// Group cleanup registers one defer per borrow so every remaining owner is
// visited even if another cleanup panics or Goexits. Published owners are a no-op.
func (s *NativeScheduler) retireDecodeBorrow(ln *schedLane, sess *model.Session) {
	// Cleanup must not replace the original forward/observer panic being
	// unwound. finish publishes a failed terminal result if cleanup aborts.
	defer func() { _ = recover() }()
	s.mu.Lock()
	defer s.mu.Unlock()
	if ln == nil || ln.inflightSess != sess {
		return
	}
	if ln.restoreSessionFromModelLocked(sess) {
		err := ln.ctx.Err()
		if err == nil {
			err = errNativeSchedulerLaneNotDecodeReady
		}
		ln.finish(nil, err)
	} else {
		ln.inflightSess = nil
		s.closeLaneSession(sess)
	}
}

func (s *NativeScheduler) stepDecodeCohort(lanes []*schedLane, ids []int, iteration uint64, gen ForwardGen) {
	execLanes := make([]*schedLane, 0, len(lanes))
	execIDs := make([]int, 0, len(lanes))
	seqs := make([]*model.Session, 0, len(lanes))
	defer func() {
		s.mu.Lock()
		pending := false
		for i, ln := range execLanes {
			pending = pending || ln.inflightSess == seqs[i]
		}
		s.mu.Unlock()
		if !pending {
			return
		}
		// Cleanup Goexit on the initiating goroutine would consume its model
		// panic. Contain only abnormal outstanding borrows in a joined child,
		// registering every retirement before any callback can exit it.
		done := make(chan struct{})
		go func() {
			defer close(done)
			for i, ln := range execLanes {
				defer s.retireDecodeBorrow(ln, seqs[i])
			}
		}()
		<-done
	}()
	for i, ln := range lanes {
		if sess := s.takeDecodeSession(ln); sess != nil {
			execLanes = append(execLanes, ln)
			execIDs = append(execIDs, ids[i])
			seqs = append(seqs, sess)
		}
	}
	for _, ln := range execLanes {
		s.beforeModelExecution(nativeSchedulerEventDecode, ln)
	}
	// Cancellation may occur in a preceding group's execution or even in the
	// callbacks above. Compact only this unsubmitted group; never replay an
	// earlier group or pull an already advanced lane back into the panel.
	live := 0
	for i, ln := range execLanes {
		s.mu.Lock()
		ready := s.runningDecodeLaneLocked(ln) && ln.inflightSess == seqs[i] && ln.ctx.Err() == nil
		s.mu.Unlock()
		if !ready {
			s.retireDecodeBorrow(ln, seqs[i])
			continue
		}
		execLanes[live], execIDs[live], seqs[live] = ln, execIDs[i], seqs[i]
		live++
	}
	execLanes, execIDs, seqs = execLanes[:live], execIDs[:live], seqs[:live]
	if live == 0 {
		return
	}
	if live == 1 {
		decodeStarted := s.now()
		logits := RunWithOp(s.coupler, OpDecode, func() []float32 {
			return gen.Impl.Step(seqs[0], execIDs[0])
		})
		elapsed := s.now().Sub(decodeStarted)
		s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, elapsed)
		s.recorder.ObserveDecodeStep(enginestep.PathSerial, 1, elapsed)
		if s.publishDecodeSession(execLanes[0], seqs[0], logits) {
			execLanes[0].retireDeferredToken(iteration)
			execLanes[0].stageDeferredToken()
		}
		return
	}
	bs := &model.BatchSession{M: s.m, Seqs: seqs}
	if s.recorder != nil {
		bs.ObserveSerialStep = s.recorder.ObserveSerialFallback
	}
	decodeStarted := s.now()
	out := RunWithOp(s.coupler, OpBatch, func() [][]float32 {
		return gen.Impl.StepBatch(bs, execIDs)
	})
	elapsed := s.now().Sub(decodeStarted)
	s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, elapsed)
	if panels := bs.LastStepSharedPanels(); panels > 0 {
		// Offered cohort size is observed by the outer iteration. These receipts
		// describe only the actual shared call, including cancellation shrinkage.
		s.recorder.ObserveDecodeStep(enginestep.PathBatched, len(execLanes), elapsed)
		s.mu.Lock()
		s.sharedBatchSteps++
		s.sharedPanels += uint64(panels)
		s.sharedMACs += uint64(bs.LastStepMACs())
		s.mu.Unlock()
	}
	if len(out) != len(execLanes) {
		panic(errNativeSchedulerLaneNotDecodeReady)
	}
	for i, ln := range execLanes {
		if s.publishDecodeSession(ln, seqs[i], copyF32(out[i])) {
			ln.retireDeferredToken(iteration)
			ln.stageDeferredToken()
		}
	}
}

// decodeBatchBucket selects the fixed-shape decode graph bucket that a live
// batch of n rows replays through: the smallest captured bucket >= n, the
// pad-to-captured-size rule from sgl-project/mini-sglang at pinned revision
// 9a91cfafe754aa85daee49998176275667eb58f2 (MIT), referenced at
// python/minisgl/engine/graph.py:16-156; the upstream file is not vendored.
// Buckets must be sorted unique positive sizes (strictly increasing); any
// other set, n < 1, or n above the largest captured size refuses with ok=false
// rather than mis-selecting a shape.
func decodeBatchBucket(n int, buckets []int) (int, bool) {
	if n < 1 || len(buckets) == 0 {
		return 0, false
	}
	for i, b := range buckets {
		if b < 1 || (i > 0 && b <= buckets[i-1]) {
			return 0, false
		}
	}
	for _, b := range buckets {
		if b >= n {
			return b, true
		}
	}
	return 0, false
}

// padDecodeRows extends a live decode batch row list to a fixed captured bucket
// shape by appending pad rows AFTER every live row, mirroring mini-sglang's
// dummy-request padding (same pin as decodeBatchBucket). Live rows keep their
// original order — the stable-UID order stepOnce establishes — and are never
// truncated or reordered; paddedSize below the live count refuses instead of
// shrinking a fixed-shape replay.
func padDecodeRows[T any](live []T, paddedSize int, pad T) ([]T, bool) {
	if paddedSize < len(live) {
		return nil, false
	}
	out := make([]T, paddedSize)
	copy(out, live)
	for i := len(live); i < paddedSize; i++ {
		out[i] = pad
	}
	return out, true
}

// takeDecodeSession replaces the live model session with a non-nil immutable
// stats shell while Step/StepBatch mutates Cache outside s.mu. Preemption runs
// only on executor boundaries, so it can never select the shell as a victim.
func (s *NativeScheduler) takeDecodeSession(ln *schedLane) *model.Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.runningDecodeLaneLocked(ln) {
		return nil
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		return nil
	}
	if ln.deferredToken != nil && !ln.deferredToken.accounted {
		ln.deferredToken.accounted = true
		ln.emitted++
	}
	return ln.takeSessionForModelLocked()
}

// publishDecodeSession restores the real session and publishes the logits only
// after the cache mutation has completed. Cancellation restores ownership first,
// then finish closes the real session exactly once under s.mu.
func (s *NativeScheduler) publishDecodeSession(ln *schedLane, sess *model.Session, logits []float32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.runningDecodeLaneLocked(ln) || !ln.restoreSessionFromModelLocked(sess) {
		if ln != nil && ln.inflightSess == sess {
			ln.inflightSess = nil
		}
		s.closeLaneSession(sess)
		return false
	}
	if err := ln.ctx.Err(); err != nil {
		ln.finish(nil, err)
		return false
	}
	if len(logits) == 0 {
		ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
		return false
	}
	ln.logits = logits
	return true
}

func (s *NativeScheduler) runningDecodeLaneLocked(ln *schedLane) bool {
	if ln == nil || ln.terminal || ln.state != schedLaneDecode {
		return false
	}
	for _, running := range s.lanes {
		if running == ln {
			return true
		}
	}
	return false
}

func (s *NativeScheduler) beforeModelExecution(kind nativeSchedulerEventKind, ln *schedLane) {
	if s != nil && s.beforeModelExecute != nil {
		s.beforeModelExecute(kind, ln)
	}
}

// schedLane is one admitted request's state + its EngineRequest handle.
// PromotionPicker is an opt-in WAITING-queue order when no live deadline exists.
// Nil preserves FIFO; deadline-bearing work uses EDF before promotion.
type PromotionPicker func([]PromotionCandidate, int, map[string]int) []int

type PromotionCandidate struct {
	Index int
	Hint  dispatchtick.WaveHint
}

// SetSessionAffinity records the worker holding an agent's warm state. The wave
// picker may prefer it; capacity is still filled by work stealing.
func (s *NativeScheduler) SetSessionAffinity(agent string, worker int) {
	s.mu.Lock()
	if s.sessionAffinity == nil {
		s.sessionAffinity = make(map[string]int)
	}
	s.sessionAffinity[agent] = worker
	s.mu.Unlock()
	s.signal()
}

func (s *NativeScheduler) SetPromotionPicker(p PromotionPicker) {
	s.mu.Lock()
	s.promotionPicker = p
	s.mu.Unlock()
	s.signal()
}

// WavePromotionPicker co-promotes a ready-set, prefers session affinity, and fills
// spare slots by work stealing. The input hints were authored by dispatchtick.
func WavePromotionPicker(c []PromotionCandidate, slots int, affinity map[string]int) []int {
	if slots <= 0 || len(c) == 0 {
		return nil
	}
	type score struct {
		wave                          string
		count, first, steps, affinity int
	}
	groups := map[string]*score{}
	for _, x := range c {
		if x.Hint.Wave == "" {
			continue
		}
		g := groups[x.Hint.Wave]
		if g == nil {
			g = &score{wave: x.Hint.Wave, first: x.Index, steps: x.Hint.StepsToExecution}
			groups[x.Hint.Wave] = g
		}
		g.count++
		if x.Hint.StepsToExecution < g.steps {
			g.steps = x.Hint.StepsToExecution
		}
		if w, ok := affinity[x.Hint.Agent]; ok && w == x.Hint.Worker {
			g.affinity++
		}
	}
	var best *score
	for _, g := range groups {
		if best == nil || (g.count <= slots && best.count > slots) || ((g.count <= slots) == (best.count <= slots) && (g.affinity > best.affinity || (g.affinity == best.affinity && (g.count > best.count || (g.count == best.count && (g.steps < best.steps || (g.steps == best.steps && g.first < best.first))))))) {
			best = g
		}
	}
	chosen := make([]int, 0, slots)
	seen := map[int]bool{}
	if best != nil {
		for _, x := range c {
			if x.Hint.Wave == best.wave && len(chosen) < slots {
				chosen = append(chosen, x.Index)
				seen[x.Index] = true
			}
		}
	}
	for _, x := range c {
		if len(chosen) >= slots {
			break
		}
		if !seen[x.Index] {
			chosen = append(chosen, x.Index)
			seen[x.Index] = true
		}
	}
	return chosen
}

type schedLane struct {
	sched  *NativeScheduler // back-pointer so Cancel can wake the run loop
	ctx    context.Context
	cancel context.CancelFunc

	// Model execution is serialized by executor. Fields consumed by live stats
	// (sess, promptLen, emitted, terminal) and completed prefill/decode publication
	// are additionally synchronized by sched.mu.
	sess         *model.Session
	statsSess    *model.Session // immutable non-nil shell exposed while inflightSess mutates
	inflightSess *model.Session // real session owned by the executor during model execution
	logits       []float32
	gen          []int
	emitted      int
	// genBudget is this lane's generated-token retirement budget; zero keeps
	// the engine-wide genTokens default. Set at admission by imported lanes
	// that carry a request-level max_tokens (P/D front door); guarded by mu.
	genBudget          int
	tool               string
	prompt             []int
	promptLen          int
	putCtx             context.Context
	tok                NLTokenizer
	q4k                bool
	prefillQ4K         bool
	state              schedLaneState
	promptCursor       int
	prefillChunkTokens int
	packedPrefill      bool
	kvReuseHits        int
	kvPinned           bool
	kvPinUntil         time.Time
	terminal           bool
	hint               dispatchtick.WaveHint
	seqNo              int64
	// First WAITING residence only; both fields are guarded by sched.mu.
	queueEnteredAt   time.Time
	queueWaitPending bool
	// Immutable submit metadata, also visible through nativeSchedulerEvent.Lane.
	// Lower values are more important, following vLLM's request-priority convention
	// (vllm/v1/core/sched/scheduler.py:735@975dca5, Apache-2.0).
	priority int

	// specAdmissionPad is the number of placeholder draft slots reserved for this
	// lane's first decode step when it was promoted into a spec-carrying decode
	// step (pad_spec_decode admission). Consumed by the first decode staging; the
	// placeholders publish no tokens. Guarded by sched.mu.
	specAdmissionPad int

	// Preemption state. A preempted lane is removed from the running set without closing
	// its token stream; readmit restores sess/logits and the stream resumes.
	preemptMode   NativePreemptionMode
	preemptRound  int64
	hostKV        []byte
	gpuDirectKV   *model.Qwen38GPUDirectDescriptor
	savedLogits   []float32
	materializing bool
	deferredToken *nativeDeferredToken

	tokens chan abi.EngineToken
	done   chan struct{}

	// terminal outputs: written once by finish() before close(done); read only
	// after <-done (close is the happens-before edge).
	res       *abi.Result
	err       error
	reclaimed bool
}

func (ln *schedLane) Tokens() <-chan abi.EngineToken { return ln.tokens }

func (ln *schedLane) Result() (*abi.Result, error) {
	<-ln.done
	return ln.res, ln.err
}

func (ln *schedLane) Cancel() {
	ln.cancel()
	if ln.sched != nil {
		ln.sched.signal() // wake the loop so the cancel is observed promptly, not only on its next step
	}
}

// Reclaimed reports whether terminal cleanup completed, including cancellation.
// It blocks until terminal; an aborted cleanup closes done but reports false.
func (ln *schedLane) Reclaimed() bool {
	<-ln.done
	return ln.reclaimed
}

// finish records the terminal state once, closes and drops the session (KV reclaim),
// and closes the stream + done edges. Called only by the serialized scheduler executor.
func (ln *schedLane) finish(res *abi.Result, err error) {
	if ln.terminal {
		return
	}
	ln.res, ln.err, ln.reclaimed = res, err, false
	retired := false
	defer func() {
		ln.reclaimed = retired
		if !retired {
			ln.res = nil
			if ln.err == nil {
				ln.err = errNativeSchedulerLaneNotDecodeReady
			}
		}
		ln.cancel()
		close(ln.tokens)
		close(ln.done)
	}()
	ln.hostKV = nil
	ln.savedLogits = nil
	ln.terminal = true
	published, inflight, shell := ln.sess, ln.inflightSess, ln.statsSess
	ln.sess, ln.inflightSess, ln.statsSess = nil, nil, nil
	descriptor := ln.gpuDirectKV
	ln.gpuDirectKV = nil
	func() {
		// Every retirement phase runs even when an earlier backend callback
		// panics or Goexits. Detach handles before handing them to callbacks.
		if ln.sched != nil {
			defer ln.sched.elastic.laneFinished(ln)
			if descriptor != nil && ln.sched.preemption.GPUDirectSwapper != nil {
				defer ln.sched.preemption.GPUDirectSwapper.FreeDescriptor(descriptor)
			}
		}
		if inflight != published {
			defer ln.closeRealSession(inflight, shell)
		}
		ln.closeRealSession(published, shell)
	}()
	retired = true
}

func (ln *schedLane) closeRealSession(sess, shell *model.Session) {
	if sess == nil || sess == shell {
		return
	}
	if ln.sched != nil {
		ln.sched.closeLaneSession(sess)
	} else {
		sess.Close()
	}
}

// takeSessionForModelLocked transfers the real session to executor-local
// ownership and leaves laneKVBlocksLocked an immutable, Cache-free view. The
// shell keeps the lane present in Running while promptLen+emitted supplies the
// synchronized live-token count.
func (ln *schedLane) takeSessionForModelLocked() *model.Session {
	if ln == nil || ln.sess == nil || ln.sess == ln.statsSess || ln.inflightSess != nil {
		return nil
	}
	real := ln.sess
	if ln.statsSess == nil {
		ln.statsSess = &model.Session{M: real.M, Quant: real.Quant, Q4K: real.Q4K}
	}
	ln.inflightSess = real
	ln.sess = ln.statsSess
	return real
}

func (ln *schedLane) restoreSessionFromModelLocked(real *model.Session) bool {
	if ln == nil || real == nil || ln.inflightSess != real || ln.statsSess == nil || ln.sess != ln.statsSess {
		return false
	}
	ln.inflightSess = nil
	ln.sess = real
	return true
}

func copyF32(v []float32) []float32 { return append([]float32(nil), v...) }

// NativeScheduler is a LifecycleEngine and each lane satisfies EngineRequest —
// the same interface the in-kernel per-request engine and the external adapter
// implement. That this compiles is the cross-shape contract the issue requires.
var (
	_ abi.LifecycleEngine = (*NativeScheduler)(nil)
	_ abi.EngineRequest   = (*schedLane)(nil)
)
