package modelengine

// Adapted from vllm-project/vllm (Apache-2.0; see THIRD_PARTY_NOTICES.md):
//   - Deferred KV block frees fence: vllm/v1/core/sched/scheduler.py:455-462,
//     1690-1694, 2974-2990 at revision ae065918640a198a56f6f17eb45e251c3229fe41
//     (FIFO of (fence_seq, blocks) gated on processed_step_seq >= fence_seq,
//     reverse drain within each entry so tail blocks evict first).
//   - AsyncScheduler output-placeholder protocol:
//     vllm/v1/core/sched/async_scheduler.py:14-78 and
//     vllm/config/scheduler.py:210-217 at the same revision (-1 placeholders,
//     num_output_placeholders tracking, stop-truncated tail resolution and
//     fenced block release).
//
// Clean-room Go adaptation for NativeScheduler; synchronous default behavior,
// seqNo stable-UID batch ordering, executor lock boundaries, and preemption
// timing are preserved. No hardware or throughput claim.

import (
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

// asyncOutputPlaceholderToken is the -1 placeholder id placed into staged
// outputs when step N+1 prepares before step N's sampled token is published.
const asyncOutputPlaceholderToken = -1

type deferredFreeEntry struct {
	fenceSeq uint64
	blocks   []int
	release  func(blkID int)
	onDone   func()
}

// DeferredFreeStats reports cumulative deferred-block-free fence activity.
type DeferredFreeStats struct {
	ProcessedStepSeq uint64
	LaunchedStepSeq  uint64
	PendingEntries   int
	PendingBlocks    int
	EnqueuedEntries  uint64
	EnqueuedBlocks   uint64
	DrainedEntries   uint64
	DrainedBlocks    uint64
	FlushedOnClose   uint64
}

// AsyncSchedStats reports cumulative async-scheduling overlap and placeholder
// lifecycle counts.
type AsyncSchedStats struct {
	StepsOverlapped      uint64
	PlaceholdersStaged   uint64
	PlaceholdersResolved uint64
	StopTruncations      uint64
}

type asyncLaneStage struct {
	lane          *schedLane
	stepSeq       uint64
	inputID       int
	toPublish     int
	isPlaceholder bool
	pubDone       bool
	countEmitted  bool
	skipForward   bool
	resolved      bool
	logits        []float32
	specAdded     int
	truncated     bool
}

type asyncStepStage struct {
	stepSeq   uint64
	iteration uint64
	entries   []*asyncLaneStage
}

// SetDeferredBlockFrees enables or disables the processedStepSeq deferred KV
// block-free fence. Disabled by default so historical synchronous tests
// preserve immediate release timing.
func (s *NativeScheduler) SetDeferredBlockFrees(enabled bool) {
	s.mu.Lock()
	s.deferredFreesEnabled = enabled
	s.mu.Unlock()
}

// SetAsyncScheduling enables or disables opt-in double-buffered async overlap
// and -1 output-placeholder staging. Enabling async scheduling also arms the
// deferred-block-frees fence required to protect blocks touched by in-flight
// steps. Disabled by default.
func (s *NativeScheduler) SetAsyncScheduling(enabled bool) {
	s.mu.Lock()
	s.asyncScheduling = enabled
	if enabled {
		s.deferredFreesEnabled = true
	}
	s.mu.Unlock()
}

// ProcessedStepSeq returns the highest step sequence whose outputs have
// completed publication.
func (s *NativeScheduler) ProcessedStepSeq() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.processedStepSeq
}

// DeferredFreeStats returns a snapshot of the deferred block-free fence state.
func (s *NativeScheduler) DeferredFreeStats() DeferredFreeStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.deferredFreeStats
	st.ProcessedStepSeq = s.processedStepSeq
	st.LaunchedStepSeq = s.launchedStepSeq
	st.PendingEntries = len(s.deferredFreeFIFO)
	for _, e := range s.deferredFreeFIFO {
		st.PendingBlocks += len(e.blocks)
	}
	return st
}

// AsyncSchedStats returns a snapshot of async overlap and placeholder stats.
func (s *NativeScheduler) AsyncSchedStats() AsyncSchedStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return AsyncSchedStats{
		StepsOverlapped:      s.asyncStepsOverlapped,
		PlaceholdersStaged:   s.asyncPlaceholdersStaged,
		PlaceholdersResolved: s.asyncPlaceholdersResolved,
		StopTruncations:      s.asyncStopTruncations,
	}
}

func (s *NativeScheduler) beginStepSeqLocked() uint64 {
	s.launchedStepSeq++
	return s.launchedStepSeq
}

func (s *NativeScheduler) beginStepSeq() uint64 {
	s.mu.Lock()
	seq := s.beginStepSeqLocked()
	s.mu.Unlock()
	return seq
}

func (s *NativeScheduler) currentFenceSeqLocked() uint64 {
	if s.launchedStepSeq > s.processedStepSeq {
		return s.launchedStepSeq
	}
	return s.processedStepSeq
}

func (s *NativeScheduler) enqueueDeferredFreeLocked(fenceSeq uint64, blocks []int, release func(int), onDone func()) {
	copied := append([]int(nil), blocks...)
	s.deferredFreeFIFO = append(s.deferredFreeFIFO, deferredFreeEntry{
		fenceSeq: fenceSeq,
		blocks:   copied,
		release:  release,
		onDone:   onDone,
	})
	s.deferredFreeStats.EnqueuedEntries++
	s.deferredFreeStats.EnqueuedBlocks += uint64(len(copied))
}

// drainDeferredFreesLocked pops FIFO entries whose fenceSeq <= processedStepSeq
// and releases each entry's blocks in reverse order so tail blocks evict first.
// Caller holds s.mu.
func (s *NativeScheduler) drainDeferredFreesLocked() {
	for len(s.deferredFreeFIFO) > 0 && s.deferredFreeFIFO[0].fenceSeq <= s.processedStepSeq {
		entry := s.deferredFreeFIFO[0]
		s.deferredFreeFIFO[0] = deferredFreeEntry{}
		s.deferredFreeFIFO = s.deferredFreeFIFO[1:]
		for i := len(entry.blocks) - 1; i >= 0; i-- {
			if entry.release != nil {
				entry.release(entry.blocks[i])
			}
		}
		if entry.onDone != nil {
			entry.onDone()
		}
		s.deferredFreeStats.DrainedEntries++
		s.deferredFreeStats.DrainedBlocks += uint64(len(entry.blocks))
	}
	if len(s.deferredFreeFIFO) == 0 {
		s.deferredFreeFIFO = nil
	}
}

// flushDeferredFreesLocked forces all pending fences to drain on Close() without
// leaking blocks. Caller holds s.mu.
func (s *NativeScheduler) flushDeferredFreesLocked() {
	if len(s.deferredFreeFIFO) == 0 {
		if s.launchedStepSeq > s.processedStepSeq {
			s.processedStepSeq = s.launchedStepSeq
		}
		return
	}
	maxFence := s.launchedStepSeq
	for _, entry := range s.deferredFreeFIFO {
		if entry.fenceSeq > maxFence {
			maxFence = entry.fenceSeq
		}
	}
	if maxFence > s.processedStepSeq {
		s.processedStepSeq = maxFence
	}
	pending := uint64(len(s.deferredFreeFIFO))
	s.drainDeferredFreesLocked()
	s.deferredFreeStats.FlushedOnClose += pending
}

func (s *NativeScheduler) completeStepPublicationLocked(stepSeq uint64) {
	if stepSeq > s.processedStepSeq {
		s.processedStepSeq = stepSeq
	}
	s.drainDeferredFreesLocked()
}

func (s *NativeScheduler) completeStepPublication(stepSeq uint64) {
	s.mu.Lock()
	s.completeStepPublicationLocked(stepSeq)
	s.mu.Unlock()
}

// finishLaneElasticBlocksLocked retires a finished lane's elastic KV blocks.
// When deferred frees are disabled (default), it preserves synchronous
// immediate release via s.elastic.laneFinished(ln). When armed, blocks enter
// deferredFreeFIFO keyed by the active fenceSeq unless the scheduler is already
// closing with no active async stage. Caller holds s.mu.
func (s *NativeScheduler) finishLaneElasticBlocksLocked(ln *schedLane) {
	if s == nil || s.elastic == nil || ln == nil {
		return
	}
	if !s.deferredFreesEnabled && !s.asyncScheduling {
		s.elastic.laneFinished(ln)
		return
	}
	s.elastic.mu.Lock()
	lane, ok := s.elastic.lanes[ln]
	if ok {
		delete(s.elastic.lanes, ln)
	}
	s.elastic.mu.Unlock()
	if !ok {
		return
	}
	releaseFn := func(blkID int) {
		if s.elastic.alloc.ReleaseCachedBlock(lane.id, blkID) == nil {
			s.elastic.mu.Lock()
			s.elastic.stats.CachedReleases++
			s.elastic.mu.Unlock()
		}
	}
	onDone := func() {
		_, _ = s.elastic.alloc.UnregisterSequence(lane.id)
	}
	if s.closed && ln.inflightSess == nil {
		for i := len(lane.blocks) - 1; i >= 0; i-- {
			releaseFn(lane.blocks[i])
		}
		onDone()
		s.deferredFreeStats.DrainedEntries++
		s.deferredFreeStats.DrainedBlocks += uint64(len(lane.blocks))
		s.deferredFreeStats.FlushedOnClose++
		return
	}
	fenceSeq := s.currentFenceSeqLocked()
	s.enqueueDeferredFreeLocked(fenceSeq, lane.blocks, releaseFn, onDone)
}

func (s *NativeScheduler) allocateLaneElasticGrowBlocksLocked(ln *schedLane, totalTokens int) {
	if s == nil || s.elastic == nil || ln == nil || totalTokens <= 0 {
		return
	}
	need := nativePromptBlocks(totalTokens, s.elastic.blockTokens)
	s.elastic.mu.Lock()
	defer s.elastic.mu.Unlock()
	lane, ok := s.elastic.lanes[ln]
	if !ok {
		return
	}
	for len(lane.blocks) < need {
		blk, err := s.elastic.alloc.AllocBlock(lane.id)
		if err != nil {
			s.elastic.stats.Shortfalls++
			break
		}
		lane.blocks = append(lane.blocks, blk)
		s.elastic.stats.AllocatedBlocks++
	}
	s.elastic.lanes[ln] = lane
}

func (s *NativeScheduler) trimLaneElasticTailBlocksLocked(ln *schedLane, keepTokens int, fenceSeq uint64) {
	if s == nil || s.elastic == nil || ln == nil {
		return
	}
	keepBlocks := 0
	if keepTokens > 0 {
		keepBlocks = nativePromptBlocks(keepTokens, s.elastic.blockTokens)
	}
	s.elastic.mu.Lock()
	lane, ok := s.elastic.lanes[ln]
	if !ok || len(lane.blocks) <= keepBlocks {
		s.elastic.mu.Unlock()
		return
	}
	trimmed := append([]int(nil), lane.blocks[keepBlocks:]...)
	lane.blocks = lane.blocks[:keepBlocks]
	s.elastic.lanes[ln] = lane
	s.elastic.mu.Unlock()

	releaseFn := func(blkID int) {
		if s.elastic.alloc.ReleaseCachedBlock(lane.id, blkID) == nil {
			s.elastic.mu.Lock()
			s.elastic.stats.CachedReleases++
			s.elastic.mu.Unlock()
		}
	}
	s.enqueueDeferredFreeLocked(fenceSeq, trimmed, releaseFn, nil)
}

func (s *NativeScheduler) laneKVBlocksWithPlaceholdersLocked(ln *schedLane) int {
	if ln == nil {
		return 0
	}
	tokens := ln.promptLen + ln.emitted + ln.numOutputPlaceholders
	if ln.sess != nil && ln.sess.Cache != nil && ln.sess.Cache.Len() > tokens {
		tokens = ln.sess.Cache.Len()
	}
	if tokens <= 0 {
		return 0
	}
	bt := s.blockTokensLocked()
	return (tokens + bt - 1) / bt
}

func (s *NativeScheduler) ensureAsyncSlotChLocked() chan struct{} {
	if s.asyncSlotCh == nil {
		s.asyncSlotCh = make(chan struct{}, 1)
	}
	return s.asyncSlotCh
}

func (s *NativeScheduler) notifyAsyncSlotLocked() {
	if s.asyncSlotCh != nil {
		select {
		case s.asyncSlotCh <- struct{}{}:
		default:
		}
	}
}

func (s *NativeScheduler) findUnresolvedAsyncEntryLocked(ln *schedLane) *asyncLaneStage {
	stages := []*asyncStepStage{s.asyncPubPending, s.asyncPubCurrent}
	for _, st := range stages {
		if st == nil {
			continue
		}
		for _, e := range st.entries {
			if e != nil && e.lane == ln && !e.truncated && !e.resolved {
				return e
			}
		}
	}
	return nil
}

func (s *NativeScheduler) findPendingEntryForLaneLocked(ln *schedLane) *asyncLaneStage {
	if s.asyncPubPending != nil {
		for _, e := range s.asyncPubPending.entries {
			if e != nil && e.lane == ln && !e.truncated {
				return e
			}
		}
	}
	return nil
}

func (ln *schedLane) resolvePlaceholderTokenLocked(realID int) bool {
	if ln.numOutputPlaceholders <= 0 {
		return false
	}
	ln.numOutputPlaceholders--
	for i, id := range ln.asyncStageTokens {
		if id == asyncOutputPlaceholderToken {
			ln.asyncStageTokens[i] = realID
			break
		}
	}
	return true
}

func (ln *schedLane) trimPlaceholderTailLocked() int {
	trimmed := 0
	for len(ln.asyncStageTokens) > 0 && ln.asyncStageTokens[len(ln.asyncStageTokens)-1] == asyncOutputPlaceholderToken {
		ln.asyncStageTokens = ln.asyncStageTokens[:len(ln.asyncStageTokens)-1]
		if ln.numOutputPlaceholders > 0 {
			ln.numOutputPlaceholders--
		}
		trimmed++
	}
	return trimmed
}

func (s *NativeScheduler) retireAsyncOnRunExit() {
	s.mu.Lock()
	pubActive := s.asyncPubActive
	pubDone := s.asyncPubDone
	s.mu.Unlock()
	if pubActive && pubDone != nil {
		<-pubDone
	}
	s.mu.Lock()
	for _, ln := range s.lanes {
		if ln != nil && !ln.terminal {
			ln.numOutputPlaceholders = 0
			ln.finish(nil, errSchedClosed)
		}
	}
	for _, ln := range s.waiting {
		if ln != nil && !ln.terminal {
			ln.numOutputPlaceholders = 0
			ln.finish(nil, errSchedClosed)
		}
	}
	for _, ln := range s.preempted {
		if ln != nil && !ln.terminal {
			ln.numOutputPlaceholders = 0
			ln.finish(nil, errSchedClosed)
		}
	}
	s.flushDeferredFreesLocked()
	s.mu.Unlock()
}

// stepAsyncDecode executes one double-buffered async decode step across active
// lanes in stable seqNo order, overlapping step N+1's forward with step N's
// host-side sampling/publication stage. Returns false without blocking if the
// 2-stage pipeline is already full so run()/drainWithDonation never deadlock on
// a full single-slot token outbox. Executor lock is held by caller.
func (s *NativeScheduler) stepAsyncDecode(active []*schedLane, iteration uint64, gen ForwardGen) bool {
	sort.SliceStable(active, func(i, j int) bool { return active[i].seqNo < active[j].seqNo })

	s.mu.Lock()
	if s.closed || (s.asyncPubActive && s.asyncPubPending != nil) {
		s.mu.Unlock()
		return false
	}
	s.drainDeferredFreesLocked()

	stageEntries := make([]*asyncLaneStage, 0, len(active))
	stepSeq := s.launchedStepSeq + 1

	for _, ln := range active {
		if !s.runningDecodeLaneLocked(ln) {
			continue
		}
		if err := ln.ctx.Err(); err != nil {
			ln.finish(nil, err)
			continue
		}
		if prevEntry := s.findUnresolvedAsyncEntryLocked(ln); prevEntry != nil && prevEntry.pubDone {
			// Terminal token already queued for async publication.
			continue
		}
		if ln.deferredToken != nil && ln.deferredToken.placeholder {
			ln.deferredToken = nil
		}
		if ln.deferredToken == nil {
			prevEntry := s.findUnresolvedAsyncEntryLocked(ln)
			if prevEntry != nil && len(prevEntry.logits) > 0 {
				// Step N's output has not finished publication yet: stage a -1
				// placeholder for step N+1 and resolve input from double buffer.
				resolvedInput := argmax(prevEntry.logits)
				ln.deferredToken = &nativeDeferredToken{
					id:          asyncOutputPlaceholderToken,
					placeholder: true,
					stepSeq:     stepSeq,
				}
				ln.numOutputPlaceholders++
				ln.asyncStageTokens = append(ln.asyncStageTokens, asyncOutputPlaceholderToken)
				s.asyncPlaceholdersStaged++
				s.asyncStepsOverlapped++
				stageEntries = append(stageEntries, &asyncLaneStage{
					lane:          ln,
					stepSeq:       stepSeq,
					inputID:       resolvedInput,
					toPublish:     asyncOutputPlaceholderToken,
					isPlaceholder: true,
				})
				continue
			}
			// No pending prior async stage for this lane: sample synchronously.
			if ln.state != schedLaneDecode || len(ln.logits) == 0 || ln.sess == nil || ln.sess == ln.statsSess {
				ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
				continue
			}
			if ln.specAdmissionPad > 0 {
				ln.specAdmissionPad = 0
			}
			next := argmax(ln.logits)
			budget := genTokens
			if ln.genBudget > 0 {
				budget = ln.genBudget
			}
			ln.deferredToken = &nativeDeferredToken{
				id:      next,
				done:    ln.sess.M.Cfg.IsEOS(next) || ln.emitted+1 >= budget,
				stepSeq: stepSeq,
			}
		}
		if ln.deferredToken.done {
			tok := ln.deferredToken.id
			countEmitted := !ln.deferredToken.accounted
			ln.deferredToken = nil
			stageEntries = append(stageEntries, &asyncLaneStage{
				lane:         ln,
				stepSeq:      stepSeq,
				toPublish:    tok,
				pubDone:      true,
				countEmitted: countEmitted,
				skipForward:  true,
			})
			continue
		}
		stageEntries = append(stageEntries, &asyncLaneStage{
			lane:          ln,
			stepSeq:       stepSeq,
			inputID:       ln.deferredToken.id,
			toPublish:     ln.deferredToken.id,
			isPlaceholder: ln.deferredToken.placeholder,
		})
	}

	if len(stageEntries) == 0 {
		s.mu.Unlock()
		return false
	}

	stepSeq = s.beginStepSeqLocked()
	finalEntries := make([]*asyncLaneStage, 0, len(stageEntries))
	execEntries := make([]*asyncLaneStage, 0, len(stageEntries))
	execIDs := make([]int, 0, len(stageEntries))
	seqs := make([]*model.Session, 0, len(stageEntries))
	for _, e := range stageEntries {
		e.stepSeq = stepSeq
		ln := e.lane
		if !s.runningDecodeLaneLocked(ln) {
			ln.deferredToken = nil
			continue
		}
		if err := ln.ctx.Err(); err != nil {
			ln.deferredToken = nil
			ln.finish(nil, err)
			continue
		}
		if e.skipForward {
			finalEntries = append(finalEntries, e)
			continue
		}
		if ln.deferredToken != nil && !ln.deferredToken.accounted {
			ln.deferredToken.accounted = true
			if !ln.deferredToken.placeholder {
				ln.emitted++
			}
		}
		sess := ln.takeSessionForModelLocked()
		if sess == nil {
			ln.deferredToken = nil
			continue
		}
		if s.elastic != nil {
			s.allocateLaneElasticGrowBlocksLocked(ln, ln.promptLen+ln.emitted+ln.numOutputPlaceholders)
		}
		ln.asyncInFlightEntry = e
		finalEntries = append(finalEntries, e)
		execEntries = append(execEntries, e)
		execIDs = append(execIDs, e.inputID)
		seqs = append(seqs, sess)
	}
	s.mu.Unlock()

	if len(finalEntries) == 0 {
		s.completeStepPublication(stepSeq)
		return false
	}
	if len(execEntries) == 0 {
		stage := &s.asyncStages[stepSeq%2]
		stage.stepSeq = stepSeq
		stage.iteration = iteration
		stage.entries = finalEntries
		s.mu.Lock()
		if !s.asyncPubActive {
			s.asyncPubActive = true
			s.asyncPubDone = make(chan struct{})
			s.asyncPubCurrent = stage
			go s.runAsyncPublish(stage)
		} else {
			s.asyncPubPending = stage
		}
		s.mu.Unlock()
		return true
	}

	defer func() {
		s.mu.Lock()
		pending := false
		for i, e := range execEntries {
			pending = pending || e.lane.inflightSess == seqs[i]
		}
		s.mu.Unlock()
		if !pending {
			return
		}
		done := make(chan struct{})
		go func() {
			defer close(done)
			for i, e := range execEntries {
				defer s.retireDecodeBorrow(e.lane, seqs[i])
			}
		}()
		<-done
	}()

	for _, e := range execEntries {
		s.beforeModelExecution(nativeSchedulerEventDecode, e.lane)
	}

	live := 0
	for i, e := range execEntries {
		ln := e.lane
		s.mu.Lock()
		ready := s.runningDecodeLaneLocked(ln) && ln.inflightSess == seqs[i] && ln.ctx.Err() == nil && !s.closed
		s.mu.Unlock()
		if !ready {
			e.truncated = true
			s.retireDecodeBorrow(ln, seqs[i])
			continue
		}
		execEntries[live], execIDs[live], seqs[live] = e, execIDs[i], seqs[i]
		live++
	}
	execEntries, execIDs, seqs = execEntries[:live], execIDs[:live], seqs[:live]
	if live == 0 {
		s.mu.Lock()
		hasPubOnly := false
		for _, e := range finalEntries {
			e.lane.asyncInFlightEntry = nil
			if e.skipForward && !e.truncated {
				hasPubOnly = true
			}
		}
		if hasPubOnly {
			stage := &s.asyncStages[stepSeq%2]
			stage.stepSeq = stepSeq
			stage.iteration = iteration
			stage.entries = finalEntries
			if !s.asyncPubActive {
				s.asyncPubActive = true
				s.asyncPubDone = make(chan struct{})
				s.asyncPubCurrent = stage
				go s.runAsyncPublish(stage)
			} else {
				s.asyncPubPending = stage
			}
			s.mu.Unlock()
			return true
		}
		s.mu.Unlock()
		s.completeStepPublication(stepSeq)
		return false
	}

	var logitsList [][]float32
	decodeStarted := s.now()
	if live == 1 {
		logits := RunWithOp(s.coupler, OpDecode, func() []float32 {
			return gen.Impl.Step(seqs[0], execIDs[0])
		})
		elapsed := s.now().Sub(decodeStarted)
		s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, elapsed)
		s.recorder.ObserveDecodeStep(enginestep.PathSerial, 1, elapsed)
		logitsList = [][]float32{copyF32(logits)}
	} else {
		bs := &model.BatchSession{M: s.m, Seqs: seqs}
		if s.recorder != nil {
			bs.ObserveSerialStep = s.recorder.ObserveSerialFallback
		}
		out := RunWithOp(s.coupler, OpBatch, func() [][]float32 {
			return gen.Impl.StepBatch(bs, execIDs)
		})
		elapsed := s.now().Sub(decodeStarted)
		s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, elapsed)
		if panels := bs.LastStepSharedPanels(); panels > 0 {
			s.recorder.ObserveDecodeStep(enginestep.PathBatched, len(execEntries), elapsed)
			s.mu.Lock()
			s.sharedBatchSteps++
			s.sharedPanels += uint64(panels)
			s.sharedMACs += uint64(bs.LastStepMACs())
			s.mu.Unlock()
		}
		if len(out) != len(execEntries) {
			panic(errNativeSchedulerLaneNotDecodeReady)
		}
		logitsList = make([][]float32, len(out))
		for i := range out {
			logitsList[i] = copyF32(out[i])
		}
	}

	stage := &s.asyncStages[stepSeq%2]
	stage.stepSeq = stepSeq
	stage.iteration = iteration
	stage.entries = finalEntries

	s.mu.Lock()
	for i, e := range execEntries {
		ln := e.lane
		sess := seqs[i]
		ln.asyncInFlightEntry = nil
		if !s.runningDecodeLaneLocked(ln) || !ln.restoreSessionFromModelLocked(sess) {
			if ln != nil && ln.inflightSess == sess {
				ln.inflightSess = nil
			}
			s.closeLaneSession(sess)
			e.truncated = true
			continue
		}
		if ln.asyncPendingFinish {
			// Prior async publication stop-truncated or finished this lane while
			// this forward was executing.
			if ln.asyncPendingTruncate > 0 && ln.sess != nil {
				ln.sess.RollbackSpeculative(ln.asyncPendingTruncate)
				if ln.trimPlaceholderTailLocked() > 0 {
					s.asyncPlaceholdersResolved++
				}
				if s.elastic != nil {
					s.trimLaneElasticTailBlocksLocked(ln, ln.promptLen+ln.emitted, stepSeq)
				}
			}
			hasTok := ln.asyncHasPendingFinalToken
			finalTok := ln.asyncPendingFinalToken
			finishErr := ln.asyncPendingErr
			res := ln.asyncPendingRes
			ln.asyncPendingFinish = false
			ln.asyncHasPendingFinalToken = false
			ln.asyncPendingTruncate = 0
			ln.asyncPendingErr = nil
			ln.asyncPendingRes = nil
			ln.deferredToken = nil
			if hasTok && finishErr == nil {
				// Publish the final stop-bounded token via runAsyncPublish so the
				// executor goroutine never blocks on a bounded token outbox.
				e.toPublish = finalTok
				e.isPlaceholder = false
				e.pubDone = true
				e.countEmitted = true
				e.truncated = false
				e.logits = nil
				e.specAdded = 0
			} else {
				e.truncated = true
				if !ln.terminal {
					ln.finish(res, finishErr)
				}
			}
			continue
		}
		if err := ln.ctx.Err(); err != nil || s.closed {
			if err == nil {
				err = errSchedClosed
			}
			ln.numOutputPlaceholders = 0
			ln.finish(nil, err)
			e.truncated = true
			continue
		}
		if len(logitsList[i]) == 0 {
			ln.numOutputPlaceholders = 0
			ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
			e.truncated = true
			continue
		}
		e.logits = logitsList[i]
		e.specAdded = 1
		ln.deferredToken = nil
	}
	if !s.asyncPubActive {
		s.asyncPubActive = true
		s.asyncPubDone = make(chan struct{})
		s.asyncPubCurrent = stage
		go s.runAsyncPublish(stage)
	} else {
		s.asyncPubPending = stage
	}
	s.mu.Unlock()
	return true
}

func (s *NativeScheduler) publishSingleAsyncToken(ln *schedLane, iteration uint64, tokID int) bool {
	var publishStarted time.Time
	if s.recorder != nil {
		publishStarted = s.now()
	}
	publishCanceled := false
	select {
	case ln.tokens <- abi.EngineToken{ID: tokID}:
	case <-ln.ctx.Done():
		publishCanceled = true
	}
	if s.recorder != nil {
		elapsed := s.now().Sub(publishStarted)
		if elapsed >= 0 && elapsed < time.Duration(1<<63-1) {
			s.recorder.ObservePhase(enginestep.PhaseTokenPublish, elapsed)
		}
	}
	if publishCanceled {
		s.mu.Lock()
		if !ln.terminal {
			ln.finish(nil, ln.ctx.Err())
		}
		s.mu.Unlock()
		return false
	}
	s.mu.Lock()
	state := ln.state
	s.mu.Unlock()
	s.observeEvent(nativeSchedulerEvent{Iteration: iteration, Kind: nativeSchedulerEventDecode, Lane: ln, State: state, Token: tokID})
	return true
}

func (s *NativeScheduler) runAsyncPublish(stage *asyncStepStage) {
	for stage != nil {
		for _, entry := range stage.entries {
			s.publishAsyncEntry(stage, entry)
		}

		s.mu.Lock()
		s.completeStepPublicationLocked(stage.stepSeq)
		if s.asyncPubPending != nil {
			next := s.asyncPubPending
			s.asyncPubPending = nil
			s.asyncPubCurrent = next
			s.notifyAsyncSlotLocked()
			s.mu.Unlock()
			s.signal()
			stage = next
			continue
		}
		s.asyncPubCurrent = nil
		s.asyncPubActive = false
		if s.closed {
			for _, ln := range s.lanes {
				if ln != nil && !ln.terminal && ln.inflightSess == nil {
					ln.numOutputPlaceholders = 0
					ln.finish(nil, errSchedClosed)
				}
			}
			s.flushDeferredFreesLocked()
		}
		if s.asyncPubDone != nil {
			close(s.asyncPubDone)
			s.asyncPubDone = nil
		}
		s.notifyAsyncSlotLocked()
		s.mu.Unlock()
		s.signal()
		return
	}
}

func (s *NativeScheduler) publishAsyncEntry(stage *asyncStepStage, entry *asyncLaneStage) {
	s.mu.Lock()
	ln := entry.lane
	if entry.truncated || ln == nil || ln.terminal {
		s.mu.Unlock()
		return
	}
	if err := ln.ctx.Err(); err != nil || s.closed {
		if err == nil {
			err = errSchedClosed
		}
		if ln.inflightSess != nil {
			ln.asyncPendingFinish = true
			ln.asyncPendingErr = err
		} else {
			ln.numOutputPlaceholders = 0
			ln.finish(nil, err)
		}
		s.mu.Unlock()
		return
	}
	tokenToSend := entry.toPublish
	isDone := entry.pubDone
	s.mu.Unlock()

	if tokenToSend != asyncOutputPlaceholderToken {
		if !s.publishSingleAsyncToken(ln, stage.iteration, tokenToSend) {
			return
		}
	}

	s.mu.Lock()
	if ln.terminal {
		s.mu.Unlock()
		return
	}
	if tokenToSend != asyncOutputPlaceholderToken {
		ln.gen = append(ln.gen, tokenToSend)
		if entry.countEmitted {
			ln.emitted++
		}
	}
	entry.resolved = true
	if isDone {
		modelName := s.modelName
		if modelName == "" {
			modelName = Default.ModelName()
		}
		res := assembleSyscallResult(ln.putCtx, ln.tool, EngineID, modelName, ln.promptLen, append([]int(nil), ln.gen...), ln.tok)
		if ln.inflightSess != nil {
			ln.asyncPendingFinish = true
			ln.asyncPendingRes = res
		} else {
			ln.finish(res, nil)
		}
		s.mu.Unlock()
		return
	}
	if len(entry.logits) == 0 || ln.sess == nil {
		s.mu.Unlock()
		return
	}

	nextID := argmax(entry.logits)
	budget := genTokens
	if ln.genBudget > 0 {
		budget = ln.genBudget
	}
	nextDone := ln.sess.M.Cfg.IsEOS(nextID) || len(ln.gen)+1 >= budget

	// Case 1: Succeeding step already finished forward and waits in asyncPubPending.
	if pendingNext := s.findPendingEntryForLaneLocked(ln); pendingNext != nil {
		if ln.resolvePlaceholderTokenLocked(nextID) {
			s.asyncPlaceholdersResolved++
		}
		if !nextDone {
			ln.emitted++
			pendingNext.toPublish = nextID
			pendingNext.isPlaceholder = false
			ln.logits = entry.logits
			s.mu.Unlock()
			return
		}
		// Stop-truncated tail: discard pendingNext, rollback speculative KV cache,
		// trim tail blocks via deferred-frees fence, then publish terminal nextID.
		pendingNext.truncated = true
		ln.asyncStopTruncated = true
		s.asyncStopTruncations++
		if pendingNext.specAdded > 0 && ln.sess != nil {
			ln.sess.RollbackSpeculative(pendingNext.specAdded)
		}
		if s.elastic != nil {
			s.trimLaneElasticTailBlocksLocked(ln, ln.promptLen+len(ln.gen), pendingNext.stepSeq)
		}
		s.mu.Unlock()

		if !s.publishSingleAsyncToken(ln, stage.iteration, nextID) {
			return
		}
		s.mu.Lock()
		if !ln.terminal {
			ln.gen = append(ln.gen, nextID)
			ln.emitted++
			modelName := s.modelName
			if modelName == "" {
				modelName = Default.ModelName()
			}
			res := assembleSyscallResult(ln.putCtx, ln.tool, EngineID, modelName, ln.promptLen, append([]int(nil), ln.gen...), ln.tok)
			ln.finish(res, nil)
		}
		s.mu.Unlock()
		return
	}

	// Case 2: Succeeding step's forward is currently running on another goroutine.
	if ln.inflightSess != nil && ln.asyncInFlightEntry != nil {
		inFlight := ln.asyncInFlightEntry
		if ln.resolvePlaceholderTokenLocked(nextID) {
			s.asyncPlaceholdersResolved++
		}
		if !nextDone {
			ln.emitted++
			inFlight.toPublish = nextID
			inFlight.isPlaceholder = false
			if ln.deferredToken != nil {
				ln.deferredToken.id = nextID
				ln.deferredToken.placeholder = false
			}
		} else {
			ln.asyncStopTruncated = true
			s.asyncStopTruncations++
			ln.asyncPendingTruncate = 1
			ln.asyncPendingFinalToken = nextID
			ln.asyncHasPendingFinalToken = true
			ln.asyncPendingFinish = true
		}
		s.mu.Unlock()
		return
	}

	// Case 3: No succeeding step overlapped yet.
	if !nextDone {
		ln.logits = entry.logits
		ln.deferredToken = &nativeDeferredToken{
			id:   nextID,
			done: false,
		}
		s.mu.Unlock()
		return
	}

	// Terminal token reached with no succeeding step overlapped: publish and finish.
	s.mu.Unlock()
	if !s.publishSingleAsyncToken(ln, stage.iteration, nextID) {
		return
	}
	s.mu.Lock()
	if !ln.terminal {
		ln.gen = append(ln.gen, nextID)
		ln.emitted++
		modelName := s.modelName
		if modelName == "" {
			modelName = Default.ModelName()
		}
		res := assembleSyscallResult(ln.putCtx, ln.tool, EngineID, modelName, ln.promptLen, append([]int(nil), ln.gen...), ln.tok)
		ln.finish(res, nil)
	}
	s.mu.Unlock()
}
