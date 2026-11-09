package modelengine

import (
	"errors"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

var errPackedPrefillDeclined = errors.New("modelengine: selected packed prefill capability changed")

// PackedPrefillReceipt describes completed model calls, not speed or token
// publication. Canceled lanes can contribute executed rows but never logits.
// Sequences and TokenRows count only prompt work; DecodeRows counts mixed decode
// work. MaxTokenRows is the complete cohort's high water, bounded by 32.
type PackedPrefillReceipt struct {
	Calls        uint64
	Sequences    uint64
	TokenRows    uint64
	MaxTokenRows int
	DecodeRows   uint64
	MixedCalls   uint64
}

func (s *NativeScheduler) PackedPrefillReceipt() PackedPrefillReceipt {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.packedPrefillStats
}

// Called under mu before materialization. Existing prefix coalescing, explicit
// prefix state, resident Q4_K and state-capture contracts keep their old route.
func (s *NativeScheduler) packedPrefillCandidateLocked(ln *schedLane) bool {
	return s.packedPrefillEnabled && ln != nil && !ln.q4k &&
		!s.inBatchDedup && !s.captureQwenState && s.getPrefixState() == nil
}

type packedPrefillWork struct {
	lane    *schedLane
	session *model.Session
	start   int
	end     int
	decode  bool
}

// advancePackedPrefill runs under the existing executor, never a second worker.
// It borrows every real session behind the existing immutable statistics shell;
// neither cancellation nor Close can free a cache while the model owns it.
// No scheduler mutex is held across model execution or recorder/test callbacks.
func (s *NativeScheduler) advancePackedPrefill(iteration uint64) []*schedLane {
	ready, _ := s.advanceMixedPackedPrefill(iteration, nil, s.fwd.Load())
	return ready
}

// advanceMixedPackedPrefill reserves at least one prompt row and lends the
// remainder of the existing 32-row panel to eligible decode owners. The initial
// scheduler generation is the ordinary native forward; a hot-swapped generation
// retains its Step/StepBatch route rather than being bypassed by PrefillPacked.
// Adapted from vLLM scheduler.py's token budget and gpu/input_batch.py's row
// mapping at fa3b9d1377219247060d3c8b49eae199c9e7e154 (Apache-2.0).
func (s *NativeScheduler) advanceMixedPackedPrefill(iteration uint64, active []*schedLane, gen ForwardGen) (ready, mixed []*schedLane) {
	var decodes []*schedLane
	if gen.Generation == 1 && gen.Impl.Name == "incumbent" {
		ordered := append([]*schedLane(nil), active...)
		sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].seqNo < ordered[j].seqNo })
		outboxes := make(map[chan abi.EngineToken]int, len(ordered))
		for _, ln := range ordered {
			outboxes[ln.tokens]++
		}
		for _, ln := range ordered {
			if len(decodes) == model.PackedPrefillMaxTokens-1 {
				break
			}
			s.mu.Lock()
			eligible := s.runningDecodeLaneLocked(ln) && ln.sess != nil && ln.inflightSess == nil && ln.sess.PackedPrefillEligible() &&
				cap(ln.tokens) > len(ln.tokens) && outboxes[ln.tokens] == 1
			s.mu.Unlock()
			if eligible && ln.stageDeferredToken() {
				if _, forward := ln.deferredForForward(); forward {
					decodes = append(decodes, ln)
				}
			}
		}
	}
	s.mu.Lock()
	lanes := make([]*schedLane, 0, model.PackedPrefillMaxTokens)
	for _, ln := range s.lanes {
		if !ln.packedPrefill || !s.livePrefillLaneLocked(ln) {
			continue
		}
		if err := ln.ctx.Err(); err != nil {
			ln.finish(nil, err)
			continue
		}
		if ln.promptCursor < 0 || ln.promptCursor >= len(ln.prompt) || ln.sess == nil {
			ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
			continue
		}
		lanes = append(lanes, ln)
		if len(lanes) == model.PackedPrefillMaxTokens-len(decodes) {
			break
		}
	}
	if len(lanes) == 0 {
		s.mu.Unlock()
		return nil, nil
	}
	work := make([]packedPrefillWork, 0, len(lanes)+len(decodes))
	prompts := make([][]int, 0, len(lanes))
	seqs := make([]*model.Session, 0, len(lanes))
	budget := model.PackedPrefillMaxTokens
	// Borrow decode owners only after finding a prompt owner. Match the panel's
	// weight mode; unsupported lanes remain on the existing decode path.
	for _, ln := range decodes {
		if !s.runningDecodeLaneLocked(ln) || ln.ctx.Err() != nil || ln.sess == nil || ln.sess.Quant != lanes[0].sess.Quant || ln.deferredToken == nil || ln.deferredToken.done {
			continue
		}
		session := ln.takeSessionForModelLocked()
		if session == nil {
			continue
		}
		if !ln.deferredToken.accounted {
			ln.deferredToken.accounted = true
			ln.emitted++
		}
		work = append(work, packedPrefillWork{lane: ln, session: session, decode: true})
		prompts = append(prompts, []int{ln.deferredToken.id})
		seqs = append(seqs, session)
		budget--
	}
	decodeRows := len(work)
	for i, ln := range lanes {
		share := budget / (len(lanes) - i)
		count := len(ln.prompt) - ln.promptCursor
		if count > share {
			count = share
		}
		session := ln.takeSessionForModelLocked()
		if session == nil {
			ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
			continue
		}
		end := ln.promptCursor + count
		work = append(work, packedPrefillWork{lane: ln, session: session, start: ln.promptCursor, end: end})
		prompts = append(prompts, ln.prompt[ln.promptCursor:end])
		seqs = append(seqs, session)
		budget -= count
	}
	s.mu.Unlock()
	if len(work) == 0 {
		return nil, nil
	}

	// A panic/Goexit during any model or observer call still returns every borrowed
	// owner to terminal retirement. Do not catch a panic and report fake success.
	retired := false
	defer func() {
		if retired {
			return
		}
		// Register every borrow before cleanup can panic or Goexit.
		done := make(chan struct{})
		go func() {
			defer close(done)
			for _, job := range work {
				defer s.retireDecodeBorrow(job.lane, job.session)
			}
		}()
		<-done
	}()

	for _, job := range work {
		kind := nativeSchedulerEventPrefill
		if job.decode {
			kind = nativeSchedulerEventDecode
		}
		s.beforeModelExecution(kind, job.lane)
	}
	bs := &model.BatchSession{M: s.m, Seqs: seqs, Quant: seqs[0].Quant}
	started := s.now()
	var logits [][]float32
	var used bool
	if decodeRows > 0 {
		s.coupler.WithOp(OpBatch, func() { logits, used = bs.PrefillPacked(prompts) })
	} else {
		logits, used = bs.PrefillPacked(prompts)
	}
	elapsed := s.now().Sub(started)

	func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if used {
			rows := model.PackedPrefillMaxTokens - budget
			s.packedPrefillStats.Calls++
			s.packedPrefillStats.Sequences += uint64(len(work) - decodeRows)
			s.packedPrefillStats.TokenRows += uint64(rows - decodeRows)
			s.packedPrefillStats.DecodeRows += uint64(decodeRows)
			if decodeRows > 0 {
				s.packedPrefillStats.MixedCalls++
			}
			if rows > s.packedPrefillStats.MaxTokenRows {
				s.packedPrefillStats.MaxTokenRows = rows
			}
		}
		for b, job := range work {
			ln := job.lane
			if !ln.restoreSessionFromModelLocked(job.session) {
				if ln.inflightSess == job.session {
					ln.inflightSess = nil
					s.closeLaneSession(job.session)
				}
				continue
			}
			if job.decode {
				switch {
				case s.closed:
					ln.finish(nil, errSchedClosed)
				case ln.ctx.Err() != nil:
					ln.finish(nil, ln.ctx.Err())
				case !s.runningDecodeLaneLocked(ln):
					ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
				case !used:
					// PrefillPacked's false result guarantees no cache effects. Retain
					// the sampled/accounted token for the ordinary decode fallback.
				case b >= len(logits) || len(logits[b]) == 0:
					ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
				default:
					ln.logits = logits[b]
				}
				if used {
					mixed = append(mixed, ln)
				}
				continue
			}
			switch {
			case !s.livePrefillLaneLocked(ln):
				ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
			case !used:
				ln.finish(nil, errPackedPrefillDeclined)
			case s.closed:
				ln.finish(nil, errSchedClosed)
			case ln.ctx.Err() != nil:
				ln.finish(nil, ln.ctx.Err())
			case b >= len(logits) || len(logits[b]) == 0:
				ln.finish(nil, errNativeSchedulerLaneNotDecodeReady)
			default:
				ln.promptCursor = job.end
				ln.promptLen = job.end
				if job.end == len(ln.prompt) {
					if err := s.setLaneStateLocked(ln, schedLaneDecode); err != nil {
						ln.finish(nil, err)
						continue
					}
					ln.logits = logits[b]
					ready = append(ready, ln)
				}
			}
		}
	}()
	retired = true

	if used {
		if elapsed >= 0 {
			s.recorder.ObservePrefillChunk(model.PackedPrefillMaxTokens-budget-decodeRows, elapsed)
		}
		if decodeRows > 0 {
			// Both observations describe the same shared call. Their elapsed
			// durations overlap and must not be added as independent wall time.
			s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhaseDecode, elapsed)
			s.recorder.ObserveCohort(decodeRows)
			s.recorder.ObserveDecodeStep(enginestep.PathBatched, decodeRows, elapsed)
		}
		for _, job := range work {
			if job.decode {
				continue
			}
			s.observeEvent(nativeSchedulerEvent{Iteration: iteration, Kind: nativeSchedulerEventPrefill,
				Lane: job.lane, State: schedLanePrefilling, ChunkStart: job.start, ChunkLen: job.end - job.start})
		}
	}
	for _, ln := range mixed {
		if ln.retireDeferredToken(iteration) {
			ln.stageDeferredToken()
		}
	}
	if used && decodeRows > 0 {
		s.noteMixedPackedSkipDrafts(iteration, work)
	}
	return ready, mixed
}

// One shared execution has one verdict. An unfinished prompt is neutral in the
// AND; a continuing hot decoder or newly ready prompt still needs its drafts.
func (s *NativeScheduler) noteMixedPackedSkipDrafts(iteration uint64, work []packedPrefillWork) {
	s.mu.Lock()
	skip := true
	var reason SkipDraftsReason
	for _, job := range work {
		unused, r := s.laneDraftsUnusedLocked(job.lane, !job.decode)
		if !unused {
			skip = false
			reason = ""
			break
		}
		if reason == "" {
			reason = r
		}
	}
	s.skipDraftsSteps++
	if skip {
		s.skipDraftsSkipped++
		switch reason {
		case SkipDraftsPrefillChunk:
			s.skipDraftsPrefillChunks++
		case SkipDraftsTerminalToken:
			s.skipDraftsTerminalTokens++
		}
	}
	obs := s.skipDraftsObserver
	s.mu.Unlock()
	if obs != nil {
		obs(SkipDraftsDecision{Iteration: iteration, Skip: skip, Reason: reason})
	}
}
