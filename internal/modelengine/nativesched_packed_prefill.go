package modelengine

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/model"
)

var errPackedPrefillDeclined = errors.New("modelengine: selected packed prefill capability changed")

// PackedPrefillReceipt describes completed model calls, not speed or token
// publication. Canceled lanes can contribute executed rows but never logits.
// MaxTokenRows is the complete cohort's high water, bounded by 32.
type PackedPrefillReceipt struct {
	Calls        uint64
	Sequences    uint64
	TokenRows    uint64
	MaxTokenRows int
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
}

// advancePackedPrefill runs under the existing executor, never a second worker.
// It borrows every real session behind the existing immutable statistics shell;
// neither cancellation nor Close can free a cache while the model owns it.
// No scheduler mutex is held across model execution or recorder/test callbacks.
func (s *NativeScheduler) advancePackedPrefill(iteration uint64) (ready []*schedLane) {
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
		if len(lanes) == model.PackedPrefillMaxTokens {
			break
		}
	}
	work := make([]packedPrefillWork, 0, len(lanes))
	prompts := make([][]int, 0, len(lanes))
	seqs := make([]*model.Session, 0, len(lanes))
	budget := model.PackedPrefillMaxTokens
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
		work = append(work, packedPrefillWork{ln, session, ln.promptCursor, end})
		prompts = append(prompts, ln.prompt[ln.promptCursor:end])
		seqs = append(seqs, session)
		budget -= count
	}
	s.mu.Unlock()
	if len(work) == 0 {
		return nil
	}

	// A panic/Goexit during any model or observer call still returns every borrowed
	// owner to terminal retirement. Do not catch a panic and report fake success.
	retired := false
	defer func() {
		if retired {
			return
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, job := range work {
			if job.lane.restoreSessionFromModelLocked(job.session) {
				job.lane.finish(nil, errPackedPrefillDeclined)
			} else if job.lane.inflightSess == job.session {
				job.lane.inflightSess = nil
				s.closeLaneSession(job.session)
			}
		}
	}()

	for _, job := range work {
		s.beforeModelExecution(nativeSchedulerEventPrefill, job.lane)
	}
	bs := &model.BatchSession{M: s.m, Seqs: seqs, Quant: seqs[0].Quant}
	started := s.now()
	logits, used := bs.PrefillPacked(prompts)
	elapsed := s.now().Sub(started)

	s.mu.Lock()
	if used {
		rows := model.PackedPrefillMaxTokens - budget
		s.packedPrefillStats.Calls++
		s.packedPrefillStats.Sequences += uint64(len(work))
		s.packedPrefillStats.TokenRows += uint64(rows)
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
				ln.logits = logits[b]
				ln.state = schedLaneDecode
				ready = append(ready, ln)
			}
		}
	}
	s.mu.Unlock()
	retired = true

	if used {
		if elapsed >= 0 {
			s.recorder.ObservePrefillChunk(model.PackedPrefillMaxTokens-budget, elapsed)
		}
		for _, job := range work {
			s.observeEvent(nativeSchedulerEvent{Iteration: iteration, Kind: nativeSchedulerEventPrefill,
				Lane: job.lane, State: schedLanePrefilling, ChunkStart: job.start, ChunkLen: job.end - job.start})
		}
	}
	return ready
}
