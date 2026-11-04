package engine

import (
	"context"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Step performs a decode-only iteration-level step: it gathers resident decodable
// slots (stable-UID order), generates one token for each simultaneously, retires
// finished slots, and pulls queued requests into freed slots. It is exactly
// stepWithBudget(ctx, 0). The pre-existing Step observable behavior is preserved
// except for two intentional refinements: the decode batch is ordered by stable
// submission sequence, and it is filtered through slotDecodableLocked so a slot
// already at its target is not advanced again.
func (cb *ContinuousBatcher) Step(ctx context.Context) (*BatchStepResult, error) {
	return cb.stepWithBudget(ctx, 0)
}

// StepPhase is the budgeted scheduler step: a PREFILL step (up to PrefillBudget
// prompt tokens, FIFO) when a slot is prefilling and the decode-starvation bound
// allows it; otherwise a DECODE step over the resident decodable slots.
func (cb *ContinuousBatcher) StepPhase(ctx context.Context) (*BatchStepResult, error) {
	return cb.stepWithBudget(ctx, cb.cfg.PrefillBudget)
}

// DecodeResident returns the M2 decode-residency roster: slots in
// SlotStateActiveDecode that are still decodable, ordered by ascending stable
// submission sequence (tie-break SessionID). The order is deterministic and
// independent of Go map iteration or slot-index reuse.
func (cb *ContinuousBatcher) DecodeResident() []*Slot {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.decodeResidentLocked()
}

// DecodeResidentUIDs returns the stable submission-sequence ids of DecodeResident,
// in the same order. It is the ordering key a caller can pin deterministically.
func (cb *ContinuousBatcher) DecodeResidentUIDs() []uint64 {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	resident := cb.decodeResidentLocked()
	uids := make([]uint64, len(resident))
	for i, slot := range resident {
		uids[i] = slot.SubmissionSeq
	}
	return uids
}

// decodeResidentLocked gathers decodable SlotStateActiveDecode slots and sorts
// them into stable-UID order. Caller MUST hold cb.mu.
func (cb *ContinuousBatcher) decodeResidentLocked() []*Slot {
	resident := make([]*Slot, 0, len(cb.slots))
	for _, slot := range cb.slots {
		slot.mu.Lock()
		ok := slot.State == SlotStateActiveDecode && slotDecodableLocked(slot)
		slot.mu.Unlock()
		if ok {
			resident = append(resident, slot)
		}
	}
	sortSlotsByStableUID(resident)
	return resident
}

// slotDecodableLocked reports whether an active-decode slot still has work under
// the same completion rule Step uses: recurrent slots are bounded by
// ExecutionDepth, ordinary slots by TargetTokens. Caller holds slot.mu.
func slotDecodableLocked(slot *Slot) bool {
	if slot.ExecutionDepth > 1 {
		return slot.CurrentDepth < slot.ExecutionDepth
	}
	return len(slot.GeneratedTokens) < slot.TargetTokens
}

// sortSlotsByStableUID orders slots by SubmissionSeq ascending, tie-break
// SessionID ascending, so the decode batch is deterministic.
func sortSlotsByStableUID(slots []*Slot) {
	sort.Slice(slots, func(i, j int) bool {
		if slots[i].SubmissionSeq != slots[j].SubmissionSeq {
			return slots[i].SubmissionSeq < slots[j].SubmissionSeq
		}
		return slots[i].SessionID < slots[j].SessionID
	})
}

// stepSnapshot captures the scheduler state observed at the start of a step:
// the promotions just applied, the once-only admission accounting, the
// per-state slot census, and the prefilling/resident rosters. The PREFILL and
// DECODE arms both report from it so they agree on slot counts within a step.
type stepSnapshot struct {
	stepStart         time.Time
	promotedIDs       []string
	prefixHits        int
	prefixReuseTokens int
	prefilling        []*Slot
	resident          []*Slot
	yieldedCount      int
	emptyCount        int
	finishedCount     int
}

// gatherStepSnapshotLocked pulls queued requests into empty slots, captures and
// resets the once-only admission accounting, and gathers the per-state census
// plus the prefilling and resident decodable rosters. Caller MUST hold cb.mu.
func (cb *ContinuousBatcher) gatherStepSnapshotLocked() *stepSnapshot {
	// 1. Pull queued requests into any empty slots before gathering batches.
	// initSlot applies the budgeted-prefill rule for ChunkedPrefill requests.
	promotedIDs := cb.tryPromoteWaitingLocked()

	// Capture and reset once-only admission accounting now that this step's
	// promotions (which may reuse a cached boundary) are included.
	prefixHits := cb.pendingPrefixHits
	prefixReuseTokens := cb.pendingPrefixReuseTokens
	cb.pendingPrefixHits = 0
	cb.pendingPrefixReuseTokens = 0

	// 2. Gather prefilling slots and the resident decodable roster, and count
	// the other states in the same pass.
	snap := &stepSnapshot{stepStart: time.Now()}
	resident := cb.decodeResidentLocked()
	for _, slot := range cb.slots {
		slot.mu.Lock()
		st := slot.State
		slot.mu.Unlock()

		switch st {
		case SlotStatePrefilling:
			snap.prefilling = append(snap.prefilling, slot)
		case SlotStateYieldedIO:
			snap.yieldedCount++
		case SlotStateFinished:
			snap.finishedCount++
		case SlotStateEmpty:
			snap.emptyCount++
		}
	}
	snap.promotedIDs = promotedIDs
	snap.prefixHits = prefixHits
	snap.prefixReuseTokens = prefixReuseTokens
	snap.resident = resident
	return snap
}

// stepWithBudget is the shared scheduler step. budget > 0 enables the PREFILL arm
// for SlotStatePrefilling slots; budget == 0 is the decode-only legacy path.
func (cb *ContinuousBatcher) stepWithBudget(ctx context.Context, budget int) (*BatchStepResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return nil, ErrBatcherClosed
	}

	snap := cb.gatherStepSnapshotLocked()

	// 3. PREFILL arm, consuming up to budget prompt tokens FIFO. The refill gate
	// adapts TGI backends/v3/src/backend.rs:186-195@b4adbf2 (Apache-2.0).
	refillDue := len(snap.resident) == 0 || cb.decodeSinceRefill >= cb.cfg.MaxWaitingDecodeSteps ||
		len(snap.prefilling) >= int(cb.cfg.WaitingServedRatio*float64(len(snap.resident)))
	if budget > 0 && len(snap.prefilling) > 0 && refillDue {
		return cb.runPrefillArmLocked(snap, budget)
	}
	return cb.runDecodeArmLocked(snap)
}

// runPrefillArmLocked runs the PREFILL arm: consume up to budget prompt tokens
// across the prefilling slots FIFO, finishing slots whose pending prompt is fully
// consumed, and return the step result. Caller MUST hold cb.mu.
func (cb *ContinuousBatcher) runPrefillArmLocked(snap *stepSnapshot, budget int) (*BatchStepResult, error) {
	cb.decodeSinceRefill = 0
	sortSlotsByStableUID(snap.prefilling)
	remaining := budget
	total := 0
	for _, slot := range snap.prefilling {
		if remaining <= 0 {
			break
		}
		slot.mu.Lock()
		pending := len(slot.PendingPrompt) - slot.PrefillPos
		if pending <= 0 {
			cb.finishPrefillLocked(slot, nil)
			slot.mu.Unlock()
			continue
		}
		chunk := remaining
		if chunk > pending {
			chunk = pending
		}
		// Capture the chunk of prompt consumed THIS step BEFORE advancing
		// PrefillPos, so the model session's KV cache advances in lockstep
		// with PrefillPos and no already-consumed token is ever re-prefilled.
		// Upstream mini-sglang prefill.py:65-151 is a resumable chunked
		// prefill; the in-repo precedent is
		// internal/modelengine/nativesched_prefill.go:481-563.
		var logits []float32
		if slot.sess != nil {
			chunkIDs := slot.PendingPrompt[slot.PrefillPos : slot.PrefillPos+chunk]
			if slot.PrefillPos+chunk == len(slot.PendingPrompt) {
				// Final chunk: its last-token distribution seeds decode.
				logits = slot.sess.Prefill(chunkIDs)
			} else {
				// Intermediate chunk: grow KV only, discard logits.
				slot.sess.PrefillNoLogits(chunkIDs)
			}
		}
		slot.PrefillPos += chunk
		remaining -= chunk
		total += chunk
		if slot.PrefillPos >= len(slot.PendingPrompt) {
			cb.finishPrefillLocked(slot, logits)
		}
		slot.mu.Unlock()
	}

	cb.iteration++
	residentUIDs := make([]uint64, len(snap.resident))
	for i, slot := range snap.resident {
		residentUIDs[i] = slot.SubmissionSeq
	}
	return &BatchStepResult{
		Iteration:            cb.iteration,
		ActiveSlots:          len(snap.resident),
		YieldedSlots:         snap.yieldedCount,
		FinishedSlots:        snap.finishedCount,
		EmptySlots:           snap.emptyCount,
		TotalSlots:           len(cb.slots),
		GeneratedTokens:      make(map[string]int),
		OperationalIntensity: cb.OperationalIntensity(len(snap.resident)),
		ArithmeticIntensity:  cb.OperationalIntensity(len(snap.resident)),
		AggregateThroughput:  cb.AggregateThroughput(len(snap.resident)),
		StepDuration:         time.Since(snap.stepStart),
		PromotedSessionIDs:   snap.promotedIDs,
		KVCacheBytesUsed:     cb.currentKVCacheBytesLocked(),
		SlotDepths:           make(map[string]int),
		Phase:                PhasePrefill,
		PrefillTokens:        total,
		DecodeResidentUIDs:   residentUIDs,
		PrefixHits:           snap.prefixHits,
		PrefixReuseTokens:    snap.prefixReuseTokens,
	}, nil
}

// runDecodeArmLocked runs the DECODE arm over the resident decodable roster:
// forward pass, token delivery, completion marking, retirement of finished
// slots, and re-promotion of queued requests into freed slots. With an empty
// roster it mirrors the legacy idle path. Caller MUST hold cb.mu.
func (cb *ContinuousBatcher) runDecodeArmLocked(snap *stepSnapshot) (*BatchStepResult, error) {
	// 4. DECODE arm. If nothing is resident, mirror the legacy empty path.
	if len(snap.resident) == 0 {
		cb.iteration++
		return &BatchStepResult{
			Iteration:          cb.iteration,
			YieldedSlots:       snap.yieldedCount,
			FinishedSlots:      snap.finishedCount,
			EmptySlots:         snap.emptyCount,
			TotalSlots:         len(cb.slots),
			GeneratedTokens:    make(map[string]int),
			StepDuration:       time.Since(snap.stepStart),
			PromotedSessionIDs: snap.promotedIDs,
			KVCacheBytesUsed:   cb.currentKVCacheBytesLocked(),
			SlotDepths:         make(map[string]int),
			Phase:              PhaseIdle,
			DecodeResidentUIDs: []uint64{},
			PrefixHits:         snap.prefixHits,
			PrefixReuseTokens:  snap.prefixReuseTokens,
		}, nil
	}

	cb.decodeSinceRefill++
	activeSlots := snap.resident
	activeCount := len(activeSlots)
	tokensThisStep := make([]int, activeCount)
	generatedTokens := make(map[string]int, activeCount)
	slotDepths := make(map[string]int, activeCount)

	// 5. Forward pass over the ordered resident slots simultaneously.
	if cb.cfg.Model != nil {
		if activeCount == 1 {
			slot := activeSlots[0]
			if len(slot.PromptTokens) > 0 && len(slot.GeneratedTokens) == 0 {
				tokensThisStep[0] = slot.LastToken
			} else {
				logits := slot.sess.Step(slot.LastToken)
				tokensThisStep[0] = argmax(logits)
			}
		} else {
			seqs := make([]*model.Session, activeCount)
			ids := make([]int, activeCount)
			stepActive := make([]bool, activeCount)
			for i, slot := range activeSlots {
				seqs[i] = slot.sess
				ids[i] = slot.LastToken
				if len(slot.PromptTokens) > 0 && len(slot.GeneratedTokens) == 0 {
					tokensThisStep[i] = slot.LastToken
					continue
				}
				stepActive[i] = true
			}
			bs := &model.BatchSession{M: cb.cfg.Model, Seqs: seqs}
			batchLogits := bs.StepBatchActive(ids, stepActive)
			for i, logits := range batchLogits {
				if stepActive[i] {
					tokensThisStep[i] = argmax(logits)
				}
			}
		}
	} else {
		// Pure scheduler mode: compute deterministic next token
		for i, slot := range activeSlots {
			seed := slot.LastToken
			if seed == 0 {
				seed = 42
			}
			nextToken := (seed*37+13+len(slot.GeneratedTokens))%32000 + 1
			tokensThisStep[i] = nextToken
		}
	}

	// 6. Deliver tokens and mark completion.
	var newlyFinished []*Slot
	for i, slot := range activeSlots {
		tok := tokensThisStep[i]
		slot.mu.Lock()
		slot.LastToken = tok
		slot.GeneratedTokens = append(slot.GeneratedTokens, tok)
		slot.CurrentDepth++
		slot.RecurrentPasses++
		cb.totalTokens++
		generatedTokens[slot.SessionID] = tok
		slotDepths[slot.SessionID] = slot.CurrentDepth

		select {
		case slot.tokenCh <- tok:
		default:
		}

		if !slotDecodableLocked(slot) {
			slot.State = SlotStateFinished
			slot.CompletedAt = time.Now()
			close(slot.doneCh)
			close(slot.tokenCh)
			slot.closeSessionLocked()
			newlyFinished = append(newlyFinished, slot)
		}
		slot.mu.Unlock()
	}

	// 7. Retire finished slots and pull queued requests into freed slots.
	var retiredIDs []string
	for _, slot := range newlyFinished {
		retiredIDs = append(retiredIDs, slot.SessionID)
		cb.completedMap[slot.SessionID] = slot
		delete(cb.sessionMap, slot.SessionID)

		idx := slot.Index
		cb.slots[idx] = &Slot{
			Index: idx,
			State: SlotStateEmpty,
		}
	}

	promotedAfterRetire := cb.tryPromoteWaitingLocked()
	promotedIDs := append(snap.promotedIDs, promotedAfterRetire...)

	cb.iteration++

	opIntensity := cb.OperationalIntensity(activeCount)
	aggTPS := cb.AggregateThroughput(activeCount)

	residentUIDs := make([]uint64, len(snap.resident))
	for i, slot := range snap.resident {
		residentUIDs[i] = slot.SubmissionSeq
	}

	return &BatchStepResult{
		Iteration:            cb.iteration,
		ActiveSlots:          activeCount,
		YieldedSlots:         snap.yieldedCount,
		FinishedSlots:        len(newlyFinished) + snap.finishedCount,
		EmptySlots:           len(cb.slots) - (activeCount - len(newlyFinished) + snap.yieldedCount),
		TotalSlots:           len(cb.slots),
		GeneratedTokens:      generatedTokens,
		TokensGenerated:      activeCount,
		OperationalIntensity: opIntensity,
		ArithmeticIntensity:  opIntensity,
		AggregateThroughput:  aggTPS,
		StepDuration:         time.Since(snap.stepStart),
		StallDuration:        0, // zero compute stalls on yield/resume
		RetiredSessionIDs:    retiredIDs,
		PromotedSessionIDs:   promotedIDs,
		KVCacheBytesUsed:     cb.currentKVCacheBytesLocked(),
		SlotDepths:           slotDepths,
		Phase:                PhaseDecode,
		DecodeTokens:         activeCount,
		DecodeResidentUIDs:   residentUIDs,
		PrefixHits:           snap.prefixHits,
		PrefixReuseTokens:    snap.prefixReuseTokens,
	}, nil
}

// finishPrefillLocked completes a budgeted-prefill chunk: it promotes the slot to
// SlotStateActiveDecode and sets the token decode starts from. On the model path the
// caller passes the final chunk's logits and the argmax must be preserved (the prompt's
// last token is the teacher-forced input, not the next-token prediction); on the
// simulation path (sess == nil) the last prompt token is the seed exactly as before.
// Caller MUST hold cb.mu AND slot.mu.
func (cb *ContinuousBatcher) finishPrefillLocked(slot *Slot, logits []float32) {
	switch {
	case slot.sess != nil && len(logits) > 0:
		slot.LastToken = argmax(logits)
	case len(slot.PendingPrompt) > 0:
		slot.LastToken = slot.PendingPrompt[len(slot.PendingPrompt)-1]
	default:
		slot.LastToken = 42 + slot.Index*31
	}
	slot.State = SlotStateActiveDecode
	// The recurrent boundary is only valid once the whole prompt has been
	// folded, so retain it only when the slot has consumed its entire pending
	// prompt. Using slot.PromptTokens (the stored full prompt) as the key is
	// exact: the snapshot captures the fold over all of it.
	if slot.PrefillPos >= len(slot.PendingPrompt) {
		cb.retainPrefixLocked(slot)
	}
}

// retainPrefixLocked stores the recurrent boundary a fully-consumed prompt
// produced, so a later conversation turn can extend it. It is a no-op unless a
// cache is enabled, a real model session exists, and the slot holds a non-empty
// prompt. Ownership of the captured complete snapshot transfers to the cache.
// An incompatible old boundary is replaced atomically, without closing and
// re-capturing the new snapshot (which would clone the full hybrid state twice).
// Caller MUST hold cb.mu and slot.mu. The one eager-path call from initSlot runs
// before the slot is published to cb.slots/sessionMap, so no other goroutine can
// observe it.
func (cb *ContinuousBatcher) retainPrefixLocked(slot *Slot) {
	if cb.prefixCache == nil || cb.cfg.Model == nil || slot.sess == nil || len(slot.PromptTokens) == 0 {
		return
	}
	snap, err := slot.sess.PrefixSnapshot()
	if err != nil || snap == nil {
		return
	}
	_ = cb.prefixCache.storeBoundary(slot.SessionID, slot.PromptTokens, snap)
}

func argmax(v []float32) int {
	if len(v) == 0 {
		return 0
	}
	m, idx := v[0], 0
	for i, x := range v[1:] {
		if x > m {
			m, idx = x, i+1
		}
	}
	return idx
}
