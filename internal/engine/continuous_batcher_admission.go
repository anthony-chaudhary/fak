package engine

import (
	"fmt"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// PrefixCacheStats returns the recurrent prefix-cache telemetry and whether the
// cache is enabled for this batcher.
func (cb *ContinuousBatcher) PrefixCacheStats() (RecurrentPrefixCacheStats, bool) {
	cb.mu.Lock()
	cache := cb.prefixCache
	cb.mu.Unlock()
	if cache == nil {
		return RecurrentPrefixCacheStats{}, false
	}
	return cache.Stats(), true
}

// PrefixCacheLen returns the number of retained conversation boundaries, or 0
// when the cache is disabled.
func (cb *ContinuousBatcher) PrefixCacheLen() int {
	cb.mu.Lock()
	cache := cb.prefixCache
	cb.mu.Unlock()
	if cache == nil {
		return 0
	}
	return cache.Len()
}

// Submit enqueues a subagent request into an available slot or waiting queue.
func (cb *ContinuousBatcher) Submit(req *SubagentRequest) (string, error) {
	return cb.submit(req, nil)
}

// submit lets the owning driver reserve IDs until their handles are drained.
// A non-nil reserved set also retires an accepted ID's old completed lookup;
// public Submit keeps its standalone historical-lookup behavior.
func (cb *ContinuousBatcher) submit(req *SubagentRequest, reserved map[string]struct{}) (string, error) {
	if req == nil {
		return "", ErrNilRequest
	}
	if req.TargetTokens <= 0 {
		if req.ExecutionDepth > 0 || req.RecurrentLoops > 0 {
			req.TargetTokens = req.effectiveDepth()
		} else {
			return "", ErrInvalidTargetTokens
		}
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.closed {
		return "", ErrBatcherClosed
	}

	sessionID := req.SessionID
	if sessionID == "" {
		for {
			id := atomic.AddUint64(&cb.seqCounter, 1)
			sessionID = fmt.Sprintf("subagent-%d", id)
			if _, exists := reserved[sessionID]; !exists {
				break
			}
		}
		req.SessionID = sessionID
	}
	if _, exists := reserved[sessionID]; exists {
		return "", ErrSessionAlreadyExists
	}

	if slot, exists := cb.sessionMap[sessionID]; exists && slot.State != SlotStateFinished {
		return "", ErrSessionAlreadyExists
	}
	for _, q := range cb.waitingQueue {
		if q.SessionID == sessionID {
			return "", ErrSessionAlreadyExists
		}
	}
	if reserved != nil {
		// All rejection checks have passed. A queued replacement must not
		// resolve to its predecessor's closed slot while awaiting promotion.
		delete(cb.completedMap, sessionID)
	}

	// Assign the stable submission-order identity only once the request is known
	// to be accepted (past the duplicate-session rejections) and before any
	// queueing decision, so a rejected duplicate does not burn a sequence number
	// while the decode-residency roster (M2) still orders by true submission time
	// even for requests that wait behind capacity.
	cb.uidSeq++
	req.SubmissionSeq = cb.uidSeq

	// Look for an available empty slot
	emptyIdx := -1
	for i, slot := range cb.slots {
		if slot.State == SlotStateEmpty {
			emptyIdx = i
			break
		}
	}

	reqBytes := cb.RequiredKVCacheBytes(req)
	canAdmit := emptyIdx != -1 && len(cb.waitingQueue) == 0 &&
		(cb.cfg.MaxKVCacheBytes <= 0 || cb.currentKVCacheBytesLocked()+reqBytes <= cb.cfg.MaxKVCacheBytes)

	if canAdmit {
		cb.initSlot(emptyIdx, req)
	} else {
		cb.waitingQueue = append(cb.waitingQueue, req)
	}

	return sessionID, nil
}

func (cb *ContinuousBatcher) initSlot(index int, req *SubagentRequest) *Slot {
	reqKVBytes := cb.RequiredKVCacheBytes(req)
	depth := req.effectiveDepth()
	slot := &Slot{
		Index:             index,
		SessionID:         req.SessionID,
		State:             SlotStateActiveDecode,
		Request:           req,
		PromptTokens:      append([]int(nil), req.PromptTokens...),
		GeneratedTokens:   make([]int, 0, req.TargetTokens),
		TargetTokens:      req.TargetTokens,
		ExecutionDepth:    depth,
		KVCacheStationary: true,
		KVCacheBytes:      reqKVBytes,
		SubmissionSeq:     req.SubmissionSeq,
		AdmittedAt:        time.Now(),
		tokenCh:           make(chan int, req.TargetTokens+16),
		doneCh:            make(chan struct{}),
	}

	// GDN recurrent prefix reuse (hybrid models only). model.PrefixSnapshot is the
	// only safe reuse unit for a Gated DeltaNet fold: the linear recurrence is an
	// irreversible fold over the whole prefix, so a whole-prefix boundary captured
	// at the token position that produced it — and carrying the full linear state
	// via KVCache.Clone → linear.clone() plus softmax KV — is the sole
	// mathematically valid reuse unit. Lookup is a take: on a hit ownership of the
	// snapshot transfers here and we prefill only req.PromptTokens[reused:].
	var (
		reused       int
		restored     bool
		restoredSess *model.Session
	)
	if cb.prefixCache != nil && cb.cfg.Model != nil && len(req.PromptTokens) > 0 {
		if matched, snap, hit := cb.prefixCache.Lookup(req.SessionID, req.PromptTokens); hit {
			// Reuse is intentionally host-only: a device session would need a
			// matching Backend, or Restore refuses closed→miss.
			sess := &model.Session{
				M:     cb.cfg.Model,
				Cache: model.NewKVCache(cb.cfg.Model.Cfg),
			}
			if err := snap.Restore(sess); err != nil {
				// A failed restore is a miss, not a hit: release the snapshot and
				// fall back to a full fresh-session prefill.
				snap.Close()
			} else {
				reused = matched
				restored = true
				restoredSess = sess
				// Once-only admission accounting: report this reuse on the next
				// emitted step, not on every step the slot stays resident.
				cb.pendingPrefixHits++
				cb.pendingPrefixReuseTokens += reused
			}
		}
	}

	if cb.cfg.PrefillBudget > 0 && req.ChunkedPrefill {
		// Budgeted chunked prefill: defer activation. The slot consumes its prompt
		// copy in budget-sized chunks across PREFILL steps, then transitions to
		// SlotStateActiveDecode. Upstream mini-sglang prefill.py:65-151 treats an
		// over-budget prompt as resumable chunks; the in-repo precedent is
		// internal/modelengine/nativesched_prefill.go (PrefillNoLogits for
		// intermediate chunks, Prefill for the final chunk). When a real model is
		// configured the session is built here exactly as the eager branch does,
		// and its KV cache is advanced in lockstep with PrefillPos so no consumed
		// token is ever re-prefilled.
		slot.State = SlotStatePrefilling
		slot.PendingPrompt = append([]int(nil), req.PromptTokens...)
		slot.PrefillPos = 0
		slot.LastToken = 0
		if cb.cfg.Model != nil {
			if restored {
				// The restored session already holds the recurrent fold for the
				// first `reused` tokens; PendingPrompt stays the full prompt so the
				// existing PREFILL arm consumes only the suffix via
				// PendingPrompt[PrefillPos:...].
				slot.sess = restoredSess
				slot.PrefillPos = reused
			} else {
				slot.sess = &model.Session{
					M:     cb.cfg.Model,
					Cache: model.NewKVCache(cb.cfg.Model.Cfg),
				}
			}
		}
		if restored {
			slot.PrefixHits = 1
			slot.PrefixMatchedTokens = reused
		}
		cb.slots[index] = slot
		cb.sessionMap[req.SessionID] = slot
		return slot
	}

	if cb.cfg.Model != nil {
		sess := restoredSess
		if sess == nil {
			sess = &model.Session{
				M:     cb.cfg.Model,
				Cache: model.NewKVCache(cb.cfg.Model.Cfg),
			}
		}
		if len(req.PromptTokens) > 0 {
			if restored && reused >= len(req.PromptTokens) {
				// Strict extend guarantees a remaining suffix, but guard the
				// impossible case rather than call Prefill with an empty slice.
				slot.LastToken = req.PromptTokens[len(req.PromptTokens)-1]
			} else {
				logits := sess.Prefill(req.PromptTokens[reused:])
				slot.LastToken = argmax(logits)
			}
		} else {
			slot.LastToken = 1
		}
		slot.sess = sess
		if restored {
			slot.PrefixHits = 1
			slot.PrefixMatchedTokens = reused
		}
		// Store the boundary this prompt produced so a later turn can extend it.
		cb.retainPrefixLocked(slot)
	} else {
		if len(req.PromptTokens) > 0 {
			slot.LastToken = req.PromptTokens[len(req.PromptTokens)-1]
		} else {
			slot.LastToken = 42 + slot.Index*31
		}
	}

	cb.slots[index] = slot
	cb.sessionMap[req.SessionID] = slot
	return slot
}

// WaitingQueueLength returns number of requests waiting for an open slot.
func (cb *ContinuousBatcher) WaitingQueueLength() int {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return len(cb.waitingQueue)
}

// RequiredKVCacheBytes calculates the required KV cache footprint for a request based on
// token dimensions, KV bytes per token, and recurrent execution depth:
// KV capacity required = (len(PromptTokens) + TargetTokens) * KVBytesPerToken * ExecutionDepth.
func (cb *ContinuousBatcher) RequiredKVCacheBytes(req *SubagentRequest) int64 {
	if req == nil {
		return 0
	}
	depth := int64(req.effectiveDepth())
	bpt := req.KVBytesPerToken
	if bpt <= 0 {
		bpt = cb.cfg.KVBytesPerToken
	}
	if bpt <= 0 {
		bpt = 1024
	}
	targetTokens := req.TargetTokens
	if targetTokens <= 0 {
		targetTokens = req.effectiveDepth()
	}
	tokens := int64(len(req.PromptTokens) + targetTokens)
	return tokens * bpt * depth
}

// CurrentKVCacheBytes returns the total KV cache memory footprint in bytes allocated across active and yielded slots.
func (cb *ContinuousBatcher) CurrentKVCacheBytes() int64 {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.currentKVCacheBytesLocked()
}

// MaxKVCacheBytes returns the configured maximum KV cache memory capacity in bytes (0 = unconstrained).
func (cb *ContinuousBatcher) MaxKVCacheBytes() int64 {
	return cb.cfg.MaxKVCacheBytes
}

func (cb *ContinuousBatcher) currentKVCacheBytesLocked() int64 {
	var total int64
	for _, s := range cb.slots {
		if s.State == SlotStatePrefilling || s.State == SlotStateActiveDecode || s.State == SlotStateYieldedIO {
			total += s.KVCacheBytes
		}
	}
	return total
}

func (cb *ContinuousBatcher) tryPromoteWaitingLocked() []string {
	if cb.closed {
		cb.waitingQueue = nil
		return nil
	}

	var promoted []string
	for len(cb.waitingQueue) > 0 {
		emptyIdx := -1
		for i, s := range cb.slots {
			if s.State == SlotStateEmpty {
				emptyIdx = i
				break
			}
		}
		if emptyIdx == -1 {
			break
		}

		req := cb.waitingQueue[0]
		reqBytes := cb.RequiredKVCacheBytes(req)
		if cb.cfg.MaxKVCacheBytes > 0 && cb.currentKVCacheBytesLocked()+reqBytes > cb.cfg.MaxKVCacheBytes {
			break
		}

		cb.waitingQueue[0] = nil
		cb.waitingQueue = cb.waitingQueue[1:]
		cb.initSlot(emptyIdx, req)
		promoted = append(promoted, req.SessionID)
	}
	return promoted
}

// WaitingQueue returns a shallow copy of the pending subagent requests in the waiting queue.
func (cb *ContinuousBatcher) WaitingQueue() []*SubagentRequest {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	res := make([]*SubagentRequest, len(cb.waitingQueue))
	copy(res, cb.waitingQueue)
	return res
}
