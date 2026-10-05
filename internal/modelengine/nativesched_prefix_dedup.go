package modelengine

import (
	"context"
	"errors"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// nativeInBatchPrefixDedupMinShared is the minimum shared token prefix (in tokens) for a
// concurrent cold admission to join an in-flight leader instead of recomputing it. Ported
// from sglang schedule_policy.py IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD (b8ec5449).
const nativeInBatchPrefixDedupMinShared = 32

// errInBatchDedupNotShareable tells CoalesceSharedPrefixNS's caller that the leader ran but
// its KV cannot be shared (e.g. the session did not populate a full cache). It is swallowed
// by the scheduler's eligibility wrapper so coalescing fails open to the ordinary prefill.
var errInBatchDedupNotShareable = errors.New("modelengine: in-batch prefix dedup KV not shareable")

// InBatchPrefixDedupStats counts how the in-batch cold-prefix dedup path resolved.
//
// Leaders and Followers are per-flight-role counters, not a partition of admissions:
// a follower whose flight fails is retried by CoalesceSharedPrefixNS and can then run
// as a leader, incrementing both. Leaders+Followers may therefore exceed the number of
// admissions. CoalescedPrefills == ExactReuses + PrefixReuses holds exactly.
type InBatchPrefixDedupStats struct {
	Leaders           uint64 // admissions that ran the shared prefill as leader
	Followers         uint64 // admissions that joined an in-flight leader
	CoalescedPrefills uint64 // follower admissions that skipped the shared prefix
	ExactReuses       uint64 // followers that reused the leader's exact logits
	PrefixReuses      uint64 // followers that adopted a truncated prefix KV and prefilled only the suffix
}

// SetInBatchPrefixDedup enables or disables in-batch cold-prefix coalescing. It is a
// configuration surface, set before the first Admit; default false.
func (s *NativeScheduler) SetInBatchPrefixDedup(enabled bool) {
	s.mu.Lock()
	s.inBatchDedup = enabled
	s.mu.Unlock()
}

// InBatchPrefixDedupStats returns a copy of the dedup counters. Safe after draining.
func (s *NativeScheduler) InBatchPrefixDedupStats() InBatchPrefixDedupStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inBatchDedupStats
}

// InBatchPrefixDedupArmed reports whether the coalescing path is armed on this
// scheduler. The counters alone cannot distinguish an armed-but-idle scheduler from
// one that was never armed, so an operator readback needs both.
func (s *NativeScheduler) InBatchPrefixDedupArmed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inBatchDedup
}

// prefixFlightGroup lazily creates the single per-scheduler flight group shared by every
// concurrent admission. One group per scheduler is required for coalescing to fire.
func (s *NativeScheduler) prefixFlightGroup() *radixkv.PrefixFlightGroup {
	s.prefixFlightsMu.Lock()
	defer s.prefixFlightsMu.Unlock()
	if s.prefixFlights == nil {
		s.prefixFlights = radixkv.NewPrefixFlightGroup(nil)
	}
	return s.prefixFlights
}

// inBatchDedupEligible reports whether this cold admission may join the coalesced path.
// Only a plain host-KV cold lane qualifies: the radixkv prefix-KV clone/truncate handoff
// is proven only for the generic host cache, so a HAL-backed (Backend != nil) or
// device/Metal-resident session stays on the ordinary prefill. A session that already
// holds KV has nothing to coalesce.
func (s *NativeScheduler) inBatchDedupEligible(sess *model.Session, prompt []int) bool {
	s.mu.Lock()
	enabled := s.inBatchDedup
	s.mu.Unlock()
	return enabled &&
		len(prompt) >= nativeInBatchPrefixDedupMinShared &&
		sess != nil &&
		sess.Cache != nil && sess.Cache.Len() == 0 &&
		sess.Backend == nil &&
		!sess.Metal && !sess.MetalQ4K
}

// coldPrefillSync runs one synchronous full-prompt prefill on sess, wrapped in the
// scheduler's op coupler and cache-phase latency recorder, exactly as the historical
// admission path did. It MUST be called with s.mu NOT held.
func (s *NativeScheduler) coldPrefillSync(sess *model.Session, ids []int) []float32 {
	started := s.now()
	logits := RunWithOp(s.coupler, OpPrefill, func() []float32 { return sess.Prefill(ids) })
	s.cachePhaseLatency.Observe(modelperfobs.CachePipelinePhasePrefill, s.now().Sub(started))
	return logits
}

// prefillCoalesced runs the shared-prefix single-flight for one cold lane and returns the
// logits for its last prompt token. It records the resolution in s.inBatchDedupStats.
// It MUST be called with s.mu NOT held. On any non-leader "fail open" branch it recomputes
// this lane's own cold prefill, so correctness never depends on a twin arriving.
func (s *NativeScheduler) prefillCoalesced(ctx context.Context, sess *model.Session, prompt []int) []float32 {
	var leaderLogits []float32
	kv, followerLogits, matched, leader, err := s.prefixFlightGroup().CoalesceSharedPrefixNS(
		ctx, "", prompt, nativeInBatchPrefixDedupMinShared,
		func(context.Context) (*model.KVCache, []float32, error) {
			if s.prefillFlightHook != nil {
				s.prefillFlightHook()
			}
			leaderLogits = s.coldPrefillSync(sess, prompt)
			if sess.Cache == nil || sess.Cache.Len() != len(prompt) {
				return nil, leaderLogits, errInBatchDedupNotShareable
			}
			return sess.Cache, leaderLogits, nil
		})
	if leader {
		s.mu.Lock()
		s.inBatchDedupStats.Leaders++
		s.mu.Unlock()
		if errors.Is(err, errInBatchDedupNotShareable) {
			return leaderLogits
		}
		if err != nil {
			// The leader prefill ran but the flight reported a non-sentinel error
			// (e.g. an incomplete IPC handoff). Never drop a completed local prefill:
			// use its logits, and only recompute if we somehow have none.
			if len(leaderLogits) > 0 {
				return leaderLogits
			}
			return s.coldPrefillSync(sess, prompt)
		}
		return leaderLogits
	}
	// Follower.
	s.mu.Lock()
	s.inBatchDedupStats.Followers++
	s.mu.Unlock()
	if err != nil {
		// The flight failed: fall open to this lane's own cold prefill.
		return s.coldPrefillSync(sess, prompt)
	}
	switch {
	case kv != nil && matched == len(prompt) && followerLogits != nil:
		// Exact twin: reuse the leader's prefix KV and logits outright.
		sess.Cache = kv
		s.mu.Lock()
		s.inBatchDedupStats.CoalescedPrefills++
		s.inBatchDedupStats.ExactReuses++
		s.mu.Unlock()
		return copyF32(followerLogits)
	case kv != nil && matched > 0 && matched < len(prompt):
		// Partial twin: adopt the prefix KV, prefill only the divergent suffix.
		sess.Cache = kv
		logits := s.coldPrefillSync(sess, prompt[matched:])
		s.mu.Lock()
		s.inBatchDedupStats.CoalescedPrefills++
		s.inBatchDedupStats.PrefixReuses++
		s.mu.Unlock()
		return logits
	default:
		// Not shareable (leader was a strict shorter/longer prefix, recurrent cache, ...):
		// fail open to a full local cold prefill.
		return s.coldPrefillSync(sess, prompt)
	}
}
