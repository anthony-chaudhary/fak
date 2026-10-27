package modelengine

import "slices"

// ChunkedPrefixPrefillReuses counts cold prompts that adopt a completed
// aligned donor-prefix checkpoint and compute only their remaining suffix.
func (s *NativeScheduler) ChunkedPrefixPrefillReuses() uint64 {
	st := s.getPrefixState()
	if st == nil {
		return 0
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	return st.chunkedPrefixReuses
}

// executor and s.mu are held. Only a fully computed aligned chunk checkpoint
// is shared; recurrent state is never truncated and no logits are invented.
// A follower waits for its last aligned common-prefix checkpoint, preserving
// its original chunk partition and avoiding an early short-prefix copy.
func (s *NativeScheduler) reuseQwenPrefillExtensionsLocked(donor *schedLane) {
	if !s.inBatchDedup || s.captureQwenState || donor == nil || donor.ctx.Err() != nil ||
		donor.terminal || (donor.state != schedLanePrefilling && donor.state != schedLaneDecode) || donor.preemptRound != 0 ||
		donor.promptCursor < nativeInBatchPrefixDedupMinShared || donor.promptCursor > len(donor.prompt) ||
		donor.prefillChunkTokens < nativeQwenPrefillMinChunkTokens || s.getPrefixHitInfo(donor) != nil {
		return
	}
	host := func(ln *schedLane) bool {
		if ln == nil || ln.sess == nil || ln.inflightSess != nil || !ln.q4k {
			return false
		}
		sess := ln.sess
		return sess.M == s.m && sess.Cache != nil && sess.Backend == nil && sess.Q4K &&
			!sess.Quant && !sess.Q4 && !sess.GPTQ && !sess.Metal && !sess.MetalQ4K &&
			!sess.Q4KGateUpOutputSlab && sess.PrecisionPolicy == nil
	}
	depth := donor.promptCursor
	if !host(donor) || donor.sess.Cache.Len() != depth {
		return
	}
	blocks := s.laneKVBlocksLocked(donor)
	var reused uint64
	for _, follower := range s.lanes {
		if follower == donor || !host(follower) || follower.terminal || follower.ctx.Err() != nil ||
			follower.state != schedLanePrefilling || follower.promptCursor != 0 || follower.preemptRound != 0 ||
			follower.sess.Cache.Len() != 0 || len(follower.logits) != 0 ||
			follower.prefillChunkTokens != donor.prefillChunkTokens || depth%follower.prefillChunkTokens != 0 ||
			len(follower.prompt) <= depth ||
			s.getPrefixHitInfo(follower) != nil {
			continue
		}
		// Probe only the next chunk first. A completely matching next chunk
		// proves this is too early without re-scanning the long shared prefix
		// after every donor chunk. Verify the whole prefix only at a possible
		// last aligned checkpoint; no pair metadata or token copies are needed.
		next := depth
		limit := min(len(donor.prompt), len(follower.prompt))
		for next < limit && next-depth < follower.prefillChunkTokens && donor.prompt[next] == follower.prompt[next] {
			next++
		}
		// Exact twins and followers with no divergent suffix retain the
		// existing final-boundary logits policy, not this no-logits handoff.
		if next-depth == follower.prefillChunkTokens || next == len(follower.prompt) ||
			!slices.Equal(donor.prompt[:depth], follower.prompt[:depth]) {
			continue
		}
		used := s.usedKVBlocksLocked()
		added := blocks - s.laneKVBlocksLocked(follower)
		if s.preemption.MaxBlocks > 0 && (added > s.preemption.MaxBlocks || used > s.preemption.MaxBlocks-added) {
			continue
		}
		follower.sess.Cache = donor.sess.Cache.Clone()
		follower.promptCursor = depth
		follower.promptLen = depth
		reused++
		if used+added > s.maxObservedKVBlocks {
			s.maxObservedKVBlocks = used + added
		}
	}
	if reused > 0 {
		st := s.getOrCreatePrefixState()
		st.mu.Lock()
		st.chunkedPrefixReuses += reused
		st.mu.Unlock()
	}
}
