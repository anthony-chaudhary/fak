package modelengine

import "testing"

// fak-test:runtime fast est=100ms
func TestChunkedPrefillStrictExtensionReusesCompletedPrefix(t *testing.T) {
	const budget = nativeQwenPrefillMinChunkTokens
	m := nativeSchedulerPrefillModel(t)
	prefix := nativeSchedulerQwenPrompt(3 * budget)
	prompt := nativeSchedulerQwenPrompt(4 * budget)
	s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{
		"prefix":    prefix,
		"extension": prompt,
	}))
	s.SetInBatchPrefixDedup(true)
	if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
		t.Fatal(err)
	}
	nativeSchedulerBeginManualDrain(t, s)
	defer nativeSchedulerEndManualDrain(s)
	donor := nativeSchedulerAdmitLane(t, s, "prefix")
	defer donor.Cancel()
	follower := nativeSchedulerAdmitLane(t, s, "extension")
	defer follower.Cancel()
	var chunks []nativeSchedulerEvent
	var finalLogits []float32
	s.observeNativeEvent = func(ev nativeSchedulerEvent) {
		if ev.Kind == nativeSchedulerEventPrefill {
			chunks = append(chunks, ev)
		}
		if ev.Lane == follower && ev.Kind == nativeSchedulerEventTransition {
			finalLogits = copyF32(follower.logits)
		}
	}
	for i := 0; i < 3; i++ {
		nativeSchedulerDriveIteration(t, s)
		nativeSchedulerDrainAvailable(donor)
		if out := nativeSchedulerDrainAvailable(follower); len(out) != 0 {
			t.Fatalf("extension emitted before suffix prefill: %v", out)
		}
	}
	if follower.state != schedLanePrefilling || follower.promptCursor != len(prefix) ||
		follower.sess.Cache.Len() != len(prefix) || len(follower.logits) != 0 {
		t.Fatalf("extension state=%d cursor=%d cache=%d logits=%d, want prefix state and no terminal logits",
			follower.state, follower.promptCursor, follower.sess.Cache.Len(), len(follower.logits))
	}
	if follower.sess.Cache == donor.sess.Cache {
		t.Fatal("extension shares mutable cache object with donor")
	}
	nativeSchedulerDriveIteration(t, s)
	out := nativeSchedulerDrainAvailable(follower)
	if len(out) != 1 {
		t.Fatalf("extension first output=%v, want one token after its single suffix chunk", out)
	}
	if len(chunks) != 4 || chunks[3].Lane != follower || chunks[3].ChunkStart != len(prefix) || chunks[3].ChunkLen != budget {
		t.Fatalf("shared-prefix workload ran %d chunks, want three donor plus one suffix", len(chunks))
	}
	if s.ChunkedPrefixPrefillReuses() != 1 || s.ChunkedExactPrefillReuses() != 0 {
		t.Fatalf("prefix/exact reuse=%d/%d, want1/0", s.ChunkedPrefixPrefillReuses(), s.ChunkedExactPrefillReuses())
	}
	control := m.NewSession()
	control.Quant = true
	control.Q4K = true
	want := copyF32(control.Prefill(prompt))
	control.Close()
	nativeSchedulerAssertLogitParity(t, finalLogits, want)
	if out[0] != argmax(want) || follower.sess.Cache.Len() != len(prompt)+1 {
		t.Fatalf("extension token/cache=%d/%d, want%d/%d", out[0], follower.sess.Cache.Len(), argmax(want), len(prompt)+1)
	}
}

// fak-test:runtime fast est=100ms
func TestChunkedPrefillDivergentPromptsReuseLastAlignedPrefix(t *testing.T) {
	const budget = nativeQwenPrefillMinChunkTokens
	m := nativeSchedulerPrefillModel(t)
	prefix := nativeSchedulerQwenPrompt(3 * budget)
	donorPrompt := nativeSchedulerQwenPrompt(4 * budget)
	followerPrompt := append([]int(nil), donorPrompt...)
	followerPrompt[len(prefix)+1] = (followerPrompt[len(prefix)+1] + 1) % m.Cfg.VocabSize
	s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{
		"donor": donorPrompt, "follower": followerPrompt,
	}))
	s.SetInBatchPrefixDedup(true)
	if err := s.SetQwenPrefillMaxTokensPerIteration(budget); err != nil {
		t.Fatal(err)
	}
	nativeSchedulerBeginManualDrain(t, s)
	defer nativeSchedulerEndManualDrain(s)
	donor := nativeSchedulerAdmitLane(t, s, "donor")
	defer donor.Cancel()
	follower := nativeSchedulerAdmitLane(t, s, "follower")
	defer follower.Cancel()
	var chunks []nativeSchedulerEvent
	var followerLogits []float32
	s.observeNativeEvent = func(ev nativeSchedulerEvent) {
		if ev.Kind == nativeSchedulerEventPrefill {
			chunks = append(chunks, ev)
		}
		if ev.Lane == follower && ev.Kind == nativeSchedulerEventTransition {
			followerLogits = copyF32(follower.logits)
		}
	}
	for i := 0; i < 3; i++ {
		nativeSchedulerDriveIteration(t, s)
		nativeSchedulerDrainAvailable(donor)
		nativeSchedulerDrainAvailable(follower)
		if i < 2 && follower.promptCursor != 0 {
			t.Fatal("follower copied an early prefix instead of its last aligned checkpoint")
		}
	}
	if donor.state != schedLanePrefilling || follower.state != schedLanePrefilling ||
		follower.promptCursor != len(prefix) || follower.sess.Cache.Len() != len(prefix) || len(follower.logits) != 0 {
		t.Fatal("divergent follower did not adopt complete aligned state before donor completion")
	}
	if follower.sess.Cache == donor.sess.Cache {
		t.Fatal("divergent requests share mutable cache ownership")
	}
	var out []int
	for i := 0; i < 2; i++ {
		nativeSchedulerDriveIteration(t, s)
		nativeSchedulerDrainAvailable(donor)
		out = append(out, nativeSchedulerDrainAvailable(follower)...)
	}
	donorChunks := len(donorPrompt) / budget // full-budget donor prefill chunks
	if len(chunks) != donorChunks+1 || chunks[donorChunks].Lane != follower || chunks[donorChunks].ChunkStart != len(prefix) || chunks[donorChunks].ChunkLen != budget {
		t.Fatalf("divergent shared-prefix chunks=%v, want %d donor plus one follower suffix", chunks, donorChunks)
	}
	if len(out) != 1 || s.ChunkedPrefixPrefillReuses() != 1 || s.ChunkedExactPrefillReuses() != 0 {
		t.Fatalf("outputs/reuse=%v/%d/%d, want one output and prefix/exact1/0", out, s.ChunkedPrefixPrefillReuses(), s.ChunkedExactPrefillReuses())
	}
	control := m.NewSession()
	control.Quant, control.Q4K = true, true
	want := copyF32(control.Prefill(followerPrompt))
	control.Close()
	nativeSchedulerAssertLogitParity(t, followerLogits, want)
	if out[0] != argmax(want) || follower.sess.Cache.Len() != len(followerPrompt)+1 {
		t.Fatal("divergent suffix changed first token or complete cache length")
	}
}

// fak-test:runtime fast est=3s
func TestChunkedPrefillExtensionReclaimsPressureBeforeNextChunk(t *testing.T) {
	t.Setenv("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP", "")
	const (
		chunkTokens  = 32
		promptTokens = 128
		commonTokens = 65
		maxBlocks    = 8
		blockTokens  = 16
	)
	m := nativeSchedulerPrefillModel(t)
	donorPrompt := nativeSchedulerQwenPrompt(promptTokens)
	followerPrompt := append([]int(nil), donorPrompt...)
	followerPrompt[commonTokens] = (followerPrompt[commonTokens] + 1) % m.Cfg.VocabSize
	type boundary struct {
		seen                                       bool
		bothPromotedCold                           bool
		used, limit, protectedBlocks, partialLanes int
		donorCursor, followerCursor                int
		preemptions                                int64
		thirdDrainUsed                             int
		prefixReuses                               uint64
	}
	exercise := func(t *testing.T, enabled bool, capacity int) boundary {
		t.Helper()
		e := New()
		e.PreloadQ4K(m)
		e.SetInBatchPrefixDedup(enabled)
		s := e.nativeScheduler()
		if s.qwenPrefillCap == nil || s.qwenPrefillTokens != chunkTokens || s.InBatchPrefixDedupArmed() != enabled {
			t.Fatal("production constructor did not preserve resident chunking and the requested dedup setting")
		}
		// Keep the ordinary Engine scheduler and real admission path. Only exact
		// token preparation and ambient worker-pressure policy are controlled.
		s.prepare = nativeSchedulerPrefillPrepare(map[string][]int{
			"donor": donorPrompt, "follower": followerPrompt,
		})
		s.SetWorkerCoupler(NewWorkerCoupler(WorkerCouplingConfig{Mode: CouplingModeDisabled}))
		s.SetMaxRunning(2)
		s.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptRecompute,
			MaxBlocks:   capacity,
			BlockTokens: blockTokens,
		})
		nativeSchedulerBeginManualDrain(t, s)
		defer finishExactScheduler(t, s)
		donor := nativeSchedulerAdmitLane(t, s, "donor")
		defer donor.Cancel()
		follower := nativeSchedulerAdmitLane(t, s, "follower")
		defer follower.Cancel()
		for _, lane := range []*schedLane{donor, follower} {
			if lane.sess == nil || lane.state != schedLanePrefilling || lane.promptCursor != 0 || lane.sess.Cache.Len() != 0 {
				t.Fatal("real admission did not materialize both cold prefilling sessions before drain")
			}
		}
		var got boundary
		drain := 0
		s.beforeModelExecute = func(kind nativeSchedulerEventKind, lane *schedLane) {
			if drain == 1 && kind == nativeSchedulerEventPrefill && lane == donor {
				s.mu.Lock()
				got.bothPromotedCold = len(s.lanes) == 2 &&
					s.runningLaneLocked(donor) && s.runningLaneLocked(follower) &&
					donor.state == schedLanePrefilling && follower.state == schedLanePrefilling &&
					donor.promptCursor == 0 && follower.promptCursor == 0 &&
					follower.sess != nil && follower.sess.Cache != nil && follower.sess.Cache.Len() == 0
				s.mu.Unlock()
			}
			if drain != 4 || kind != nativeSchedulerEventPrefill || lane != donor {
				return
			}
			// runIteration has already enforced pressure at this boundary. The
			// donor now exposes its stats shell; published promptLen remains the
			// synchronized accounting authority until the next chunk completes.
			// Capture only: Fatal/Goexit inside this callback would strand the
			// manual drain's executor lock rather than report a clean failure.
			s.mu.Lock()
			got.seen = true
			got.used = s.usedKVBlocksLocked()
			got.limit = s.preemption.MaxBlocks
			got.donorCursor = donor.promptCursor
			got.followerCursor = follower.promptCursor
			got.preemptions = s.preemptStats.Preemptions
			for _, running := range s.lanes {
				if running.terminal || running.state != schedLanePrefilling {
					continue
				}
				got.protectedBlocks += s.laneKVBlocksLocked(running)
				if running.promptCursor > 0 && running.promptCursor < len(running.prompt) {
					got.partialLanes++
				}
			}
			s.mu.Unlock()
		}
		for drain = 1; drain <= 4; drain++ {
			nativeSchedulerDriveIteration(t, s)
			nativeSchedulerDrainAvailable(donor)
			nativeSchedulerDrainAvailable(follower)
			if drain == 3 {
				// Transient post-chunk excess is allowed by the existing boundary
				// policy. It is diagnostic only, not this regression's assertion.
				got.thirdDrainUsed = s.KVPreemptionStats().UsedBlocks
			}
		}
		got.prefixReuses = s.ChunkedPrefixPrefillReuses()
		if !got.bothPromotedCold || !got.seen || got.donorCursor != 3*chunkTokens || got.limit != capacity {
			t.Fatalf("fourth-drain donor prefill boundary was not reached: %+v", got)
		}
		return got
	}
	t.Run("dedup-off-control", func(t *testing.T) {
		got := exercise(t, false, maxBlocks)
		if got.used != 6 || got.protectedBlocks != 6 || got.followerCursor != 0 || got.partialLanes != 1 || got.preemptions != 0 {
			t.Fatalf("same-partition control changed its ordinary prefill/accounting path: %+v", got)
		}
	})
	t.Run("dedup-on", func(t *testing.T) {
		got := exercise(t, true, maxBlocks)
		if got.used > got.limit && got.protectedBlocks > got.limit {
			t.Fatalf("next prefill started after boundary enforcement with unreclaimable partial-prefix pressure: %+v", got)
		}
	})
	t.Run("dedup-on-safe-headroom", func(t *testing.T) {
		got := exercise(t, true, 12)
		if got.used != 10 || got.protectedBlocks != 10 || got.followerCursor != 2*chunkTokens || got.partialLanes != 2 || got.preemptions != 0 || got.prefixReuses != 1 {
			t.Fatalf("safe finite-budget partial checkpoint was not reused exactly once: %+v", got)
		}
	})
}

// fak-test:runtime fast est=1s
func TestChunkedPrefillExtensionLeavesHeadroomForEarlierLongFollower(t *testing.T) {
	t.Setenv("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP", "")
	m := nativeSchedulerPrefillModel(t)
	donorPrompt := nativeSchedulerQwenPrompt(128)
	followerPrompt := nativeSchedulerQwenPrompt(256)
	followerPrompt[65] = (followerPrompt[65] + 1) % m.Cfg.VocabSize
	e := New()
	e.PreloadQ4K(m)
	e.SetInBatchPrefixDedup(true)
	s := e.nativeScheduler()
	if s.qwenPrefillTokens != 32 || !s.InBatchPrefixDedupArmed() {
		t.Fatal("production constructor did not arm the expected chunked dedup route")
	}
	s.prepare = nativeSchedulerPrefillPrepare(map[string][]int{
		"donor": donorPrompt, "earlier": followerPrompt, "later": followerPrompt,
	})
	s.SetWorkerCoupler(NewWorkerCoupler(WorkerCouplingConfig{Mode: CouplingModeDisabled}))
	s.SetMaxRunning(3)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{
		Mode: NativePreemptRecompute, MaxBlocks: 16, BlockTokens: 16,
	})
	nativeSchedulerBeginManualDrain(t, s)
	defer finishExactScheduler(t, s)
	donor := nativeSchedulerAdmitLane(t, s, "donor")
	defer donor.Cancel()
	earlier := nativeSchedulerAdmitLane(t, s, "earlier")
	defer earlier.Cancel()
	later := nativeSchedulerAdmitLane(t, s, "later")
	defer later.Cancel()
	for _, lane := range []*schedLane{donor, earlier, later} {
		if lane.sess == nil || lane.state != schedLanePrefilling || lane.promptCursor != 0 || lane.sess.Cache.Len() != 0 {
			t.Fatal("real admission did not materialize all three cold prefilling sessions")
		}
	}
	for i := 0; i < 2; i++ {
		nativeSchedulerDriveIteration(t, s)
		for _, lane := range []*schedLane{donor, earlier, later} {
			if out := nativeSchedulerDrainAvailable(lane); len(out) != 0 {
				t.Fatalf("lane %s emitted before prompt completion: %v", lane.tool, out)
			}
		}
	}
	// Donor-only growth would permit both 64-token clones: 8+4+4=16.
	// Once the donor finishes, however, the earlier 256-token follower needs
	// all 16 blocks beside the later follower's protected four-block prefix.
	// Keep the useful first clone and decline only that unsafe second clone.
	if len(s.lanes) != 3 || donor.promptCursor != 64 || earlier.promptCursor != 64 || earlier.sess.Cache.Len() != 64 || later.promptCursor != 0 || later.sess.Cache.Len() != 0 || s.ChunkedPrefixPrefillReuses() != 1 {
		t.Fatalf("ordered checkpoint admission donor/earlier/later=%d/%d/%d reuses=%d, want64/64/0 and one copy",
			donor.promptCursor, earlier.promptCursor, later.promptCursor, s.ChunkedPrefixPrefillReuses())
	}
}
