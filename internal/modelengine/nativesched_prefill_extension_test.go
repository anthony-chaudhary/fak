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
	if len(chunks) != 5 || chunks[4].Lane != follower || chunks[4].ChunkStart != len(prefix) || chunks[4].ChunkLen != budget {
		t.Fatalf("divergent shared-prefix chunks=%v, want four donor plus one follower suffix", chunks)
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
