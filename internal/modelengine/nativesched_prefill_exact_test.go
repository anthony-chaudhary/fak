package modelengine

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=2s
func TestChunkedExactPrefillConfiguredRoute(t *testing.T) {
	t.Setenv("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP", "")
	m := nativeSchedulerPrefillModel(t)
	prompt := nativeSchedulerQwenPrompt(nativeServingPrefillTokensPerIteration + 16)
	e := New()
	e.PreloadQ4K(m)
	e.SetInBatchPrefixDedup(true)
	s := e.nativeScheduler()
	if !s.InBatchPrefixDedupArmed() || s.qwenPrefillCap == nil || s.qwenPrefillTokens != nativeServingPrefillTokensPerIteration {
		t.Fatal("production constructor did not arm both chunking and exact dedup")
	}
	// Preserve production scheduler/config; exact token preparation avoids tokenizer noise.
	s.prepare = nativeSchedulerPrefillPrepare(map[string][]int{"a": prompt, "b": prompt, "c": prompt, "d": prompt})
	nativeSchedulerBeginManualDrain(t, s)
	defer finishExactScheduler(t, s)
	oracle := m.NewSession()
	oracle.Q4K = true
	defer oracle.Close()
	oracle.PrefillNoLogits(prompt[:nativeServingPrefillTokensPerIteration])
	want := copyF32(oracle.Prefill(prompt[nativeServingPrefillTokensPerIteration:]))
	cache := oracle.Cache
	nextOracle := m.NewSession()
	nextOracle.Q4K = true
	nextOracle.Cache = cache.Clone()
	defer nextOracle.Close()
	wantNext := copyF32(nextOracle.Step(3))
	chunks := 0
	transitions := map[string]int{}
	s.observeNativeEvent = func(ev nativeSchedulerEvent) {
		if ev.Kind == nativeSchedulerEventPrefill {
			chunks++
		}
		if ev.Kind == nativeSchedulerEventTransition {
			transitions[ev.Lane.tool]++
			// The donor remains the original cache; followers receive full Clone snapshots.
			// Compare each exact representation without normalizing or dropping fields.
			wantCache := cache
			if ev.Lane.tool != "a" {
				wantCache = cache.Clone()
			}
			if !reflect.DeepEqual(ev.Lane.logits, want) || !reflect.DeepEqual(ev.Lane.sess.Cache, wantCache) || ev.Lane.sess.SequencePosition() != len(prompt) {
				t.Errorf("%s full prompt/logits/state differs from same-partition oracle", ev.Lane.tool)
				return
			}
			clone := m.NewSession()
			clone.Q4K = true
			clone.Cache = ev.Lane.sess.Cache.Clone()
			if got := clone.Step(3); !reflect.DeepEqual(got, wantNext) {
				t.Errorf("%s next-Step differs", ev.Lane.tool)
			}
			clone.Close()
		}
	}
	lanes := []*schedLane{}
	for _, name := range []string{"a", "b", "c", "d"} {
		lanes = append(lanes, nativeSchedulerAdmitLane(t, s, name))
	}
	for i := 0; i < 2; i++ {
		nativeSchedulerDriveIteration(t, s)
		for _, ln := range lanes {
			nativeSchedulerDrainAvailable(ln)
		}
	}
	if chunks != 2 || len(transitions) != 4 {
		t.Fatalf("chunks=%d transitions=%v; want one 2-chunk walk and 4 ready lanes", chunks, transitions)
	}
	for _, ln := range lanes {
		if ln.state != schedLaneDecode || transitions[ln.tool] != 1 {
			t.Fatalf("%s not published once", ln.tool)
		}
	}
	counter, ok := any(s).(interface{ ChunkedExactPrefillReuses() uint64 })
	if !ok || counter.ChunkedExactPrefillReuses() != 3 {
		t.Fatal("missing distinct exact-copy accounting")
	}
	if got := s.InBatchPrefixDedupStats(); got != (InBatchPrefixDedupStats{}) {
		t.Fatalf("asynchronous copies polluted synchronous flight roles: %+v", got)
	}
	// Each follower owns detached cache state even after the donor starts decode.
	if lanes[1].sess.Cache == lanes[2].sess.Cache || lanes[1].sess.Cache == lanes[0].sess.Cache {
		t.Fatal("shared mutable cache")
	}
	before := lanes[2].sess.Cache.Clone()
	lanes[1].sess.Step(7)
	if !reflect.DeepEqual(lanes[2].sess.Cache, before) {
		t.Fatal("follower mutation changed sibling")
	}
}

// fak-test:runtime fast est=2s
func TestChunkedExactPrefillExclusions(t *testing.T) {
	for _, mode := range []string{"disabled", "divergent", "partition", "partial", "device", "budget", "prefix-donor", "prefix-follower"} {
		t.Run(mode, func(t *testing.T) {
			m := nativeSchedulerPrefillModel(t)
			prompt := nativeSchedulerQwenPrompt(32)
			s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"a": prompt, "b": prompt}))
			s.SetInBatchPrefixDedup(mode != "disabled")
			if err := s.SetQwenPrefillMaxTokensPerIteration(16); err != nil {
				t.Fatal(err)
			}
			nativeSchedulerBeginManualDrain(t, s)
			defer finishExactScheduler(t, s)
			a, b := nativeSchedulerAdmitLane(t, s, "a"), nativeSchedulerAdmitLane(t, s, "b")
			switch mode {
			case "divergent":
				b.prompt[len(b.prompt)-1]++
			case "partition":
				b.prefillChunkTokens = 32
			case "partial":
				b.sess.PrefillNoLogits(b.prompt[:16])
				b.promptCursor = 16
				b.promptLen = 16
			case "device":
				b.sess.MetalQ4K = true
			case "budget":
				s.preemption.MaxBlocks = 3
				s.preemption.BlockTokens = 16
			case "prefix-donor", "prefix-follower":
				target := a
				if mode == "prefix-follower" {
					target = b
				}
				st := s.getOrCreatePrefixState()
				st.holderLookups[target.sess] = &prefixHitInfo{applied: true}
			}
			for i := 0; i < 2; i++ {
				nativeSchedulerDriveIteration(t, s)
				nativeSchedulerDrainAvailable(a)
			}
			if b.state != schedLanePrefilling {
				t.Fatalf("%s unexpectedly copied", mode)
			}
			if mode == "device" {
				b.sess.MetalQ4K = false
			} // no device execution during cleanup
		})
	}
}

// fak-test:runtime fast est=2s
func TestChunkedExactPrefillCancellation(t *testing.T) {
	for _, mode := range []string{"donor-before-final", "follower-before-final", "donor-after-copy", "follower-after-copy"} {
		t.Run(mode, func(t *testing.T) {
			m := nativeSchedulerPrefillModel(t)
			prompt := nativeSchedulerQwenPrompt(32)
			s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"a": prompt, "b": prompt}))
			s.SetInBatchPrefixDedup(true)
			if err := s.SetQwenPrefillMaxTokensPerIteration(16); err != nil {
				t.Fatal(err)
			}
			nativeSchedulerBeginManualDrain(t, s)
			defer finishExactScheduler(t, s)
			closes := map[*model.Session]int{}
			s.closeSession = func(sess *model.Session) { closes[sess]++; sess.Close() }
			a, b := nativeSchedulerAdmitLane(t, s, "a"), nativeSchedulerAdmitLane(t, s, "b")
			sa, sb := a.sess, b.sess
			if mode == "donor-after-copy" || mode == "follower-after-copy" {
				s.observeNativeEvent = func(ev nativeSchedulerEvent) {
					if ev.Kind == nativeSchedulerEventPrefill && ev.ChunkStart == 16 {
						if mode == "follower-after-copy" {
							b.cancel()
						} else {
							a.cancel()
						}
					}
				}
			}
			nativeSchedulerDriveIteration(t, s)
			if mode == "donor-before-final" {
				a.cancel()
			}
			if mode == "follower-before-final" {
				b.cancel()
			}
			for i := 0; i < 5; i++ {
				nativeSchedulerDriveIteration(t, s)
				nativeSchedulerDrainAvailable(a)
				nativeSchedulerDrainAvailable(b)
			}
			survivor := b
			if mode == "follower-before-final" || mode == "follower-after-copy" {
				survivor = a
			}
			if survivor.terminal && survivor.err != nil {
				t.Fatalf("survivor failed: %v", survivor.err)
			}
			if survivor.promptCursor != len(prompt) {
				t.Fatal("survivor did not finish prefill")
			}
			a.Cancel()
			b.Cancel()
			s.Close()
			for i := 0; i < 3; i++ {
				nativeSchedulerDriveIteration(t, s)
				nativeSchedulerDrainAvailable(a)
				nativeSchedulerDrainAvailable(b)
			}
			if closes[sa] != 1 || closes[sb] != 1 {
				t.Fatalf("close counts %v/%v (%s)", closes[sa], closes[sb], mode)
			}
		})
	}
}

func finishExactScheduler(t *testing.T, s *NativeScheduler) {
	t.Helper()
	nativeSchedulerEndManualDrain(s)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := s.CloseAndWait(ctx); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=2s
func TestChunkedExactPrefillTerminalParity(t *testing.T) {
	m := nativeSchedulerPrefillModel(t)
	prompt := nativeSchedulerQwenPrompt(48)
	run := func(enabled bool) (out [][]int, meta []map[string]string, chunks int) {
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"a": prompt, "b": prompt}))
		s.SetInBatchPrefixDedup(enabled)
		if err := s.SetQwenPrefillMaxTokensPerIteration(16); err != nil {
			t.Fatal(err)
		}
		nativeSchedulerBeginManualDrain(t, s)
		defer finishExactScheduler(t, s)
		s.observeNativeEvent = func(ev nativeSchedulerEvent) {
			if ev.Kind == nativeSchedulerEventPrefill {
				chunks++
			}
		}
		lanes := []*schedLane{nativeSchedulerAdmitLane(t, s, "a"), nativeSchedulerAdmitLane(t, s, "b")}
		out = make([][]int, 2)
		for i := 0; i < 100; i++ {
			nativeSchedulerDriveIteration(t, s)
			done := true
			for j, ln := range lanes {
				out[j] = append(out[j], nativeSchedulerDrainAvailable(ln)...)
				done = done && ln.terminal
			}
			if done {
				break
			}
		}
		for _, ln := range lanes {
			if !ln.terminal || ln.err != nil || ln.res == nil {
				t.Fatalf("terminal state=%v error=%v", ln.terminal, ln.err)
			}
			meta = append(meta, ln.res.Meta)
		}
		return
	}
	off, offMeta, offChunks := run(false)
	on, onMeta, onChunks := run(true)
	if !reflect.DeepEqual(off, on) || !reflect.DeepEqual(offMeta, onMeta) {
		t.Fatalf("off/on token or usage mismatch: %v/%v %v/%v", off, on, offMeta, onMeta)
	}
	if offChunks != 6 || onChunks != 3 {
		t.Fatalf("real model chunks off/on=%d/%d want6/3", offChunks, onChunks)
	}
}

// fak-test:runtime fast est=2s
func TestChunkedExactPrefillPartialBudgetAndObserverClose(t *testing.T) {
	for _, closeAtCopy := range []bool{false, true} {
		t.Run(map[bool]string{false: "one-copy-budget", true: "observer-close"}[closeAtCopy], func(t *testing.T) {
			m := nativeSchedulerPrefillModel(t)
			prompt := nativeSchedulerQwenPrompt(32)
			s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"a": prompt, "b": prompt, "c": prompt}))
			s.SetInBatchPrefixDedup(true)
			if err := s.SetQwenPrefillMaxTokensPerIteration(16); err != nil {
				t.Fatal(err)
			}
			nativeSchedulerBeginManualDrain(t, s)
			defer finishExactScheduler(t, s)
			a, b, c := nativeSchedulerAdmitLane(t, s, "a"), nativeSchedulerAdmitLane(t, s, "b"), nativeSchedulerAdmitLane(t, s, "c")
			s.preemption.MaxBlocks = 4
			s.preemption.BlockTokens = 16
			chunks := map[string]int{}
			transitions := map[string]int{}
			s.observeNativeEvent = func(ev nativeSchedulerEvent) {
				if ev.Kind == nativeSchedulerEventPrefill {
					chunks[ev.Lane.tool]++
					if closeAtCopy && ev.ChunkStart == 16 {
						s.Close()
					}
				}
				if ev.Kind == nativeSchedulerEventTransition {
					transitions[ev.Lane.tool]++
					if ev.Lane.terminal || ev.Lane.sess == nil {
						t.Fatal("transition for retired session")
					}
				}
			}
			for i := 0; i < 2; i++ {
				nativeSchedulerDriveIteration(t, s)
				nativeSchedulerDrainAvailable(a)
				nativeSchedulerDrainAvailable(b)
				nativeSchedulerDrainAvailable(c)
			}
			if closeAtCopy {
				if len(transitions) != 0 {
					t.Fatalf("transition after Close: %v", transitions)
				}
				return
			}
			if b.state != schedLaneDecode || c.state != schedLanePrefilling || s.maxObservedKVBlocks > 4 {
				t.Fatalf("partial budget b/c=%d/%d high=%d", b.state, c.state, s.maxObservedKVBlocks)
			}
			// Retire completed donor/twin before ordinary prefill of the excluded lane.
			s.mu.Lock()
			a.finish(nil, context.Canceled)
			b.finish(nil, context.Canceled)
			s.mu.Unlock()
			for i := 0; i < 5; i++ {
				nativeSchedulerDriveIteration(t, s)
				nativeSchedulerDrainAvailable(c)
			}
			if chunks["b"] != 0 || chunks["c"] != 2 {
				t.Fatalf("budget-excluded lane not ordinary: %v", chunks)
			}
		})
	}
}
