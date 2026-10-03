package modelengine

import (
	"context"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// prefixDedupPlainPrepare admits exact token prompts through the historical
// unquantized session path, so tests can exercise the partial (prefix-truncating)
// coalesce on a full-attention model whose KV cache supports eviction. A hybrid
// Gated-DeltaNet cache cannot discard a leader-only suffix (radixkv fails that
// follower open to cold), so PrefixReuses is only reachable on an evictable cache.
func prefixDedupPlainPrepare(prompts map[string][]int) schedPrepareFunc {
	return func(_ context.Context, call *abi.ToolCall, _ *model.Model) schedPrepare {
		return schedPrepare{prompt: append([]int(nil), prompts[call.Tool]...)}
	}
}

// prefixDedupAwait polls a predicate with a hard deadline. The dedup seam is
// asynchronous (a follower joins a leader registered on another goroutine), so a
// deterministic test needs a bounded spin rather than a fixed sleep.
func prefixDedupAwait(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

// prefixDedupDrainAll drives the manual scheduler until every lane is terminal
// and returns each lane's emitted token IDs. Driving stops the moment all lanes
// terminal, so the final buffered tokens are still collected.
func prefixDedupDrainAll(t *testing.T, s *NativeScheduler, lanes ...*schedLane) [][]int {
	t.Helper()
	outs := make([][]int, len(lanes))
	for iter := 0; iter < 5000; iter++ {
		alive := false
		for _, ln := range lanes {
			if !ln.terminal {
				alive = true
				break
			}
		}
		if alive {
			nativeSchedulerDriveIteration(t, s)
		}
		for i, ln := range lanes {
			outs[i] = append(outs[i], nativeSchedulerDrainAvailable(ln)...)
		}
		if !alive {
			return outs
		}
	}
	t.Fatalf("lanes did not terminate within 5000 scheduler iterations")
	return outs
}

// prefixDedupAdmitPairGated admits two lanes concurrently with the leader wedged
// inside the shared-prefix flight (via prefillFlightHook) until the follower has
// joined, so exact/prefix coalescing is exercised deterministically rather than
// by luck of goroutine scheduling. It returns the two lanes and the number of
// times the leader prefill hook ran (exactly one leader prefill is expected).
func prefixDedupAdmitPairGated(t *testing.T, s *NativeScheduler, toolA, toolB string) ([2]*schedLane, int64) {
	t.Helper()
	entered := make(chan struct{})
	release := make(chan struct{})
	var hookOnce sync.Once
	var hookCalls atomic.Int64
	s.prefillFlightHook = func() {
		hookCalls.Add(1)
		hookOnce.Do(func() { close(entered) })
		<-release
	}

	var (
		wg    sync.WaitGroup
		lanes [2]*schedLane
		errs  [2]error
	)
	admit := func(i int, tool string) {
		defer wg.Done()
		req, err := s.Admit(context.Background(), inlineCall(tool, `{}`))
		if err != nil {
			errs[i] = err
			return
		}
		lanes[i] = req.(*schedLane)
	}

	// Leader first: it registers the in-flight promise and blocks in the hook.
	wg.Add(1)
	go admit(0, toolA)
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("leader never entered the shared-prefix prefill flight")
	}

	// Follower second: it must join the leader's flight rather than lead.
	wg.Add(1)
	go admit(1, toolB)
	prefixDedupAwait(t, "follower to join the in-flight leader", func() bool {
		return s.prefixFlightGroup().Coalesced() >= 1
	})
	close(release)
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("admit %d: %v", i, err)
		}
	}
	return lanes, hookCalls.Load()
}

// TestNativeSchedulerInBatchPrefixDedupExactCoalesces proves spec clauses 2 and 3:
// with dedup enabled, two concurrent cold admissions of an identical >=32-token
// prompt run exactly ONE leader prefill, the twin joins as a follower, and the
// follower reuses the leader's exact KV and logits (ExactReuses).
//
// fak-test:runtime fast est=1100ms
func TestNativeSchedulerInBatchPrefixDedupExactCoalesces(t *testing.T) {
	m := nativeSchedulerPrefillModel(t)
	prompt := nativeSchedulerQwenPrompt(40) // >= nativeInBatchPrefixDedupMinShared (32)
	prepare := nativeSchedulerPrefillPrepare(map[string][]int{"a": prompt, "b": prompt})

	s := newNativeScheduler(m, prepare)
	s.SetInBatchPrefixDedup(true)
	nativeSchedulerBeginManualDrain(t, s)

	lanes, hookCalls := prefixDedupAdmitPairGated(t, s, "a", "b")

	if hookCalls != 1 {
		t.Fatalf("leader prefill ran %d times, want exactly 1 (shared prefill computed once)", hookCalls)
	}
	stats := s.InBatchPrefixDedupStats()
	if stats.Leaders != 1 || stats.Followers < 1 {
		t.Fatalf("dedup stats = %+v, want Leaders=1 Followers>=1", stats)
	}
	if stats.ExactReuses < 1 || stats.CoalescedPrefills != stats.ExactReuses+stats.PrefixReuses {
		t.Fatalf("dedup stats = %+v, want ExactReuses>=1 and CoalescedPrefills==ExactReuses+PrefixReuses", stats)
	}

	// Both lanes adopted a full-length cache: the leader computed it, the twin
	// adopted the leader's cloned KV rather than recomputing it.
	for i, ln := range lanes {
		if ln.sess == nil || ln.sess.Cache == nil || ln.sess.Cache.Len() != len(prompt) {
			t.Fatalf("lane %d cache = %v, want full prompt length %d", i, ln.sess.Cache, len(prompt))
		}
	}

	outs := prefixDedupDrainAll(t, s, lanes[0], lanes[1])
	for i, ln := range lanes {
		if len(outs[i]) != genTokens {
			t.Fatalf("lane %d emitted %d tokens, want %d", i, len(outs[i]), genTokens)
		}
		res, err := ln.Result()
		if err != nil || res == nil || res.Status != abi.StatusOK {
			t.Fatalf("lane %d result = %+v err = %v, want OK", i, res, err)
		}
	}
	if !reflect.DeepEqual(outs[0], outs[1]) {
		t.Fatalf("exact twins diverged: leader=%v follower=%v", outs[0], outs[1])
	}
	nativeSchedulerEndManualDrain(s)
}

// TestNativeSchedulerInBatchPrefixDedupPrefixSuffixReuse proves spec clause 4:
// two prompt twins that share a >=32-token prefix but diverge in the suffix make
// the follower adopt the leader's prefix KV and prefill only its divergent suffix
// (PrefixReuses). It also proves fusion-independence for that path: the suffix
// prefill on the adopted prefix yields the same last-prompt-token logits as an
// independent full cold prefill of the same prompt.
//
// fak-test:runtime fast est=1100ms
func TestNativeSchedulerInBatchPrefixDedupPrefixSuffixReuse(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig()) // full-attention cache: suffix truncation is supported
	common := nativeSchedulerQwenPrompt(40)    // >= 32 shared tokens
	promptA := append(append([]int(nil), common...), 7, 11, 13)
	promptB := append(append([]int(nil), common...), 9, 17, 19, 23)
	prepare := prefixDedupPlainPrepare(map[string][]int{"a": promptA, "b": promptB})

	s := newNativeScheduler(m, prepare)
	s.SetInBatchPrefixDedup(true)
	nativeSchedulerBeginManualDrain(t, s)

	lanes, hookCalls := prefixDedupAdmitPairGated(t, s, "a", "b")
	if hookCalls != 1 {
		t.Fatalf("shared-prefix leader prefill ran %d times, want exactly 1", hookCalls)
	}
	stats := s.InBatchPrefixDedupStats()
	if stats.Leaders != 1 || stats.Followers != 1 {
		t.Fatalf("dedup stats = %+v, want Leaders=1 Followers=1", stats)
	}
	if stats.PrefixReuses != 1 || stats.ExactReuses != 0 || stats.CoalescedPrefills != 1 {
		t.Fatalf("dedup stats = %+v, want PrefixReuses=1 ExactReuses=0 CoalescedPrefills=1", stats)
	}
	// The follower's cache ends at its own full prompt length: it adopted the
	// prefix clone then prefilled only the divergent suffix.
	if got, want := lanes[1].sess.Cache.Len(), len(promptB); got != want {
		t.Fatalf("prefix-reuse follower cache length = %d, want %d", got, want)
	}

	// Fusion-independence: run the same prompts with dedup OFF and compare the
	// follower's post-admission (prefill) logits against the coalesced run.
	offS := newNativeScheduler(m, prepare)
	nativeSchedulerBeginManualDrain(t, offS)
	offA := nativeSchedulerAdmitLane(t, offS, "a")
	offB := nativeSchedulerAdmitLane(t, offS, "b")
	nativeSchedulerAssertLogitParity(t, lanes[1].logits, offB.logits)

	outs := prefixDedupDrainAll(t, s, lanes[0], lanes[1])
	for i := range outs {
		if len(outs[i]) != genTokens {
			t.Fatalf("dedup lane %d emitted %d tokens, want %d", i, len(outs[i]), genTokens)
		}
	}
	offOuts := prefixDedupDrainAll(t, offS, offA, offB)
	if !reflect.DeepEqual(outs[1], offOuts[1]) {
		t.Fatalf("prefix-reuse follower tokens = %v, independent cold tokens = %v", outs[1], offOuts[1])
	}
	nativeSchedulerEndManualDrain(s)
	nativeSchedulerEndManualDrain(offS)
}

// TestNativeSchedulerInBatchPrefixDedupShortPromptsDoNotCoalesce proves spec
// clause 2's negative: a prompt below the 32-token floor, and two prompts that
// share fewer than 32 tokens, must NOT coalesce (CoalescedPrefills==0) and must
// still decode correctly. The first pair fails eligibility; the second pair is
// eligible by length but the flight group refuses the <32-token shared prefix.
//
// fak-test:runtime fast est=250ms
func TestNativeSchedulerInBatchPrefixDedupShortPromptsDoNotCoalesce(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig()) // plain full-attention: isolate the 32-token threshold

	short := nativeSchedulerQwenPrompt(20) // < 32
	divA := nativeSchedulerQwenPrompt(40)
	divB := append(append([]int(nil), divA[:10]...), nativeSchedulerQwenPrompt(40)[10:]...) // share only 10
	if len(divB) != 40 {
		t.Fatalf("divergent fixture length = %d, want 40", len(divB))
	}

	prepare := prefixDedupPlainPrepare(map[string][]int{
		"short-a": short,
		"short-b": short,
		"div-a":   divA,
		"div-b":   divB,
	})
	s := newNativeScheduler(m, prepare)
	s.SetInBatchPrefixDedup(true)
	nativeSchedulerBeginManualDrain(t, s)

	shortPair := prefixDedupAdmitSequential(t, s, "short-a", "short-b")
	divPair := prefixDedupAdmitSequential(t, s, "div-a", "div-b")

	stats := s.InBatchPrefixDedupStats()
	if stats.CoalescedPrefills != 0 || stats.ExactReuses != 0 || stats.PrefixReuses != 0 {
		t.Fatalf("short/divergent admissions coalesced = %+v, want all zero coalescing", stats)
	}

	all := append(append([]*schedLane(nil), shortPair...), divPair...)
	outs := prefixDedupDrainAll(t, s, all...)
	for i, ln := range all {
		if len(outs[i]) != genTokens {
			t.Fatalf("lane %d emitted %d tokens, want %d", i, len(outs[i]), genTokens)
		}
		if res, err := ln.Result(); err != nil || res == nil || res.Status != abi.StatusOK {
			t.Fatalf("lane %d result = %+v err = %v, want OK", i, res, err)
		}
	}
	nativeSchedulerEndManualDrain(s)
}

// TestNativeSchedulerInBatchPrefixDedupDecodeFusionIndependent proves spec
// clause 5 end to end: for the same prompts, enabling dedup (with a forced
// leader/twin coalesce) produces byte-identical generated-token sequences and
// identical last-prompt-token logits as dedup disabled. This is the
// fusion/coalescing-independence witness under temp-0 deterministic decoding.
//
// fak-test:runtime fast est=500ms
func TestNativeSchedulerInBatchPrefixDedupDecodeFusionIndependent(t *testing.T) {
	m := nativeSchedulerPrefillModel(t)
	prompt := nativeSchedulerQwenPrompt(48)
	prepare := nativeSchedulerPrefillPrepare(map[string][]int{"a": prompt, "b": prompt})

	// Baseline: dedup never enabled.
	offS := newNativeScheduler(m, prepare)
	nativeSchedulerBeginManualDrain(t, offS)
	offLanes := prefixDedupAdmitSequential(t, offS, "a", "b")
	offPre := [2][]float32{copyF32(offLanes[0].logits), copyF32(offLanes[1].logits)}
	offOuts := prefixDedupDrainAll(t, offS, offLanes[0], offLanes[1])
	nativeSchedulerEndManualDrain(offS)

	// Dedup enabled, forced coalesce of the two identical twins.
	onS := newNativeScheduler(m, prepare)
	onS.SetInBatchPrefixDedup(true)
	nativeSchedulerBeginManualDrain(t, onS)
	onLanes, hookCalls := prefixDedupAdmitPairGated(t, onS, "a", "b")
	if hookCalls != 1 {
		t.Fatalf("leader prefill ran %d times, want exactly 1 under coalescing", hookCalls)
	}
	if stats := onS.InBatchPrefixDedupStats(); stats.ExactReuses < 1 {
		t.Fatalf("dedup stats = %+v, want ExactReuses>=1 for identical twins", stats)
	}
	onPre := [2][]float32{copyF32(onLanes[0].logits), copyF32(onLanes[1].logits)}
	onOuts := prefixDedupDrainAll(t, onS, onLanes[0], onLanes[1])
	nativeSchedulerEndManualDrain(onS)

	for i := 0; i < 2; i++ {
		if !reflect.DeepEqual(onOuts[i], offOuts[i]) {
			t.Fatalf("lane %d dedup tokens = %v, no-dedup tokens = %v", i, onOuts[i], offOuts[i])
		}
		nativeSchedulerAssertLogitParity(t, onPre[i], offPre[i])
	}
}

// prefixDedupAdmitSequential admits lanes one at a time through the ordinary
// synchronous path (no in-flight twin can exist between admissions).
func prefixDedupAdmitSequential(t *testing.T, s *NativeScheduler, tools ...string) []*schedLane {
	t.Helper()
	lanes := make([]*schedLane, len(tools))
	for i, tool := range tools {
		lanes[i] = nativeSchedulerAdmitLane(t, s, tool)
	}
	return lanes
}
