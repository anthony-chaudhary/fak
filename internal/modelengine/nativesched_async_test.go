package modelengine

// fak-test:runtime fast est=1s

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeSchedulerAsyncPlaceholder(t *testing.T) {
	syncOracle := func(m *model.Model, prompt []int, budget int) []int {
		sess := m.NewSession()
		defer sess.Close()
		logits := sess.Prefill(prompt)
		out := make([]int, 0, budget)
		for len(out) < budget {
			next := argmax(logits)
			out = append(out, next)
			if m.Cfg.IsEOS(next) || len(out) >= budget {
				break
			}
			logits = sess.Step(next)
		}
		return out
	}

	t.Run("two_steps_overlap_with_placeholder_and_match_sync", func(t *testing.T) {
		m := model.NewSynthetic(SyntheticConfig())
		prompt := []int{1, 2, 3}
		want := syncOracle(m, prompt, genTokens)
		if len(want) < 3 {
			t.Fatalf("oracle generated %d tokens, need at least 3 for overlap test", len(want))
		}

		s := NewNativeScheduler(m)
		s.SetAsyncScheduling(true)
		nativeSchedulerBeginManualDrain(t, s)
		t.Cleanup(func() { nativeSchedulerEndManualDrain(s) })

		s.prepare = func(_ context.Context, _ *abi.ToolCall, _ *model.Model) schedPrepare {
			return schedPrepare{prompt: append([]int(nil), prompt...)}
		}
		ln := nativeSchedulerAdmitLane(t, s, "async-overlap")

		pubEntered := make(chan struct{})
		pubRelease := make(chan struct{})
		var pubReturned atomic.Bool
		var pubOnce sync.Once
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventDecode && event.Lane == ln && event.Token == want[0] {
				pubOnce.Do(func() { close(pubEntered) })
				<-pubRelease
				pubReturned.Store(true)
			}
		}

		var forwardCount atomic.Int64
		var step2Observed struct {
			beforePubReturned bool
			stagedID          int
			placeholders      int
			processedSeq      uint64
		}
		s.beforeModelExecute = func(kind nativeSchedulerEventKind, lane *schedLane) {
			if kind != nativeSchedulerEventDecode || lane != ln {
				return
			}
			n := forwardCount.Add(1)
			if n == 2 {
				step2Observed.beforePubReturned = !pubReturned.Load()
				s.mu.Lock()
				if lane.deferredToken != nil {
					step2Observed.stagedID = lane.deferredToken.id
				}
				step2Observed.placeholders = lane.numOutputPlaceholders
				step2Observed.processedSeq = s.processedStepSeq
				s.mu.Unlock()
			}
		}

		// Step 1: launches forward 1 and begins publication on separate stage.
		nativeSchedulerDriveIteration(t, s)
		select {
		case <-pubEntered:
		case <-time.After(3 * time.Second):
			t.Fatal("step 1 never entered async publication stage")
		}

		// Step 2: prepares with -1 placeholder and starts forward 2 BEFORE
		// step 1's publication returns.
		nativeSchedulerDriveIteration(t, s)

		if !step2Observed.beforePubReturned {
			t.Fatal("step 2 forward did not start before step 1 publication returned")
		}
		if step2Observed.stagedID != asyncOutputPlaceholderToken || step2Observed.placeholders != 1 {
			t.Fatalf("step 2 forward observed stagedID=%d placeholders=%d, want -1 / 1",
				step2Observed.stagedID, step2Observed.placeholders)
		}
		if step2Observed.processedSeq != 0 {
			t.Fatalf("step 2 forward saw processedStepSeq=%d, want 0 (step 1 pub in flight)", step2Observed.processedSeq)
		}

		// Release step 1 publication and drive remaining tokens to completion.
		close(pubRelease)

		var got []int
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			got = append(got, nativeSchedulerDrainAvailable(ln)...)
			s.mu.Lock()
			term := ln.terminal
			s.mu.Unlock()
			if term {
				break
			}
			didWork, _, _ := nativeSchedulerDriveIteration(t, s)
			if !didWork {
				time.Sleep(time.Millisecond)
			}
		}
		got = append(got, nativeSchedulerDrainAvailable(ln)...)
		s.mu.Lock()
		term := ln.terminal
		s.mu.Unlock()
		if !term {
			t.Fatalf("lane did not complete; got=%v", got)
		}
		res, err := ln.Result()
		if err != nil || res == nil || res.Status != abi.StatusOK {
			t.Fatalf("Result = (%v, %v), want StatusOK", res, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("async tokens = %v, want sync parity %v", got, want)
		}
		s.mu.Lock()
		remPlaceholders := ln.numOutputPlaceholders
		s.mu.Unlock()
		if remPlaceholders != 0 {
			t.Fatalf("remaining numOutputPlaceholders = %d, want 0", remPlaceholders)
		}
		stats := s.AsyncSchedStats()
		if stats.StepsOverlapped < 1 || stats.PlaceholdersStaged < 1 || stats.PlaceholdersResolved < 1 {
			t.Fatalf("async stats = %+v, want >=1 overlapped, staged, resolved", stats)
		}
	})

	t.Run("stop_truncated_tail_resolves_placeholders_and_frees_right_blocks", func(t *testing.T) {
		m := model.NewSynthetic(SyntheticConfig())
		prompt := []int{1, 2, 3}
		want := syncOracle(m, prompt, 2)
		if len(want) != 2 {
			t.Fatalf("oracle len = %d, want 2", len(want))
		}

		s := NewNativeScheduler(m)
		var err error
		s.elastic, err = newNativeElasticLedger(16, 1) // 1 token per block
		if err != nil {
			t.Fatalf("newNativeElasticLedger: %v", err)
		}
		s.SetAsyncScheduling(true)
		nativeSchedulerBeginManualDrain(t, s)
		t.Cleanup(func() { nativeSchedulerEndManualDrain(s) })

		var capturedCacheLen int
		var sessionCloses atomic.Int64
		s.closeSession = func(sess *model.Session) {
			sessionCloses.Add(1)
			if sess != nil && sess.Cache != nil {
				capturedCacheLen = sess.Cache.Len()
			}
			sess.Close()
		}

		s.prepare = func(_ context.Context, _ *abi.ToolCall, _ *model.Model) schedPrepare {
			return schedPrepare{prompt: append([]int(nil), prompt...)}
		}
		ln := nativeSchedulerAdmitLane(t, s, "stop-trunc")
		s.mu.Lock()
		ln.genBudget = 2 // token 1 (0-based) is terminal stop boundary
		s.mu.Unlock()

		pubEntered := make(chan struct{})
		pubRelease := make(chan struct{})
		var pubOnce sync.Once
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventDecode && event.Lane == ln && event.Token == want[0] {
				pubOnce.Do(func() { close(pubEntered) })
				<-pubRelease
			}
		}

		// Step 1 launches forward 1 (Cache.Len = 4, blocks = 4) and enters pub.
		nativeSchedulerDriveIteration(t, s)
		select {
		case <-pubEntered:
		case <-time.After(3 * time.Second):
			t.Fatal("step 1 did not enter publication")
		}

		// Step 2 overlaps with -1 placeholder and grows Cache.Len to 5 (spec tail block!).
		nativeSchedulerDriveIteration(t, s)
		if es := s.elastic.snapshot(); es.Allocator.AllocatedBlocks != 5 {
			t.Fatalf("mid-overlap AllocatedBlocks = %d, want 5 (3 prompt + 1 step1 + 1 speculative step2)", es.Allocator.AllocatedBlocks)
		}

		// Release step 1 publication: token 2 hits budget=2 -> step 2's tail is
		// stop-truncated, placeholders resolve to 0, Cache rolls back to 4, and
		// fenced tail + lane blocks release cleanly once step 2's fence clears.
		// Capture pubDone before draining so we can wait for step 2's fence drain
		// to complete after ln.Tokens() closes.
		s.mu.Lock()
		pubDone := s.asyncPubDone
		s.mu.Unlock()
		close(pubRelease)

		var got []int
		for tok := range ln.Tokens() {
			got = append(got, tok.ID)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("emitted token stream = %v, want stop-bounded %v", got, want)
		}
		if pubDone != nil {
			select {
			case <-pubDone:
			case <-time.After(3 * time.Second):
				t.Fatal("timed out waiting for async publication drain")
			}
		}
		res, err := ln.Result()
		if err != nil || res == nil || res.Status != abi.StatusOK {
			t.Fatalf("Result = (%v, %v), want StatusOK", res, err)
		}
		if !ln.Reclaimed() || sessionCloses.Load() != 1 {
			t.Fatalf("reclaimed=%t closes=%d, want true/1", ln.Reclaimed(), sessionCloses.Load())
		}
		if capturedCacheLen != len(prompt)+1 {
			t.Fatalf("KV cache len at finish = %d, want %d (speculative tail rolled back)", capturedCacheLen, len(prompt)+1)
		}
		s.mu.Lock()
		gen := append([]int(nil), ln.gen...)
		placeholders := ln.numOutputPlaceholders
		truncated := ln.asyncStopTruncated
		s.mu.Unlock()
		if !reflect.DeepEqual(gen, want) {
			t.Fatalf("generated tokens = %v, want stop-bounded %v", gen, want)
		}
		if placeholders != 0 || !truncated {
			t.Fatalf("placeholders=%d truncated=%t, want 0/true", placeholders, truncated)
		}
		if stats := s.AsyncSchedStats(); stats.StopTruncations != 1 || stats.PlaceholdersResolved != 1 {
			t.Fatalf("async stats = %+v, want StopTruncations=1 PlaceholdersResolved=1", stats)
		}
		df := s.DeferredFreeStats()
		es := s.elastic.snapshot()
		if df.PendingEntries != 0 || es.Allocator.AllocatedBlocks != 0 || es.CachedReleases != es.AllocatedBlocks {
			t.Fatalf("block accounting leaked: deferred=%+v elastic=%+v", df, es)
		}
		if err := s.elastic.alloc.VerifyInvariant(); err != nil {
			t.Fatalf("VerifyInvariant: %v", err)
		}
	})

	t.Run("close_mid_overlap_retires_both_steps_without_double_free", func(t *testing.T) {
		m := model.NewSynthetic(SyntheticConfig())
		s := NewNativeScheduler(m)
		var err error
		s.elastic, err = newNativeElasticLedger(16, 1)
		if err != nil {
			t.Fatalf("newNativeElasticLedger: %v", err)
		}
		s.SetAsyncScheduling(true)

		var closesMu sync.Mutex
		closes := make(map[*model.Session]int)
		s.closeSession = func(sess *model.Session) {
			closesMu.Lock()
			closes[sess]++
			closesMu.Unlock()
			sess.Close()
		}

		manualActive := true
		nativeSchedulerBeginManualDrain(t, s)
		defer func() {
			if manualActive {
				nativeSchedulerEndManualDrain(s)
			}
		}()

		s.prepare = func(_ context.Context, _ *abi.ToolCall, _ *model.Model) schedPrepare {
			return schedPrepare{prompt: []int{1, 2, 3}}
		}
		ln := nativeSchedulerAdmitLane(t, s, "close-mid-overlap")
		sess := ln.sess

		pubEntered := make(chan struct{})
		var pubOnce sync.Once
		s.observeNativeEvent = func(event nativeSchedulerEvent) {
			if event.Kind == nativeSchedulerEventDecode && event.Lane == ln {
				pubOnce.Do(func() { close(pubEntered) })
				// Unblock cleanly on lane cancellation when s.Close() runs.
				<-ln.ctx.Done()
			}
		}

		// Step 1 enters publication and blocks on ctx.Done().
		nativeSchedulerDriveIteration(t, s)
		select {
		case <-pubEntered:
		case <-time.After(3 * time.Second):
			t.Fatal("step 1 never entered publication")
		}

		// Step 2 overlaps while step 1 publication is blocked -> both steps active!
		nativeSchedulerDriveIteration(t, s)

		s.mu.Lock()
		pubActive := s.asyncPubActive
		hasPending := s.asyncPubPending != nil
		s.mu.Unlock()
		if !pubActive || !hasPending {
			t.Fatalf("mid-overlap state pubActive=%t hasPending=%t, want true/true", pubActive, hasPending)
		}

		// End manual drain ownership so s.run() can exit once Close cancels lanes.
		manualActive = false
		s.Close()
		s.endBlockedDrain()

		// Close mid-overlap: both in-flight steps must retire cleanly without double-free.
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if err := s.CloseAndWait(ctx); err != nil {
			t.Fatalf("CloseAndWait: %v", err)
		}

		if !ln.Reclaimed() {
			t.Fatal("lane was not reclaimed on mid-overlap Close")
		}
		if _, err := ln.Result(); !errors.Is(err, context.Canceled) && !errors.Is(err, errSchedClosed) {
			t.Fatalf("Result err = %v, want cancellation or sched closed", err)
		}
		closesMu.Lock()
		sessCloseCount := closes[sess]
		totalSessions := len(closes)
		closesMu.Unlock()
		if sessCloseCount != 1 || totalSessions != 1 {
			t.Fatalf("session closes = %d across %d sessions, want exactly 1 close (no double-free)", sessCloseCount, totalSessions)
		}
		df := s.DeferredFreeStats()
		es := s.elastic.snapshot()
		if df.PendingEntries != 0 || df.PendingBlocks != 0 || es.Allocator.AllocatedBlocks != 0 || es.CachedReleases != es.AllocatedBlocks {
			t.Fatalf("mid-overlap Close leaked blocks: deferred=%+v elastic=%+v", df, es)
		}
		if err := s.elastic.alloc.VerifyInvariant(); err != nil {
			t.Fatalf("VerifyInvariant: %v", err)
		}
	})

	t.Run("live_loop_multilane_parity_and_clean_retire", func(t *testing.T) {
		m := model.NewSynthetic(SyntheticConfig())
		prompt0 := []int{1, 2, 3}
		prompt1 := []int{4, 5, 6}
		want0 := syncOracle(m, prompt0, genTokens)
		want1 := syncOracle(m, prompt1, genTokens)

		s := NewNativeScheduler(m)
		var err error
		s.elastic, err = newNativeElasticLedger(16, 2)
		if err != nil {
			t.Fatalf("newNativeElasticLedger: %v", err)
		}
		s.SetAsyncScheduling(true)
		s.prepare = func(_ context.Context, call *abi.ToolCall, _ *model.Model) schedPrepare {
			if call != nil && call.Tool == "lane-1" {
				return schedPrepare{prompt: append([]int(nil), prompt1...)}
			}
			return schedPrepare{prompt: append([]int(nil), prompt0...)}
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		defer func() { _ = s.CloseAndWait(ctx) }()

		req0, err := s.Admit(ctx, inlineCall("lane-0", `{}`))
		if err != nil {
			t.Fatalf("Admit lane-0: %v", err)
		}
		req1, err := s.Admit(ctx, inlineCall("lane-1", `{}`))
		if err != nil {
			t.Fatalf("Admit lane-1: %v", err)
		}

		collect := func(req abi.EngineRequest) ([]int, *abi.Result, error) {
			var got []int
			for tok := range req.Tokens() {
				got = append(got, tok.ID)
			}
			res, rerr := req.Result()
			return got, res, rerr
		}

		var got0, got1 []int
		var res0, res1 *abi.Result
		var err0, err1 error
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			got0, res0, err0 = collect(req0)
		}()
		go func() {
			defer wg.Done()
			got1, res1, err1 = collect(req1)
		}()
		wg.Wait()

		if err0 != nil || res0 == nil || res0.Status != abi.StatusOK {
			t.Fatalf("lane-0 Result = (%v, %v), want StatusOK", res0, err0)
		}
		if err1 != nil || res1 == nil || res1.Status != abi.StatusOK {
			t.Fatalf("lane-1 Result = (%v, %v), want StatusOK", res1, err1)
		}
		if !reflect.DeepEqual(got0, want0) {
			t.Fatalf("lane-0 tokens = %v, want %v", got0, want0)
		}
		if !reflect.DeepEqual(got1, want1) {
			t.Fatalf("lane-1 tokens = %v, want %v", got1, want1)
		}
		if err := s.CloseAndWait(ctx); err != nil {
			t.Fatalf("CloseAndWait: %v", err)
		}
		df := s.DeferredFreeStats()
		es := s.elastic.snapshot()
		if df.PendingEntries != 0 || df.PendingBlocks != 0 || es.Allocator.AllocatedBlocks != 0 || es.CachedReleases != es.AllocatedBlocks {
			t.Fatalf("live loop leaked blocks: deferred=%+v elastic=%+v", df, es)
		}
		if err := s.elastic.alloc.VerifyInvariant(); err != nil {
			t.Fatalf("VerifyInvariant: %v", err)
		}
	})
}
