package observer

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestClosedVocabularyAndWitnessVerdicts(t *testing.T) {
	t.Run("StepVerdicts", func(t *testing.T) {
		for _, v := range []StepVerdict{STEP_ADVANCE, STEP_CHURN, STEP_REGRESS} {
			if !v.IsValid() {
				t.Fatalf("expected verdict %s to be valid", v)
			}
		}
		if StepVerdict("STEP_UNKNOWN").IsValid() {
			t.Fatal("expected arbitrary verdict to be invalid")
		}
	})

	t.Run("WitnessVerdicts", func(t *testing.T) {
		for _, w := range []WitnessVerdict{WITNESS_DIFF_CONFIRMED, WITNESS_UNWITNESSED_CLAIM} {
			if !w.IsValid() {
				t.Fatalf("expected witness %s to be valid", w)
			}
		}
		if WitnessVerdict("WITNESS_UNKNOWN").IsValid() {
			t.Fatal("expected arbitrary witness to be invalid")
		}
	})
}

func TestAsyncShadowingReadOnly(t *testing.T) {
	cfg := Config{
		WorkerCount:          4,
		QueueSize:            128,
		BarrierTimeout:       20 * time.Millisecond,
		MaxHistoryPerSession: 20,
		SimulateKVCache:      true,
	}
	p := NewPool(cfg)
	if err := p.Start(); err != nil {
		t.Fatalf("failed to start pool: %v", err)
	}
	defer p.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	readTools := []string{"Read", "Grep", "Glob"}
	sessionID := "sess-readonly-1"

	for i, tool := range readTools {
		obs := StepObservation{
			SessionID: sessionID,
			Tool:      tool,
			Args:      map[string]any{"path": fmt.Sprintf("/src/file%d.go", i)},
			Result:    "found content",
		}

		start := time.Now()
		ch := p.ObserveAsync(ctx, obs)
		dispatchLatency := time.Since(start)

		// Near-zero latency: async dispatch must not block caller (<1ms, effectively 0ms)
		if dispatchLatency > 1*time.Millisecond {
			t.Fatalf("async dispatch took %s, expected <1ms near-zero latency", dispatchLatency)
		}

		select {
		case res, ok := <-ch:
			if !ok {
				t.Fatalf("channel closed prematurely for tool %s", tool)
			}
			if res.StepVerdict != STEP_ADVANCE {
				t.Fatalf("expected STEP_ADVANCE, got %s", res.StepVerdict)
			}
			if res.WitnessVerdict != WITNESS_DIFF_CONFIRMED {
				t.Fatalf("expected WITNESS_DIFF_CONFIRMED, got %s", res.WitnessVerdict)
			}
			// Turn 0 is cold prefix; turn 1+ should hit warm KV-cache prefix reuse
			if i == 0 && res.CachedPrefix {
				t.Fatalf("turn 0 expected cold cache prefix, got warm")
			}
			if i > 0 && !res.CachedPrefix {
				t.Fatalf("turn %d expected warm KV-cache prefix hit, got cold", i)
			}
		case <-time.After(500 * time.Millisecond):
			t.Fatalf("timed out waiting for async observation on tool %s", tool)
		}
	}

	total, asyncCount, _, hits, misses, _, _ := p.Stats()
	if total != 3 || asyncCount != 3 {
		t.Fatalf("expected 3 total async observations, got total=%d, async=%d", total, asyncCount)
	}
	if hits < 2 || misses < 1 {
		t.Fatalf("expected KV cache prefix hits>=2, misses>=1, got hits=%d, misses=%d", hits, misses)
	}
}

func TestHardSeamPromotionMutating(t *testing.T) {
	cfg := Config{
		WorkerCount:    4,
		QueueSize:      128,
		BarrierTimeout: 50 * time.Millisecond,
	}
	p := NewPool(cfg)
	if err := p.Start(); err != nil {
		t.Fatalf("failed to start pool: %v", err)
	}
	defer p.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	mutatingTools := []struct {
		tool string
		diff string
	}{
		{"Edit", "--- a/main.go\n+++ b/main.go\n@@ -1 +1 @@\n-old\n+new"},
		{"Write", "wrote 42 bytes to /path/to/file.go"},
		{"git commit", "[main 1a2b3c] fix: resolve issue\n 1 file changed, 1 insertion(+)"},
	}

	for _, tc := range mutatingTools {
		obs := StepObservation{
			SessionID: "sess-mutating",
			Tool:      tc.tool,
			Diff:      tc.diff,
			Result:    tc.diff,
		}

		res, err := p.ObserveSyncBarrier(ctx, obs)
		if err != nil {
			t.Fatalf("unexpected error for %s: %v", tc.tool, err)
		}
		if res.StepVerdict != STEP_ADVANCE {
			t.Fatalf("expected STEP_ADVANCE for %s, got %s", tc.tool, res.StepVerdict)
		}
		if res.WitnessVerdict != WITNESS_DIFF_CONFIRMED {
			t.Fatalf("expected WITNESS_DIFF_CONFIRMED for %s, got %s", tc.tool, res.WitnessVerdict)
		}

		// Hard-seam promotion requirement: <2ms barrier latency
		if res.BarrierLatency > 2*time.Millisecond {
			t.Fatalf("barrier latency for %s was %s, expected <2ms", tc.tool, res.BarrierLatency)
		}
	}

	// Also verify that calling ObserveAsync with a mutating tool automatically promotes to barrier
	obsAsyncMut := StepObservation{
		SessionID: "sess-mutating-async",
		Tool:      "Edit",
		Diff:      "--- a/foo.go\n+++ b/foo.go\n@@ -1 +1 @@\n+added",
	}
	ch := p.ObserveAsync(ctx, obsAsyncMut)
	select {
	case res := <-ch:
		if res.StepVerdict != STEP_ADVANCE {
			t.Fatalf("expected promoted STEP_ADVANCE, got %s", res.StepVerdict)
		}
		if res.BarrierLatency > 2*time.Millisecond {
			t.Fatalf("promoted barrier latency was %s, expected <2ms", res.BarrierLatency)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for promoted mutating observation")
	}
}

func TestDeterministicRefusalChurnLoop(t *testing.T) {
	cfg := Config{
		WorkerCount:      4,
		QueueSize:        64,
		ChurnThreshold:   3,
		RegressThreshold: 5,
	}
	p := NewPool(cfg)
	if err := p.Start(); err != nil {
		t.Fatalf("failed to start pool: %v", err)
	}
	defer p.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	t.Run("RepeatedErrors", func(t *testing.T) {
		sessionID := "sess-churn-errors"
		for i := 1; i <= 2; i++ {
			obs := StepObservation{
				SessionID: sessionID,
				Tool:      "Edit",
				Error:     "oldString not found in content",
			}
			res, err := p.ObserveSyncBarrier(ctx, obs)
			if err != nil {
				t.Fatalf("turn %d unexpectedly failed: %v", i, err)
			}
			if res.StepVerdict != STEP_ADVANCE {
				t.Fatalf("turn %d expected STEP_ADVANCE before threshold, got %s", i, res.StepVerdict)
			}
		}

		obsChurn := StepObservation{
			SessionID: sessionID,
			Tool:      "Edit",
			Error:     "oldString not found in content",
		}
		resChurn, err := p.ObserveSyncBarrier(ctx, obsChurn)
		if !errors.Is(err, ErrChurnRefused) {
			t.Fatalf("expected ErrChurnRefused, got %v", err)
		}
		if resChurn.StepVerdict != STEP_CHURN {
			t.Fatalf("expected STEP_CHURN, got %s", resChurn.StepVerdict)
		}
	})

	t.Run("RepeatedQueries", func(t *testing.T) {
		sessionID := "sess-query-loop"
		for i := 1; i <= 2; i++ {
			obs := StepObservation{
				SessionID: sessionID,
				Tool:      "Grep",
				Args:      "pattern=samePattern",
				Result:    "no matches",
			}
			res, err := p.ObserveSyncBarrier(ctx, obs)
			if err != nil {
				t.Fatalf("turn %d unexpectedly failed: %v", i, err)
			}
			if res.StepVerdict != STEP_ADVANCE {
				t.Fatalf("turn %d expected STEP_ADVANCE, got %s", i, res.StepVerdict)
			}
		}

		obsQueryChurn := StepObservation{
			SessionID: sessionID,
			Tool:      "Grep",
			Args:      "pattern=samePattern",
			Result:    "no matches",
		}
		resQueryChurn, err := p.ObserveSyncBarrier(ctx, obsQueryChurn)
		if !errors.Is(err, ErrChurnRefused) {
			t.Fatalf("expected ErrChurnRefused on repeated query loop, got %v", err)
		}
		if resQueryChurn.StepVerdict != STEP_CHURN {
			t.Fatalf("expected STEP_CHURN, got %s", resQueryChurn.StepVerdict)
		}
	})
}

func TestDeterministicRefusalRegressLoop(t *testing.T) {
	cfg := Config{
		WorkerCount:      4,
		QueueSize:        64,
		ChurnThreshold:   3,
		RegressThreshold: 4,
	}
	p := NewPool(cfg)
	if err := p.Start(); err != nil {
		t.Fatalf("failed to start pool: %v", err)
	}
	defer p.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sessionID := "sess-regress"

	for i := 1; i <= 2; i++ {
		obs := StepObservation{
			SessionID: sessionID,
			Tool:      "bash",
			Error:     "command failed",
		}
		_, err := p.ObserveSyncBarrier(ctx, obs)
		if err != nil {
			t.Fatalf("turn %d failed: %v", i, err)
		}
	}

	obs3 := StepObservation{
		SessionID: sessionID,
		Tool:      "bash",
		Error:     "command failed",
	}
	_, err := p.ObserveSyncBarrier(ctx, obs3)
	if !errors.Is(err, ErrChurnRefused) {
		t.Fatalf("turn 3 expected ErrChurnRefused, got %v", err)
	}

	obs4 := StepObservation{
		SessionID: sessionID,
		Tool:      "bash",
		Error:     "command failed",
	}
	res4, err := p.ObserveSyncBarrier(ctx, obs4)
	if !errors.Is(err, ErrRegressRefused) {
		t.Fatalf("turn 4 expected ErrRegressRefused, got %v", err)
	}
	if res4.StepVerdict != STEP_REGRESS {
		t.Fatalf("expected STEP_REGRESS, got %s", res4.StepVerdict)
	}
	if res4.CachedPrefix {
		t.Fatal("expected cached prefix to be invalidated on STEP_REGRESS")
	}
}

func TestMutatingWitnessVerification(t *testing.T) {
	cfg := Config{
		WorkerCount:        4,
		QueueSize:          64,
		RequireWitnessDiff: true,
	}
	p := NewPool(cfg)
	if err := p.Start(); err != nil {
		t.Fatalf("failed to start pool: %v", err)
	}
	defer p.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Mutating tool with confirmed diff
	obsConfirmed := StepObservation{
		SessionID: "sess-witness",
		Tool:      "Edit",
		Diff:      "@@ -1,1 +1,1 @@\n-old\n+new",
	}
	resConfirmed, err := p.ObserveSyncBarrier(ctx, obsConfirmed)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resConfirmed.WitnessVerdict != WITNESS_DIFF_CONFIRMED {
		t.Fatalf("expected WITNESS_DIFF_CONFIRMED, got %s", resConfirmed.WitnessVerdict)
	}

	// Mutating tool with NO diff or empty result -> unwitnessed claim refusal
	obsUnwitnessed := StepObservation{
		SessionID: "sess-witness",
		Tool:      "Edit",
		Args:      map[string]any{"path": "file.go"},
		Result:    nil,
	}
	resUnwitnessed, err := p.ObserveSyncBarrier(ctx, obsUnwitnessed)
	if !errors.Is(err, ErrUnwitnessedDiff) {
		t.Fatalf("expected ErrUnwitnessedDiff, got %v", err)
	}
	if resUnwitnessed.WitnessVerdict != WITNESS_UNWITNESSED_CLAIM {
		t.Fatalf("expected WITNESS_UNWITNESSED_CLAIM, got %s", resUnwitnessed.WitnessVerdict)
	}
}

func TestConcurrencySafetyHighThroughput(t *testing.T) {
	cfg := Config{
		WorkerCount:          8,
		QueueSize:            512,
		BarrierTimeout:       100 * time.Millisecond,
		MaxHistoryPerSession: 50,
	}
	p := NewPool(cfg)
	if err := p.Start(); err != nil {
		t.Fatalf("failed to start pool: %v", err)
	}
	defer p.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	const numGoroutines = 50
	const opsPerGoroutine = 20

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(gID int) {
			defer wg.Done()
			sessionID := fmt.Sprintf("concurrent-sess-%d", gID%10)

			for i := 0; i < opsPerGoroutine; i++ {
				if i%4 == 0 {
					// Mutating barrier
					obs := StepObservation{
						SessionID: sessionID,
						Tool:      "Write",
						Args:      fmt.Sprintf("file_%d_%d.go", gID, i),
						Diff:      fmt.Sprintf("diff for worker %d turn %d", gID, i),
					}
					res, err := p.ObserveSyncBarrier(ctx, obs)
					if err != nil {
						t.Errorf("worker %d turn %d barrier error: %v", gID, i, err)
						return
					}
					if res.BarrierLatency > 20*time.Millisecond {
						t.Errorf("worker %d turn %d excessive latency: %s", gID, i, res.BarrierLatency)
						return
					}
				} else {
					// Read-only async
					obs := StepObservation{
						SessionID: sessionID,
						Tool:      "Read",
						Args:      fmt.Sprintf("path/file_%d_%d.go", gID, i),
					}
					ch := p.ObserveAsync(ctx, obs)
					select {
					case res, ok := <-ch:
						if !ok {
							t.Errorf("worker %d turn %d channel closed", gID, i)
							return
						}
						if !res.StepVerdict.IsValid() {
							t.Errorf("worker %d turn %d invalid verdict: %s", gID, i, res.StepVerdict)
							return
						}
					case <-ctx.Done():
						t.Errorf("worker %d turn %d context cancelled", gID, i)
						return
					}
				}
			}
		}(g)
	}

	wg.Wait()

	total, asyncCount, barriers, _, _, _, _ := p.Stats()
	expectedTotal := int64(numGoroutines * opsPerGoroutine)
	if total != expectedTotal {
		t.Fatalf("expected total %d observations, got %d (async=%d, barriers=%d)",
			expectedTotal, total, asyncCount, barriers)
	}
}

func TestLifecycle(t *testing.T) {
	p := NewPool(Config{WorkerCount: 2, QueueSize: 16})

	// Start idempotency
	if err := p.Start(); err != nil {
		t.Fatalf("first start failed: %v", err)
	}
	if err := p.Start(); err != nil {
		t.Fatalf("second start should be idempotent: %v", err)
	}

	// Stop idempotency
	if err := p.Stop(); err != nil {
		t.Fatalf("first stop failed: %v", err)
	}
	if err := p.Stop(); err != nil {
		t.Fatalf("second stop should be idempotent: %v", err)
	}

	// Close idempotency
	if err := p.Close(); err != nil {
		t.Fatalf("first close failed: %v", err)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("second close should be idempotent: %v", err)
	}

	// Refusal after close
	ctx := context.Background()
	_, err := p.ObserveSyncBarrier(ctx, StepObservation{Tool: "Read"})
	if !errors.Is(err, ErrPoolClosed) {
		t.Fatalf("expected ErrPoolClosed, got %v", err)
	}
}

func BenchmarkObserveAsyncReadOnly(b *testing.B) {
	p := NewPool(Config{WorkerCount: 8, QueueSize: 2048})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()
	obs := StepObservation{
		SessionID: "bench-async",
		Tool:      "Read",
		Args:      "file.go",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ch := p.ObserveAsync(ctx, obs)
		<-ch
	}
}

func BenchmarkObserveSyncBarrierMutating(b *testing.B) {
	p := NewPool(Config{WorkerCount: 8, QueueSize: 2048})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()
	obs := StepObservation{
		SessionID: "bench-barrier",
		Tool:      "Edit",
		Diff:      "@@ -1 +1 @@\n-old\n+new",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = p.ObserveSyncBarrier(ctx, obs)
	}
}

func TestClosedVocabularyStrictEnforcement(t *testing.T) {
	t.Run("StepVerdicts", func(t *testing.T) {
		svs := StepVerdicts()
		if len(svs) != 3 {
			t.Fatalf("expected 3 StepVerdicts, got %d", len(svs))
		}
		for _, sv := range svs {
			if !sv.IsValid() {
				t.Fatalf("expected StepVerdict %s to be valid", sv)
			}
		}
	})

	t.Run("ParseStepVerdict", func(t *testing.T) {
		validSteps := []string{"STEP_ADVANCE", "step_advance", "STEP_CHURN", "step_churn", "STEP_REGRESS", "step_regress"}
		for _, s := range validSteps {
			v, err := ParseStepVerdict(s)
			if err != nil {
				t.Fatalf("expected %q to parse successfully, got err: %v", s, err)
			}
			if !v.IsValid() {
				t.Fatalf("parsed verdict %s is invalid", v)
			}
		}

		invalidSteps := []string{"STEP_UNKNOWN", "UNKNOWN", "INVALID", "", "   ", "STEP_HALT"}
		for _, s := range invalidSteps {
			_, err := ParseStepVerdict(s)
			if !errors.Is(err, ErrInvalidStepVerdict) {
				t.Fatalf("expected ErrInvalidStepVerdict for %q, got: %v", s, err)
			}
		}
	})

	t.Run("WitnessVerdicts", func(t *testing.T) {
		wvs := WitnessVerdicts()
		if len(wvs) != 2 {
			t.Fatalf("expected 2 WitnessVerdicts, got %d", len(wvs))
		}
		for _, wv := range wvs {
			if !wv.IsValid() {
				t.Fatalf("expected WitnessVerdict %s to be valid", wv)
			}
		}
	})

	t.Run("ParseWitnessVerdict", func(t *testing.T) {
		validWitnesses := []string{"WITNESS_DIFF_CONFIRMED", "witness_diff_confirmed", "WITNESS_UNWITNESSED_CLAIM", "witness_unwitnessed_claim"}
		for _, s := range validWitnesses {
			w, err := ParseWitnessVerdict(s)
			if err != nil {
				t.Fatalf("expected %q to parse successfully, got err: %v", s, err)
			}
			if !w.IsValid() {
				t.Fatalf("parsed witness %s is invalid", w)
			}
		}

		invalidWitnesses := []string{"WITNESS_UNKNOWN", "UNKNOWN", "INVALID", "", "   ", "WITNESS_SPECULATIVE"}
		for _, s := range invalidWitnesses {
			_, err := ParseWitnessVerdict(s)
			if !errors.Is(err, ErrInvalidWitnessVerdict) {
				t.Fatalf("expected ErrInvalidWitnessVerdict for %q, got: %v", s, err)
			}
		}
	})

	t.Run("ObservationValidate", func(t *testing.T) {
		obsValid := StepObservation{
			StepVerdict:    STEP_ADVANCE,
			WitnessVerdict: WITNESS_DIFF_CONFIRMED,
		}
		if err := obsValid.Validate(); err != nil {
			t.Fatalf("expected valid observation to pass Validate(), got %v", err)
		}

		obsEmpty := StepObservation{}
		if err := obsEmpty.Validate(); err != nil {
			t.Fatalf("expected empty observation to pass Validate(), got %v", err)
		}

		obsBadStep := StepObservation{
			StepVerdict: StepVerdict("BAD_STEP"),
		}
		if err := obsBadStep.Validate(); !errors.Is(err, ErrInvalidStepVerdict) {
			t.Fatalf("expected ErrInvalidStepVerdict, got %v", err)
		}

		obsBadWitness := StepObservation{
			WitnessVerdict: WitnessVerdict("BAD_WITNESS"),
		}
		if err := obsBadWitness.Validate(); !errors.Is(err, ErrInvalidWitnessVerdict) {
			t.Fatalf("expected ErrInvalidWitnessVerdict, got %v", err)
		}
	})
}

func TestObserverSemanticScreen_InterfaceConformance(t *testing.T) {
	var _ abi.SemanticScreen = ObserverSemanticScreen{}
	var _ abi.SemanticScreen = (*ObserverSemanticScreen)(nil)

	screen := NewObserverSemanticScreen(nil)
	if screen == nil || screen.Pool() == nil {
		t.Fatal("expected non-nil screen and initialized pool")
	}
}

func TestObserverSemanticScreen_ScreenResult_MutatingUnwitnessed(t *testing.T) {
	pool := NewPool(Config{RequireWitnessDiff: true})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()

	// Mutating tool without diff witness in meta or body
	call := &abi.ToolCall{
		Tool:    "Edit",
		TraceID: "sess-screen-unwitnessed",
		Meta:    map[string]string{"args": `{"filePath": "main.go"}`},
	}
	body := []byte("replaced lines successfully")

	advice := screen.ScreenResult(ctx, call, body)
	if advice.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine for unwitnessed mutating claim, got %v", advice.Disposition)
	}
	if advice.Reason != abi.ReasonUnwitnessed {
		t.Fatalf("expected ReasonUnwitnessed, got %v", advice.Reason)
	}
	if advice.By != "observer:unwitnessed_claim" {
		t.Fatalf("expected By='observer:unwitnessed_claim', got %q", advice.By)
	}
}

func TestObserverSemanticScreen_ScreenResult_MutatingConfirmed(t *testing.T) {
	pool := NewPool(Config{RequireWitnessDiff: true})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()

	// Mutating tool with confirmed diff in metadata
	call := &abi.ToolCall{
		Tool:    "Edit",
		TraceID: "sess-screen-confirmed",
		Meta: map[string]string{
			"diff": "@@ -1,1 +1,1 @@\n-old\n+new",
		},
	}
	body := []byte("applied edit successfully")

	advice := screen.ScreenResult(ctx, call, body)
	if advice.Disposition != abi.ScreenAllow {
		t.Fatalf("expected ScreenAllow for confirmed mutating diff, got %v", advice.Disposition)
	}

	// Mutating tool with diff evidence directly in body
	call2 := &abi.ToolCall{
		Tool:    "git commit",
		TraceID: "sess-screen-confirmed-2",
	}
	body2 := []byte("[main a1b2c3d] fix: resolved defect\n 1 file changed, 2 insertions(+)")

	advice2 := screen.ScreenResult(ctx, call2, body2)
	if advice2.Disposition != abi.ScreenAllow {
		t.Fatalf("expected ScreenAllow for git commit diff body, got %v", advice2.Disposition)
	}
}

func TestObserverSemanticScreen_ScreenResult_ChurnLoop(t *testing.T) {
	pool := NewPool(Config{
		ChurnThreshold:   3,
		RegressThreshold: 5,
	})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()

	sessionID := "sess-screen-churn"
	call := &abi.ToolCall{
		Tool:    "Edit",
		TraceID: sessionID,
		Meta:    map[string]string{"error": "oldString not found"},
	}

	// Turns 1 and 2: under threshold, allow
	for i := 1; i <= 2; i++ {
		advice := screen.ScreenResult(ctx, call, []byte("error: oldString not found"))
		if advice.Disposition != abi.ScreenAllow {
			t.Fatalf("turn %d expected ScreenAllow, got %v", i, advice.Disposition)
		}
	}

	// Turn 3: reaches ChurnThreshold (3) -> ScreenQuarantine with ReasonTrustViolation
	adviceChurn := screen.ScreenResult(ctx, call, []byte("error: oldString not found"))
	if adviceChurn.Disposition != abi.ScreenQuarantine {
		t.Fatalf("turn 3 expected ScreenQuarantine for churn loop, got %v", adviceChurn.Disposition)
	}
	if adviceChurn.Reason != abi.ReasonTrustViolation {
		t.Fatalf("expected ReasonTrustViolation, got %v", adviceChurn.Reason)
	}
	if adviceChurn.By != "observer:step_churn" {
		t.Fatalf("expected By='observer:step_churn', got %q", adviceChurn.By)
	}
}

func TestObserverSemanticScreen_ScreenResult_RegressionLoop(t *testing.T) {
	pool := NewPool(Config{
		ChurnThreshold:   2,
		RegressThreshold: 3,
	})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()

	sessionID := "sess-screen-regress"
	call := &abi.ToolCall{
		Tool:    "bash",
		TraceID: sessionID,
		Meta:    map[string]string{"error": "command exit 1"},
	}

	// Turn 1: allow
	adv1 := screen.ScreenResult(ctx, call, []byte("command exit 1"))
	if adv1.Disposition != abi.ScreenAllow {
		t.Fatalf("turn 1 expected ScreenAllow, got %v", adv1.Disposition)
	}

	// Turn 2: churn
	adv2 := screen.ScreenResult(ctx, call, []byte("command exit 1"))
	if adv2.Disposition != abi.ScreenQuarantine || adv2.Reason != abi.ReasonTrustViolation {
		t.Fatalf("turn 2 expected churn quarantine, got disp=%v reason=%v", adv2.Disposition, adv2.Reason)
	}

	// Turn 3: reaches RegressThreshold (3) -> ScreenQuarantine with ReasonIntegrityRefuted
	adv3 := screen.ScreenResult(ctx, call, []byte("command exit 1"))
	if adv3.Disposition != abi.ScreenQuarantine {
		t.Fatalf("turn 3 expected ScreenQuarantine for regression loop, got %v", adv3.Disposition)
	}
	if adv3.Reason != abi.ReasonIntegrityRefuted {
		t.Fatalf("expected ReasonIntegrityRefuted, got %v", adv3.Reason)
	}
	if adv3.By != "observer:step_regress" {
		t.Fatalf("expected By='observer:step_regress', got %q", adv3.By)
	}
}

func TestObserverSemanticScreen_ScreenResult_ReadOnly(t *testing.T) {
	pool := NewPool(Config{RequireWitnessDiff: true})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()

	tools := []string{"Read", "Grep", "Glob"}
	for _, tool := range tools {
		call := &abi.ToolCall{
			Tool:    tool,
			TraceID: "sess-screen-readonly",
			Meta:    map[string]string{"args": "query"},
		}
		advice := screen.ScreenResult(ctx, call, []byte("matched 42 lines"))
		if advice.Disposition != abi.ScreenAllow {
			t.Fatalf("expected ScreenAllow for read-only tool %s, got %v", tool, advice.Disposition)
		}
	}
}

func TestObserverSemanticScreen_BarrierTimeoutQuarantined(t *testing.T) {
	pool := NewPool(Config{
		WorkerCount:        1,
		QueueSize:          16,
		BarrierTimeout:     10 * time.Millisecond,
		RequireWitnessDiff: true,
	})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()
	sessionID := "sess-screen-barrier-timeout"

	sess := pool.getOrCreateSession(sessionID)
	// Simulate an un-settled in-flight task that triggers barrier timeout
	atomic.StoreInt64(&sess.inFlight, 1)
	defer atomic.StoreInt64(&sess.inFlight, 0)

	call := &abi.ToolCall{
		Tool:    "Edit",
		TraceID: sessionID,
		Meta: map[string]string{
			"diff": "@@ -1 +1 @@\n-old\n+new",
		},
	}
	body := []byte("applied edit successfully")

	advice := screen.ScreenResult(ctx, call, body)
	if advice.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine on barrier timeout, got %v", advice.Disposition)
	}
	if advice.Reason != abi.ReasonIntegrityRefuted {
		t.Fatalf("expected ReasonIntegrityRefuted, got %v", advice.Reason)
	}
	if advice.By != "observer:barrier_timeout" {
		t.Fatalf("expected By='observer:barrier_timeout', got %q", advice.By)
	}
}

func TestObserverSemanticScreen_BarrierTimeoutReadOnlyAllowed(t *testing.T) {
	pool := NewPool(Config{
		WorkerCount:        1,
		QueueSize:          16,
		BarrierTimeout:     10 * time.Millisecond,
		RequireWitnessDiff: true,
	})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()
	sessionID := "sess-screen-barrier-timeout-readonly"

	sess := pool.getOrCreateSession(sessionID)
	// Simulate an un-settled in-flight task that triggers barrier timeout
	atomic.StoreInt64(&sess.inFlight, 1)
	defer atomic.StoreInt64(&sess.inFlight, 0)

	readTools := []string{"Read", "Grep", "Glob", "fak_read"}
	for _, tool := range readTools {
		call := &abi.ToolCall{
			Tool:    tool,
			TraceID: sessionID,
			Meta: map[string]string{
				"args": "query",
			},
		}
		body := []byte("read contents safely")

		advice := screen.ScreenResult(ctx, call, body)
		if advice.Disposition != abi.ScreenAllow {
			t.Fatalf("expected ScreenAllow on barrier timeout for read-only tool %s, got %v (reason: %v)", tool, advice.Disposition, advice.Reason)
		}
	}
}

func TestObserverSemanticScreen_BarrierTimeout_FailClosedOnUnknownTools(t *testing.T) {
	pool := NewPool(Config{
		WorkerCount:        1,
		QueueSize:          16,
		BarrierTimeout:     10 * time.Millisecond,
		RequireWitnessDiff: true,
	})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()
	sessionID := "sess-screen-barrier-timeout-unknown"

	sess := pool.getOrCreateSession(sessionID)
	// Simulate an un-settled in-flight task that triggers barrier timeout
	atomic.StoreInt64(&sess.inFlight, 1)
	defer atomic.StoreInt64(&sess.inFlight, 0)

	unknownTools := []string{"powershell", "pwsh", "cmd", "sh", "zsh", "delete_file", "custom_mcp_op"}
	for _, tool := range unknownTools {
		call := &abi.ToolCall{
			Tool:    tool,
			TraceID: sessionID,
			Meta: map[string]string{
				"args": "exec",
			},
		}
		body := []byte("command output")

		advice := screen.ScreenResult(ctx, call, body)
		if advice.Disposition != abi.ScreenQuarantine {
			t.Fatalf("expected ScreenQuarantine on barrier timeout for unclassified tool %s, got %v", tool, advice.Disposition)
		}
		if advice.Reason != abi.ReasonIntegrityRefuted {
			t.Fatalf("expected ReasonIntegrityRefuted for %s, got %v", tool, advice.Reason)
		}
		if advice.By != "observer:barrier_timeout" {
			t.Fatalf("expected By='observer:barrier_timeout' for %s, got %q", tool, advice.By)
		}
	}

	// Also verify that a read-only tool on a flagged session is quarantined on barrier timeout
	flaggedSessionID := "sess-screen-barrier-timeout-flagged"
	flaggedSess := pool.getOrCreateSession(flaggedSessionID)
	flaggedSess.mu.Lock()
	flaggedSess.flaggedChurn = true
	flaggedSess.mu.Unlock()
	atomic.StoreInt64(&flaggedSess.inFlight, 1)
	defer atomic.StoreInt64(&flaggedSess.inFlight, 0)

	readCall := &abi.ToolCall{
		Tool:    "Read",
		TraceID: flaggedSessionID,
	}
	advice := screen.ScreenResult(ctx, readCall, []byte("content"))
	if advice.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine on barrier timeout for flagged session, got %v", advice.Disposition)
	}
	if advice.Reason != abi.ReasonIntegrityRefuted {
		t.Fatalf("expected ReasonIntegrityRefuted for flagged session, got %v", advice.Reason)
	}
	if advice.By != "observer:barrier_timeout_flagged" {
		t.Fatalf("expected By='observer:barrier_timeout_flagged' for flagged session, got %q", advice.By)
	}
}

func TestObserverSemanticScreen_BarrierTimeout_PreservesFlaggedQuarantine(t *testing.T) {
	pool := NewPool(Config{
		WorkerCount:        1,
		QueueSize:          16,
		BarrierTimeout:     10 * time.Millisecond,
		RequireWitnessDiff: true,
	})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()

	// 1. Churn flagged session
	sessionChurnID := "sess-screen-preserve-quarantine-churn"
	sessChurn := pool.getOrCreateSession(sessionChurnID)
	sessChurn.mu.Lock()
	sessChurn.flaggedChurn = true
	sessChurn.mu.Unlock()

	atomic.StoreInt64(&sessChurn.inFlight, 1)
	defer atomic.StoreInt64(&sessChurn.inFlight, 0)

	callChurn := &abi.ToolCall{
		Tool:    "Read",
		TraceID: sessionChurnID,
	}
	adviceChurn := screen.ScreenResult(ctx, callChurn, []byte("content"))
	if adviceChurn.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine on barrier timeout for churn-flagged session, got %v", adviceChurn.Disposition)
	}
	if adviceChurn.Reason != abi.ReasonIntegrityRefuted {
		t.Fatalf("expected ReasonIntegrityRefuted for churn-flagged session, got %v", adviceChurn.Reason)
	}
	if adviceChurn.By != "observer:barrier_timeout_flagged" {
		t.Fatalf("expected By='observer:barrier_timeout_flagged', got %q", adviceChurn.By)
	}
	if !sessChurn.isFlagged() {
		t.Fatal("expected session to remain flagged after barrier timeout")
	}

	// Verify that evaluation happened on timeout: observation history is updated
	histChurn := pool.GetSessionHistory(sessionChurnID)
	if len(histChurn) != 1 {
		t.Fatalf("expected 1 evaluated observation in history on barrier timeout, got %d", len(histChurn))
	}
	if histChurn[0].Tool != "Read" {
		t.Fatalf("expected evaluated tool 'Read', got %s", histChurn[0].Tool)
	}
	if histChurn[0].StepVerdict != StepChurn {
		t.Fatalf("expected evaluated StepVerdict=StepChurn, got %s", histChurn[0].StepVerdict)
	}

	// Second read call maintains repeat count and preserves quarantine
	adviceChurn2 := screen.ScreenResult(ctx, callChurn, []byte("content 2"))
	if adviceChurn2.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine on second barrier timeout, got %v", adviceChurn2.Disposition)
	}
	if adviceChurn2.By != "observer:barrier_timeout_flagged" {
		t.Fatalf("expected By='observer:barrier_timeout_flagged' on second timeout, got %q", adviceChurn2.By)
	}
	histChurn2 := pool.GetSessionHistory(sessionChurnID)
	if len(histChurn2) != 2 {
		t.Fatalf("expected 2 evaluated observations in history on second barrier timeout, got %d", len(histChurn2))
	}
	sessChurn.mu.Lock()
	repCount := sessChurn.repeatCount
	sessChurn.mu.Unlock()
	if repCount != 2 {
		t.Fatalf("expected repeat count=2 after second evaluated timeout call, got %d", repCount)
	}

	// 2. Regress flagged session
	sessionRegressID := "sess-screen-preserve-quarantine-regress"
	sessRegress := pool.getOrCreateSession(sessionRegressID)
	sessRegress.mu.Lock()
	sessRegress.flaggedRegress = true
	sessRegress.mu.Unlock()

	atomic.StoreInt64(&sessRegress.inFlight, 1)
	defer atomic.StoreInt64(&sessRegress.inFlight, 0)

	callRegress := &abi.ToolCall{
		Tool:    "Grep",
		TraceID: sessionRegressID,
	}
	adviceRegress := screen.ScreenResult(ctx, callRegress, []byte("content"))
	if adviceRegress.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine on barrier timeout for regress-flagged session, got %v", adviceRegress.Disposition)
	}
	if adviceRegress.Reason != abi.ReasonIntegrityRefuted {
		t.Fatalf("expected ReasonIntegrityRefuted for regress-flagged session, got %v", adviceRegress.Reason)
	}
	if adviceRegress.By != "observer:barrier_timeout_flagged" {
		t.Fatalf("expected By='observer:barrier_timeout_flagged' for regress session, got %q", adviceRegress.By)
	}
	if !sessRegress.isFlagged() {
		t.Fatal("expected regress session to remain flagged after barrier timeout")
	}
	sessRegress.mu.Lock()
	if sessRegress.kvPrefixWarm {
		sessRegress.mu.Unlock()
		t.Fatal("expected regress session kvPrefixWarm to remain false after barrier timeout")
	}
	sessRegress.mu.Unlock()

	// Verify evaluation happened on timeout for regress session
	histRegress := pool.GetSessionHistory(sessionRegressID)
	if len(histRegress) != 1 {
		t.Fatalf("expected 1 evaluated observation in history on barrier timeout for regress session, got %d", len(histRegress))
	}
	if histRegress[0].Tool != "Grep" {
		t.Fatalf("expected evaluated tool 'Grep', got %s", histRegress[0].Tool)
	}
	if histRegress[0].StepVerdict != StepRegress {
		t.Fatalf("expected evaluated StepVerdict=StepRegress, got %s", histRegress[0].StepVerdict)
	}
}

func TestObserverSemanticScreen_ScreenResult_ContextDeadlineExceeded(t *testing.T) {
	pool := NewPool(Config{
		WorkerCount:        1,
		QueueSize:          16,
		BarrierTimeout:     500 * time.Millisecond,
		RequireWitnessDiff: true,
	})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	sessionID := "sess-screen-context-deadline"

	sess := pool.getOrCreateSession(sessionID)
	// Simulate an un-settled in-flight task that forces the barrier to wait
	atomic.StoreInt64(&sess.inFlight, 1)
	defer atomic.StoreInt64(&sess.inFlight, 0)

	call := &abi.ToolCall{
		Tool:    "Edit",
		TraceID: sessionID,
		Meta: map[string]string{
			"diff": "@@ -1 +1 @@\n-old\n+new",
		},
	}
	body := []byte("applied edit successfully")

	// 1. Context deadline exceeded during barrier wait
	ctxDeadline, cancelDeadline := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancelDeadline()

	adviceDeadline := screen.ScreenResult(ctxDeadline, call, body)
	if adviceDeadline.Disposition == abi.ScreenAllow && adviceDeadline.By == "observer:advance" {
		t.Fatalf("expected context deadline exceeded not to return ScreenAllow with observer:advance, got %+v", adviceDeadline)
	}
	if adviceDeadline.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine on context deadline exceeded, got %v", adviceDeadline.Disposition)
	}
	if adviceDeadline.Reason != abi.ReasonIntegrityRefuted {
		t.Fatalf("expected ReasonIntegrityRefuted on context deadline exceeded, got %v", adviceDeadline.Reason)
	}

	// 2. Context cancellation during barrier wait
	ctxCancel, cancelNow := context.WithCancel(context.Background())
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancelNow()
	}()

	adviceCancel := screen.ScreenResult(ctxCancel, call, body)
	if adviceCancel.Disposition == abi.ScreenAllow && adviceCancel.By == "observer:advance" {
		t.Fatalf("expected context cancellation not to return ScreenAllow with observer:advance, got %+v", adviceCancel)
	}
	if adviceCancel.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine on context cancellation, got %v", adviceCancel.Disposition)
	}
	if adviceCancel.Reason != abi.ReasonIntegrityRefuted {
		t.Fatalf("expected ReasonIntegrityRefuted on context cancellation, got %v", adviceCancel.Reason)
	}
}

func TestObserverSemanticScreen_VerifyToolCall_PreExecution(t *testing.T) {
	pool := NewPool(Config{ChurnThreshold: 2, RegressThreshold: 3})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()

	sessionID := "sess-pre-exec"

	t.Run("CleanCallAllowed", func(t *testing.T) {
		callClean := &abi.ToolCall{
			Tool:    "Edit",
			TraceID: sessionID,
		}
		advClean := screen.VerifyToolCall(ctx, callClean)
		if advClean.Disposition != abi.ScreenAllow {
			t.Fatalf("expected ScreenAllow on clean pre-execution call, got %v", advClean.Disposition)
		}
	})

	t.Run("QuarantinedContextMMU", func(t *testing.T) {
		callQuarantined := &abi.ToolCall{
			Tool:    "Write",
			TraceID: sessionID,
			Meta:    map[string]string{"mmu_quarantined": "true"},
		}
		advQ := screen.VerifyToolCall(ctx, callQuarantined)
		if advQ.Disposition != abi.ScreenQuarantine || advQ.Reason != abi.ReasonTrustViolation {
			t.Fatalf("expected ScreenQuarantine with ReasonTrustViolation for quarantined MMU state, got disp=%v reason=%v",
				advQ.Disposition, advQ.Reason)
		}
	})

	t.Run("UnwitnessedMutatingCall", func(t *testing.T) {
		callUnwitnessed := &abi.ToolCall{
			Tool:    "Edit",
			TraceID: sessionID,
			Meta:    map[string]string{"require_diff": "true"},
		}
		advUnwit := screen.VerifyToolCall(ctx, callUnwitnessed)
		if advUnwit.Disposition != abi.ScreenQuarantine || advUnwit.Reason != abi.ReasonUnwitnessed {
			t.Fatalf("expected ScreenQuarantine with ReasonUnwitnessed for unwitnessed mutating call, got disp=%v reason=%v",
				advUnwit.Disposition, advUnwit.Reason)
		}
	})

	t.Run("FlaggedChurnPreExecutionQuarantine", func(t *testing.T) {
		callClean := &abi.ToolCall{
			Tool:    "Edit",
			TraceID: sessionID,
		}
		churnCall := &abi.ToolCall{
			Tool:    "bash",
			TraceID: sessionID,
			Meta:    map[string]string{"error": "fail"},
		}
		_ = screen.ScreenResult(ctx, churnCall, []byte("fail"))
		_ = screen.ScreenResult(ctx, churnCall, []byte("fail"))

		advChurnPre := screen.VerifyToolCall(ctx, callClean)
		if advChurnPre.Disposition != abi.ScreenQuarantine || advChurnPre.Reason != abi.ReasonTrustViolation {
			t.Fatalf("expected pre-execution quarantine once churn tripped, got disp=%v reason=%v",
				advChurnPre.Disposition, advChurnPre.Reason)
		}
	})
}

func TestObserveSyncBarrier_ContextCancelled(t *testing.T) {
	p := NewPool(Config{WorkerCount: 2, QueueSize: 16})
	_ = p.Start()
	defer p.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	obs := StepObservation{
		SessionID: "sess-canceled",
		Tool:      "Edit",
		Diff:      "@@ -1 +1 @@\n+added",
	}

	_, err := p.ObserveSyncBarrier(ctx, obs)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestObserveSyncBarrier_InFlightTaskWait(t *testing.T) {
	p := NewPool(Config{
		WorkerCount:    1,
		QueueSize:      16,
		BarrierTimeout: 200 * time.Millisecond,
	})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()
	sessionID := "sess-inflight-wait"
	sess := p.getOrCreateSession(sessionID)

	// Simulate an in-flight async task
	sess.incInFlight()

	// In a background goroutine, simulate task finishing after 5ms
	go func() {
		time.Sleep(5 * time.Millisecond)
		sess.decInFlight()
	}()

	obs := StepObservation{
		SessionID: sessionID,
		Tool:      "Write",
		Diff:      "@@ -1 +1 @@\n+added",
	}

	start := time.Now()
	res, err := p.ObserveSyncBarrier(ctx, obs)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("unexpected barrier error: %v", err)
	}
	if res.StepVerdict != STEP_ADVANCE {
		t.Fatalf("expected STEP_ADVANCE, got %s", res.StepVerdict)
	}
	if elapsed < 4*time.Millisecond {
		t.Fatalf("expected barrier to wait for in-flight task, elapsed %s", elapsed)
	}
}

func TestObserveSyncBarrier_ConditionVariableNotification(t *testing.T) {
	p := NewPool(Config{
		WorkerCount:    2,
		QueueSize:      16,
		BarrierTimeout: 200 * time.Millisecond,
	})
	if err := p.Start(); err != nil {
		t.Fatalf("failed to start pool: %v", err)
	}
	defer p.Close()

	ctx := context.Background()
	sessionID := "sess-cond-var"
	sess := p.getOrCreateSession(sessionID)

	// 1. Verify sync.Cond on sess.mu is signaled/broadcast when inFlight drops to 0
	sess.incInFlight()
	condWoken := make(chan struct{})
	go func() {
		sess.mu.Lock()
		for atomic.LoadInt64(&sess.inFlight) > 0 {
			sess.cond.Wait()
		}
		sess.mu.Unlock()
		close(condWoken)
	}()

	time.Sleep(5 * time.Millisecond)
	sess.decInFlight()

	select {
	case <-condWoken:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("timed out waiting for sess.cond.Wait() condition broadcast")
	}

	// 2. Real async exploration turn settles barrier without 50µs polling timer
	readCh := p.ObserveAsync(ctx, StepObservation{
		SessionID: sessionID,
		Tool:      "Read",
		Args:      "main.go",
		Result:    "package main",
	})

	writeObs := StepObservation{
		SessionID: sessionID,
		Tool:      "Write",
		Diff:      "@@ -1 +1 @@\n+new",
	}
	res, err := p.ObserveSyncBarrier(ctx, writeObs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.StepVerdict != STEP_ADVANCE {
		t.Fatalf("expected STEP_ADVANCE, got %s", res.StepVerdict)
	}
	<-readCh
}

func TestObserveSyncBarrier_ReadOnlyBypassesInFlightWait(t *testing.T) {
	p := NewPool(Config{
		WorkerCount:    1,
		QueueSize:      16,
		BarrierTimeout: 100 * time.Millisecond,
	})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()
	sessionID := "sess-readonly-bypass-inflight"
	sess := p.getOrCreateSession(sessionID)

	// Simulate an active in-flight async task that does not finish
	atomic.StoreInt64(&sess.inFlight, 1)
	defer atomic.StoreInt64(&sess.inFlight, 0)

	readTools := []string{"Read", "Grep", "Glob", "fak_read"}
	for _, tool := range readTools {
		obs := StepObservation{
			SessionID: sessionID,
			Tool:      tool,
			Args:      "path/to/target",
			Result:    "sample content",
		}

		start := time.Now()
		res, err := p.ObserveSyncBarrier(ctx, obs)
		elapsed := time.Since(start)

		if err != nil {
			t.Fatalf("expected read-only tool %s to succeed without timeout error, got %v", tool, err)
		}
		if elapsed >= 10*time.Millisecond {
			t.Fatalf("expected read-only tool %s to bypass in-flight wait (<10ms), took %s", tool, elapsed)
		}
		if res.BarrierLatency > 2*time.Millisecond {
			t.Fatalf("expected BarrierLatency < 2ms for read-only bypass on %s, got %s", tool, res.BarrierLatency)
		}
		if res.StepVerdict != STEP_ADVANCE {
			t.Fatalf("expected STEP_ADVANCE for %s, got %s", tool, res.StepVerdict)
		}
	}

	if got := p.BarrierTimeouts(); got != 0 {
		t.Fatalf("expected BarrierTimeouts to be 0 for read-only bypasses, got %d", got)
	}
	if got := p.DetailedStats().BarrierTimeouts; got != 0 {
		t.Fatalf("expected DetailedStats().BarrierTimeouts to be 0, got %d", got)
	}
}

func TestObserveSyncBarrier_TimeoutWaitingForInFlight(t *testing.T) {
	p := NewPool(Config{
		WorkerCount:    1,
		QueueSize:      16,
		BarrierTimeout: 10 * time.Millisecond,
	})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()
	sessionID := "sess-inflight-timeout"
	sess := p.getOrCreateSession(sessionID)

	// Set in-flight task that never finishes
	atomic.StoreInt64(&sess.inFlight, 1)
	defer atomic.StoreInt64(&sess.inFlight, 0)

	obs := StepObservation{
		SessionID: sessionID,
		Tool:      "Write",
		Diff:      "@@ -1 +1 @@\n+added",
	}

	_, err := p.ObserveSyncBarrier(ctx, obs)
	if !errors.Is(err, ErrBarrierTimeout) {
		t.Fatalf("expected ErrBarrierTimeout, got %v", err)
	}
	if got := p.BarrierTimeouts(); got != 1 {
		t.Fatalf("expected BarrierTimeouts to be 1, got %d", got)
	}
	if got := p.DetailedStats().BarrierTimeouts; got != 1 {
		t.Fatalf("expected DetailedStats().BarrierTimeouts to be 1, got %d", got)
	}
}

func TestObserveSyncBarrier_IncrementsBarrierTimeouts(t *testing.T) {
	p := NewPool(Config{
		WorkerCount:    1,
		QueueSize:      16,
		BarrierTimeout: 10 * time.Millisecond,
	})
	_ = p.Start()
	defer p.Close()

	if p.BarrierTimeouts() != 0 {
		t.Fatalf("expected initial BarrierTimeouts=0, got %d", p.BarrierTimeouts())
	}
	if p.DetailedStats().BarrierTimeouts != 0 {
		t.Fatalf("expected initial DetailedStats().BarrierTimeouts=0, got %d", p.DetailedStats().BarrierTimeouts)
	}

	ctx := context.Background()

	// 1. Successful barrier does not increment timeout counter
	obsSuccess := StepObservation{
		SessionID: "sess-timeouts",
		Tool:      "Edit",
		Diff:      "@@ -1 +1 @@\n+added",
	}
	if _, err := p.ObserveSyncBarrier(ctx, obsSuccess); err != nil {
		t.Fatalf("unexpected error on successful barrier: %v", err)
	}
	if p.BarrierTimeouts() != 0 {
		t.Fatalf("expected BarrierTimeouts=0 after successful barrier, got %d", p.BarrierTimeouts())
	}

	// 2. Timeout increment
	sess := p.getOrCreateSession("sess-timeouts")
	atomic.StoreInt64(&sess.inFlight, 1)
	defer atomic.StoreInt64(&sess.inFlight, 0)

	obsTimeout := StepObservation{
		SessionID: "sess-timeouts",
		Tool:      "Write",
		Diff:      "@@ -1 +1 @@\n+added",
	}

	_, err := p.ObserveSyncBarrier(ctx, obsTimeout)
	if !errors.Is(err, ErrBarrierTimeout) {
		t.Fatalf("expected ErrBarrierTimeout, got %v", err)
	}
	if got := p.BarrierTimeouts(); got != 1 {
		t.Fatalf("expected BarrierTimeouts=1 after first timeout, got %d", got)
	}

	// 3. Second timeout increment
	_, err = p.ObserveSyncBarrier(ctx, obsTimeout)
	if !errors.Is(err, ErrBarrierTimeout) {
		t.Fatalf("expected ErrBarrierTimeout on second call, got %v", err)
	}
	if got := p.BarrierTimeouts(); got != 2 {
		t.Fatalf("expected BarrierTimeouts=2 after second timeout, got %d", got)
	}

	// 4. Verify DetailedStats and backwards-compatible Stats()
	st := p.DetailedStats()
	if st.BarrierTimeouts != 2 {
		t.Fatalf("expected DetailedStats().BarrierTimeouts=2, got %d", st.BarrierTimeouts)
	}
	if st.BarriersTotal != 3 {
		t.Fatalf("expected DetailedStats().BarriersTotal=3, got %d", st.BarriersTotal)
	}

	total, asyncCount, barriers, _, _, _, _ := p.Stats()
	if barriers != 3 || total != 3 || asyncCount != 0 {
		t.Fatalf("expected Stats() total=3, barriers=3, got total=%d, async=%d, barriers=%d", total, asyncCount, barriers)
	}

	// 5. Nil safety
	var nilPool *Pool
	if nilPool.BarrierTimeouts() != 0 {
		t.Fatalf("expected nil pool BarrierTimeouts()=0, got %d", nilPool.BarrierTimeouts())
	}
	if nilPool.DetailedStats().BarrierTimeouts != 0 {
		t.Fatalf("expected nil pool DetailedStats().BarrierTimeouts=0, got %d", nilPool.DetailedStats().BarrierTimeouts)
	}
	nTotal, _, _, _, _, _, _ := nilPool.Stats()
	if nTotal != 0 {
		t.Fatalf("expected nil pool Stats() total=0, got %d", nTotal)
	}
}

func TestObserveSyncBarrier_ContextCancelWhileWaiting(t *testing.T) {
	p := NewPool(Config{
		WorkerCount:    1,
		QueueSize:      16,
		BarrierTimeout: 500 * time.Millisecond,
	})
	_ = p.Start()
	defer p.Close()

	ctx, cancel := context.WithCancel(context.Background())
	sessionID := "sess-inflight-cancel"
	sess := p.getOrCreateSession(sessionID)

	// Set in-flight task
	atomic.StoreInt64(&sess.inFlight, 1)
	defer atomic.StoreInt64(&sess.inFlight, 0)

	// Cancel context after 10ms
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	obs := StepObservation{
		SessionID: sessionID,
		Tool:      "Write",
		Diff:      "@@ -1 +1 @@\n+added",
	}

	_, err := p.ObserveSyncBarrier(ctx, obs)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestSessionHistory_RetentionCap(t *testing.T) {
	const maxHistory = 10
	p := NewPool(Config{
		WorkerCount:          2,
		QueueSize:            32,
		MaxHistoryPerSession: maxHistory,
	})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()
	sessionID := "sess-retention-test"

	for i := 0; i < 25; i++ {
		obs := StepObservation{
			SessionID: sessionID,
			Tool:      "Read",
			Args:      fmt.Sprintf("path/file_%d.go", i),
			Result:    fmt.Sprintf("content_%d", i),
		}
		_, err := p.ObserveSyncBarrier(ctx, obs)
		if err != nil {
			t.Fatalf("unexpected barrier error on step %d: %v", i, err)
		}
	}

	history := p.GetSessionHistory(sessionID)
	if len(history) != maxHistory {
		t.Fatalf("expected exactly %d history entries, got %d", maxHistory, len(history))
	}

	lastEntry := history[len(history)-1]
	expectedArg := "path/file_24.go"
	if lastEntry.Args != expectedArg {
		t.Fatalf("expected newest entry arg %q, got %v", expectedArg, lastEntry.Args)
	}
}

func TestSessionHistory_TrailingElementsZeroedForGC(t *testing.T) {
	const maxHistory = 5
	p := NewPool(Config{
		WorkerCount:          2,
		QueueSize:            32,
		MaxHistoryPerSession: maxHistory,
	})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()
	sessionID := "sess-gc-zero-test"

	// Fill and overflow history
	for i := 0; i < 15; i++ {
		obs := StepObservation{
			SessionID: sessionID,
			Tool:      "Read",
			Args:      map[string]any{"index": i, "data": make([]byte, 1024)},
			Result:    fmt.Sprintf("result-%d", i),
		}
		if _, err := p.ObserveSyncBarrier(ctx, obs); err != nil {
			t.Fatalf("unexpected error on turn %d: %v", i, err)
		}
	}

	sess := p.getOrCreateSession(sessionID)
	sess.mu.Lock()
	defer sess.mu.Unlock()

	if len(sess.history) != maxHistory {
		t.Fatalf("expected len(history) == %d, got %d", maxHistory, len(sess.history))
	}

	c := cap(sess.history)
	if c <= maxHistory {
		t.Fatalf("expected cap(history) > maxHistory, got cap=%d, maxHistory=%d", c, maxHistory)
	}

	backingSlice := sess.history[:c]
	for i := maxHistory; i < c; i++ {
		elem := backingSlice[i]
		if elem.Args != nil || elem.Result != nil || elem.Tool != "" || elem.Diff != "" {
			t.Fatalf("slot %d beyond len in backing array was not zeroed: %+v", i, elem)
		}
	}
}

func TestObservation_ByteSliceResultsAndErrors(t *testing.T) {
	p := NewPool(Config{
		RequireWitnessDiff: true,
		ChurnThreshold:     2,
	})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()

	// 1. Read tool with byte-slice error is recognized and tracks failure
	obsErr1 := StepObservation{
		SessionID: "sess-bytes-err",
		Tool:      "Read",
		Result:    []byte("error: file not found"),
	}
	resErr1, err := p.ObserveSyncBarrier(ctx, obsErr1)
	if err != nil {
		t.Fatalf("unexpected error on read byte error: %v", err)
	}
	if resErr1.StepVerdict != STEP_ADVANCE {
		t.Fatalf("expected STEP_ADVANCE on first error, got %s", resErr1.StepVerdict)
	}

	// Second byte-slice error reaches ChurnThreshold (2) -> ErrChurnRefused
	obsErr2 := StepObservation{
		SessionID: "sess-bytes-err",
		Tool:      "Read",
		Result:    []byte("error: file not found"),
	}
	resErr2, err := p.ObserveSyncBarrier(ctx, obsErr2)
	if !errors.Is(err, ErrChurnRefused) {
		t.Fatalf("expected ErrChurnRefused on second byte error, got %v", err)
	}
	if resErr2.StepVerdict != STEP_CHURN {
		t.Fatalf("expected STEP_CHURN, got %s", resErr2.StepVerdict)
	}

	// 2. Mutating tool with byte-slice diff confirmation
	obsDiff := StepObservation{
		SessionID: "sess-bytes-diff",
		Tool:      "Edit",
		Result:    []byte("1 file changed, 3 insertions(+)"),
	}
	resDiff, err := p.ObserveSyncBarrier(ctx, obsDiff)
	if err != nil {
		t.Fatalf("unexpected error on byte diff observation: %v", err)
	}
	if resDiff.WitnessVerdict != WITNESS_DIFF_CONFIRMED {
		t.Fatalf("expected WITNESS_DIFF_CONFIRMED, got %s", resDiff.WitnessVerdict)
	}

	// 3. Mutating tool with byte-slice error and no diff -> ErrUnwitnessedDiff
	obsMutErr := StepObservation{
		SessionID: "sess-bytes-mut-err",
		Tool:      "bash",
		Result:    []byte("error: command not found"),
	}
	resMutErr, err := p.ObserveSyncBarrier(ctx, obsMutErr)
	if !errors.Is(err, ErrUnwitnessedDiff) {
		t.Fatalf("expected ErrUnwitnessedDiff on mutating byte error, got %v", err)
	}
	if resMutErr.WitnessVerdict != WITNESS_UNWITNESSED_CLAIM {
		t.Fatalf("expected WITNESS_UNWITNESSED_CLAIM, got %s", resMutErr.WitnessVerdict)
	}
}

func TestStepObservation_ToolClassification(t *testing.T) {
	readOnly := []string{
		"Read", "read", "Grep", "grep", "Glob", "glob", "fak_read",
		"list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource",
	}
	for _, tool := range readOnly {
		if !IsReadOnlyTool(tool) {
			t.Errorf("expected %q to be read-only", tool)
		}
		if IsMutatingTool(tool) {
			t.Errorf("expected %q not to be mutating", tool)
		}
		obs := StepObservation{Tool: tool}
		if !obs.IsReadOnly() || obs.IsMutating() {
			t.Errorf("expected StepObservation(%q) to be read-only", tool)
		}
	}

	mutating := []string{
		"Edit", "edit", "Write", "write", "bash", "git commit", "commit", "fak_syscall",
		"git push", "git checkout",
	}
	for _, tool := range mutating {
		if !IsMutatingTool(tool) {
			t.Errorf("expected %q to be mutating", tool)
		}
		if IsReadOnlyTool(tool) {
			t.Errorf("expected %q not to be read-only", tool)
		}
		obs := StepObservation{Tool: tool}
		if !obs.IsMutating() || obs.IsReadOnly() {
			t.Errorf("expected StepObservation(%q) to be mutating", tool)
		}
	}

	gitReadOnly := []string{"git status", "git diff", "git log", "git rev-parse"}
	for _, tool := range gitReadOnly {
		if IsMutatingTool(tool) {
			t.Errorf("expected git subcommand %q not to be mutating", tool)
		}
	}
}

func TestObserverSemanticScreen_NilAndContextHandling(t *testing.T) {
	screen := NewObserverSemanticScreen(nil)

	adv := screen.ScreenResult(context.Background(), nil, []byte("ok"))
	if adv.Disposition != abi.ScreenAllow {
		t.Fatalf("expected ScreenAllow on nil ToolCall, got %v", adv.Disposition)
	}

	advPre := screen.VerifyToolCall(context.Background(), nil)
	if advPre.Disposition != abi.ScreenAllow {
		t.Fatalf("expected ScreenAllow on nil ToolCall in VerifyToolCall, got %v", advPre.Disposition)
	}

	ctxCanceled, cancel := context.WithCancel(context.Background())
	cancel()

	call := &abi.ToolCall{Tool: "Edit"}
	advCancel := screen.ScreenResult(ctxCanceled, call, []byte("ok"))
	if advCancel.Disposition != abi.ScreenAllow {
		t.Fatalf("expected ScreenAllow on canceled context in ScreenResult, got %v", advCancel.Disposition)
	}

	advCancelPre := screen.VerifyToolCall(ctxCanceled, call)
	if advCancelPre.Disposition != abi.ScreenAllow {
		t.Fatalf("expected ScreenAllow on canceled context in VerifyToolCall, got %v", advCancelPre.Disposition)
	}
}

func TestPool_ResetAndGetSession(t *testing.T) {
	p := NewPool(Config{WorkerCount: 2, QueueSize: 16})
	_ = p.Start()
	defer p.Close()

	if h := p.GetSessionHistory("non-existent"); h != nil {
		t.Fatalf("expected nil history for non-existent session, got %v", h)
	}

	p.ResetSession("non-existent")

	ctx := context.Background()
	_, err := p.ObserveSyncBarrier(ctx, StepObservation{
		SessionID: "sess-temp",
		Tool:      "Read",
	})
	if err != nil {
		t.Fatalf("unexpected barrier error: %v", err)
	}
	if h := p.GetSessionHistory("sess-temp"); len(h) != 1 {
		t.Fatalf("expected 1 history item, got %d", len(h))
	}

	p.ResetSession("sess-temp")
	if h := p.GetSessionHistory("sess-temp"); h != nil {
		t.Fatalf("expected nil history after reset, got %v", h)
	}
}

func BenchmarkObserveAsyncParallel(b *testing.B) {
	p := NewPool(Config{WorkerCount: 8, QueueSize: 4096})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()
	obs := StepObservation{
		SessionID: "bench-async-par",
		Tool:      "Read",
		Args:      "file.go",
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			ch := p.ObserveAsync(ctx, obs)
			<-ch
		}
	})
}

func BenchmarkObserveSyncBarrierParallel(b *testing.B) {
	p := NewPool(Config{WorkerCount: 8, QueueSize: 4096})
	_ = p.Start()
	defer p.Close()

	ctx := context.Background()
	obs := StepObservation{
		SessionID: "bench-barrier-par",
		Tool:      "Edit",
		Diff:      "@@ -1 +1 @@\n-old\n+new",
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = p.ObserveSyncBarrier(ctx, obs)
		}
	})
}

func BenchmarkObserverSemanticScreen_ScreenResult_ReadOnly(b *testing.B) {
	p := NewPool(Config{WorkerCount: 8, QueueSize: 2048})
	_ = p.Start()
	defer p.Close()

	screen := NewObserverSemanticScreen(p)
	ctx := context.Background()
	call := &abi.ToolCall{
		Tool:    "Read",
		TraceID: "bench-screen-readonly",
		Meta:    map[string]string{"args": "query"},
	}
	body := []byte("search results matched 10 lines")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = screen.ScreenResult(ctx, call, body)
	}
}

func BenchmarkObserverSemanticScreen_ScreenResult_Mutating(b *testing.B) {
	p := NewPool(Config{WorkerCount: 8, QueueSize: 2048, RequireWitnessDiff: true})
	_ = p.Start()
	defer p.Close()

	screen := NewObserverSemanticScreen(p)
	ctx := context.Background()
	call := &abi.ToolCall{
		Tool:    "Edit",
		TraceID: "bench-screen-mutating",
		Meta:    map[string]string{"diff": "@@ -1 +1 @@\n-old\n+new"},
	}
	body := []byte("replaced content successfully")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = screen.ScreenResult(ctx, call, body)
	}
}

func BenchmarkObserverSemanticScreen_VerifyToolCall(b *testing.B) {
	p := NewPool(Config{WorkerCount: 8, QueueSize: 2048})
	_ = p.Start()
	defer p.Close()

	screen := NewObserverSemanticScreen(p)
	ctx := context.Background()
	call := &abi.ToolCall{
		Tool:    "Edit",
		TraceID: "bench-screen-verify",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = screen.VerifyToolCall(ctx, call)
	}
}

func BenchmarkParseStepVerdict(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ParseStepVerdict("STEP_ADVANCE")
	}
}

func BenchmarkStepObservation_Validate(b *testing.B) {
	obs := StepObservation{
		StepVerdict:    STEP_ADVANCE,
		WitnessVerdict: WITNESS_DIFF_CONFIRMED,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = obs.Validate()
	}
}
