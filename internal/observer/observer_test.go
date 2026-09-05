package observer

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestClosedVocabularyAndWitnessVerdicts(t *testing.T) {
	// Step classification closed vocabulary
	verdicts := []StepVerdict{STEP_ADVANCE, STEP_CHURN, STEP_REGRESS}
	for _, v := range verdicts {
		if !v.IsValid() {
			t.Fatalf("expected verdict %s to be valid", v)
		}
	}
	if StepVerdict("STEP_UNKNOWN").IsValid() {
		t.Fatal("expected arbitrary verdict to be invalid")
	}

	// Witness verification closed vocabulary
	witnesses := []WitnessVerdict{WITNESS_DIFF_CONFIRMED, WITNESS_UNWITNESSED_CLAIM}
	for _, w := range witnesses {
		if !w.IsValid() {
			t.Fatalf("expected witness %s to be valid", w)
		}
	}
	if WitnessVerdict("WITNESS_UNKNOWN").IsValid() {
		t.Fatal("expected arbitrary witness to be invalid")
	}
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

	sessionID := "sess-churn"

	// 1. Repeated failing calls
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

	// 3rd failure reaches churn threshold (3) -> deterministic refusal
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

	// 2. Repeated identical queries trigger churn
	p.ResetSession("sess-query-loop")
	for i := 1; i <= 2; i++ {
		obs := StepObservation{
			SessionID: "sess-query-loop",
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

	// 3rd identical query -> churn
	obsQueryChurn := StepObservation{
		SessionID: "sess-query-loop",
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

	// Turns 1 and 2: advance
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

	// Turn 3: churn
	obs3 := StepObservation{
		SessionID: sessionID,
		Tool:      "bash",
		Error:     "command failed",
	}
	_, err := p.ObserveSyncBarrier(ctx, obs3)
	if !errors.Is(err, ErrChurnRefused) {
		t.Fatalf("turn 3 expected ErrChurnRefused, got %v", err)
	}

	// Turn 4: reaches RegressThreshold (4) -> STEP_REGRESS refusal
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
	// On regression, warm KV-cache prefix is invalidated
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
	// 1. Verify StepVerdicts() returns complete closed vocabulary
	svs := StepVerdicts()
	if len(svs) != 3 {
		t.Fatalf("expected 3 StepVerdicts, got %d", len(svs))
	}
	for _, sv := range svs {
		if !sv.IsValid() {
			t.Fatalf("expected StepVerdict %s to be valid", sv)
		}
	}

	// 2. ParseStepVerdict validation
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

	// 3. Verify WitnessVerdicts() returns complete closed vocabulary
	wvs := WitnessVerdicts()
	if len(wvs) != 2 {
		t.Fatalf("expected 2 WitnessVerdicts, got %d", len(wvs))
	}
	for _, wv := range wvs {
		if !wv.IsValid() {
			t.Fatalf("expected WitnessVerdict %s to be valid", wv)
		}
	}

	// 4. ParseWitnessVerdict validation
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

	// 5. StepObservation.Validate()
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

func TestObserverSemanticScreen_VerifyToolCall_PreExecution(t *testing.T) {
	pool := NewPool(Config{ChurnThreshold: 2, RegressThreshold: 3})
	_ = pool.Start()
	defer pool.Close()

	screen := NewObserverSemanticScreen(pool)
	ctx := context.Background()

	sessionID := "sess-pre-exec"

	// 1. Fresh call: allowed
	callClean := &abi.ToolCall{
		Tool:    "Edit",
		TraceID: sessionID,
	}
	advClean := screen.VerifyToolCall(ctx, callClean)
	if advClean.Disposition != abi.ScreenAllow {
		t.Fatalf("expected ScreenAllow on clean pre-execution call, got %v", advClean.Disposition)
	}

	// 2. Call with quarantined context MMU metadata
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

	// 3. Mutating call with require_diff set and no diff/args
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

	// 4. Pre-execution barrier once churn tripped
	churnCall := &abi.ToolCall{
		Tool:    "bash",
		TraceID: sessionID,
		Meta:    map[string]string{"error": "fail"},
	}
	_ = screen.ScreenResult(ctx, churnCall, []byte("fail"))
	_ = screen.ScreenResult(ctx, churnCall, []byte("fail")) // trips churn

	advChurnPre := screen.VerifyToolCall(ctx, callClean)
	if advChurnPre.Disposition != abi.ScreenQuarantine || advChurnPre.Reason != abi.ReasonTrustViolation {
		t.Fatalf("expected pre-execution quarantine once churn tripped, got disp=%v reason=%v",
			advChurnPre.Disposition, advChurnPre.Reason)
	}
}
