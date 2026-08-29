package coordination

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type serveObserverFunc func(context.Context) (ServeSourceSnapshot, error)

func (f serveObserverFunc) SnapshotServe(ctx context.Context) (ServeSourceSnapshot, error) {
	return f(ctx)
}

type legacyServeFunc func(context.Context, ServeObservation, ServePlanRequest) (ServeDecision, error)

func (f legacyServeFunc) PlanLegacyServe(ctx context.Context, observation ServeObservation, request ServePlanRequest) (ServeDecision, error) {
	return f(ctx, observation, request)
}

type serveExecutorFunc func(context.Context, ServeDecision) error

func (f serveExecutorFunc) ApplyServe(ctx context.Context, decision ServeDecision) error {
	return f(ctx, decision)
}

func serveTestTime() time.Time {
	return time.Unix(1_700_000_000, 0).UTC()
}

func validServeSource(identity string, now time.Time) ServeSourceCandidate {
	return ServeSourceCandidate{
		Identity:          []byte(identity),
		Engine:            EngineFakNative,
		Generation:        3,
		CapacitySnapshot:  11,
		Ready:             true,
		QueueDepth:        1,
		QueueCapacity:     10,
		BatchSize:         1,
		BatchCapacity:     8,
		PrefillAvailable:  4096,
		DecodeAvailable:   1024,
		AdmissionMillis:   25,
		CancellationReady: true,
		CacheAffinity:     1,
		Provenance:        ServeMeasured,
		ObservedAt:        now.Add(-time.Second),
		ValidUntil:        now.Add(time.Minute),
	}
}

func serveObservationFromSources(t *testing.T, now time.Time, sources ...ServeSourceCandidate) ServeObservation {
	t.Helper()
	adapter := NewServeAdapter(true, serveObserverFunc(func(context.Context) (ServeSourceSnapshot, error) {
		return ServeSourceSnapshot{Generation: 7, CapturedAt: now.Add(-time.Second), Candidates: sources}, nil
	}), nil, nil)
	adapter.now = func() time.Time { return now }
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatalf("Observe: %v", err)
	}
	return observation
}

func validServeRequest(now time.Time) ServePlanRequest {
	return ServePlanRequest{
		IdempotencyKey: "serve-test",
		Now:            now,
		Deadline:       now.Add(time.Second),
		Intent: NeutralHarnessIntent{
			WorkID:        "work-1",
			CorrelationID: "corr-1",
			Fanout:        4,
			Concurrency:   4,
		},
		Pressure: HarnessPressure{Concurrency: 4},
		Requirements: ServeProjectionRequirements{
			PrefillTokens: 128,
			DecodeTokens:  32,
			BatchSize:     1,
		},
	}
}

func planServe(t *testing.T, now time.Time, observation ServeObservation, request ServePlanRequest) ServeDecision {
	t.Helper()
	adapter := NewServeAdapter(true, nil, nil, nil)
	adapter.now = func() time.Time { return now }
	decision, err := adapter.Plan(context.Background(), observation, request)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	return decision
}

func TestServeAdapterObserveProjectsBoundedContentFreeEvidence(t *testing.T) {
	now := serveTestTime()
	source := validServeSource("private-worker-identity", now)
	observation := serveObservationFromSources(t, now, source)
	if observation.Generation != 7 || !observation.ObservedAt.Equal(now.Add(-time.Second)) {
		t.Fatalf("observation identity = generation %d at %v", observation.Generation, observation.ObservedAt)
	}
	if len(observation.Candidates) != 1 {
		t.Fatalf("candidate count = %d", len(observation.Candidates))
	}
	got := observation.Candidates[0]
	if got.StableID == "" || got.StableID == string(source.Identity) || strings.Contains(got.StableID, "worker") {
		t.Fatalf("StableID leaked or omitted source identity: %q", got.StableID)
	}
	if got.Provenance != ServeMeasured || !got.ObservedAt.Equal(source.ObservedAt) || !got.ValidUntil.Equal(source.ValidUntil) || got.Generation != source.Generation || !got.Fresh {
		t.Fatalf("projected evidence = %#v", got)
	}
	wantStableIDSum := sha256.Sum256(source.Identity)
	wantStableID := hex.EncodeToString(wantStableIDSum[:sha256.Size/2])
	if got.StableID != wantStableID {
		t.Fatalf("StableID = %q, want content-free source identity %q", got.StableID, wantStableID)
	}
}

func TestServeAdapterObserveFailsClosed(t *testing.T) {
	now := serveTestTime()
	tests := []struct {
		name   string
		source func() ServeSourceCandidate
	}{
		{"stale", func() ServeSourceCandidate { c := validServeSource("stale", now); c.ValidUntil = now; return c }},
		{"malformed identity", func() ServeSourceCandidate { c := validServeSource("x", now); c.Identity = nil; return c }},
		{"content-bearing identity over limit", func() ServeSourceCandidate {
			c := validServeSource("x", now)
			c.Identity = []byte(strings.Repeat("payload", 40))
			return c
		}},
		{"unknown provenance", func() ServeSourceCandidate {
			c := validServeSource("unknown", now)
			c.Provenance = ServeProvenance("observed-ish")
			return c
		}},
		{"non fak native", func() ServeSourceCandidate { c := validServeSource("foreign", now); c.Engine = "llama_cpp"; return c }},
		{"invalid capacity", func() ServeSourceCandidate {
			c := validServeSource("capacity", now)
			c.QueueDepth = c.QueueCapacity + 1
			return c
		}},
		{"missing generation", func() ServeSourceCandidate { c := validServeSource("generation", now); c.Generation = 0; return c }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := NewServeAdapter(true, serveObserverFunc(func(context.Context) (ServeSourceSnapshot, error) {
				return ServeSourceSnapshot{Generation: 7, CapturedAt: now, Candidates: []ServeSourceCandidate{tt.source()}}, nil
			}), nil, nil)
			adapter.now = func() time.Time { return now }
			if _, err := adapter.Observe(context.Background()); err == nil {
				t.Fatal("Observe admitted invalid evidence")
			}
		})
	}

	tooMany := make([]ServeSourceCandidate, maxServeCandidates+1)
	for i := range tooMany {
		tooMany[i] = validServeSource(string(rune(i+1)), now)
	}
	adapter := NewServeAdapter(true, serveObserverFunc(func(context.Context) (ServeSourceSnapshot, error) {
		return ServeSourceSnapshot{Generation: 7, CapturedAt: now, Candidates: tooMany}, nil
	}), nil, nil)
	adapter.now = func() time.Time { return now }
	if _, err := adapter.Observe(context.Background()); err == nil {
		t.Fatal("Observe admitted an over-limit candidate set")
	}
}

func TestServeAdapterQueuePressureReducesFanoutAndAlternativePlacement(t *testing.T) {
	now := serveTestTime()
	t.Run("queue backpressure propagates into harness pressure", func(t *testing.T) {
		candidate := validServeSource("pressured", now)
		candidate.Backpressure = true
		observation := serveObservationFromSources(t, now, candidate)
		decision := planServe(t, now, observation, validServeRequest(now))
		if decision.Action != ServeDefer || decision.HarnessPressure.Concurrency != 2 {
			t.Fatalf("decision = action %q pressure %#v", decision.Action, decision.HarnessPressure)
		}
		if decision.Reason != "queue backpressure requires delayed fan-out" {
			t.Fatalf("reason = %q", decision.Reason)
		}
	})

	t.Run("lower pressure selects alternative compute", func(t *testing.T) {
		first := validServeSource("first", now)
		second := validServeSource("second", now)
		second.CapacitySnapshot = 12
		observation := serveObservationFromSources(t, now, first, second)
		observation.Candidates[0].QueueDepth = 8
		observation.Candidates[1].QueueDepth = 0
		decision := planServe(t, now, observation, validServeRequest(now))
		if decision.Action != ServeReroute || decision.Selected == nil || decision.Selected.StableID != observation.Candidates[1].StableID {
			t.Fatalf("alternative placement decision = %#v", decision)
		}
	})
}

func TestServeAdapterCacheAffinityBreaksEquivalentReadyTie(t *testing.T) {
	now := serveTestTime()
	low := validServeSource("low-affinity", now)
	high := validServeSource("high-affinity", now)
	high.CapacitySnapshot = 12
	observation := serveObservationFromSources(t, now, low, high)
	observation.Candidates[0].CacheAffinity = 1
	observation.Candidates[1].CacheAffinity = 99
	decision := planServe(t, now, observation, validServeRequest(now))
	if decision.Selected == nil || decision.Selected.StableID != observation.Candidates[1].StableID {
		t.Fatalf("selected = %#v, want high-affinity candidate %#v", decision.Selected, observation.Candidates[1])
	}
	if decision.Action != ServeReroute {
		t.Fatalf("action = %q, want %q", decision.Action, ServeReroute)
	}
}

func TestServeAdapterDecisionCarriesDeadlineBudgetAndRejectionClass(t *testing.T) {
	now := serveTestTime()
	candidate := validServeSource("decision", now)
	observation := serveObservationFromSources(t, now, candidate)

	request := validServeRequest(now)
	decision := planServe(t, now, observation, request)
	if !decision.Deadline.Equal(request.Deadline) || decision.BudgetImpact.Tokens != 160 || decision.BudgetImpact.Concurrency != 4 || decision.BudgetImpact.Delay != 25*time.Millisecond {
		t.Fatalf("decision impact = deadline %v budget %#v", decision.Deadline, decision.BudgetImpact)
	}
	if decision.Rejection != ServeNotRejected {
		t.Fatalf("admission rejection = %q", decision.Rejection)
	}

	request.Deadline = now.Add(10 * time.Millisecond)
	retryable := planServe(t, now, observation, request)
	if retryable.Action != ServeReject || retryable.Rejection != ServeRetryable {
		t.Fatalf("deadline rejection = %#v", retryable)
	}

	request = validServeRequest(now)
	request.Pressure.Cancelled = true
	terminal := planServe(t, now, observation, request)
	if terminal.Action != ServeReject || terminal.Rejection != ServeTerminal {
		t.Fatalf("cancelled rejection = %#v", terminal)
	}
}

func TestServeAdapterApplyIdempotent(t *testing.T) {
	now := serveTestTime()
	source := validServeSource("apply", now)
	observer := serveObserverFunc(func(context.Context) (ServeSourceSnapshot, error) {
		return ServeSourceSnapshot{Generation: 7, CapturedAt: now, Candidates: []ServeSourceCandidate{source}}, nil
	})
	var calls atomic.Int32
	adapter := NewServeAdapter(true, observer, nil, serveExecutorFunc(func(context.Context, ServeDecision) error {
		calls.Add(1)
		return nil
	}))
	adapter.now = func() time.Time { return now }
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	decision, err := adapter.Plan(context.Background(), observation, validServeRequest(now))
	if err != nil {
		t.Fatal(err)
	}
	first := adapter.Apply(context.Background(), decision)
	second := adapter.Apply(context.Background(), decision)
	if first.Status != ServeApplied || second.Status != ServeApplied || calls.Load() != 1 {
		t.Fatalf("apply results = %#v, %#v; executor calls = %d", first, second, calls.Load())
	}
}

func TestServeAdapterApplyDetectsChangedGenerationAndCapacity(t *testing.T) {
	now := serveTestTime()
	for _, tt := range []struct {
		name       string
		mutate     func(*ServeSourceSnapshot)
		wantStatus ServeApplyStatus
	}{
		{"generation superseded", func(snapshot *ServeSourceSnapshot) { snapshot.Generation++ }, ServeSuperseded},
		{"capacity unavailable", func(snapshot *ServeSourceSnapshot) { snapshot.Candidates[0].CapacitySnapshot++ }, ServeUnavailable},
	} {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := ServeSourceSnapshot{Generation: 7, CapturedAt: now, Candidates: []ServeSourceCandidate{validServeSource(tt.name, now)}}
			observer := serveObserverFunc(func(context.Context) (ServeSourceSnapshot, error) { return snapshot, nil })
			adapter := NewServeAdapter(true, observer, nil, nil)
			adapter.now = func() time.Time { return now }
			observation, err := adapter.Observe(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			decision, err := adapter.Plan(context.Background(), observation, validServeRequest(now))
			if err != nil {
				t.Fatal(err)
			}
			tt.mutate(&snapshot)
			if got := adapter.Apply(context.Background(), decision); got.Status != tt.wantStatus {
				t.Fatalf("Apply status = %q (%s), want %q", got.Status, got.Error, tt.wantStatus)
			}
		})
	}
}

func TestServeAdapterApplySameKeyRaceExecutesOnce(t *testing.T) {
	now := serveTestTime()
	snapshot := ServeSourceSnapshot{Generation: 7, CapturedAt: now, Candidates: []ServeSourceCandidate{validServeSource("race", now)}}
	entered := make(chan struct{}, 2)
	release := make(chan struct{})
	var calls atomic.Int32
	adapter := NewServeAdapter(true, serveObserverFunc(func(context.Context) (ServeSourceSnapshot, error) {
		return snapshot, nil
	}), nil, serveExecutorFunc(func(context.Context, ServeDecision) error {
		calls.Add(1)
		entered <- struct{}{}
		<-release
		return nil
	}))
	adapter.now = func() time.Time { return now }
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	decision, err := adapter.Plan(context.Background(), observation, validServeRequest(now))
	if err != nil {
		t.Fatal(err)
	}

	results := make(chan ServeApplyResult, 2)
	var wg sync.WaitGroup
	wg.Add(2)
	for range 2 {
		go func() {
			defer wg.Done()
			results <- adapter.Apply(context.Background(), decision)
		}()
	}
	<-entered
	select {
	case <-entered:
		// Both calls entered the side-effecting executor. The assertion below
		// preserves the issue regression until Apply serializes same-key work.
	case <-time.After(250 * time.Millisecond):
	}
	close(release)
	wg.Wait()
	close(results)
	for result := range results {
		if result.Status != ServeApplied {
			t.Fatalf("racing Apply status = %q (%s)", result.Status, result.Error)
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("same-key racing Apply executed %d times, want exactly once", calls.Load())
	}
}

func TestServeAdapterCoordinationOffDelegatesLegacyAdmission(t *testing.T) {
	now := serveTestTime()
	observation := serveObservationFromSources(t, now, validServeSource("legacy", now))
	request := validServeRequest(now)
	called := false
	legacy := legacyServeFunc(func(_ context.Context, gotObservation ServeObservation, gotRequest ServePlanRequest) (ServeDecision, error) {
		called = true
		if gotObservation.Generation != observation.Generation || gotRequest.IdempotencyKey != request.IdempotencyKey {
			t.Fatalf("legacy inputs = %#v, %#v", gotObservation, gotRequest)
		}
		return finishServeDecision(ServeDecision{
			Action:         ServeAdmit,
			Reason:         "legacy admission",
			Deadline:       request.Deadline,
			Rejection:      ServeNotRejected,
			Observation:    observation.Generation,
			IdempotencyKey: request.IdempotencyKey,
		}), nil
	})
	adapter := NewServeAdapter(false, nil, legacy, nil)
	adapter.now = func() time.Time { return now }
	decision, err := adapter.Plan(context.Background(), observation, request)
	if err != nil {
		t.Fatal(err)
	}
	if !called || !decision.LegacyDelegated || decision.Reason != "legacy admission" {
		t.Fatalf("legacy decision = %#v, called = %v", decision, called)
	}
}

func TestServeAdapterCapturedTwoWorkerFixtureChangesAction(t *testing.T) {
	now := serveTestTime()
	cold := validServeSource("cold-primary", now)
	cold.CacheAffinity = 0
	warm := validServeSource("warm-alternative", now)
	warm.CapacitySnapshot = 12
	warm.CacheAffinity = 50
	warm.PrefillAvailable = 2048
	warm.DecodeAvailable = 512
	observation := serveObservationFromSources(t, now, cold, warm)

	request := validServeRequest(now)
	request.Intent.Workers = []HarnessWorker{{ID: "capture", Role: "context"}, {ID: "serve", Role: "compute"}}
	request.Intent.Fanout = 2
	request.Intent.Concurrency = 2
	request.Pressure.Concurrency = 2
	request.Intent.Requirements = NeutralRequirements{Cache: []string{"shared-context"}, Placement: []string{EngineFakNative}, Serve: []string{"admit"}}
	request.Context = ServeContextCapture{Captured: true, Generation: 5, ReusableBytes: 1024, CacheKey: "capture-5"}
	request.Requirements = ServeProjectionRequirements{PrefillTokens: 1024, DecodeTokens: 256, BatchSize: 2}

	decision := planServe(t, now, observation, request)
	if decision.Action != ServeReroute || decision.Selected == nil || decision.Selected.StableID != observation.Candidates[1].StableID {
		t.Fatalf("captured two-worker fixture decision = %#v", decision)
	}
	if decision.BudgetImpact.Concurrency != 2 || decision.BudgetImpact.Tokens != 1280 {
		t.Fatalf("harness projection impact = %#v", decision.BudgetImpact)
	}
}

func TestServeAdapterPlanFailsClosedOnMissingOrStaleEvidence(t *testing.T) {
	now := serveTestTime()
	base := serveObservationFromSources(t, now, validServeSource("plan-validation", now))
	request := validServeRequest(now)
	tests := []struct {
		name   string
		mutate func(*ServeObservation, *ServePlanRequest)
	}{
		{"non fak native", func(o *ServeObservation, _ *ServePlanRequest) { o.Candidates[0].Engine = "foreign" }},
		{"stale", func(o *ServeObservation, _ *ServePlanRequest) { o.Candidates[0].ValidUntil = now }},
		{"malformed identity", func(o *ServeObservation, _ *ServePlanRequest) { o.Candidates[0].StableID = "" }},
		{"missing generation", func(o *ServeObservation, _ *ServePlanRequest) { o.Candidates[0].Generation = 0 }},
		{"missing capacity evidence", func(o *ServeObservation, _ *ServePlanRequest) { o.Candidates[0].CapacitySnapshot = 0 }},
		{"missing captured context evidence", func(_ *ServeObservation, r *ServePlanRequest) {
			r.Context = ServeContextCapture{Captured: true, ReusableBytes: 1}
		}},
		{"invalid requirements", func(_ *ServeObservation, r *ServePlanRequest) { r.Requirements.PrefillTokens = maxServeCapacity + 1 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			observation := base
			observation.Candidates = append([]ServeCandidate(nil), base.Candidates...)
			gotRequest := request
			tt.mutate(&observation, &gotRequest)
			adapter := NewServeAdapter(true, nil, nil, nil)
			adapter.now = func() time.Time { return now }
			if _, err := adapter.Plan(context.Background(), observation, gotRequest); err == nil {
				t.Fatal("Plan admitted incomplete or invalid evidence")
			}
		})
	}
}

func TestServeAdapterSelfCheckDeterministicAndContentFree(t *testing.T) {
	adapter := NewServeAdapter(true, nil, nil, nil)
	first := adapter.SelfCheck()
	second := adapter.SelfCheck()
	if !first.Passed || first.Action != ServeDefer || first.Error != "" {
		t.Fatalf("SelfCheck = %#v", first)
	}
	digest, err := hex.DecodeString(first.Digest)
	if first != second || err != nil || len(digest) != sha256.Size {
		t.Fatalf("SelfCheck not deterministic: first %#v second %#v", first, second)
	}
	for _, forbidden := range []string{"prompt", "payload", "private", "context"} {
		if strings.Contains(strings.ToLower(first.Digest), forbidden) {
			t.Fatalf("SelfCheck digest contains content marker %q: %q", forbidden, first.Digest)
		}
	}
}
