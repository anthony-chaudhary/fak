package modelperfobs

import (
	"context"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestStateCampaignProvesColdWarmInvalidatedAndPressureEvicted(t *testing.T) {
	ctx := context.Background()
	clock := newStepClock()
	backend := NewLocalWorkflowBackend(clock.Now)
	prefix := []int{11, 22, 33, 44}
	primeStateBackend(t, ctx, backend, prefix)
	report := RunStateCampaign(ctx, StateRunner{Backend: backend, Now: clock.Now}, StateLayerWorkflow, prefix, []StateTransition{
		TransitionColdStart,
		TransitionWarmAdmit,
		TransitionExplicitInvalidate,
		TransitionPressureEvict,
	}, testStateProvenance(clock.Now()))
	if err := ValidateStateReport(report); err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "admitted" {
		t.Fatalf("verdict = %q, want admitted", report.Verdict)
	}
	want := []StateTransition{
		TransitionColdStart,
		TransitionWarmAdmit,
		TransitionExplicitInvalidate,
		TransitionPressureEvict,
	}
	got := make([]StateTransition, 0, len(report.Arms))
	for _, arm := range report.Arms {
		got = append(got, arm.Receipt.Transition)
		if !arm.MeasurementIncluded || arm.Receipt.Result != TransitionProved {
			t.Fatalf("arm %q was not admitted: %+v", arm.Label, arm)
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("transitions = %v, want %v", got, want)
	}
	cold, warm := report.Arms[0].Receipt, report.Arms[1].Receipt
	if cold.PreMetrics == nil || cold.PreMetrics.Entries != 1 || cold.Probes[0].Reused {
		t.Fatalf("cold receipt did not prove warm-to-cold reset: %+v", cold)
	}
	if len(warm.Probes) != 2 || warm.Probes[0].TokenPrefixDigest != warm.Probes[1].TokenPrefixDigest || !warm.Probes[1].Reused {
		t.Fatalf("warm receipt did not prove pinned-prefix reuse: %+v", warm)
	}
	evicted := report.Arms[3].Receipt
	if evicted.PostMetrics == nil || evicted.PreMetrics == nil || evicted.PostMetrics.Evictions <= evicted.PreMetrics.Evictions || !strings.Contains(evicted.Mechanism.Detail, "competing prefix") {
		t.Fatalf("pressure receipt did not prove capacity eviction: %+v", evicted)
	}
	t.Logf("STATE_TRANSITION_WITNESS verdict=%s transitions=%v provenance=%s scope=%s", report.Verdict, got, report.Provenance.EvidenceKind, report.Provenance.Scope)
}

func TestStateIneffectiveZeroExitResetIsUnproved(t *testing.T) {
	ctx := context.Background()
	clock := newStepClock()
	base := NewLocalWorkflowBackend(clock.Now)
	prefix := []int{8, 4, 2, 6}
	primeStateBackend(t, ctx, base, prefix)
	backend := ineffectiveColdBackend{StateBackend: base}
	receipt := (StateRunner{Backend: backend, Now: clock.Now}).RunTransition(ctx, stateTransitionRequest(TransitionColdStart, prefix))
	if !receipt.Mechanism.ExitOK {
		t.Fatal("fixture must model a reset command that exits zero")
	}
	if receipt.Result != TransitionUnproved || receipt.Reason != "reuse_observed_after_cold" {
		t.Fatalf("ineffective reset = %s/%s, want transition_unproved/reuse_observed_after_cold", receipt.Result, receipt.Reason)
	}
	report := RunStateCampaign(ctx, StateRunner{Backend: backend, Now: clock.Now}, StateLayerWorkflow, prefix, []StateTransition{TransitionColdStart}, testStateProvenance(clock.Now()))
	if report.Arms[0].MeasurementIncluded || report.Verdict != "invalid_arms" {
		t.Fatalf("unproved cold arm counted: %+v", report.Arms[0])
	}
}

func TestStateRejectsStaleMetrics(t *testing.T) {
	ctx := context.Background()
	clock := newStepClock()
	base := NewLocalWorkflowBackend(clock.Now)
	prefix := []int{1, 2, 3}
	primeStateBackend(t, ctx, base, prefix)
	backend := &staleSnapshotBackend{StateBackend: base}
	receipt := (StateRunner{Backend: backend, Now: clock.Now}).RunTransition(ctx, stateTransitionRequest(TransitionColdStart, prefix))
	if receipt.Result != TransitionUnproved || receipt.Reason != "stale_metrics" {
		t.Fatalf("stale metrics = %s/%s, want transition_unproved/stale_metrics", receipt.Result, receipt.Reason)
	}
}

func TestStateRejectsCounterReset(t *testing.T) {
	ctx := context.Background()
	clock := newStepClock()
	base := NewLocalWorkflowBackend(clock.Now)
	prefix := []int{1, 3, 5}
	primeStateBackend(t, ctx, base, prefix)
	backend := &counterResetBackend{StateBackend: base}
	receipt := (StateRunner{Backend: backend, Now: clock.Now}).RunTransition(ctx, stateTransitionRequest(TransitionColdStart, prefix))
	if receipt.Result != TransitionUnproved || receipt.Reason != "counter_reset" {
		t.Fatalf("counter reset = %s/%s, want transition_unproved/counter_reset", receipt.Result, receipt.Reason)
	}
}

func TestStateRejectsConcurrentTrafficContamination(t *testing.T) {
	ctx := context.Background()
	clock := newStepClock()
	base := NewLocalWorkflowBackend(clock.Now)
	prefix := []int{2, 4, 6}
	primeStateBackend(t, ctx, base, prefix)
	backend := &contaminatingProbeBackend{StateBackend: base}
	receipt := (StateRunner{Backend: backend, Now: clock.Now}).RunTransition(ctx, stateTransitionRequest(TransitionColdStart, prefix))
	if receipt.Result != TransitionUnproved || receipt.Reason != "concurrent_traffic_contamination" {
		t.Fatalf("contaminated transition = %s/%s, want transition_unproved/concurrent_traffic_contamination", receipt.Result, receipt.Reason)
	}
}

func TestStateUnsupportedLayerStaysUnsupported(t *testing.T) {
	clock := newStepClock()
	backend := NewLocalWorkflowBackend(clock.Now)
	receipt := (StateRunner{Backend: backend, Now: clock.Now}).RunTransition(context.Background(), TransitionRequest{
		Layer:      StateLayerProviderPrompt,
		Transition: TransitionExplicitInvalidate,
		Probe:      ProbeRequest{TokenPrefix: []int{1}},
	})
	if receipt.Result != TransitionUnsupported || receipt.Reason != "backend_does_not_support_transition" {
		t.Fatalf("unsupported transition = %s/%s", receipt.Result, receipt.Reason)
	}
	expiry := (StateRunner{Backend: backend, Now: clock.Now}).RunTransition(context.Background(), TransitionRequest{
		Layer:      StateLayerWorkflow,
		Transition: TransitionNaturalExpiry,
		Probe:      ProbeRequest{TokenPrefix: []int{1}},
	})
	if expiry.Result != TransitionUnsupported {
		t.Fatalf("unobservable natural expiry = %s, want unsupported", expiry.Result)
	}
}

func TestCapturedStateWitnessIsProvenanceHonest(t *testing.T) {
	f, err := os.Open("testdata/cache-state-witness.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	report, err := ReadStateReport(f)
	if err != nil {
		t.Fatal(err)
	}
	if report.Provenance.EvidenceKind != EvidenceObserved || report.Provenance.Scope != "in_process_fak_workflow_cache" {
		t.Fatalf("captured provenance overclaims its scope: %+v", report.Provenance)
	}
	if report.Provenance.ExternalBackendClaims {
		t.Fatal("hermetic witness claimed an external backend")
	}
	if !strings.Contains(report.Provenance.Note, "does not claim external KV or provider-cache behavior") {
		t.Fatalf("missing provenance boundary: %q", report.Provenance.Note)
	}
}

func TestStateWitnessVerifierRejectsTamperedWarmProbe(t *testing.T) {
	f, err := os.Open("testdata/cache-state-witness.json")
	if err != nil {
		t.Fatal(err)
	}
	report, err := ReadStateReport(f)
	_ = f.Close()
	if err != nil {
		t.Fatal(err)
	}
	report.Arms[1].Receipt.Probes[1].Reused = false
	report.Arms[1].Receipt.Probes[1].ReusedTokens = 0
	if err := ValidateStateReport(report); err == nil || !strings.Contains(err.Error(), "does not reproduce") {
		t.Fatalf("tampered warm receipt validation error = %v", err)
	}
}

func BenchmarkStateTransitionCampaign(b *testing.B) {
	ctx := context.Background()
	for i := 0; i < b.N; i++ {
		backend := NewLocalWorkflowBackend(time.Now)
		prefix := []int{101, 8426, 17, 23, 42, 99}
		_, _ = backend.Probe(ctx, StateLayerWorkflow, ProbeRequest{TokenPrefix: prefix})
		report := RunStateCampaign(ctx, StateRunner{Backend: backend}, StateLayerWorkflow, prefix, []StateTransition{
			TransitionColdStart,
			TransitionWarmAdmit,
			TransitionExplicitInvalidate,
			TransitionPressureEvict,
		}, testStateProvenance(time.Now()))
		if report.Verdict != "admitted" {
			b.Fatalf("benchmark campaign verdict = %q", report.Verdict)
		}
	}
}

type stepClock struct {
	now time.Time
}

func newStepClock() *stepClock {
	return &stepClock{now: time.Date(2026, 8, 21, 0, 0, 0, 0, time.UTC)}
}

func (c *stepClock) Now() time.Time {
	c.now = c.now.Add(time.Millisecond)
	return c.now
}

func testStateProvenance(capturedAt time.Time) StateProvenance {
	return StateProvenance{
		EvidenceKind: EvidenceObserved,
		Scope:        "in_process_fak_workflow_cache",
		Command:      "go test ./internal/modelperfobs",
		GoVersion:    "test",
		GOOS:         "test",
		GOARCH:       "test",
		CodeState:    "test",
		CapturedAt:   capturedAt,
		Note:         "Observed from the running hermetic workflow-cache backend; it does not claim external KV or provider-cache behavior.",
	}
}

func stateTransitionRequest(transition StateTransition, prefix []int) TransitionRequest {
	return TransitionRequest{
		Layer:      StateLayerWorkflow,
		Transition: transition,
		Probe:      ProbeRequest{TokenPrefix: append([]int(nil), prefix...)},
	}
}

func primeStateBackend(t *testing.T, ctx context.Context, backend StateBackend, prefix []int) {
	t.Helper()
	if _, err := backend.Probe(ctx, StateLayerWorkflow, ProbeRequest{TokenPrefix: prefix}); err != nil {
		t.Fatal(err)
	}
}

type ineffectiveColdBackend struct {
	StateBackend
}

func (b ineffectiveColdBackend) Apply(ctx context.Context, req TransitionRequest) (MechanismReceipt, error) {
	if req.Transition == TransitionColdStart {
		return MechanismReceipt{Name: "ineffective-zero-exit-reset", ExitOK: true}, nil
	}
	return b.StateBackend.Apply(ctx, req)
}

type staleSnapshotBackend struct {
	StateBackend
	calls         int
	first         time.Time
	firstSequence uint64
}

func (b *staleSnapshotBackend) Snapshot(ctx context.Context, layer StateLayer) (MetricSnapshot, error) {
	snapshot, err := b.StateBackend.Snapshot(ctx, layer)
	if err != nil {
		return snapshot, err
	}
	b.calls++
	if b.calls == 1 {
		b.first = snapshot.CapturedAt
		b.firstSequence = snapshot.SampleSequence
	} else {
		snapshot.CapturedAt = b.first
		snapshot.SampleSequence = b.firstSequence
	}
	return snapshot, nil
}

type counterResetBackend struct {
	StateBackend
	calls int
}

func (b *counterResetBackend) Snapshot(ctx context.Context, layer StateLayer) (MetricSnapshot, error) {
	snapshot, err := b.StateBackend.Snapshot(ctx, layer)
	b.calls++
	if b.calls > 1 {
		snapshot.CounterEpoch = "reset-process-2"
	}
	return snapshot, err
}

type contaminatingProbeBackend struct {
	StateBackend
}

func (b *contaminatingProbeBackend) Probe(ctx context.Context, layer StateLayer, req ProbeRequest) (ProbeObservation, error) {
	probe, err := b.StateBackend.Probe(ctx, layer, req)
	if err != nil {
		return probe, err
	}
	_, _ = b.StateBackend.Probe(ctx, layer, ProbeRequest{TokenPrefix: []int{999, 1000}})
	return probe, nil
}
