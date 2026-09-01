package wipreadiness

import (
	"context"
	"reflect"
	"testing"
	"time"
)

type countingScanner struct {
	observation Observation
	calls       int
}

func (s *countingScanner) Scan(context.Context) (Observation, error) {
	s.calls++
	return s.observation, nil
}

func healthyObservation(now time.Time) Observation {
	source := func(name, schema string, total int) Source {
		return Source{Name: name, Schema: schema, ExpectedSchema: schema, ObservedAt: now, Available: true, Complete: true, Summary: Summary{Total: total}}
	}
	return Observation{
		ObservedAt: now,
		Queue:      source("queue", "fak-wip-queue/1", 3), Inventory: source("inventory", "fak-wip-inventory/1", 4),
		Lifecycle: source("lifecycle", "fak-wip-lifecycle/1", 2), Capacity: source("capacity", "fak-wip-capacity/1", 5),
		Hosts: HostCoverage{Expected: []string{"local", "gpu"}, Observed: []string{"local", "gpu"}},
	}
}

func TestCurrentReceiptCoversCanonicalEvidenceAndAdmitsFreshStart(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	receipt := Build(healthyObservation(now), now, 5*time.Minute)
	if receipt.Schema != Schema || receipt.Verdict != VerdictCurrent || receipt.ObservedAt != now || receipt.ExpiresAt != now.Add(5*time.Minute) {
		t.Fatalf("unexpected receipt header: %+v", receipt)
	}
	if receipt.Queue.Summary.Total != 3 || receipt.Inventory.Summary.Total != 4 || receipt.Lifecycle.Summary.Total != 2 || receipt.Capacity.Summary.Total != 5 {
		t.Fatalf("receipt omitted a canonical summary: %+v", receipt)
	}
	if len(receipt.Hosts.Observed) != 2 || !receipt.EvidenceOnly || len(receipt.Diagnostics) != 0 {
		t.Fatalf("receipt omitted coverage or diagnostics state: %+v", receipt)
	}
	if got := Admit(&receipt, AdmissionRequest{Intent: IntentFreshStart}); !got.Admitted || got.Exempt || got.Overridden {
		t.Fatalf("healthy fresh start not admitted: %+v", got)
	}
}

func TestReadinessFailuresHaveStableReasons(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name   string
		mutate func(*Observation)
		reason ReasonCode
	}{
		{"stale", func(o *Observation) { o.ObservedAt = now.Add(-10 * time.Minute) }, ReasonStale},
		{"partial host", func(o *Observation) { o.Hosts.Observed = []string{"local"} }, ReasonPartialHost},
		{"unknown schema", func(o *Observation) { o.Inventory.Schema = "future/9" }, ReasonUnknownSchema},
		{"unavailable", func(o *Observation) { o.Capacity.Available = false }, ReasonUnavailable},
		{"diagnostically incomplete", func(o *Observation) { o.Lifecycle.Diagnostics = []Diagnostic{{Code: "corrupt-row"}} }, ReasonDiagnosticallyIncomplete},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := healthyObservation(now)
			tt.mutate(&o)
			r := Build(o, now, 5*time.Minute)
			if r.Verdict != VerdictBlocked || !containsReason(r.Reasons, tt.reason) {
				t.Fatalf("got verdict=%q reasons=%v, want %q", r.Verdict, r.Reasons, tt.reason)
			}
		})
	}
	if got := Admit(nil, AdmissionRequest{Intent: IntentFreshStart}); got.Admitted || !reflect.DeepEqual(got.Reasons, []ReasonCode{ReasonMissing}) {
		t.Fatalf("missing receipt decision = %+v", got)
	}
	unknown := Build(healthyObservation(now), now, time.Minute)
	unknown.Schema = "future/9"
	if got := Admit(&unknown, AdmissionRequest{Intent: IntentFreshStart}); got.Admitted || !containsReason(got.Reasons, ReasonUnknownSchema) {
		t.Fatalf("unknown receipt schema decision = %+v", got)
	}
}

func TestRetirementIntentsRemainAdmittedAndFreshStartRequiresWitnessedOverride(t *testing.T) {
	failure := Receipt{Schema: Schema, Verdict: VerdictBlocked, Reasons: []ReasonCode{ReasonStale}}
	for _, intent := range []Intent{IntentRecovery, IntentLanding, IntentSafety, IntentParking, IntentAlreadyOwnedContinuation} {
		if got := Admit(&failure, AdmissionRequest{Intent: intent}); !got.Admitted || !got.Exempt {
			t.Fatalf("intent %q deadlocked: %+v", intent, got)
		}
	}
	if got := Admit(&failure, AdmissionRequest{Intent: IntentFreshStart}); got.Admitted {
		t.Fatalf("unwitnessed override admitted: %+v", got)
	}
	got := Admit(&failure, AdmissionRequest{Intent: IntentFreshStart, OverrideReason: "operator accepted local-only evidence"})
	if !got.Admitted || !got.Overridden || got.OverrideReason == "" || !containsReason(got.Reasons, ReasonStale) {
		t.Fatalf("explicit override not recorded: %+v", got)
	}
}

func TestCacheReusesScanDuringValidityWindow(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	clock := now
	scanner := &countingScanner{observation: healthyObservation(now)}
	cache := NewCacheWithClock(scanner, 5*time.Minute, func() time.Time { return clock })
	for range 4 {
		r := cache.Receipt(context.Background())
		if !Admit(&r, AdmissionRequest{Intent: IntentFreshStart}).Admitted {
			t.Fatal("cached healthy receipt was not admitted")
		}
	}
	if scanner.calls != 1 {
		t.Fatalf("scanner calls=%d, want 1", scanner.calls)
	}
	clock = now.Add(6 * time.Minute)
	cache.Receipt(context.Background())
	if scanner.calls != 2 {
		t.Fatalf("scanner calls after expiry=%d, want 2", scanner.calls)
	}
}

func TestDirtyAndRemoteWorkIsEvidenceOnlyAndPreserved(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	o := healthyObservation(now)
	o.Work = []Work{
		{ID: "clean-local", Ownership: OwnershipLocal},
		{ID: "dirty-local", Dirty: true, Ownership: OwnershipLocal},
		{ID: "remote-owned", Ownership: OwnershipRemote, Host: "gpu"},
	}
	r := Build(o, now, time.Minute)
	got := r.PreservedWork()
	want := []Work{o.Work[1], o.Work[2]}
	if !r.EvidenceOnly || !reflect.DeepEqual(got, want) {
		t.Fatalf("preserved work=%+v evidence_only=%v, want %+v", got, r.EvidenceOnly, want)
	}
}

func containsReason(reasons []ReasonCode, want ReasonCode) bool {
	for _, reason := range reasons {
		if reason == want {
			return true
		}
	}
	return false
}
