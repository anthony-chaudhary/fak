package issuepolicy

import (
	"testing"
	"time"
)

func ev(dim, op string, value float64, unit string) EnvelopeValue {
	return EnvelopeValue{Dimension: dim, Operator: op, Value: value, Unit: unit}
}

func TestReconcileScopeExpansionInvalidatesOldProductionCredit(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	r := ReconcileScope(ScopeSnapshot{Key: "model", PreviousTarget: []EnvelopeValue{ev("concurrency", ">=", 100, "requests")}, Target: []EnvelopeValue{ev("concurrency", ">=", 1000, "requests")}, Witnessed: []EnvelopeValue{ev("concurrency", "=", 100, "requests")}, Observed: []EnvelopeValue{ev("concurrency", "=", 800, "requests")}, WitnessedAt: now.Add(-time.Hour), MaxAge: "24h"}, now)
	if r.Status != ScopeExpanded || r.ProductionCreditCurrent || len(r.Changes) == 0 {
		t.Fatalf("reconcile=%+v, want expanded without credit", r)
	}
}

func TestReconcileScopeStableSingleUserRemainsAligned(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	target := []EnvelopeValue{ev("concurrency", ">=", 1, "user")}
	r := ReconcileScope(ScopeSnapshot{Key: "cli", PreviousTarget: target, Target: target, Witnessed: target, Observed: target, WitnessedAt: now.Add(-time.Hour), MaxAge: "168h"}, now)
	if r.Status != ScopeAligned || !r.ProductionCreditCurrent || len(r.Changes) != 0 {
		t.Fatalf("reconcile=%+v, want stable aligned scope", r)
	}
}

func TestReconcileScopeStaleRetainsTimestampButWithholdsCredit(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	target := []EnvelopeValue{ev("concurrency", ">=", 1, "user")}
	witnessedAt := now.Add(-25 * time.Hour)
	r := ReconcileScope(ScopeSnapshot{Target: target, Witnessed: target, WitnessedAt: witnessedAt, MaxAge: "24h"}, now)
	if r.Status != ScopeStale || r.ProductionCreditCurrent || !r.WitnessedAt.Equal(witnessedAt) {
		t.Fatalf("reconcile=%+v, want retained stale witness", r)
	}
}

func TestReconcileScopeContractionRequiresAudit(t *testing.T) {
	now := time.Now().UTC()
	r := ReconcileScope(ScopeSnapshot{PreviousTarget: []EnvelopeValue{ev("concurrency", ">=", 1000, "requests")}, Target: []EnvelopeValue{ev("concurrency", ">=", 100, "requests")}, Witnessed: []EnvelopeValue{ev("concurrency", "=", 100, "requests")}, WitnessedAt: now, MaxAge: "24h"}, now)
	if r.Status != ScopeContracted || r.ProductionCreditCurrent {
		t.Fatalf("reconcile=%+v, want audited contraction", r)
	}
}

func TestReconcileScopeUnknownOnObservedUnitMismatch(t *testing.T) {
	now := time.Now().UTC()
	target := []EnvelopeValue{ev("concurrency", ">=", 1000, "requests")}
	r := ReconcileScope(ScopeSnapshot{Target: target, Witnessed: target, Observed: []EnvelopeValue{ev("concurrency", "=", 1000, "users")}, WitnessedAt: now, MaxAge: "24h"}, now)
	if r.Status != ScopeUnknown || r.ProductionCreditCurrent || len(r.Unknown) == 0 {
		t.Fatalf("reconcile=%+v, want unknown unit mismatch", r)
	}
}

func TestReconcileScopeWitnessGap(t *testing.T) {
	now := time.Now().UTC()
	r := ReconcileScope(ScopeSnapshot{Target: []EnvelopeValue{ev("concurrency", ">=", 1000, "requests")}, Witnessed: []EnvelopeValue{ev("concurrency", "=", 100, "requests")}, WitnessedAt: now, MaxAge: "24h"}, now)
	if r.Status != ScopeGap || len(r.Gaps) != 1 || r.ProductionCreditCurrent {
		t.Fatalf("reconcile=%+v, want gap", r)
	}
}
