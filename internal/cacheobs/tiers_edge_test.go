package cacheobs

import (
	"math"
	"testing"
	"time"
)

func TestRejectedTierObservationEdgeTable(t *testing.T) {
	const oversized = 1 << 20
	invalid := CacheTier(255)

	tests := []struct {
		name         string
		access       TierAccess
		wantRejected uint64
		wantRequests uint64
	}{
		{name: "empty access uses the closed-vocabulary zero values", access: TierAccess{}, wantRequests: 1},
		{name: "oversized tier", wantRejected: 1, access: TierAccess{Tier: CacheTier(oversized), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory}},
		{name: "oversized operation", wantRejected: 1, access: TierAccess{Tier: TierLocalPrefix, Op: TierOp(oversized), Outcome: OutcomeHit, Backend: BackendMemory}},
		{name: "oversized outcome", wantRejected: 1, access: TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: TierOutcome(oversized), Backend: BackendMemory}},
		{name: "oversized backend", wantRejected: 1, access: TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: BackendClass(oversized)}},
		{name: "negative tier", wantRejected: 1, access: TierAccess{Tier: CacheTier(-1), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory}},
		{name: "negative operation", wantRejected: 1, access: TierAccess{Tier: TierLocalPrefix, Op: TierOp(-1), Outcome: OutcomeHit, Backend: BackendMemory}},
		{name: "negative outcome", wantRejected: 1, access: TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: TierOutcome(-1), Backend: BackendMemory}},
		{name: "negative backend", wantRejected: 1, access: TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: BackendClass(-1)}},
		{name: "every dimension malformed", wantRejected: 1, access: TierAccess{Tier: invalid, Op: TierOp(255), Outcome: TierOutcome(255), Backend: BackendClass(255), Bytes: math.MaxInt64, BytesKnown: true, Latency: time.Duration(math.MaxInt64), LatencyKnown: true}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var o Observer
			o.ObserveTier(tt.access)

			stats := o.Snapshot()
			report := o.TierSnapshot()
			if stats.RejectedTierAccesses != tt.wantRejected || report.RejectedTierAccesses != tt.wantRejected {
				t.Fatalf("rejected count mismatch: stats=%d tier=%d want=%d", stats.RejectedTierAccesses, report.RejectedTierAccesses, tt.wantRejected)
			}
			if report.Total.Requests != tt.wantRequests {
				t.Fatalf("request accounting mismatch: total=%+v want_requests=%d", report.Total, tt.wantRequests)
			}
			if tt.wantRejected == 1 {
				for _, tier := range report.Tiers {
					if tier.TierCounters != (TierCounters{}) || len(tier.Ops) != 0 {
						t.Fatalf("invalid access leaked into tier %q: total=%+v rows=%+v", tier.Tier, tier.TierCounters, tier.Ops)
					}
				}
			}
		})
	}
}

func TestRejectedTierObservationAdversarialAccounting(t *testing.T) {
	valid := TierAccess{
		Tier:         TierLocalPrefix,
		Op:           OpRead,
		Outcome:      OutcomeHit,
		Backend:      BackendMemory,
		Bytes:        64,
		BytesKnown:   true,
		Latency:      2 * time.Millisecond,
		LatencyKnown: true,
	}
	invalid := []TierAccess{
		{Tier: CacheTier(-1), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: TierOp(-1), Outcome: OutcomeHit, Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: OpRead, Outcome: TierOutcome(-1), Backend: BackendMemory},
		{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: BackendClass(-1)},
		{Tier: CacheTier(255), Op: TierOp(255), Outcome: TierOutcome(255), Backend: BackendClass(255), Bytes: math.MaxInt64, BytesKnown: true, Latency: time.Duration(math.MaxInt64), LatencyKnown: true},
	}

	var o Observer
	o.ObserveTier(valid)
	for _, access := range invalid {
		o.ObserveTier(access)
	}
	o.ObserveTier(valid)

	stats := o.Snapshot()
	report := o.TierSnapshot()
	wantRejected := uint64(len(invalid))
	if stats.RejectedTierAccesses != wantRejected || report.RejectedTierAccesses != wantRejected {
		t.Fatalf("rejected count mismatch: stats=%d tier=%d want=%d", stats.RejectedTierAccesses, report.RejectedTierAccesses, wantRejected)
	}
	wantTotal := TierCounters{Requests: 2, Hits: 2, Bytes: 128, SizedAccesses: 2, LatencyNanos: uint64(4 * time.Millisecond), TimedAccesses: 2}
	if report.Total.Requests != wantTotal.Requests || report.Total.Hits != wantTotal.Hits || report.Total.Bytes != wantTotal.Bytes || report.Total.SizedAccesses != wantTotal.SizedAccesses || report.Total.LatencyNanos != wantTotal.LatencyNanos || report.Total.TimedAccesses != wantTotal.TimedAccesses {
		t.Fatalf("valid observations changed by hostile rejects: got=%+v want=%+v", report.Total, wantTotal)
	}
}
