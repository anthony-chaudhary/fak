package contextq

import (
	"testing"
	"time"
)

// TestCapabilityLedger_ResidencyTracking tests C3: per-capability residency tracking.
func TestCapabilityLedger_ResidencyTracking(t *testing.T) {
	ledger := NewCapabilityLedger()

	ref1 := CapRef{Name: "test-skill", Source: "skill", Card: 0, IsQuery: false}
	now := time.Now().UnixNano()
	ledger.RecordFault(ref1, now)

	snap := ledger.Query()
	if len(snap.Spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(snap.Spans))
	}
	if snap.Spans[0].State != CapStateResident {
		t.Fatalf("expected state %q, got %q", CapStateResident, snap.Spans[0].State)
	}
	if snap.Spans[0].Faults != 1 {
		t.Fatalf("expected faults 1, got %d", snap.Spans[0].Faults)
	}

	ledger.RecordFault(ref1, now)
	snap = ledger.Query()
	if snap.Spans[0].Faults != 2 {
		t.Fatalf("expected faults 2, got %d", snap.Spans[0].Faults)
	}

	ledger.MarkEvictable(ref1)
	snap = ledger.Query()
	if snap.Spans[0].State != CapStateEvictable {
		t.Fatalf("expected state %q, got %q", CapStateEvictable, snap.Spans[0].State)
	}

	ledger.RecordEvict(ref1)
	snap = ledger.Query()
	if snap.Spans[0].State != CapStateHeld {
		t.Fatalf("expected state %q, got %q", CapStateHeld, snap.Spans[0].State)
	}
}

// TestCapabilityLedger_EvictColdest tests eviction by fault count (coldest-first).
func TestCapabilityLedger_EvictColdest(t *testing.T) {
	ledger := NewCapabilityLedger()

	refHot := CapRef{Name: "hot-skill", Source: "skill", Card: 0, IsQuery: false}
	refCold := CapRef{Name: "cold-skill", Source: "skill", Card: 0, IsQuery: false}
	refMed := CapRef{Name: "med-skill", Source: "skill", Card: 0, IsQuery: false}

	now := time.Now().UnixNano()

	ledger.RecordFault(refCold, now)
	ledger.MarkEvictable(refCold)

	ledger.RecordFault(refMed, now)
	ledger.RecordFault(refMed, now)
	ledger.MarkEvictable(refMed)

	ledger.RecordFault(refHot, now)
	ledger.RecordFault(refHot, now)
	ledger.RecordFault(refHot, now)

	evicted := ledger.EvictColdest(1)
	if len(evicted) != 1 || evicted[0] != refCold {
		t.Fatalf("expected to evict cold-skill, got %v", evicted)
	}

	snap := ledger.Query()
	for _, span := range snap.Spans {
		switch span.CapRef.Name {
		case "cold-skill":
			if span.State != CapStateHeld {
				t.Fatalf("cold-skill should be held, got %q", span.State)
			}
		case "med-skill":
			if span.State != CapStateEvictable {
				t.Fatalf("med-skill should still be evictable, got %q", span.State)
			}
		case "hot-skill":
			if span.State != CapStateResident {
				t.Fatalf("hot-skill should still be resident, got %q", span.State)
			}
		}
	}
}

// TestCapabilityLedger_EvictUnderBudget tests pressure-driven eviction.
func TestCapabilityLedger_EvictUnderBudget(t *testing.T) {
	ledger := NewCapabilityLedger()

	now := time.Now().UnixNano()
	for i := 0; i < 3; i++ {
		name := string(rune('a' + i))
		ref := CapRef{Name: "skill-" + name, Source: "skill", Card: 0, IsQuery: false}
		ledger.RecordFault(ref, now)
		ledger.MarkEvictable(ref)
	}

	// Each skill is estimated at ~8 bytes (name length * 4): "skill-a" = 7*4 = 28, etc.
	// Actually the formula is len(ref.Name) * 4, so:
	// skill-a = 7*4 = 28
	// skill-b = 7*4 = 28
	// skill-c = 7*4 = 28
	// Total = 84 bytes, budget of 40 bytes should evict at least 1
	evicted := ledger.EvictUnderBudget(40)
	if len(evicted) < 1 {
		t.Fatalf("expected at least 1 evicted, got %d", len(evicted))
	}
}