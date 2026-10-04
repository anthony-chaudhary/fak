package ctxmmu

import "testing"

// countingBacking is a ResidencyBacking that counts every Load, so "no payload re-read" is a
// count assertion rather than a belief.
type countingBacking struct {
	loads int
	bytes int64
}

func (b *countingBacking) Load(layer, expert int) (ExpertDescriptor, bool) {
	b.loads++
	return ExpertDescriptor{Layer: layer, Expert: expert, Bytes: b.bytes, Payload: struct{}{}}, true
}

func smallResidencyPolicy(t *testing.T, hot, trans int64, backing ResidencyBacking) *ExpertResidencyPolicy {
	t.Helper()
	p, err := NewExpertResidencyPolicy(ExpertResidencyOptions{HotBytes: hot, TransientBytes: trans}, backing)
	if err != nil {
		t.Fatalf("NewExpertResidencyPolicy: %v", err)
	}
	return p
}

// TestExpertResidencyPrefillSweepLeavesHotSetIntact is acceptance criterion 1: a prefill sweep
// evicts only from the transient ring and never the resident decode hot set.
func TestExpertResidencyPrefillSweepLeavesHotSetIntact(t *testing.T) {
	backing := &countingBacking{bytes: 100}
	p := smallResidencyPolicy(t, 300, 200, backing)

	// Seed a 3-entry resident hot set by decoding three experts.
	for e := 0; e < 3; e++ {
		if _, ok := p.Decode(0, e); !ok {
			t.Fatalf("decode seed expert %d not resident", e)
		}
	}

	// A broad prefill sweep of ten cold experts must churn only the transient ring.
	for e := 0; e < 10; e++ {
		if _, ok := p.Prefill(1, e); !ok {
			t.Fatalf("prefill expert %d missed", e)
		}
	}

	for e := 0; e < 3; e++ {
		if got := p.Tier(0, e); got != TierResident {
			t.Fatalf("hot expert (0,%d) tier=%q after prefill sweep, want resident", e, got)
		}
	}
	st := p.Stats()
	if st.HotEvictions != 0 {
		t.Fatalf("prefill sweep evicted %d hot-set entries, want 0", st.HotEvictions)
	}
	if st.TransientEvicts == 0 {
		t.Fatalf("prefill sweep churned no transient entries; the sweep did not exercise the ring")
	}
	if st.ResidentEntries != 3 {
		t.Fatalf("resident entries=%d after sweep, want 3", st.ResidentEntries)
	}
}

// TestExpertResidencyPromotionNoPayloadReRead is acceptance criterion 2: transient->resident
// promotion moves the descriptor without consulting backing again.
func TestExpertResidencyPromotionNoPayloadReRead(t *testing.T) {
	backing := &countingBacking{bytes: 100}
	p := smallResidencyPolicy(t, 400, 400, backing)

	if _, ok := p.Prefill(2, 7); !ok {
		t.Fatal("prefill (2,7) missed")
	}
	if backing.loads != 1 {
		t.Fatalf("backing loads after prefill=%d, want 1", backing.loads)
	}
	if got := p.Tier(2, 7); got != TierTransient {
		t.Fatalf("tier after prefill=%q, want transient", got)
	}

	if _, ok := p.Decode(2, 7); !ok {
		t.Fatal("decode (2,7) missed after transient staging")
	}
	if backing.loads != 1 {
		t.Fatalf("promotion re-read backing: loads=%d, want 1 (no payload re-read)", backing.loads)
	}
	st := p.Stats()
	if st.Promotions != 1 {
		t.Fatalf("promotions=%d, want 1", st.Promotions)
	}
	if got := p.Tier(2, 7); got != TierResident {
		t.Fatalf("tier after promotion=%q, want resident", got)
	}
	if st.TransientEntries != 0 {
		t.Fatalf("transient entries=%d after promotion, want 0 (entry moved, not copied)", st.TransientEntries)
	}
}

// TestExpertResidencyWarmStartBeatsColdStart is acceptance criterion 4: a trace-seeded warm
// start yields a higher resident-hit count than a cold start over the same decode sequence,
// with the hit/miss/eviction/transient counters exposed.
func TestExpertResidencyWarmStartBeatsColdStart(t *testing.T) {
	// A trace whose layer 0 routes experts 0,1,2 heavily and 3..9 once each.
	var accesses []ExpertAccess
	for i := 0; i < 20; i++ {
		for e := 0; e < 3; e++ {
			accesses = append(accesses, ExpertAccess{Layer: 0, Expert: e})
		}
	}
	for e := 3; e < 10; e++ {
		accesses = append(accesses, ExpertAccess{Layer: 0, Expert: e})
	}

	decodeSeq := []int{0, 1, 2, 0, 1, 2}

	coldBacking := &countingBacking{bytes: 100}
	cold := smallResidencyPolicy(t, 300, 200, coldBacking)
	for _, e := range decodeSeq {
		cold.Decode(0, e)
	}
	coldHits := cold.Stats().ResidentHits

	warmBacking := &countingBacking{bytes: 100}
	warm := smallResidencyPolicy(t, 300, 200, warmBacking)
	if seeded := warm.WarmStart(accesses); seeded == 0 {
		t.Fatal("warm start seeded no experts")
	}
	if warm.Tier(0, 0) != TierResident || warm.Tier(0, 1) != TierResident || warm.Tier(0, 2) != TierResident {
		t.Fatal("warm start did not seed the trace-hot resident experts")
	}
	for _, e := range decodeSeq {
		warm.Decode(0, e)
	}
	warmHits := warm.Stats().ResidentHits

	if warmHits <= coldHits {
		t.Fatalf("warm-start resident hits=%d not greater than cold-start hits=%d", warmHits, coldHits)
	}
	st := warm.Stats()
	if st.ColdMisses == 0 {
		// Warm start seeds from backing, so warm demand loads are 0 here; that is the point.
		t.Logf("warm stats: hits=%d misses=%d promotions=%d hotEvict=%d", st.ResidentHits, st.ColdMisses, st.Promotions, st.HotEvictions)
	}
}
