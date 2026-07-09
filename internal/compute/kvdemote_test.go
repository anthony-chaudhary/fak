package compute

import "testing"

// demoteLadder is the drop-only ladder — the single evict rung that makes PlanKVDemotion
// degenerate to PickEvictionVictim.
var demoteLadder = []KVTierProfile{{Action: ActionEvict, Available: true}}

// spillRung prices a host tier at restore cost `perByte` per byte of the span.
func spillRung(perByte float64) KVTierProfile {
	return KVTierProfile{Action: ActionSpillHost, Available: true, RestoreCostPerByte: perByte}
}

func TestKVTierActionString(t *testing.T) {
	for _, tc := range []struct {
		action KVTierAction
		want   string
	}{
		{ActionKeep, "keep"},
		{ActionSpillHost, "spill-host"},
		{ActionEvict, "evict"},
		{KVTierAction(200), "kvaction?"},
	} {
		if got := tc.action.String(); got != tc.want {
			t.Errorf("KVTierAction(%d).String() = %q, want %q", tc.action, got, tc.want)
		}
	}
}

// TestPlanKVDemotionSingleTierReducesToPickEvictionVictim is the issue's reduction requirement:
// with only the evict rung configured, the min-cost ACTION planner picks exactly the victim the
// binary drop-only evictor picks — cost ranking, LRU tie-break, and pin/lease exclusions all
// identical. Demotion is a strict generalization of eviction, never a divergence from it.
func TestPlanKVDemotionSingleTierReducesToPickEvictionVictim(t *testing.T) {
	// KVEvictionCost = Tokens*(Hits+1)/Bytes:
	//   0: 100*10/100 = 10.0
	//   1: 100*1/200  =  0.5   \ tie on cost...
	//   2:  50*2/100  =  1.0   |
	//   3: 100*1/200  =  0.5   / ...broken by the older LastUsed (1 < 3) ⇒ span 3 is the victim
	//   4: pinned — the cheapest cost of all (0.001), and excluded by both functions
	spans := []KVSpanStats{
		{Tokens: 100, Bytes: 100, Hits: 9, LastUsed: 5},
		{Tokens: 100, Bytes: 200, Hits: 0, LastUsed: 3},
		{Tokens: 50, Bytes: 100, Hits: 1, LastUsed: 9},
		{Tokens: 100, Bytes: 200, Hits: 0, LastUsed: 1},
		{Tokens: 1, Bytes: 1000, Hits: 0, LastUsed: 0, Pinned: true},
	}
	victim := PickEvictionVictim(spans)
	if victim != 3 {
		t.Fatalf("PickEvictionVictim = %d, want 3 (cheapest cost, oldest on the tie)", victim)
	}

	// The reduction must hold for ANY positive per-token recompute price: it scales every
	// evict-rung score by the same positive constant, which cannot reorder the ranking.
	for _, recompute := range []float64{1.0, 7.5, 0.001} {
		plan := PlanKVDemotion(spans, 1, demoteLadder, recompute)
		if len(plan) != 1 {
			t.Fatalf("recompute=%v: plan has %d decisions, want 1 (a 1-byte deficit needs one span)", recompute, len(plan))
		}
		got := plan[0]
		if got.SpanIndex != victim {
			t.Errorf("recompute=%v: demotion picked span %d, want PickEvictionVictim's %d", recompute, got.SpanIndex, victim)
		}
		if got.Action != ActionEvict {
			t.Errorf("recompute=%v: action = %v, want evict (the only configured rung)", recompute, got.Action)
		}
		if got.BytesFreed != 200 {
			t.Errorf("recompute=%v: freed %d bytes, want the span's whole 200-byte footprint", recompute, got.BytesFreed)
		}
		if want := 100 * recompute; got.RestoreCost != want {
			t.Errorf("recompute=%v: restore cost = %v, want %v (full re-prefill of 100 tokens)", recompute, got.RestoreCost, want)
		}
	}

	// A deficit spanning several victims drains them in ascending cost-of-losing order, with the
	// same LRU tie-break: 3 (0.5, oldest), 1 (0.5), 2 (1.0). Span 0 is too expensive; 4 is pinned.
	plan := PlanKVDemotion(spans, 500, demoteLadder, 1.0)
	gotOrder := make([]int, len(plan))
	for i, d := range plan {
		gotOrder[i] = d.SpanIndex
	}
	wantOrder := []int{3, 1, 2}
	if len(gotOrder) != len(wantOrder) {
		t.Fatalf("plan order = %v, want %v", gotOrder, wantOrder)
	}
	for i := range wantOrder {
		if gotOrder[i] != wantOrder[i] {
			t.Fatalf("plan order = %v, want %v (ascending cost-of-losing, LRU tie-break)", gotOrder, wantOrder)
		}
	}
}

// TestPlanKVDemotionSpillsInsteadOfDroppingWhenRestoreUndercutsRecompute is the headline
// demote-not-drop witness: the SAME span that drops under drop-only economics instead spills to
// host once a rung is priced below full re-prefill — and the deficit is still cleared. The win is
// the recompute the ladder saves: a transfer-back restore instead of a re-prefill.
func TestPlanKVDemotionSpillsInsteadOfDroppingWhenRestoreUndercutsRecompute(t *testing.T) {
	// A 1000-token span costing 2000 bytes. Re-prefill costs 1000 (1.0/token); a host restore
	// costs 2000 x 0.1 = 200. Restore undercuts re-prefill 5x.
	spans := []KVSpanStats{{Tokens: 1000, Bytes: 2000, Hits: 0, LastUsed: 5}}
	const recompute = 1.0
	const deficit = 2000

	dropOnly := PlanKVDemotion(spans, deficit, demoteLadder, recompute)
	if len(dropOnly) != 1 || dropOnly[0].Action != ActionEvict {
		t.Fatalf("drop-only economics must evict, got %+v", dropOnly)
	}
	if dropOnly[0].RestoreCost != 1000 {
		t.Fatalf("drop books restore cost %v, want the full 1000-token re-prefill", dropOnly[0].RestoreCost)
	}

	ladder := append([]KVTierProfile{spillRung(0.1)}, demoteLadder...)
	demoted := PlanKVDemotion(spans, deficit, ladder, recompute)
	if len(demoted) != 1 {
		t.Fatalf("plan has %d decisions, want 1", len(demoted))
	}
	if demoted[0].Action != ActionSpillHost {
		t.Fatalf("action = %v, want spill-host (its restore undercuts re-prefill)", demoted[0].Action)
	}
	if demoted[0].RestoreCost != 200 {
		t.Fatalf("spill books restore cost %v, want 200 (2000 bytes x 0.1)", demoted[0].RestoreCost)
	}
	// The deficit is still cleared — demoting is not a way of doing less work on the pressure.
	if demoted[0].BytesFreed < deficit {
		t.Fatalf("spill freed %d bytes, want >= the %d-byte deficit", demoted[0].BytesFreed, deficit)
	}
	// The witnessed win: effective recompute saved by demoting instead of dropping.
	if saved := dropOnly[0].RestoreCost - demoted[0].RestoreCost; saved != 800 {
		t.Fatalf("effective recompute saved = %v, want 800 (1000 re-prefill - 200 restore)", saved)
	}

	// The converse must also hold — the flip is driven by economics, not by the rung's presence.
	// A host tier priced ABOVE re-prefill (2000 x 1.0 = 2000 > 1000) must still drop the span.
	expensive := PlanKVDemotion(spans, deficit, append([]KVTierProfile{spillRung(1.0)}, demoteLadder...), recompute)
	if len(expensive) != 1 || expensive[0].Action != ActionEvict {
		t.Fatalf("a tier whose restore exceeds re-prefill must not attract the span, got %+v", expensive)
	}
}

// TestPlanKVDemotionUnpricedTierStillEvicts is the refute guard: fak never demotes into a tier it
// cannot cost. An unavailable rung, an unpriced rung, and a missing recompute unit bridge each
// collapse the ladder back to drop-only rather than guessing a price.
func TestPlanKVDemotionUnpricedTierStillEvicts(t *testing.T) {
	spans := []KVSpanStats{{Tokens: 1000, Bytes: 2000, Hits: 0, LastUsed: 5}}
	const deficit = 2000

	for _, tc := range []struct {
		name      string
		tiers     []KVTierProfile
		recompute float64
	}{
		{
			// Priced at zero: an unpriced rung is not a free rung.
			name:      "unpriced rung (RestoreCostPerByte=0)",
			tiers:     []KVTierProfile{{Action: ActionSpillHost, Available: true, RestoreCostPerByte: 0}, demoteLadder[0]},
			recompute: 1.0,
		},
		{
			// Negative price is nonsense, not a bargain.
			name:      "negatively priced rung",
			tiers:     []KVTierProfile{{Action: ActionSpillHost, Available: true, RestoreCostPerByte: -5}, demoteLadder[0]},
			recompute: 1.0,
		},
		{
			// Absurdly cheap but NOT configured: availability is the caller's attestation and
			// compute never overrides it.
			name:      "unavailable rung despite a cheap price",
			tiers:     []KVTierProfile{{Action: ActionSpillHost, Available: false, RestoreCostPerByte: 0.0001}, demoteLadder[0]},
			recompute: 1.0,
		},
		{
			// No unit bridge: byte-priced spill and token-priced evict are incommensurable, so
			// the spill rung is dropped rather than compared on a guessed conversion.
			name:      "priced rung but no recompute unit bridge",
			tiers:     []KVTierProfile{spillRung(0.1), demoteLadder[0]},
			recompute: 0,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := PlanKVDemotion(spans, deficit, tc.tiers, tc.recompute)
			if len(plan) != 1 {
				t.Fatalf("plan has %d decisions, want 1", len(plan))
			}
			if plan[0].Action != ActionEvict {
				t.Fatalf("action = %v, want evict — a tier fak cannot cost must never attract a span", plan[0].Action)
			}
		})
	}

	// And with no evict rung either, an uncostable ladder demotes nothing at all rather than
	// inventing a fate.
	if plan := PlanKVDemotion(spans, deficit, []KVTierProfile{spillRung(0.1)}, 0); plan != nil {
		t.Fatalf("an uncostable ladder with no evict rung must plan nothing, got %+v", plan)
	}
}

// TestPlanKVDemotionSkipsPinnedAndLeased extends the evictor's hard exclusions to EVERY rung: a
// span being served cannot be dropped, moved, or requantized mid-decode — not even to a cheap,
// well-priced tier.
func TestPlanKVDemotionSkipsPinnedAndLeased(t *testing.T) {
	ladder := append([]KVTierProfile{spillRung(0.01)}, demoteLadder...)

	locked := []KVSpanStats{
		{Tokens: 100, Bytes: 1000, LastUsed: 1, Pinned: true},
		{Tokens: 100, Bytes: 1000, LastUsed: 2, Leased: true},
	}
	if plan := PlanKVDemotion(locked, 5000, ladder, 1.0); plan != nil {
		t.Fatalf("an all-locked resident set must plan nothing (the -1 analogue), got %+v", plan)
	}

	mixed := []KVSpanStats{
		{Tokens: 100, Bytes: 1000, LastUsed: 1, Pinned: true},
		{Tokens: 100, Bytes: 1000, LastUsed: 2, Leased: true},
		{Tokens: 100, Bytes: 1000, LastUsed: 3},
	}
	plan := PlanKVDemotion(mixed, 5000, ladder, 1.0)
	if len(plan) != 1 || plan[0].SpanIndex != 2 {
		t.Fatalf("only the unlocked span may be demoted, got %+v", plan)
	}

	// A span with an unknown footprint frees nothing measurable, so it is never a candidate —
	// the same fail-open KVEvictionCost makes by scoring it +Inf.
	unsized := []KVSpanStats{{Tokens: 100, Bytes: 0, LastUsed: 1}}
	if plan := PlanKVDemotion(unsized, 5000, ladder, 1.0); plan != nil {
		t.Fatalf("an unknown-footprint span must never be demoted, got %+v", plan)
	}
}

// TestPlanKVDemotionFreesAtLeastDeficit pins the planner's contract with the pressured pool: the
// plan clears the deficit whenever the admissible actions can, names each span at most once, and
// stops as soon as the deficit is met rather than over-demoting.
func TestPlanKVDemotionFreesAtLeastDeficit(t *testing.T) {
	spans := []KVSpanStats{
		{Tokens: 100, Bytes: 1000, Hits: 0, LastUsed: 1},
		{Tokens: 100, Bytes: 1000, Hits: 5, LastUsed: 2},
		{Tokens: 100, Bytes: 1000, Hits: 2, LastUsed: 3},
		{Tokens: 100, Bytes: 1000, Hits: 9, LastUsed: 4, Pinned: true},
	}
	ladder := append([]KVTierProfile{spillRung(0.05)}, demoteLadder...)

	const deficit = 2500
	plan := PlanKVDemotion(spans, deficit, ladder, 1.0)

	var freed int64
	seen := map[int]bool{}
	for _, d := range plan {
		if seen[d.SpanIndex] {
			t.Fatalf("span %d named twice — a span meets exactly one fate: %+v", d.SpanIndex, plan)
		}
		seen[d.SpanIndex] = true
		if d.Action == ActionKeep {
			t.Fatalf("plan emitted ActionKeep; keeping is the implicit default: %+v", plan)
		}
		if spans[d.SpanIndex].Pinned {
			t.Fatalf("plan named the pinned span %d: %+v", d.SpanIndex, plan)
		}
		freed += d.BytesFreed
	}
	if freed < deficit {
		t.Fatalf("plan freed %d bytes, want >= the %d-byte deficit", freed, deficit)
	}
	// Three 1000-byte spans are evictable; a 2500-byte deficit needs exactly three of them, and
	// the planner must not demote a fourth it does not need.
	if len(plan) != 3 {
		t.Fatalf("plan has %d decisions, want 3 (stop as soon as the deficit is cleared)", len(plan))
	}

	// Best-effort when the deficit outruns the pool: every admissible span is demoted and the
	// caller sees the shortfall by summing BytesFreed, rather than the planner refusing or
	// touching a pinned span.
	over := PlanKVDemotion(spans, 1<<40, ladder, 1.0)
	if len(over) != 3 {
		t.Fatalf("an unclearable deficit must still demote all 3 admissible spans, got %+v", over)
	}
	var overFreed int64
	for _, d := range over {
		overFreed += d.BytesFreed
	}
	if overFreed != 3000 {
		t.Fatalf("best-effort plan freed %d bytes, want the 3000 the unlocked spans hold", overFreed)
	}
}

// TestPlanKVDemotionEmptyOnNonPositiveDeficit is the fail-open floor: no pressure, no demotion.
func TestPlanKVDemotionEmptyOnNonPositiveDeficit(t *testing.T) {
	spans := []KVSpanStats{{Tokens: 100, Bytes: 1000, LastUsed: 1}}
	ladder := append([]KVTierProfile{spillRung(0.01)}, demoteLadder...)
	for _, deficit := range []int64{0, -1, -1 << 30} {
		if plan := PlanKVDemotion(spans, deficit, ladder, 1.0); plan != nil {
			t.Errorf("deficit %d must demote nothing, got %+v", deficit, plan)
		}
	}
	if plan := PlanKVDemotion(nil, 1000, ladder, 1.0); plan != nil {
		t.Errorf("an empty resident set must demote nothing, got %+v", plan)
	}
}
