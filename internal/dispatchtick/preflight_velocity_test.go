package dispatchtick

import "testing"

// TestEvaluateVelocityPriorHotLaneRaisesPrior: a hot fixture lane (60 revs over a 4-week
// window = 15 revs/week, past the default 10/week hot floor) earns a RAISED collision prior
// carrying the closed COLLISION_RISK class -- the advisory signal the issue's done condition
// names, never a hold.
func TestEvaluateVelocityPriorHotLaneRaisesPrior(t *testing.T) {
	got := EvaluateVelocityPrior(VelocityCheck{Lane: "gateway", RevDelta: 60, Weeks: 4, Present: true})
	if got.RevsPerWeek != 15 {
		t.Fatalf("revs_per_week = %v, want 15 (60 revs / 4 weeks)", got.RevsPerWeek)
	}
	if !got.Hot {
		t.Fatalf("hot = false, want true (15 revs/week >= 10 hot floor)")
	}
	if got.Token != CollisionRisk {
		t.Fatalf("token = %q, want %q", got.Token, CollisionRisk)
	}
	if got.Prior != 1.5 {
		t.Fatalf("prior = %v, want 1.5 (15 / 10 hot floor)", got.Prior)
	}
	if got.Reason == "" {
		t.Fatalf("reason empty, want a COLLISION_RISK citation for the hot lane")
	}
}

// TestEvaluateVelocityPriorMapSurfacesSignal is the DONE-CONDITION witness: the arbitration
// output (Map) for a hot fixture lane SHOWS the velocity signal -- the derived revs/week, the
// raised prior, the hot flag, and the COLLISION_RISK token.
func TestEvaluateVelocityPriorMapSurfacesSignal(t *testing.T) {
	m := EvaluateVelocityPrior(VelocityCheck{Lane: "modver", RevDelta: 40, Weeks: 2, Present: true}).Map()
	if m["hot"] != true {
		t.Fatalf("map hot = %v, want true (20 revs/week)", m["hot"])
	}
	if m["revs_per_week"] != 20.0 {
		t.Fatalf("map revs_per_week = %v, want 20 (40 revs / 2 weeks)", m["revs_per_week"])
	}
	if m["prior"] != 2.0 {
		t.Fatalf("map prior = %v, want 2.0 (20 / 10 hot floor)", m["prior"])
	}
	if m["token"] != CollisionRisk {
		t.Fatalf("map token = %v, want %q", m["token"], CollisionRisk)
	}
	if m["lane"] != "modver" {
		t.Fatalf("map lane = %v, want modver", m["lane"])
	}
}

// TestEvaluateVelocityPriorDormantLaneByteIdentical: a dormant lane (2 revs over 4 weeks =
// 0.5 revs/week, well below the floor) raises NO prior and attaches NO token -- the arbitration
// output is byte-identical to today (the caller attaches the block only when Hot).
func TestEvaluateVelocityPriorDormantLaneByteIdentical(t *testing.T) {
	got := EvaluateVelocityPrior(VelocityCheck{Lane: "appversion", RevDelta: 2, Weeks: 4, Present: true})
	if got.Hot {
		t.Fatalf("hot = true, want false (0.5 revs/week < 10 hot floor)")
	}
	if got.Prior != 0 || got.Token != "" || got.Reason != "" {
		t.Fatalf("dormant lane emitted prior=%v token=%q reason=%q, want zero/empty (byte-identical)", got.Prior, got.Token, got.Reason)
	}
}

// TestEvaluateVelocityPriorNoLedgerAbstains: with Present false (the ledger folded no rows for
// this lane) the term abstains even if a RevDelta is wired -- no ledger, no slander.
func TestEvaluateVelocityPriorNoLedgerAbstains(t *testing.T) {
	got := EvaluateVelocityPrior(VelocityCheck{Lane: "gateway", RevDelta: 99, Weeks: 1, Present: false})
	if got.Hot || got.Prior != 0 || got.Token != "" {
		t.Fatalf("no-ledger lane = %+v, want not hot / zero prior / no token", got)
	}
}

// TestEvaluateVelocityPriorAtFloorExactlyIsHot: the predicate is RevsPerWeek >= Threshold, so a
// lane exactly at the hot floor is hot with a prior of exactly 1.0 (at, not over).
func TestEvaluateVelocityPriorAtFloorExactlyIsHot(t *testing.T) {
	got := EvaluateVelocityPrior(VelocityCheck{Lane: "engine", RevDelta: 10, Weeks: 1, Present: true})
	if !got.Hot {
		t.Fatalf("hot = false, want true (10 revs/week == 10 hot floor)")
	}
	if got.Prior != 1.0 {
		t.Fatalf("prior = %v, want 1.0 (exactly at the floor)", got.Prior)
	}
}

// TestEvaluateVelocityPriorThresholdOverride: a tick can raise the hot floor so a lane that is
// hot under the default is dormant under a stricter override (the shell's FAK_HOT_REVS_PER_WEEK).
func TestEvaluateVelocityPriorThresholdOverride(t *testing.T) {
	got := EvaluateVelocityPrior(VelocityCheck{Lane: "gateway", RevDelta: 12, Weeks: 1, Threshold: 20, Present: true})
	if got.Hot {
		t.Fatalf("hot = true, want false (12 revs/week < 20 override floor)")
	}
	if got.Threshold != 20 {
		t.Fatalf("threshold = %v, want 20 (override echoed)", got.Threshold)
	}
}
