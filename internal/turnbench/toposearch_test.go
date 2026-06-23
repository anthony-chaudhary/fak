package turnbench

import (
	"context"
	"encoding/json"
	"testing"
)

// TestTopologySearch_OracleFindsBetterTopologyAtZeroModel is the headline #541 proof: over a
// frozen corpus of recorded fan-out runs, the model-free STRUCTURE-search finds a fleet
// topology that strictly BEATS the hand-frozen baseline (more measured savings at no worse
// arbiter collision) — and spends ZERO model calls doing it. The oracle demonstrably works
// on the fleet graph, not the policy table.
func TestTopologySearch_OracleFindsBetterTopologyAtZeroModel(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	cfg := TopologySearchConfig{
		Profile:       FanoutResearch,
		FrontierWidth: 16, // the corpus recorded fan-outs up to width 16
		// The hand-frozen shape: a narrow master->4-worker graph, all on ONE lane.
		Baseline:  TopologyGenome{Width: 4, SubTurns: 4, Lanes: 1},
		WidthGrid: []int{1, 2, 4, 8, 16},
		LaneGrid:  []int{1, 2, 4, 8, 16},
		Trials:    6,
		Seed:      1337,
		TopK:      0,
	}

	rep, err := RunTopologySearch(ctx, cfg, cm)
	if err != nil {
		t.Fatalf("RunTopologySearch: %v", err)
	}

	// $0 model spend is the whole point — the structure-search is model-free replay.
	if rep.ModelCallsSpent != 0 {
		t.Errorf("topology search must spend ZERO model calls, got %d", rep.ModelCallsSpent)
	}

	// THE ORACLE WORKS: the search's best topology strictly improves credited savings over
	// the hand-frozen baseline, without paying more arbiter collision.
	if rep.Best.Fitness.CreditedSavingsTokens <= rep.Baseline.Fitness.CreditedSavingsTokens {
		t.Fatalf("search must IMPROVE credited savings over the baseline (%d), got best=%d",
			rep.Baseline.Fitness.CreditedSavingsTokens, rep.Best.Fitness.CreditedSavingsTokens)
	}
	if rep.Best.Fitness.ArbiterCollisionCost > rep.Baseline.Fitness.ArbiterCollisionCost {
		t.Errorf("the best topology must not cost MORE arbiter collision than the baseline (%d), got %d",
			rep.Baseline.Fitness.ArbiterCollisionCost, rep.Best.Fitness.ArbiterCollisionCost)
	}
	// The win is via MEASURED savings (exact prefix geometry + real dedup), not a resolve-rate.
	if rep.Best.Fitness.PrefixTokensSaved <= 0 {
		t.Errorf("the best topology must save via measured prefix geometry, got %d", rep.Best.Fitness.PrefixTokensSaved)
	}
	// The best is on the Pareto frontier and is WITNESSED (not an extrapolated projection).
	if !rep.Best.OnFrontier {
		t.Errorf("the best topology must be on the Pareto frontier")
	}
	if rep.Best.Fitness.Bounded {
		t.Errorf("the best topology must be witnessed (within the corpus frontier), got bounded width=%d", rep.Best.Genome.Width)
	}
	// The discovered structure spreads workers across lanes to dodge serialization: the
	// max-savings frontier width with zero collision is the EvoAgentX-style structure win.
	if rep.Best.Fitness.ArbiterCollisionCost != 0 {
		t.Errorf("the best topology should spread workers to 0 arbiter collision, got %d", rep.Best.Fitness.ArbiterCollisionCost)
	}
}

// TestTopologySearch_DivergenceGateRefusesPastFrontierWin is the load-bearing honesty proof:
// a genome cannot win by fanning out WIDER than the frozen corpus ever recorded. A width past
// the corpus frontier is bounded; its savings beyond the frontier is REFUSED credit (capped),
// it is flagged NeedsLiveRevalidation, and it can NOT be crowned best by an unrecorded-scale
// projection.
func TestTopologySearch_DivergenceGateRefusesPastFrontierWin(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	frontier := 16

	// A genome at width 64 — far past the corpus frontier (16).
	g := TopologyGenome{Width: 64, SubTurns: 4, Lanes: 64}
	sc := scoreTopology(ctx, FanoutResearch, g, frontier, 6, 7, cm)

	// It is BOUNDED (width > frontier) and flagged for live re-validation with a note.
	if !sc.Fitness.Bounded {
		t.Fatalf("a width past the corpus frontier must be BOUNDED, got witnessed")
	}
	if !sc.NeedsLiveRevalidation || sc.RevalidationNote == "" {
		t.Errorf("a bounded genome must be flagged NeedsLiveRevalidation with a note")
	}
	// THE GATE: credit is capped at the frontier width — its measured projection at width 64
	// exceeds its credited savings, and the difference is REFUSED.
	if sc.Fitness.CreditedWidth != frontier {
		t.Errorf("credited width must be capped at the frontier %d, got %d", frontier, sc.Fitness.CreditedWidth)
	}
	if sc.Fitness.MeasuredSavingsTokens <= sc.Fitness.CreditedSavingsTokens {
		t.Fatalf("a past-frontier genome's raw projection (%d) must exceed its credited savings (%d)",
			sc.Fitness.MeasuredSavingsTokens, sc.Fitness.CreditedSavingsTokens)
	}
	if sc.Fitness.RefusedProjectedSavingsTokens != sc.Fitness.MeasuredSavingsTokens-sc.Fitness.CreditedSavingsTokens {
		t.Errorf("refused = measured - credited must hold; got refused=%d measured=%d credited=%d",
			sc.Fitness.RefusedProjectedSavingsTokens, sc.Fitness.MeasuredSavingsTokens, sc.Fitness.CreditedSavingsTokens)
	}
	if sc.Fitness.RefusedProjectedSavingsTokens <= 0 {
		t.Errorf("the past-frontier savings must be REFUSED (>0 refused), got %d", sc.Fitness.RefusedProjectedSavingsTokens)
	}

	// The credited savings of the width-64 genome equals a WITNESSED width-16 genome's — the
	// gate holds it to exactly what the corpus recorded, so it cannot out-credit the witnessed
	// frontier topology.
	w16 := scoreTopology(ctx, FanoutResearch, TopologyGenome{Width: 16, SubTurns: 4, Lanes: 16}, frontier, 6, 7, cm)
	if sc.Fitness.CreditedSavingsTokens != w16.Fitness.CreditedSavingsTokens {
		t.Errorf("the bounded width-64 genome must be credited exactly the witnessed width-16 savings: got %d vs %d",
			sc.Fitness.CreditedSavingsTokens, w16.Fitness.CreditedSavingsTokens)
	}

	// And in a full search that INCLUDES the past-frontier width, Best must NOT be the bounded
	// genome (the gate forbids crowning a counterfactual-scale win).
	cfg := TopologySearchConfig{
		Profile:       FanoutResearch,
		FrontierWidth: frontier,
		Baseline:      TopologyGenome{Width: 4, SubTurns: 4, Lanes: 1},
		WidthGrid:     []int{8, 16, 64},
		LaneGrid:      []int{1, 16, 64},
		Trials:        6,
		Seed:          7,
	}
	rep, err := RunTopologySearch(ctx, cfg, cm)
	if err != nil {
		t.Fatalf("RunTopologySearch: %v", err)
	}
	if rep.Best.Fitness.Bounded {
		t.Errorf("the gate must NOT crown a past-frontier topology as best; best width=%d bounded=%v",
			rep.Best.Genome.Width, rep.Best.Fitness.Bounded)
	}
}

// TestTopologySearch_ArbiterCollisionCostIsRealAndMonotone proves the cost axis is the exact
// dos_arbitrate serialization rule: N workers on ONE lane all serialize (C(N,2) pairs), N
// workers on N lanes never collide (0), and spreading across more lanes strictly relieves it.
// This is the lever the hand-authored dos.toml partition freezes and the search learns to set.
func TestTopologySearch_ArbiterCollisionCostIsRealAndMonotone(t *testing.T) {
	// One lane: every pair of the N workers serializes -> C(N,2).
	for _, n := range []int{1, 2, 4, 8, 16} {
		want := n * (n - 1) / 2
		if got := arbiterCollisionCost(n, 1); got != want {
			t.Errorf("collision(width=%d, lanes=1)=%d, want C(N,2)=%d", n, got, want)
		}
		// One lane per worker: fully parallel, zero serialization.
		if got := arbiterCollisionCost(n, n); got != 0 {
			t.Errorf("collision(width=%d, lanes=%d)=%d, want 0 (fully parallel)", n, n, got)
		}
	}
	// More lanes never increases collision and (for a contended width) strictly decreases it.
	prev := arbiterCollisionCost(16, 1)
	for _, l := range []int{2, 4, 8, 16} {
		got := arbiterCollisionCost(16, l)
		if got > prev {
			t.Fatalf("collision rose with more lanes (lanes->%d): %d > %d", l, got, prev)
		}
		prev = got
	}
	// 16 workers across 4 lanes => 4 lanes of 4 => 4*C(4,2) = 4*6 = 24.
	if got := arbiterCollisionCost(16, 4); got != 24 {
		t.Errorf("collision(16,4)=%d, want 24 (4 lanes of 4, 4*C(4,2))", got)
	}
}

// TestTopologySearch_NoResolveRateInFitness is the first-hard-fence proof: the topology
// fitness uses ONLY the replayable measured savings + the pure-data collision axis and NEVER
// a resolve-rate / completion term. The fitness JSON exposes no such key, and the search
// ranks on measured savings, not on which topology "gets more done".
func TestTopologySearch_NoResolveRateInFitness(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	cfg := TopologySearchConfig{
		Profile:       FanoutResearch,
		FrontierWidth: 16,
		Baseline:      TopologyGenome{Width: 4, SubTurns: 4, Lanes: 1},
		WidthGrid:     []int{4, 8, 16},
		LaneGrid:      []int{1, 4, 16},
		Trials:        4,
		Seed:          99,
	}
	rep, err := RunTopologySearch(ctx, cfg, cm)
	if err != nil {
		t.Fatalf("RunTopologySearch: %v", err)
	}
	b, _ := json.Marshal(rep.Best.Fitness)
	for _, banned := range []string{"resolve", "completion", "resolve_rate", "throughput"} {
		if containsKey(string(b), banned) {
			t.Errorf("TopologyFitness must NOT expose a %q axis (resolve-rate is not a fitness term): %s", banned, b)
		}
	}
}

// TestTopologySearch_FrontierAndRevalidationFlag proves the report surface mirrors
// PolicySearchReport: a Pareto frontier of credited-savings-vs-arbiter-collision sorted by
// savings, each genome carrying its divergence status, the past-frontier genomes flagged for
// LIVE re-validation (a flag, not an executed model run), and a clear completion note.
func TestTopologySearch_FrontierAndRevalidationFlag(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	cfg := TopologySearchConfig{
		Profile:       FanoutResearch,
		FrontierWidth: 16,
		Baseline:      TopologyGenome{Width: 4, SubTurns: 4, Lanes: 1},
		WidthGrid:     []int{2, 8, 16, 32}, // 32 is past the frontier -> bounded
		LaneGrid:      []int{1, 8, 32},
		Trials:        6,
		Seed:          5,
		TopK:          10,
	}
	rep, err := RunTopologySearch(ctx, cfg, cm)
	if err != nil {
		t.Fatalf("RunTopologySearch: %v", err)
	}

	if len(rep.Frontier) == 0 {
		t.Fatalf("the report must surface a non-empty Pareto frontier")
	}
	// The frontier is sorted by credited savings DESCENDING (the trade-off surface).
	for i := 1; i < len(rep.Frontier); i++ {
		if rep.Frontier[i].Fitness.CreditedSavingsTokens > rep.Frontier[i-1].Fitness.CreditedSavingsTokens {
			t.Errorf("frontier not sorted by credited_savings descending at %d", i)
		}
	}
	// Every genome's revalidation flag tracks its bounded (past-frontier) status.
	for _, c := range rep.Candidates {
		if c.Fitness.Bounded != c.NeedsLiveRevalidation {
			t.Errorf("genome %q: NeedsLiveRevalidation (%v) must track Bounded (%v)",
				c.Name, c.NeedsLiveRevalidation, c.Fitness.Bounded)
		}
		if c.NeedsLiveRevalidation && c.RevalidationNote == "" {
			t.Errorf("flagged genome %q must carry a revalidation note", c.Name)
		}
	}
	// The width-32 genomes are past the frontier -> at least one frontier genome flagged.
	if len(rep.FlaggedForRevalidation) == 0 {
		t.Errorf("expected at least one frontier genome flagged for live re-validation (width past the frontier)")
	}
	// The flag is the ONLY effect — the search ran no model.
	if rep.ModelCallsSpent != 0 {
		t.Errorf("flagging for revalidation must NOT run a model; ModelCallsSpent=%d", rep.ModelCallsSpent)
	}
	if rep.CompletionNote == "" {
		t.Errorf("the report must carry a completion note (sound only within the corpus frontier)")
	}
	if len(rep.JSON()) == 0 {
		t.Fatal("empty JSON artifact")
	}
}

// TestTopologySearch_Deterministic asserts the search is a regenerable artifact: a fixed
// (corpus, grids, seed) yields byte-identical JSON across runs (modulo host provenance), so
// the frontier is reproducible, not a sample.
func TestTopologySearch_Deterministic(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	mk := func() TopologySearchConfig {
		return TopologySearchConfig{
			Profile:       FanoutResearch,
			FrontierWidth: 16,
			Baseline:      TopologyGenome{Width: 4, SubTurns: 4, Lanes: 1},
			WidthGrid:     []int{2, 8, 16, 32},
			LaneGrid:      []int{1, 8, 32},
			Trials:        6,
			Seed:          2024,
			TopK:          3,
		}
	}
	a, err := RunTopologySearch(ctx, mk(), cm)
	if err != nil {
		t.Fatalf("run a: %v", err)
	}
	b, err := RunTopologySearch(ctx, mk(), cm)
	if err != nil {
		t.Fatalf("run b: %v", err)
	}
	a.Provenance, b.Provenance = Provenance{}, Provenance{}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Errorf("topology search report drifted across runs:\n a=%s\n b=%s", ja, jb)
	}
}
