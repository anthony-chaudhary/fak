package bench

import (
	"context"
	"testing"

	// Wire the real kernel (engine, vdso, adjudicator, blob backend, ctx-MMU, ...) so the
	// turnbench fan-out scoring inside RunFanScale records genuine tier-2 dedup events.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

const defaultPrefixTokens = 2048 // turnbench.DefaultFanoutCostModel().PrefixTokens

// Issue #11 (D-001), host smoke: RunFanScale at a tiny N=8 fan-out exercises the whole
// harness so `go test ./internal/bench/...` proves it builds and the outputs are sane —
// provenance stamped, the N=1 baseline priced, coordination overhead and cross-agent
// reuse uplift recorded, and the published headline run honestly named as deferred.
func TestRunFanScale_SmokeN8(t *testing.T) {
	rep := RunFanScale(context.Background(), FanScaleOptions{
		Grid: []int{8}, SubTurns: 2, Trials: 3,
	})

	// Provenance stamped.
	if rep.AppVersion == "" || rep.GoVersion == "" || rep.OS == "" {
		t.Errorf("provenance not stamped: ver=%q go=%q os=%q", rep.AppVersion, rep.GoVersion, rep.OS)
	}
	if rep.GeneratedBy != "fak/internal/bench (fanscale)" {
		t.Errorf("GeneratedBy = %q, want the fanscale harness", rep.GeneratedBy)
	}
	if rep.Profile != "research-goal" {
		t.Errorf("default profile = %q, want %q", rep.Profile, "research-goal")
	}

	// The grid is normalized to {1,8}: the baseline is always present so overhead/uplift
	// are priced against one agent.
	if got, want := rep.Grid, []int{1, 8}; !equalInts(got, want) {
		t.Fatalf("grid = %v, want %v", got, want)
	}
	if len(rep.Points) != len(rep.Grid) {
		t.Fatalf("points = %d, want one per grid N (%d)", len(rep.Points), len(rep.Grid))
	}

	// Baseline: N=1, zero coordination overhead (it is the reference point), zero prefix
	// reuse (a single agent shares nothing).
	if rep.Baseline.Agents != 1 {
		t.Errorf("baseline Agents = %d, want 1", rep.Baseline.Agents)
	}
	if rep.Baseline.CoordOverheadTurns != 0 || rep.Baseline.CoordOverheadFrac != 0 {
		t.Errorf("baseline overhead = %.3f turns / %.3f frac, want 0/0",
			rep.Baseline.CoordOverheadTurns, rep.Baseline.CoordOverheadFrac)
	}
	if rep.Baseline.PrefixTokensSaved != 0 {
		t.Errorf("baseline PrefixTokensSaved = %d, want 0", rep.Baseline.PrefixTokensSaved)
	}

	// The N=8 fan-out point.
	p8 := rep.Points[1]
	if p8.Agents != 8 {
		t.Fatalf("points[1].Agents = %d, want 8", p8.Agents)
	}
	// Coordination overhead: the synchronous-join/fold tax can only add depth vs the
	// baseline, never subtract.
	if p8.CoordOverheadTurns < 0 {
		t.Errorf("N=8 coord overhead = %.3f, want >= 0 (fold tax cannot reduce depth)", p8.CoordOverheadTurns)
	}
	// Exact shared-prefix-KV-reuse geometry: (N-1)*prefix.
	if want := (8 - 1) * defaultPrefixTokens; p8.PrefixTokensSaved != want {
		t.Errorf("N=8 PrefixTokensSaved = %d, want (8-1)*%d = %d", p8.PrefixTokensSaved, defaultPrefixTokens, want)
	}
	// Cross-agent reuse uplift is well-defined (sibling dedup is non-negative for the
	// read-shared research profile) and the modeled clawback is a fraction in [0,1].
	if p8.CrossUpliftP50 < 0 {
		t.Errorf("N=8 cross_uplift = %d, want >= 0 for the research profile", p8.CrossUpliftP50)
	}
	if p8.TaxClawedBackFrac < 0 || p8.TaxClawedBackFrac > 1 {
		t.Errorf("N=8 tax_clawed_back = %.3f, want in [0,1]", p8.TaxClawedBackFrac)
	}

	// The published headline run must be named as deferred, not fabricated.
	if rep.DeferredRun == "" {
		t.Error("DeferredRun is empty: the headline bench-node run must be named, not silently dropped")
	}
}

// The acceptance criteria name N=100/500/1000; assert the documented default grid carries
// exactly those scale points (plus the N=1 baseline).
func TestCanonicalFanScaleGrid_AcceptancePoints(t *testing.T) {
	for _, n := range []int{1, 100, 500, 1000} {
		if !containsInt(CanonicalFanScaleGrid, n) {
			t.Errorf("CanonicalFanScaleGrid %v is missing the acceptance point N=%d", CanonicalFanScaleGrid, n)
		}
	}
}

// Issue #11 requires the harness be ">=1024 capable". Witness it with a real (but cheap:
// trials=1, sub-turns=1) measured pass at N=1024 — the harness produces a 1024-agent
// point with the exact (1023)*prefix reuse geometry and a strictly positive fold/join
// coordination overhead vs the baseline.
func TestRunFanScale_Capable1024(t *testing.T) {
	rep := RunFanScale(context.Background(), FanScaleOptions{
		Grid: []int{1024}, SubTurns: 1, Trials: 1,
	})
	if got, want := rep.Grid, []int{1, 1024}; !equalInts(got, want) {
		t.Fatalf("grid = %v, want %v", got, want)
	}
	var p1024 *FanScalePoint
	for i := range rep.Points {
		if rep.Points[i].Agents == 1024 {
			p1024 = &rep.Points[i]
		}
	}
	if p1024 == nil {
		t.Fatal("no N=1024 point produced; harness is not >=1024 capable")
	}
	if want := (1024 - 1) * defaultPrefixTokens; p1024.PrefixTokensSaved != want {
		t.Errorf("N=1024 PrefixTokensSaved = %d, want %d", p1024.PrefixTokensSaved, want)
	}
	// The fold/join tax grows with N, so at N=1024 the critical path strictly exceeds the
	// single-agent baseline — coordination overhead must be positive.
	if p1024.CoordOverheadTurns <= 0 {
		t.Errorf("N=1024 coord overhead = %.3f, want > 0 (fold tax grows with N)", p1024.CoordOverheadTurns)
	}
}

// Anti-inflation control: with the no-share profile there is nothing for cross-agent
// dedup to serve, so the MEASURED sibling uplift MUST be exactly 0 at every N. A non-zero
// uplift here is a harness bug, not a result — this guards the measured half from
// silently double-counting a single agent's own intra-agent dedup.
func TestRunFanScale_NoShareUpliftIsZero(t *testing.T) {
	rep := RunFanScale(context.Background(), FanScaleOptions{
		Profile: turnbench.FanoutNoShare,
		Grid:    []int{4, 16}, SubTurns: 2, Trials: 4,
	})
	if rep.Profile != "no-share" {
		t.Fatalf("profile = %q, want %q", rep.Profile, "no-share")
	}
	for _, p := range rep.Points {
		if p.CrossUpliftP50 != 0 {
			t.Errorf("no-share cross_uplift at N=%d = %d, want exactly 0", p.Agents, p.CrossUpliftP50)
		}
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func containsInt(xs []int, v int) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}
