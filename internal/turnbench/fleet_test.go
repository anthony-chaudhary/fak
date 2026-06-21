package turnbench

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vdso"

	// Wire the full ABI (resolver, vDSO, ctx-MMU, adjudicator, ifc, engines) before
	// agent.Configure / the kernel run inside RunFleetCell — same as the other tests.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// TestFleet_NoShareHasZeroCrossUplift is the anti-inflation control at the source.
// A profile with NO shared reads gives cross-agent dedup nothing to serve: running
// A agents together must save the SAME as running them apart, so the cross uplift
// median is exactly 0. If this is ever positive the harness is crediting the fleet
// for sharing that did not happen.
func TestFleet_NoShareHasZeroCrossUplift(t *testing.T) {
	cm := DefaultCostModel()
	c := RunFleetCell(context.Background(), FleetNoShare, 16, 8, 40, fleetSeed, cm)
	if c.CrossUplift.P50 != 0 {
		t.Errorf("no-share cross uplift p50 = %d, want 0 (private-only work shares nothing)", c.CrossUplift.P50)
	}
	if c.CrossUplift.Max > 0 {
		t.Errorf("no-share cross uplift max = %d, want <= 0 (no trial may gain from sharing)", c.CrossUplift.Max)
	}
}

// TestFleet_SharedReadsGiveCrossUplift is the live proof the agent-count lever is
// real: with shared reference reads, an A-agent fleet sharing one world deletes
// strictly MORE turns than the same A agents isolated — because a later agent's
// read of an earlier agent's reference data is a genuine tier-2 hit. The uplift
// equals the extra VDSOHits the shared epoch produced (it is a measured path swap,
// not arithmetic).
func TestFleet_SharedReadsGiveCrossUplift(t *testing.T) {
	cm := DefaultCostModel()
	c := RunFleetCell(context.Background(), FleetReadHeavy, 20, 16, 40, fleetSeed, cm)
	if !(c.CrossUplift.P50 > 0) {
		t.Errorf("read-heavy cross uplift p50 = %d, want > 0 (shared reads must dedup across agents)", c.CrossUplift.P50)
	}
	if !(c.SharedSaved.P50 > c.IsolatedSaved.P50) {
		t.Errorf("shared (%d) must beat isolated (%d) for a shared-read fleet", c.SharedSaved.P50, c.IsolatedSaved.P50)
	}
}

// TestFleet_Determinism: a fixed (profile, T, A, trials, seed) yields the identical
// cell — the surface is reproducible byte-for-byte.
func TestFleet_Determinism(t *testing.T) {
	cm := DefaultCostModel()
	a := RunFleetCell(context.Background(), FleetReadHeavy, 12, 8, 24, fleetSeed, cm)
	b := RunFleetCell(context.Background(), FleetReadHeavy, 12, 8, 24, fleetSeed, cm)
	if a.SharedSaved != b.SharedSaved || a.CrossUplift != b.CrossUplift || a.IsolatedSaved != b.IsolatedSaved {
		t.Fatalf("same seed produced different cells:\n a=%+v\n b=%+v", a, b)
	}
}

func TestFleetSweepVersions(t *testing.T) {
	sw := RunFleetSweep(context.Background(), FleetReadHeavy, []int{1}, []int{1}, 1, fleetSeed, DefaultCostModel(), nil)
	if sw.AppVersion == "" {
		t.Fatal("fleet sweep app_version is empty")
	}
	if sw.Profile.Version != BenchmarkConceptVersion {
		t.Fatalf("fleet profile version=%q, want %q", sw.Profile.Version, BenchmarkConceptVersion)
	}
	if sw.Cost.Version != CostModelVersion {
		t.Fatalf("fleet cost model version=%q, want %q", sw.Cost.Version, CostModelVersion)
	}
}

// TestFleet_MonotoneInAgents: holding T fixed, the total turns the fleet deletes
// grows with the agent count (more agents => more work => more deletable turns).
// This is the A-axis of the surface; the test asserts the median is non-decreasing
// across a few agent counts.
func TestFleet_MonotoneInAgents(t *testing.T) {
	cm := DefaultCostModel()
	prev := -1
	for _, A := range []int{1, 2, 4, 8, 16} {
		c := RunFleetCell(context.Background(), FleetReadHeavy, 16, A, 30, fleetSeed, cm)
		if c.SharedSaved.P50 < prev {
			t.Errorf("shared_saved not monotone in agents: A=%d p50=%d < prev=%d", A, c.SharedSaved.P50, prev)
		}
		prev = c.SharedSaved.P50
	}
}

// TestFleet_MonotoneInTurns: holding A fixed, longer sessions delete more turns
// (more calls => more dedup/grammar/serve opportunities). The T-axis of the surface.
func TestFleet_MonotoneInTurns(t *testing.T) {
	cm := DefaultCostModel()
	prev := -1
	for _, T := range []int{1, 4, 8, 16, 32} {
		c := RunFleetCell(context.Background(), FleetReadHeavy, T, 8, 30, fleetSeed, cm)
		if c.SharedSaved.P50 < prev {
			t.Errorf("shared_saved not monotone in turns: T=%d p50=%d < prev=%d", T, c.SharedSaved.P50, prev)
		}
		prev = c.SharedSaved.P50
	}
}

// TestFleet_SingleAgentHasNoCrossUplift: with A=1 there is no other agent to share
// with, so shared == isolated and the cross uplift is 0 (a sanity floor on the
// ablation — the fleet benefit is a strictly multi-agent phenomenon).
func TestFleet_SingleAgentHasNoCrossUplift(t *testing.T) {
	cm := DefaultCostModel()
	c := RunFleetCell(context.Background(), FleetReadHeavy, 24, 1, 30, fleetSeed, cm)
	if c.CrossUplift.P50 != 0 || c.CrossUplift.Max != 0 || c.CrossUplift.Min != 0 {
		t.Errorf("A=1 cross uplift must be 0 across the whole distribution, got %+v", c.CrossUplift)
	}
}

// TestFleet_TimingProbe is not an assertion — it measures the wall-clock cost of a
// representative large cell so the full-sweep grid can be sized to a time budget.
// It logs cells/sec; run with -v to read it. (Skipped under -short.)
func TestFleet_TimingProbe(t *testing.T) {
	if testing.Short() {
		t.Skip("timing probe skipped under -short")
	}
	cm := DefaultCostModel()
	// A worst-ish cell: long sessions × a big fleet × the sweep's trial count.
	const T, A, trials = 50, 50, 24
	t0 := time.Now()
	c := RunFleetCell(context.Background(), FleetReadHeavy, T, A, trials, fleetSeed, cm)
	d := time.Since(t0)
	t.Logf("cell T=%d A=%d trials=%d calls/trial=%d took %v (%.1f ms/cell)",
		T, A, trials, c.Calls, d, float64(d.Milliseconds()))
	t.Logf("  shared_saved p50=%d isolated p50=%d cross_uplift p50=%d per-agent=%.2f",
		c.SharedSaved.P50, c.IsolatedSaved.P50, c.CrossUplift.P50, float64(c.SharedPerAgentMilli.P50)/1000)
	// Extrapolate a full 50×50 grid at this per-cell cost.
	full := d * 2500
	t.Logf("  => a full 50×50 grid at trials=%d ≈ %v", trials, full.Round(time.Second))
}

// TestFleet_FinerEraserPushesCrossoverOut is the headline measured claim of the
// "finer eraser" work: under the v0.1 GLOBAL flush a write-mixed fleet loses its
// cross-agent benefit (one booking strands every agent's warmed reads), but the
// RESOURCE eraser — which strands only the booked route's reads — keeps the
// cross-uplift strictly higher and positive. It is a real path swap through the live
// kernel at three eraser settings over the SAME generated work, not a model.
func TestFleet_FinerEraserPushesCrossoverOut(t *testing.T) {
	cm := DefaultCostModel()
	// A write-mixed read fleet. Under the global flush this ~3% write rate sits well
	// past the measured <1% crossover, so global cross-uplift is throttled (often
	// negative); the finer eraser must recover it.
	p := FleetReadHeavy
	p.PWrite = 0.03
	const T, A, trials = 24, 30, 16

	defer SetInvalidation(vdso.Global) // restore the package default for sibling tests

	run := func(g vdso.Granularity) FleetCell {
		SetInvalidation(g)
		return RunFleetCell(context.Background(), p, T, A, trials, fleetSeed, cm)
	}
	global := run(vdso.Global)
	namespace := run(vdso.Namespace)
	resource := run(vdso.Resource)

	t.Logf("cross_uplift p50 @ write_rate=%.2f T=%d A=%d: global=%d namespace=%d resource=%d",
		p.PWrite, T, A, global.CrossUplift.P50, namespace.CrossUplift.P50, resource.CrossUplift.P50)

	if !(resource.CrossUplift.P50 > global.CrossUplift.P50) {
		t.Errorf("resource eraser cross-uplift (%d) must exceed global (%d): the finer eraser must recover sharing under writes",
			resource.CrossUplift.P50, global.CrossUplift.P50)
	}
	if resource.CrossUplift.P50 < namespace.CrossUplift.P50 {
		t.Errorf("resource (%d) must be >= namespace (%d): a finer eraser cannot lose cross-uplift",
			resource.CrossUplift.P50, namespace.CrossUplift.P50)
	}
	if !(resource.CrossUplift.P50 > 0) {
		t.Errorf("resource eraser cross-uplift (%d) must be > 0: a write-mixed fleet should still gain from sharing",
			resource.CrossUplift.P50)
	}
}

// TestFleet_FinerEraserNoShareStillZero is the anti-inflation control at finer
// granularity: a no-shared-reads fleet must still show EXACTLY zero cross-uplift under
// the resource eraser. If finer invalidation ever fabricated a positive uplift here,
// the headline gain would be a harness artifact rather than real cross-agent dedup.
func TestFleet_FinerEraserNoShareStillZero(t *testing.T) {
	cm := DefaultCostModel()
	defer SetInvalidation(vdso.Global)
	SetInvalidation(vdso.Resource)
	c := RunFleetCell(context.Background(), FleetNoShare, 16, 8, 40, fleetSeed, cm)
	if c.CrossUplift.P50 != 0 || c.CrossUplift.Max > 0 {
		t.Errorf("resource-eraser no-share cross uplift must be 0 (got p50=%d max=%d): finer invalidation must not fabricate sharing",
			c.CrossUplift.P50, c.CrossUplift.Max)
	}
}

// TestFleet_RouteEndpointsDistinct pins the load-bearing route-generator invariant:
// the first 8 routes are byte-identical to the v0.1 catalog (so every default-pool
// sweep is unchanged), and routes past 8 are genuinely DISTINCT (so a --shared-pool > 8
// is not silently collapsed back to 8 by a modulo-8 args collision — the bug the
// "distinct past 8" change fixed).
func TestFleet_RouteEndpointsDistinct(t *testing.T) {
	wantO := []string{"SFO", "LAX", "BOS", "JFK", "ORD", "SEA", "ATL", "DFW"}
	wantD := []string{"JFK", "ORD", "SEA", "SFO", "LAX", "BOS", "DFW", "ATL"}
	for i := 0; i < 8; i++ {
		o, d := routeEndpoints(i)
		if o != wantO[i] || d != wantD[i] {
			t.Errorf("routeEndpoints(%d) = %s-%s, want %s-%s (first 8 must match the v0.1 catalog)", i, o, d, wantO[i], wantD[i])
		}
	}
	seen := map[string]int{}
	for i := 0; i < 128; i++ {
		o, d := routeEndpoints(i)
		k := o + "-" + d
		if prev, ok := seen[k]; ok {
			t.Fatalf("routeEndpoints(%d) collides with routeEndpoints(%d) (both %q): pool > 8 is not distinct", i, prev, k)
		}
		seen[k] = i
	}
}

const fleetSeed int64 = 0x5EED_F1EE
