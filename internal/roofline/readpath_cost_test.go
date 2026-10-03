package roofline

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// TestDecomposeReadPathCostFitsRecoversKnownCosts feeds a sweep synthesized
// from a known additive model and asserts the solve recovers it. This is the
// "buffered path" direction: the model holds, so it must fit — and the split,
// the no-disk ceiling and `base` must all be consistent with the injected costs.
func TestDecomposeReadPathCostFitsRecoversKnownCosts(t *testing.T) {
	// base = 100 ms/call (10 tok/s resident), cRAM = 2 ms/miss, cDisk = 20 ms/miss.
	const resident = 10.0
	const base = 1000.0 / resident
	ramCost, diskCost := 2.0, 20.0

	synth := func(m, f float64) ReadPathTierPoint {
		stall := base + m*f*ramCost + m*(1-f)*diskCost
		return ReadPathTierPoint{L2SizeBytes: int64(16 + m), MeanStallMillis: stall, MissesPerCall: m, RAMFraction: f}
	}
	points := []ReadPathTierPoint{
		synth(10, 0.8), // a=8,  b=2,  y=56
		synth(10, 0.2), // a=2,  b=8,  y=164
		synth(20, 0.5), // a=10, b=10, y=220
	}

	m, err := DecomposeReadPathCost(points, resident)
	if err != nil {
		t.Fatalf("unexpected refusal on a usable envelope: %v", err)
	}
	if m.Verdict != ReadPathFits {
		t.Fatalf("want verdict %q, got %q (%s)", ReadPathFits, m.Verdict, m.Reason)
	}
	if math.Abs(m.BaseStallMillis-base) > 1e-9 {
		t.Errorf("base: want %.6g (1000/resident, measured), got %.6g", base, m.BaseStallMillis)
	}
	if math.Abs(m.RAMMissCostMillis-ramCost) > 1e-6 {
		t.Errorf("RAM per-miss cost: want %.6g, got %.6g", ramCost, m.RAMMissCostMillis)
	}
	if math.Abs(m.DiskMissCostMillis-diskCost) > 1e-6 {
		t.Errorf("disk per-miss cost: want %.6g, got %.6g", diskCost, m.DiskMissCostMillis)
	}
	if m.Identifiability <= 0 {
		t.Errorf("identifiability: want > 0 for a split-varying sweep, got %g", m.Identifiability)
	}
	if m.ComputeFraction <= 0 {
		t.Errorf("compute fraction: want > 0, got %g", m.ComputeFraction)
	}
	if got := m.ComputeFraction + m.RAMFraction + m.DiskFraction; math.Abs(got-1) > 1e-9 {
		t.Errorf("time split must sum to 1, got %g", got)
	}
	if m.NoDiskTokensPerSec > m.AllResidentTokensPerSec {
		t.Errorf("no-disk ceiling must sit at or below the all-resident ceiling: noDisk=%g allResident=%g",
			m.NoDiskTokensPerSec, m.AllResidentTokensPerSec)
	}
	if m.ObservedTokensPerSec <= 0 {
		t.Errorf("observed rate at the sweep mean: want > 0, got %g", m.ObservedTokensPerSec)
	}
}

// TestDecomposeReadPathCostFalsifiesNegativeCost is the load-bearing direction
// from the source finding: when the measurements imply a physically impossible
// (negative) per-miss cost, the model must return Falsified — the additive
// assumption does not hold for that read path — and must NOT coerce a split.
func TestDecomposeReadPathCostFalsifiesNegativeCost(t *testing.T) {
	// A read path where the measured stalls *fall* as the RAM-tier share rises
	// (reads overlap with each other and with compute): the fitted RAM cost is
	// negative, exactly the O_DIRECT direct-path falsification the source found.
	points := []ReadPathTierPoint{
		{L2SizeBytes: 16, MeanStallMillis: 90, MissesPerCall: 10, RAMFraction: 0.2},
		{L2SizeBytes: 24, MeanStallMillis: 60, MissesPerCall: 10, RAMFraction: 0.8},
		{L2SizeBytes: 48, MeanStallMillis: 50, MissesPerCall: 20, RAMFraction: 0.5},
	}
	m, err := DecomposeReadPathCost(points, 5.0) // base = 200 ms
	if err != nil {
		t.Fatalf("a falsified model over a usable envelope must return nil error, got %v", err)
	}
	if m.Verdict != ReadPathFalsified {
		t.Fatalf("want verdict %q, got %q", ReadPathFalsified, m.Verdict)
	}
	if m.Reason == "" {
		t.Error("a falsified verdict must carry a reason")
	}
	if m.ComputeFraction != 0 || m.RAMFraction != 0 || m.DiskFraction != 0 {
		t.Errorf("a falsified model must not emit a split, got compute=%g ram=%g disk=%g",
			m.ComputeFraction, m.RAMFraction, m.DiskFraction)
	}
}

// TestDecomposeReadPathCostFalsifiesInvariantSplit pins the hit-rate-invariance
// precondition: a sweep whose points all share one RAM/disk split cannot
// identify the two per-miss costs, so it is Falsified rather than fit to noise.
func TestDecomposeReadPathCostFalsifiesInvariantSplit(t *testing.T) {
	points := []ReadPathTierPoint{
		{L2SizeBytes: 16, MeanStallMillis: 120, MissesPerCall: 2, RAMFraction: 0.5},
		{L2SizeBytes: 32, MeanStallMillis: 140, MissesPerCall: 4, RAMFraction: 0.5},
		{L2SizeBytes: 64, MeanStallMillis: 180, MissesPerCall: 8, RAMFraction: 0.5},
	}
	m, err := DecomposeReadPathCost(points, 10.0)
	if err != nil {
		t.Fatalf("unexpected refusal: %v", err)
	}
	if m.Verdict != ReadPathFalsified {
		t.Fatalf("want verdict %q for a ratio-invariant sweep, got %q", ReadPathFalsified, m.Verdict)
	}
	if !strings.Contains(m.Reason, "hit-rate invariance") {
		t.Errorf("reason should name the failed precondition, got %q", m.Reason)
	}
}

// TestDecomposeReadPathCostRefusesBadEnvelope asserts the input guards: a
// degenerate envelope is a typed refusal (ErrReadPathCostModel), distinct from
// a Falsified model over a usable envelope.
func TestDecomposeReadPathCostRefusesBadEnvelope(t *testing.T) {
	good := ReadPathTierPoint{L2SizeBytes: 16, MeanStallMillis: 100, MissesPerCall: 4, RAMFraction: 0.5}
	cases := []struct {
		name   string
		points []ReadPathTierPoint
		res    float64
	}{
		{"too few points", []ReadPathTierPoint{good}, 10},
		{"zero resident rate", []ReadPathTierPoint{good, good}, 0},
		{"negative resident rate", []ReadPathTierPoint{good, good}, -1},
		{"non-positive L2", []ReadPathTierPoint{{L2SizeBytes: 0, MeanStallMillis: 1, MissesPerCall: 1, RAMFraction: 0.5}, good}, 10},
		{"negative stall", []ReadPathTierPoint{{L2SizeBytes: 16, MeanStallMillis: -1, MissesPerCall: 1, RAMFraction: 0.5}, good}, 10},
		{"RAM fraction out of range", []ReadPathTierPoint{{L2SizeBytes: 16, MeanStallMillis: 1, MissesPerCall: 1, RAMFraction: 1.5}, good}, 10},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := DecomposeReadPathCost(tc.points, tc.res); !errors.Is(err, ErrReadPathCostModel) {
				t.Fatalf("want ErrReadPathCostModel, got %v", err)
			}
		})
	}
}
