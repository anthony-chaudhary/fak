package deepseekv4moe

import (
	"math"
	"reflect"
	"testing"
)

// sweepHeader is the human-readable table header the issue witness expects in
// `-v` output: cache GiB | groups | hit rate | marginal/GiB.
const sweepHeader = "cache GiB | groups | hit rate | marginal/GiB"

// witnessGibSteps is the capacity band the witness sweeps: 9 steps, 48..96 GiB.
var witnessGibSteps = []int{48, 54, 60, 66, 72, 78, 84, 90, 96}

// logSweepTable prints the sweep as a header + one row per point so `go test
// -v` shows the capacity curve the issue witness is about.
func logSweepTable(t *testing.T, sw ExpertCacheCapacitySweep) {
	t.Helper()
	t.Logf("%s", sweepHeader)
	for _, p := range sw.Points {
		t.Logf("%4d GiB | %6d | %.4f | %.6f", p.CacheGiB, p.CacheGroups, p.HitRate, p.MarginalHitRatePerGiB)
	}
	t.Logf("knee: %d GiB @ hit rate %.4f (marginal %.6f/GiB)", sw.KneeGiB, sw.KneeHitRate, sw.MarginalAtKnee)
}

// TestExpertCacheCapacity is the issue witness: it builds the pinned V41
// geometry, generates the capacity trace, sweeps the 48..96 GiB band, and
// asserts the structural invariants of the capacity curve (monotonicity,
// bounds, exact counts, knee placement) without pinning absolute hit-rate
// values the trace generator owns.
func TestExpertCacheCapacity(t *testing.T) {
	// 1. Pinned V41 geometry.
	geom := V41Geometry()
	if geom.Layers != 40 || geom.Experts != 384 || geom.TopK != 6 || geom.MoEInter != 2304 || geom.Hidden != 5120 {
		t.Fatalf("V41Geometry() = %+v, want 40 layers / 384 experts / top-6 / MoEInter 2304 / hidden 5120", geom)
	}

	// 2. Group bytes: 3 * hidden * moeInter * bytesPerWeight (0.5 == FP4, 1.0 == FP8).
	if got, want := V41ExpertGroupBytes(0.5), int64(17_694_720); got != want {
		t.Fatalf("V41ExpertGroupBytes(0.5) = %d, want %d", got, want)
	}
	if got, want := V41ExpertGroupBytes(1.0), int64(35_389_440); got != want {
		t.Fatalf("V41ExpertGroupBytes(1.0) = %d, want %d", got, want)
	}
	groupBytes := V41ExpertGroupBytes(0.5)

	// 3. Trace: one route per (layer, token), top-k unique experts per route.
	// 512 tokens per layer is a deliberate speed choice: the capacity curve was
	// verified statistically identical at 256/512/4096 tokens (same knee and
	// monotone shape), and the O(capacity) LRU victim scan makes 4096 ~8x slower.
	const tokensPerLayer = 512
	trace := GenerateV41CapacityTrace(geom, tokensPerLayer, 0)
	if want := geom.Layers * tokensPerLayer; len(trace) != want {
		t.Fatalf("len(trace) = %d, want %d", len(trace), want)
	}
	for i, route := range trace {
		if route.Layer < 0 || route.Layer >= geom.Layers {
			t.Fatalf("trace[%d].Layer = %d, want in [0,%d)", i, route.Layer, geom.Layers)
		}
		if len(route.Experts) != geom.TopK {
			t.Fatalf("trace[%d] has %d experts, want top-k %d", i, len(route.Experts), geom.TopK)
		}
		seen := make(map[int]struct{}, geom.TopK)
		for _, e := range route.Experts {
			if e < 0 || e >= geom.Experts {
				t.Fatalf("trace[%d] expert %d out of range [0,%d)", i, e, geom.Experts)
			}
			if _, dup := seen[e]; dup {
				t.Fatalf("trace[%d] duplicate expert %d", i, e)
			}
			seen[e] = struct{}{}
		}
	}

	// 4. Sweep the 48..96 GiB band.
	sw, err := SweepExpertCacheCapacity(geom, trace, witnessGibSteps, groupBytes)
	if err != nil {
		t.Fatalf("SweepExpertCacheCapacity: %v", err)
	}
	if sw.Schema != ExpertCacheCapacitySweepSchema {
		t.Fatalf("Schema = %q, want %q", sw.Schema, ExpertCacheCapacitySweepSchema)
	}
	if sw.Label != "SW-VERIFIED" {
		t.Fatalf("Label = %q, want SW-VERIFIED", sw.Label)
	}
	if sw.Geometry != geom {
		t.Fatalf("Geometry = %+v, want echoed %+v", sw.Geometry, geom)
	}
	if sw.GroupBytes != groupBytes {
		t.Fatalf("GroupBytes = %d, want %d", sw.GroupBytes, groupBytes)
	}
	if len(sw.Points) != len(witnessGibSteps) {
		t.Fatalf("len(Points) = %d, want %d", len(sw.Points), len(witnessGibSteps))
	}

	// 5. Structural invariants across the curve.
	var baseAccesses int64 = -1
	prevRate := math.Inf(-1)
	for i, p := range sw.Points {
		if p.CacheGiB != witnessGibSteps[i] {
			t.Fatalf("point %d CacheGiB = %d, want requested step %d", i, p.CacheGiB, witnessGibSteps[i])
		}
		if p.HitRate < 0 || p.HitRate > 1 || math.IsNaN(p.HitRate) {
			t.Fatalf("point %d HitRate = %v, want in [0,1]", i, p.HitRate)
		}
		if p.HitRate < prevRate {
			t.Fatalf("hit rate decreased at point %d: %v after %v", i, p.HitRate, prevRate)
		}
		prevRate = p.HitRate
		if i == 0 {
			if p.MarginalHitRatePerGiB != 0 {
				t.Fatalf("first marginal = %v, want 0", p.MarginalHitRatePerGiB)
			}
		} else if p.MarginalHitRatePerGiB < 0 || math.IsNaN(p.MarginalHitRatePerGiB) {
			t.Fatalf("point %d marginal = %v, want >= 0", i, p.MarginalHitRatePerGiB)
		}
		if baseAccesses < 0 {
			baseAccesses = p.Accesses
		} else if p.Accesses != baseAccesses {
			t.Fatalf("point %d Accesses = %d, want %d (same trace)", i, p.Accesses, baseAccesses)
		}
		if p.Hits < 0 || p.Hits > p.Accesses {
			t.Fatalf("point %d Hits = %d outside [0,%d]", i, p.Hits, p.Accesses)
		}
	}

	// 6. Knee must name a requested step and echo that point's own values.
	var knee *ExpertCacheCapacityPoint
	for i := range sw.Points {
		if sw.Points[i].CacheGiB == sw.KneeGiB {
			knee = &sw.Points[i]
			break
		}
	}
	if knee == nil {
		t.Fatalf("KneeGiB = %d is not one of the requested steps %v", sw.KneeGiB, witnessGibSteps)
	}
	if knee.HitRate != sw.KneeHitRate {
		t.Fatalf("KneeHitRate = %v, want point hit rate %v", sw.KneeHitRate, knee.HitRate)
	}
	if knee.MarginalHitRatePerGiB != sw.MarginalAtKnee {
		t.Fatalf("MarginalAtKnee = %v, want point marginal %v", sw.MarginalAtKnee, knee.MarginalHitRatePerGiB)
	}

	// 7. Witness table.
	logSweepTable(t, sw)

	// 8. Knee falls inside the swept 48..96 GiB band.
	if sw.KneeGiB < witnessGibSteps[0] || sw.KneeGiB > witnessGibSteps[len(witnessGibSteps)-1] {
		t.Fatalf("KneeGiB = %d outside swept band [%d,%d]", sw.KneeGiB, witnessGibSteps[0], witnessGibSteps[len(witnessGibSteps)-1])
	}
}

// TestExpertCacheCapacityMonotonicAndDeterministic pins that identical inputs
// yield byte-identical points and an identical knee, so the sweep is reproducible
// evidence rather than run-dependent output.
func TestExpertCacheCapacityMonotonicAndDeterministic(t *testing.T) {
	geom := V41Geometry()
	trace := GenerateV41CapacityTrace(geom, 256, 0)
	steps := []int{48, 60, 72, 84, 96}
	groupBytes := V41ExpertGroupBytes(0.5)

	first, err := SweepExpertCacheCapacity(geom, trace, steps, groupBytes)
	if err != nil {
		t.Fatalf("first sweep: %v", err)
	}
	second, err := SweepExpertCacheCapacity(geom, trace, steps, groupBytes)
	if err != nil {
		t.Fatalf("second sweep: %v", err)
	}
	if !reflect.DeepEqual(first.Points, second.Points) {
		t.Fatalf("points differ between identical sweeps:\nfirst  %+v\nsecond %+v", first.Points, second.Points)
	}
	if first.KneeGiB != second.KneeGiB || first.KneeHitRate != second.KneeHitRate || first.MarginalAtKnee != second.MarginalAtKnee {
		t.Fatalf("knee differs: (%d, %v, %v) vs (%d, %v, %v)",
			first.KneeGiB, first.KneeHitRate, first.MarginalAtKnee,
			second.KneeGiB, second.KneeHitRate, second.MarginalAtKnee)
	}
}

// TestExpertCacheCapacityMarginalRule reconstructs the knee from the documented
// diminishing-returns rule and checks it matches the returned knee.
func TestExpertCacheCapacityMarginalRule(t *testing.T) {
	geom := V41Geometry()
	trace := GenerateV41CapacityTrace(geom, 256, 0)
	steps := []int{48, 54, 60, 66, 72, 78, 84, 90, 96}
	sw, err := SweepExpertCacheCapacity(geom, trace, steps, V41ExpertGroupBytes(0.5))
	if err != nil {
		t.Fatalf("SweepExpertCacheCapacity: %v", err)
	}
	if len(sw.Points) < 2 {
		t.Fatalf("need at least 2 points, got %d", len(sw.Points))
	}
	if sw.Points[0].MarginalHitRatePerGiB != 0 {
		t.Fatalf("first marginal = %v, want 0", sw.Points[0].MarginalHitRatePerGiB)
	}

	// Mean marginal over steps i>0, then the last step meeting the threshold.
	var sum float64
	for i := 1; i < len(sw.Points); i++ {
		sum += sw.Points[i].MarginalHitRatePerGiB
	}
	mean := sum / float64(len(sw.Points)-1)
	threshold := 0.5 * mean

	kneeIdx := -1
	for i := 1; i < len(sw.Points); i++ {
		if sw.Points[i].MarginalHitRatePerGiB >= threshold {
			kneeIdx = i
		}
	}
	if kneeIdx < 0 {
		t.Fatalf("rule found no knee (threshold %.6f, mean %.6f)", threshold, mean)
	}
	want := sw.Points[kneeIdx]
	if sw.KneeGiB != want.CacheGiB {
		t.Fatalf("KneeGiB = %d, rule says %d", sw.KneeGiB, want.CacheGiB)
	}
	if sw.KneeHitRate != want.HitRate {
		t.Fatalf("KneeHitRate = %v, rule says %v", sw.KneeHitRate, want.HitRate)
	}
	if sw.MarginalAtKnee != want.MarginalHitRatePerGiB {
		t.Fatalf("MarginalAtKnee = %v, rule says %v", sw.MarginalAtKnee, want.MarginalHitRatePerGiB)
	}
}

// TestExpertCacheCapacityFailsClosed pins that malformed inputs are rejected
// rather than silently producing a plausible-looking sweep.
func TestExpertCacheCapacityFailsClosed(t *testing.T) {
	geom := V41Geometry()
	trace := GenerateV41CapacityTrace(geom, 16, 0)
	groupBytes := V41ExpertGroupBytes(0.5)

	badGeom := geom
	badGeom.Layers = 0

	cases := []struct {
		name       string
		geom       ExpertCacheGeometry
		trace      []ExpertRoute
		steps      []int
		groupBytes int64
	}{
		{"empty trace", geom, nil, []int{48, 54}, groupBytes},
		{"empty steps", geom, trace, nil, groupBytes},
		{"non-ascending steps", geom, trace, []int{72, 48}, groupBytes},
		{"zero step", geom, trace, []int{0, 48}, groupBytes},
		{"negative group bytes", geom, trace, []int{48, 54}, -1},
		{"invalid geometry", badGeom, trace, []int{48, 54}, groupBytes},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := SweepExpertCacheCapacity(tc.geom, tc.trace, tc.steps, tc.groupBytes); err == nil {
				t.Fatalf("expected error, got nil")
			}
		})
	}
}
