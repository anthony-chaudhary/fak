package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// fak-test:runtime fast est=20ms lane=default
func TestRegimeLatencyHistogramsReconcileWithUnlabeled(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	if out := renderInference(m); strings.Contains(out, "_by_regime_seconds") {
		t.Fatalf("idle scrape published regime rows:\n%s", out)
	}
	// uncached prompt, completion, cached, cache-create, dur, ttft
	m.observeInferenceTimed(100, 20, 0, 0, "stop", 2*time.Second, time.Second)                // cold
	m.observeInferenceTimed(40, 20, 60, 0, "stop", time.Second, 300*time.Millisecond)         // partial
	m.observeInferenceTimed(5, 20, 95, 0, "stop", 500*time.Millisecond, 20*time.Millisecond)  // frozen
	m.observeInferenceTimed(0, 20, 100, 0, "stop", 400*time.Millisecond, 10*time.Millisecond) // frozen
	m.observeInferenceTimed(0, 20, 0, 0, "stop", 300*time.Millisecond, 0)                     // unknown, buffered

	snap := m.inferenceSnapshotData()
	if len(snap.regimeHists) != 4 {
		t.Fatalf("regimes booked = %d, want 4: %+v", len(snap.regimeHists), snap.regimeHists)
	}
	families := []struct {
		name   string
		global latencySnapshot
		pick   func(regimeLatencySnapshot) latencySnapshot
	}{
		{"ttft", snap.ttftHist, func(r regimeLatencySnapshot) latencySnapshot { return r.ttft }},
		{"tpot", snap.tpotHist, func(r regimeLatencySnapshot) latencySnapshot { return r.tpot }},
		{"e2e", snap.e2eHist, func(r regimeLatencySnapshot) latencySnapshot { return r.e2e }},
	}
	for _, fam := range families {
		var count uint64
		var sum float64
		buckets := make([]uint64, len(fam.global.buckets))
		for _, regime := range cacheobs.Regimes {
			row := fam.pick(snap.regimeHists[regime])
			count += row.count
			sum += row.sum
			for i, v := range row.buckets {
				buckets[i] += v
			}
		}
		if count != fam.global.count || absDiff(sum, fam.global.sum) > 1e-9 {
			t.Fatalf("%s regime rows count/sum = %d/%v, unlabeled = %d/%v", fam.name, count, sum, fam.global.count, fam.global.sum)
		}
		for i := range buckets {
			if buckets[i] != fam.global.buckets[i] {
				t.Fatalf("%s bucket %d: regime rows %d != unlabeled %d", fam.name, i, buckets[i], fam.global.buckets[i])
			}
		}
	}
	if got := snap.regimeHists[cacheobs.RegimeFrozen].ttft.count; got != 2 {
		t.Fatalf("frozen ttft count = %d, want 2", got)
	}
	if got := snap.regimeHists[cacheobs.RegimeUnknown].ttft.count; got != 0 {
		t.Fatalf("unknown buffered turn landed in ttft: count %d", got)
	}

	out := renderInference(m)
	for _, want := range []string{
		`fak_gateway_inference_ttft_by_regime_seconds_count{regime="cold"} 1`,
		`fak_gateway_inference_ttft_by_regime_seconds_count{regime="frozen"} 2`,
		`fak_gateway_inference_e2e_by_regime_seconds_count{regime="unknown"} 1`,
		"fak_gateway_inference_ttft_seconds_count 4\n",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("scrape missing %q\n%s", want, out)
		}
	}
}

func absDiff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}
