package compute

import (
	"math"
	"testing"
)

// Closed in-memory oracle for state-spaces/mamba selective_scan_fwd_kernel.cuh (ADAPT).
// [SW-VERIFIED] structure and oracle only: associative (a,b) chunked prefix scan with
// running-prefix carry across chunks and final-state extraction.
type scanPair struct{ a, b float64 }

func scanCombine(prev, cur scanPair) scanPair {
	return scanPair{a: cur.a * prev.a, b: cur.a*prev.b + cur.b}
}

func TestSelectiveScanPrefill(t *testing.T) {
	t.Parallel()
	if cuhDefine(t, "selective_scan_prefill.cuh", "FAK_MAMBA_MAX_DSTATE") != 256 ||
		cuhDefine(t, "selective_scan_prefill.cuh", "FAK_MAMBA_SM_MIN") != 75 ||
		cuhDefine(t, "selective_scan_prefill.cuh", "FAK_MAMBA_BF16_SM_MIN") != 80 ||
		cuhDefine(t, "selective_scan_prefill.cuh", "FAK_MAMBA_SCAN_OP_PREFIX") != 1 {
		t.Fatal("selective_scan_prefill.cuh constants drifted")
	}

	pairs := []scanPair{
		{0.9, 1.0}, {0.8, -0.5}, {0.7, 2.0}, {0.6, 0.25}, {0.95, -1.5}, {0.5, 3.0},
	}
	const chunk = 2
	// Chunked associative scan with running-prefix carry.
	got := make([]float64, len(pairs))
	carry := scanPair{a: 1.0, b: 0.0}
	for start := 0; start < len(pairs); start += chunk {
		end := min(start+chunk, len(pairs))
		local := scanPair{a: 1.0, b: 0.0}
		for i := start; i < end; i++ {
			local = scanCombine(local, pairs[i])
			full := scanCombine(carry, local)
			got[i] = full.b
		}
		carry = scanCombine(carry, local)
	}
	finalState := carry.b

	// Sequential recurrence oracle: h_t = a_t * h_{t-1} + b_t.
	h := 0.0
	for i, p := range pairs {
		h = p.a*h + p.b
		if math.Abs(got[i]-h) > 1e-12 {
			t.Fatalf("step %d: chunked=%v, seq=%v", i, got[i], h)
		}
	}
	if math.Abs(finalState-h) > 1e-12 {
		t.Fatalf("finalState=%v, want %v", finalState, h)
	}
}
