package agent

import (
	"math"
	"sort"
	"testing"
)

// referenceClusterEligible is intentionally separate from the production selector:
// it states the issue contract directly rather than calling production helpers.
func referenceClusterEligible(vocabSize int, topP float64, topK int, clusterCapable, deviceResident bool) bool {
	wideVocabulary := vocabSize >= 65536
	truncationRequested := (topK > 0 && topK < vocabSize) || (topP > 0 && topP < 1)
	return wideVocabulary && truncationRequested && clusterCapable && deviceResident
}

// referenceCPUTruncation independently implements top-k then top-p over index-aligned
// probabilities. It is the correctness oracle for the explicit CPU fallback.
func referenceCPUTruncation(in []float64, topP float64, topK int) ([]float64, float64) {
	out := append([]float64(nil), in...)
	order := make([]int, len(out))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		if out[order[i]] == out[order[j]] {
			return order[i] < order[j]
		}
		return out[order[i]] > out[order[j]]
	})

	keep := make([]bool, len(out))
	for i := range keep {
		keep[i] = true
	}
	if topK > 0 && topK < len(out) {
		for i := range keep {
			keep[i] = false
		}
		for _, idx := range order[:topK] {
			keep[idx] = true
		}
	}

	if topP > 0 && topP < 1 {
		var available float64
		for i, p := range out {
			if keep[i] {
				available += p
			}
		}
		target := topP * available
		var cumulative float64
		for rank, idx := range order {
			if !keep[idx] {
				continue
			}
			if rank > 0 && cumulative >= target {
				keep[idx] = false
				continue
			}
			cumulative += out[idx]
		}
	}

	var sum float64
	for i := range out {
		if keep[i] {
			sum += out[i]
		} else {
			out[i] = 0
		}
	}
	return out, sum
}

func TestClusterSamplingSelectorMatchesIndependentCPUOracle(t *testing.T) {
	cases := []struct {
		name                           string
		vocabSize, topK                int
		topP                           float64
		clusterCapable, deviceResident bool
	}{
		{name: "wide eligible top-k", vocabSize: 131072, topK: 64, clusterCapable: true, deviceResident: true},
		{name: "wide eligible top-p", vocabSize: 65536, topP: 0.9, clusterCapable: true, deviceResident: true},
		{name: "narrow vocabulary", vocabSize: 65535, topK: 64, clusterCapable: true, deviceResident: true},
		{name: "cluster unavailable", vocabSize: 131072, topK: 64, deviceResident: true},
		{name: "host resident", vocabSize: 131072, topK: 64, clusterCapable: true},
		{name: "no truncation", vocabSize: 131072, clusterCapable: true, deviceResident: true},
		{name: "disabled cutoffs", vocabSize: 131072, topK: 131072, topP: 1, clusterCapable: true, deviceResident: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			wantCluster := referenceClusterEligible(tc.vocabSize, tc.topP, tc.topK, tc.clusterCapable, tc.deviceResident)
			got := selectSamplingTruncationPath(tc.vocabSize, tc.topP, tc.topK, samplingTruncationEnvelope{
				clusterCapable: tc.clusterCapable,
				deviceResident: tc.deviceResident,
			})
			if (got == samplingTruncationCluster) != wantCluster {
				t.Fatalf("path=%v, want cluster=%v", got, wantCluster)
			}
		})
	}

	probs := []float64{0.31, 0.27, 0.19, 0.13, 0.07, 0.03}
	wantProbs, wantSum := referenceCPUTruncation(probs, 0.75, 4)
	gotProbs := append([]float64(nil), probs...)
	gotSum := applyCPUSamplingTruncation(gotProbs, 1, 0.75, 4)
	if math.Abs(gotSum-wantSum) > 1e-12 {
		t.Fatalf("CPU fallback sum=%v, oracle=%v", gotSum, wantSum)
	}
	for i := range gotProbs {
		if math.Abs(gotProbs[i]-wantProbs[i]) > 1e-12 {
			t.Fatalf("CPU fallback probs[%d]=%v, oracle=%v (got=%v want=%v)", i, gotProbs[i], wantProbs[i], gotProbs, wantProbs)
		}
	}
}
