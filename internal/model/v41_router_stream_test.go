package model

import (
	"errors"
	"math"
	"sort"
	"testing"
)

// Runtime estimate only; unmeasured.
// fak-test:runtime fast est=100ms lane=default
func TestV41StreamingRouterMatchesStableReference(t *testing.T) {
	// Keep the former full-width algorithm independent of the production
	// selector, including both score-publication and normalization contracts.
	reference := func(logits, bias []float32, full bool) ([]routePick, error) {
		raw, choice := make([]float32, len(logits)), make([]float32, len(logits))
		indices := make([]int, len(logits))
		for i, z := range logits {
			if math.IsNaN(float64(z)) || math.IsInf(float64(z), 0) {
				return nil, &v4RouteError{Field: "logits"}
			}
			zf := float64(z)
			softplus := math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf)))
			if full {
				softplus = float64(float32(softplus))
			}
			raw[i] = float32(math.Sqrt(softplus))
			choice[i], indices[i] = raw[i], i
			if bias != nil {
				if math.IsNaN(float64(bias[i])) || math.IsInf(float64(bias[i]), 0) {
					return nil, &v4RouteError{Field: "correction_bias"}
				}
				choice[i] += bias[i]
			}
		}
		sort.SliceStable(indices, func(i, j int) bool { return choice[indices[i]] > choice[indices[j]] })
		out := make([]routePick, V41RouterTopK)
		var sum float32
		for i := range out {
			expert := indices[i]
			out[i] = routePick{expert: expert, weight: raw[expert]}
			sum += raw[expert]
		}
		if !full && sum == 0 {
			return nil, &v4RouteError{Field: "normalization"}
		}
		denominator := sum
		if full {
			denominator = float32(sum + float32(1e-20))
		}
		for i := range out {
			if full {
				out[i].weight = float32(out[i].weight / denominator)
				out[i].weight = float32(out[i].weight * V41RouterRouteScale)
			} else {
				out[i].weight = out[i].weight / sum * V41RouterRouteScale
			}
		}
		return out, nil
	}
	previous := v4BitonicTopKEnabled()
	defer SetV4BitonicTopK(previous)
	for _, partial := range []bool{false, true} {
		SetV4BitonicTopK(partial)
		for fixture := 0; fixture < 9; fixture++ {
			logits := make([]float32, V41RouterExperts)
			bias := make([]float32, V41RouterExperts)
			seed := uint32(14043)
			for i := range logits {
				seed = seed*1664525 + 1013904223
				switch fixture {
				case 0:
					logits[i] = float32(int(seed%4096)-2048) / 64
				case 1:
					logits[i] = float32(i % 7) // Ties cross the sixth-place boundary.
				case 2:
					logits[i] = float32(V41RouterExperts - i)
				case 3:
					logits[i] = float32(i)
				case 4:
					logits[i] = -120 // Full and reduced score publication diverge.
				case 5:
					logits[i] = -1000 // All-zero scores are legal only in full mode.
				case 6:
					logits[i] = math.MaxFloat32
				case 7, 8:
					logits[i] = 0
				}
				bias[i] = float32(int(seed%13)-6) / 4
			}
			if fixture == 7 || fixture == 8 {
				logits[23] = float32(math.NaN())
				if fixture == 8 {
					bias[6] = float32(math.NaN()) // Earlier bias must win over later logit.
				}
			}
			for _, correction := range [][]float32{nil, bias} {
				for _, full := range []bool{false, true} {
					want, wantErr := reference(logits, correction, full)
					got, gotErr := v41RouteForGeometry(logits, correction, v41DefaultRouterConfig(), full)
					if wantErr != nil || gotErr != nil {
						var wantRoute, gotRoute *v4RouteError
						if !errors.As(wantErr, &wantRoute) || !errors.As(gotErr, &gotRoute) || gotRoute.Field != wantRoute.Field || got != nil {
							t.Fatalf("partial=%v fixture=%d full=%v: fault got %v, want %v", partial, fixture, full, gotErr, wantErr)
						}
						continue
					}
					if len(got) != len(want) {
						t.Fatalf("route length got %d, want %d", len(got), len(want))
					}
					for i := range want {
						if got[i].expert != want[i].expert || math.Float32bits(got[i].weight) != math.Float32bits(want[i].weight) {
							t.Fatalf("partial=%v fixture=%d full=%v slot=%d: got %+v, want %+v", partial, fixture, full, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}
