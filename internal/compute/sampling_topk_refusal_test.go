package compute_test

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=50ms lane=default
func TestDeviceSamplingRefusalIsTransactional(t *testing.T) {
	t.Parallel()
	negZero := math.Copysign(0, -1)
	cases := []struct {
		name       string
		row        []float64
		mass, topP float64
		topK       int
	}{
		{name: "empty", mass: 1},
		{name: "negative would be excluded", row: []float64{2, -1, 1}, mass: 2, topK: 1},
		{name: "negative without truncation", row: []float64{2, -1}, mass: 1},
		{name: "nan payload", row: []float64{negZero, 2, math.Float64frombits(0x7ff8000000000123)}, mass: 2, topK: 1},
		{name: "positive infinity", row: []float64{2, math.Inf(1)}, mass: 2, topK: 1},
		{name: "negative infinity", row: []float64{2, math.Inf(-1)}, mass: 2, topK: 1},
		{name: "zero incoming mass", row: []float64{1, 2}, mass: 0, topK: 1},
		{name: "negative incoming mass", row: []float64{1, 2}, mass: -1, topK: 1},
		{name: "nan incoming mass", row: []float64{1, 2}, mass: math.NaN(), topP: 0.5},
		{name: "infinite incoming mass", row: []float64{1, 2}, mass: math.Inf(1), topK: 1},
		{name: "empty support with stale mass", row: []float64{negZero, 0}, mass: 1},
		// Deliberately stale finite incoming mass isolates the late failure:
		// top-k overflows and would have already overwritten the final 1.
		{name: "top-k overflow", row: []float64{math.MaxFloat64, math.MaxFloat64, 1}, mass: 1, topK: 2},
		{name: "pipeline overflow", row: []float64{math.MaxFloat64, math.MaxFloat64, 1}, mass: 1, topK: 2, topP: 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := append([]float64(nil), tc.row...)
			mass, err := compute.DeviceSamplingTruncation(tc.row, tc.mass, tc.topP, tc.topK)
			if !errors.Is(err, compute.ErrNonFiniteLogits) || mass != 0 {
				t.Fatalf("mass=%g err=%v, want zero and ErrNonFiniteLogits", mass, err)
			}
			for i := range before {
				if math.Float64bits(tc.row[i]) != math.Float64bits(before[i]) {
					t.Fatalf("refusal changed row[%d]: bits %016x -> %016x", i, math.Float64bits(before[i]), math.Float64bits(tc.row[i]))
				}
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDeviceSamplingRefusalFixPreservesExactLaw(t *testing.T) {
	t.Parallel()
	negZero := math.Copysign(0, -1)
	cases := []struct {
		name string
		row  []float64
		topP float64
		topK int
	}{
		{name: "signed-zero rank tie", row: []float64{negZero, 0, 1}, topK: 2},
		{name: "signed-zero inert", row: []float64{negZero, 0, 1}},
		{name: "non-associative mass", row: []float64{1, 0x1p-53, 0x1p-53, 0.5, 0.25, 0x1p-52}, topK: 5, topP: 0.99},
		{name: "minimal tied prefix", row: []float64{0, 1, 1, 1, 1, 0}, topK: 3, topP: 0.5},
		{name: "subnormal target", row: []float64{0, math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64}, topP: math.SmallestNonzeroFloat64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var mass float64
			for _, p := range tc.row {
				mass += p
			}
			want := append([]float64(nil), tc.row...)
			// Existing independent full-sort oracle, not production radix.
			wantMass := oracleCPUTruncation(want, mass, tc.topP, tc.topK)
			got := append([]float64(nil), tc.row...)
			gotMass, err := compute.DeviceSamplingTruncation(got, mass, tc.topP, tc.topK)
			if err != nil || math.Float64bits(gotMass) != math.Float64bits(wantMass) {
				t.Fatalf("mass=%g err=%v, want bits %016x", gotMass, err, math.Float64bits(wantMass))
			}
			for i := range want {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
					t.Fatalf("row[%d] bits %016x != oracle %016x", i, math.Float64bits(got[i]), math.Float64bits(want[i]))
				}
			}
			coins := []float64{0, math.Nextafter(1, 0)}
			var cumulative float64
			for _, p := range want {
				if p <= 0 {
					continue
				}
				cumulative += p
				boundary := cumulative / wantMass
				for _, c := range []float64{math.Nextafter(boundary, 0), boundary, math.Nextafter(boundary, 1)} {
					if c >= 0 && c < 1 {
						coins = append(coins, c)
					}
				}
			}
			// Replayed host coins test this row's law, not a GPU RNG or the
			// request-local absolute-position/cancellation protocol of #14198.
			rng := rand.New(rand.NewSource(14198))
			for i := 0; i < 256; i++ {
				coins = append(coins, rng.Float64())
			}
			for _, coin := range coins {
				wantToken := len(want) - 1
				var cdf float64
				for i, p := range want {
					if p <= 0 {
						continue
					}
					wantToken = i
					cdf += p
					if cdf > coin*wantMass {
						break
					}
				}
				if token := compute.DrawToken(got, gotMass, coin); token != wantToken {
					t.Fatalf("coin=%g token=%d, want %d", coin, token, wantToken)
				}
			}
		})
	}
}
