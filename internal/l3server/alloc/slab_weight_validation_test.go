package alloc

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestSlabPlanRejectsInvalidWeights(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name    string
		weights map[uint64]float64
		wantErr bool
		budget  uint64
		want    []slabClassPlan
	}{
		{name: "negative selected weight", weights: map[uint64]float64{256: -1, 512: 2}, wantErr: true},
		{name: "NaN selected weight", weights: map[uint64]float64{256: math.NaN(), 512: 1}, wantErr: true},
		{name: "positive infinity", weights: map[uint64]float64{256: math.Inf(1), 512: 1}, wantErr: true},
		{name: "negative infinity", weights: map[uint64]float64{256: math.Inf(-1), 512: 1}, wantErr: true},
		{name: "zero total", weights: map[uint64]float64{256: 0, 512: 0}, wantErr: true},
		// Class 256 gets fallback weight 0.5, then is filtered for too few
		// slots. Explicitly zero-weight model class 512 is retained alone.
		{name: "filter leaves zero total", weights: map[uint64]float64{512: 0}, budget: 1, wantErr: true},
		{name: "finite weights overflow total", weights: map[uint64]float64{256: math.MaxFloat64, 512: math.MaxFloat64}, wantErr: true},
		{name: "default dedicated weights", want: []slabClassPlan{
			{size: 256, weight: 5, regionBytes: 52224},
			{size: 512, weight: 95, regionBytes: 995840},
		}},
		{name: "empty explicit map", weights: map[uint64]float64{}, want: []slabClassPlan{
			{size: 256, weight: 0.5, regionBytes: 524288},
			{size: 512, weight: 0.5, regionBytes: 524288},
		}},
		{name: "zero class keeps minimum", weights: map[uint64]float64{256: 0, 512: 1}, want: []slabClassPlan{
			{size: 256, weight: 0, regionBytes: 1024},
			{size: 512, weight: 1, regionBytes: 1048576},
		}},
		{name: "subnormal positive weights", weights: map[uint64]float64{256: math.SmallestNonzeroFloat64, 512: math.SmallestNonzeroFloat64}, want: []slabClassPlan{
			{size: 256, weight: math.SmallestNonzeroFloat64, regionBytes: 524288},
			{size: 512, weight: math.SmallestNonzeroFloat64, regionBytes: 524288},
		}},
		{name: "largest finite total", weights: map[uint64]float64{256: math.MaxFloat64, 512: 0}, want: []slabClassPlan{
			{size: 256, weight: math.MaxFloat64, regionBytes: 1048576},
			{size: 512, weight: 0, regionBytes: 2048},
		}},
		{name: "unselected weight stays ignored", weights: map[uint64]float64{256: 1, 512: 1, 999: -1}, want: []slabClassPlan{
			{size: 256, weight: 1, regionBytes: 524288},
			{size: 512, weight: 1, regionBytes: 524288},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			budget := tc.budget
			if budget == 0 {
				budget = 1 << 20
			}
			plans, err := planSlabClasses(SlabConfig{
				MaxMemoryBytes: budget,
				ModelPageBytes: 512,
				Dedicated:      true,
				ClassWeights:   tc.weights,
			})
			if tc.wantErr {
				if err == nil || !strings.Contains(err.Error(), "weight") || plans != nil {
					t.Fatalf("plans=%v err=%v, want nil plans and weight error", plans, err)
				}
				return
			}
			if err != nil || !reflect.DeepEqual(plans, tc.want) {
				t.Fatalf("plans=%v err=%v, want %v", plans, err, tc.want)
			}
		})
	}
}
