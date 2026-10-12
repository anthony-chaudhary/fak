package ggufload

import (
	"math"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestActivatedExpertFitRejectsExpertLayerProductOverflow(t *testing.T) {
	t.Parallel()
	if strconv.IntSize < 64 {
		t.Skip("two positive 32-bit int counts cannot overflow an int64 product")
	}
	for _, tc := range []struct {
		name      string
		experts   int64
		layers, k int
		overflow  bool
	}{
		{"negative_wrapped_product", 1 << 62, 2, 2, true},
		{"zero_wrapped_product", 1 << 62, 4, 2, true},
		{"positive_wrapped_product", (1 << 62) + 1, 4, 2, true},
		{"largest_safe_four_layer_count", math.MaxInt64 / 4, 4, 2, false},
		{"unknown_k_keeps_whole_band", (1 << 62) + 1, 4, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			as := RoutedExpertActiveSet{
				NumExperts:        int(tc.experts),
				MoELayers:         tc.layers,
				ExpertsUsed:       tc.k,
				NonExpertResident: 8,
				RoutedResident:    64,
				ActivePerToken:    16,
			}
			got, ok, err := activatedExpertFit(as, 24)
			if tc.overflow {
				const wantErr = "gguf: activated expert count times MoE layers overflow int64"
				if err == nil || err.Error() != wantErr || ok || got != (ActivatedExpertFit{}) {
					t.Fatalf("activatedExpertFit = (%+v, %v, %v), want empty fit, false, %q", got, ok, err, wantErr)
				}
				return
			}
			want := ActivatedExpertFit{
				MoELayers:           tc.layers,
				NumExperts:          int(tc.experts),
				ExpertsUsed:         tc.k,
				DeviceBaseBytes:     8,
				RoutedBandBytes:     64,
				ActivatedLayerBytes: 2,
				ActivatedTokenBytes: 16,
				DeviceBudgetBytes:   24,
				RingBytes:           16,
				HostBandBytes:       48,
				MinFits:             true,
				TokenFits:           true,
			}
			if tc.k == 0 {
				want.ActivatedLayerBytes = 64
				want.ActivatedTokenBytes = 64
				want.RingBytes = 64
				want.HostBandBytes = 0
				want.MinFits = false
				want.TokenFits = false
			}
			if err != nil || !ok || got != want {
				t.Fatalf("activatedExpertFit = (%+v, %v, %v), want (%+v, true, nil)", got, ok, err, want)
			}
		})
	}
}
