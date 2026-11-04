package ggufload

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestAssignLayersTensorSplitUpperBound(t *testing.T) {
	// Pinned placement from llama.cpp src/llama-model.cpp:1496-1517@83078fec:
	// 32 layers, 24 offloaded, split {3,1} -> cumulative {0.75, 1.0}; the first
	// 8 layers stay on the CPU, offloaded layer i lands on the first device
	// whose cumulative share exceeds i/24, so layers 8..25 take device 0 and
	// layers 26..31 take device 1.
	want := make([]int, 32)
	for i := range want {
		want[i] = -1
	}
	for i := 8; i < 26; i++ {
		want[i] = 0
	}
	for i := 26; i < 32; i++ {
		want[i] = 1
	}
	got, err := AssignLayers(32, 24, []float64{3, 1})
	if err != nil {
		t.Fatalf("AssignLayers(32, 24, {3,1}): %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("AssignLayers(32, 24, {3,1}) = %v, want %v", got, want)
	}

	cases := []struct {
		name    string
		nLayer  int
		nGPU    int
		split   []float64
		wantDev []int
		wantErr error
	}{
		{
			name:   "equal two-way split halves the offloaded range",
			nLayer: 32, nGPU: 24, split: []float64{1, 1},
			wantDev: concatDevs(8, -1, 12, 0, 12, 1),
		},
		{
			name:   "single device takes every offloaded layer",
			nLayer: 10, nGPU: 4, split: []float64{1},
			wantDev: concatDevs(6, -1, 4, 0),
		},
		{
			name:   "zero gpu layers keeps everything on the cpu",
			nLayer: 8, nGPU: 0, split: []float64{1, 1},
			wantDev: concatDevs(8, -1),
		},
		{
			name:   "gpu budget above the layer count clamps to all layers",
			nLayer: 4, nGPU: 40, split: []float64{1, 1},
			wantDev: concatDevs(2, 0, 2, 1),
		},
		{
			name:   "empty split degrades to the cpu for every layer",
			nLayer: 6, nGPU: 3, split: nil,
			wantDev: concatDevs(6, -1),
		},
		{
			name: "negative fraction", nLayer: 4, nGPU: 2, split: []float64{1, -1}, wantErr: ErrSplitNegative,
		},
		{
			name: "all-zero fractions", nLayer: 4, nGPU: 2, split: []float64{0, 0}, wantErr: ErrSplitAllZero,
		},
		{
			name: "nan fraction", nLayer: 4, nGPU: 2, split: []float64{math.NaN(), 1}, wantErr: ErrSplitNonFinite,
		},
		{
			name: "negative layer count", nLayer: -1, nGPU: 0, split: nil, wantErr: ErrAssignLayersArgs,
		},
		{
			name: "negative gpu layer count", nLayer: 4, nGPU: -2, split: nil, wantErr: ErrAssignLayersArgs,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := AssignLayers(tc.nLayer, tc.nGPU, tc.split)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("AssignLayers: error = %v, want errors.Is %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("AssignLayers: %v", err)
			}
			if !reflect.DeepEqual(got, tc.wantDev) {
				t.Fatalf("AssignLayers = %v, want %v", got, tc.wantDev)
			}
		})
	}
}

// concatDevs builds a device list from alternating (count, device) pairs.
func concatDevs(pairs ...int) []int {
	var out []int
	for i := 0; i < len(pairs); i += 2 {
		for j := 0; j < pairs[i]; j++ {
			out = append(out, pairs[i+1])
		}
	}
	return out
}
