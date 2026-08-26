package main

import (
	"strings"
	"testing"
)

func TestMetalGGUFPeakCapacity(t *testing.T) {
	const gib = int64(1 << 30)
	tests := []struct {
		name   string
		metal  bool
		steady int64
		total  int64
		known  bool
		refuse bool
	}{
		{
			name:   "Qwen3.8 27B Q4_K_M refuses 36 GiB Mac",
			metal:  true,
			steady: 1592 * gib / 100,
			total:  36 * gib,
			known:  true,
			refuse: true,
		},
		{
			name:   "larger unified memory admits same checkpoint",
			metal:  true,
			steady: 1592 * gib / 100,
			total:  64 * gib,
			known:  true,
			refuse: false,
		},
		{
			name:   "non-Metal remains unchanged",
			metal:  false,
			steady: 1592 * gib / 100,
			total:  36 * gib,
			known:  true,
			refuse: false,
		},
		{
			name:   "unknown host memory remains unchanged",
			metal:  true,
			steady: 1592 * gib / 100,
			total:  0,
			known:  false,
			refuse: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			peak, refuse := metalGGUFPeakCapacity(tt.metal, tt.steady, tt.total, tt.known)
			if refuse != tt.refuse {
				t.Fatalf("refuse = %v, want %v (peak=%d)", refuse, tt.refuse, peak)
			}
		})
	}
}

func TestStreamedQ4KMetalCapacityUsesMeasuredNativePeak(t *testing.T) {
	if streamedQ4KMeasuredPeakBytes != 22754885632 {
		t.Fatalf("measured peak = %d, want canonical no-FREE_CPU receipt's /usr/bin/time RSS", streamedQ4KMeasuredPeakBytes)
	}
	if streamedQ4KMetalCapacityBytes < streamedQ4KMeasuredPeakBytes {
		t.Fatalf("capacity bound %d is below measured peak %d", streamedQ4KMetalCapacityBytes, streamedQ4KMeasuredPeakBytes)
	}

	tests := []struct {
		name      string
		total     int64
		known     bool
		required  int64
		refuse    bool
		wantError bool
	}{
		{name: "one byte below measured bound refuses", total: streamedQ4KMetalCapacityBytes - 1, known: true, required: streamedQ4KMetalCapacityBytes, refuse: true, wantError: true},
		{name: "measured bound admits", total: streamedQ4KMetalCapacityBytes, known: true, required: streamedQ4KMetalCapacityBytes},
		{name: "36 GiB receipt host refuses after positive swap", total: 36 << 30, known: true, required: streamedQ4KMetalCapacityBytes, refuse: true, wantError: true},
		{name: "44 GiB measured bound admits", total: 44 << 30, known: true, required: streamedQ4KMetalCapacityBytes},
		{name: "unknown host memory preserves existing behavior", known: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			required, refuse := streamedQ4KMetalCapacity(tt.total, tt.known)
			if required != tt.required || refuse != tt.refuse {
				t.Fatalf("capacity = (%d, %v), want (%d, %v)", required, refuse, tt.required, tt.refuse)
			}
			err := refuseStreamedQ4KMetalCapacity(tt.total, tt.known)
			if (err != nil) != tt.wantError {
				t.Fatalf("error = %v, wantError %v", err, tt.wantError)
			}
			if err != nil {
				for _, want := range []string{"METAL_STREAM_Q4K_PEAK_TOO_BIG", "requires 47244640256 bytes"} {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error %q does not contain %q", err, want)
					}
				}
			}
		})
	}
}
