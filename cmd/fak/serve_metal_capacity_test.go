package main

import (
	"fmt"
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
	if streamedQ4KFreeCPUMetalCapacityBytes != 36<<30 {
		t.Fatalf("FREE_CPU capacity = %d, want exact #8964 36 GiB host envelope (not the old 18 GiB RSS)", streamedQ4KFreeCPUMetalCapacityBytes)
	}

	tests := []struct {
		name      string
		total     int64
		known     bool
		freeCPU   bool
		required  int64
		mode      string
		refuse    bool
		wantError bool
	}{
		{name: "FREE_CPU one byte below 36 GiB refuses", total: (36 << 30) - 1, known: true, freeCPU: true, required: 36 << 30, mode: streamedQ4KModeFreeCPU, refuse: true, wantError: true},
		{name: "FREE_CPU 36 GiB proceeds", total: 36 << 30, known: true, freeCPU: true, required: 36 << 30, mode: streamedQ4KModeFreeCPU},
		{name: "retained CPU 36 GiB refuses", total: 36 << 30, known: true, required: 44 << 30, mode: streamedQ4KModeRetainedCPU, refuse: true, wantError: true},
		{name: "retained CPU 44 GiB proceeds", total: 44 << 30, known: true, required: 44 << 30, mode: streamedQ4KModeRetainedCPU},
		{name: "unknown host memory preserves existing behavior", freeCPU: true, mode: streamedQ4KModeFreeCPU},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			required, refuse, mode := streamedQ4KMetalCapacity(tt.total, tt.known, tt.freeCPU)
			if required != tt.required || refuse != tt.refuse || mode != tt.mode {
				t.Fatalf("capacity = (%d, %v, %q), want (%d, %v, %q)", required, refuse, mode, tt.required, tt.refuse, tt.mode)
			}
			err := refuseStreamedQ4KMetalCapacity(tt.total, tt.known, tt.freeCPU)
			if (err != nil) != tt.wantError {
				t.Fatalf("error = %v, wantError %v", err, tt.wantError)
			}
			if err != nil {
				for _, want := range []string{"METAL_STREAM_Q4K_PEAK_TOO_BIG", "mode=" + tt.mode, fmt.Sprintf("requires %d bytes", tt.required)} {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error %q does not contain %q", err, want)
					}
				}
			}
		})
	}
}

func TestNonStreamingMetalCapacityIsUnchangedByFREECPUDeclaration(t *testing.T) {
	const gib = int64(1 << 30)
	t.Setenv("FAK_Q4K_FREE_CPU", "1")
	peak, refuse := metalGGUFPeakCapacity(true, 1592*gib/100, 36*gib, true)
	if peak != int64(float64(1592*gib/100)*metalGGUFObservedPeakMultiplier) || !refuse {
		t.Fatalf("non-streaming capacity = (%d, %v), want unchanged generic Metal refusal", peak, refuse)
	}
}
