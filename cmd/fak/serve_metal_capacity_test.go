package main

import "testing"

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
