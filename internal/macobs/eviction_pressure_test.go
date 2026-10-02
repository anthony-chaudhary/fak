package macobs

import (
	"math"
	"testing"
)

func TestBuildEvictionPressure(t *testing.T) {
	const gib = uint64(1) << 30

	// Paired host fixture from #13537: kern.memorystatus_level=42 and
	// kern.memorystatus_vm_pressure_level=1 (normal). These helpers consume
	// the percentage, not the separate pressure enum: an input of 1 is 1%
	// available and must not be mistaken for the enum's normal state.
	tests := []struct {
		name          string
		recoveryCount int
		level         int
		wiredLimit    uint64
		resident      uint64
		reserve       uint64
		wantHeadroom  uint64
		wantFraction  float64
		wantAtRisk    bool
	}{
		{
			name:         "normal headroom is not at risk",
			level:        42,
			wiredLimit:   16 * gib,
			resident:     8 * gib,
			reserve:      gib,
			wantHeadroom: 8 * gib,
			wantFraction: 0.5,
			wantAtRisk:   false,
		},
		{
			name:         "headroom below reserve is at risk",
			level:        42,
			wiredLimit:   16 * gib,
			resident:     16*gib - gib/2,
			reserve:      gib,
			wantHeadroom: gib / 2,
			wantFraction: 0.03125,
			wantAtRisk:   true,
		},
		{
			name:          "one percent available is at risk even with ample headroom",
			recoveryCount: 42,
			level:         1,
			wiredLimit:    64 * gib,
			resident:      8 * gib,
			reserve:       gib,
			wantHeadroom:  56 * gib,
			wantFraction:  0.875,
			wantAtRisk:    true,
		},
		{
			name:         "two percent available is at risk",
			level:        2,
			wiredLimit:   16 * gib,
			resident:     4 * gib,
			reserve:      gib,
			wantHeadroom: 12 * gib,
			wantFraction: 0.75,
			wantAtRisk:   true,
		},
		{
			name:         "resident exceeds wired limit clamps headroom to zero without wraparound",
			level:        42,
			wiredLimit:   4 * gib,
			resident:     8 * gib,
			reserve:      gib,
			wantHeadroom: 0,
			wantFraction: 0,
			wantAtRisk:   true,
		},
		{
			name:         "resident exactly at wired limit yields zero headroom",
			level:        42,
			wiredLimit:   4 * gib,
			resident:     4 * gib,
			reserve:      gib,
			wantHeadroom: 0,
			wantFraction: 0,
			wantAtRisk:   true,
		},
		{
			name:         "zero wired limit has zero fraction and does not panic",
			level:        42,
			wiredLimit:   0,
			resident:     0,
			reserve:      gib,
			wantHeadroom: 0,
			wantFraction: 0,
			wantAtRisk:   false,
		},
		{
			name:         "zero wired limit with zero reserve is not at risk",
			level:        42,
			wiredLimit:   0,
			resident:     0,
			reserve:      0,
			wantHeadroom: 0,
			wantFraction: 0,
			wantAtRisk:   false,
		},
		{
			name:         "three percent available meets the eviction floor",
			level:        3,
			wiredLimit:   16 * gib,
			resident:     8 * gib,
			reserve:      gib,
			wantHeadroom: 8 * gib,
			wantFraction: 0.5,
			wantAtRisk:   false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := BuildEvictionPressure(tc.recoveryCount, tc.level, tc.wiredLimit, tc.resident, tc.reserve)
			if got.HeadroomBytes != tc.wantHeadroom {
				t.Errorf("HeadroomBytes = %d, want %d", got.HeadroomBytes, tc.wantHeadroom)
			}
			if math.Abs(got.HeadroomFraction-tc.wantFraction) > 1e-9 {
				t.Errorf("HeadroomFraction = %v, want %v", got.HeadroomFraction, tc.wantFraction)
			}
			if got.AtRisk != tc.wantAtRisk {
				t.Errorf("AtRisk = %v, want %v", got.AtRisk, tc.wantAtRisk)
			}
			if got.RecoveryCount != tc.recoveryCount {
				t.Errorf("RecoveryCount = %d, want %d", got.RecoveryCount, tc.recoveryCount)
			}
			if got.MemorystatusLevel != tc.level {
				t.Errorf("MemorystatusLevel = %d, want %d", got.MemorystatusLevel, tc.level)
			}
		})
	}
}

func TestIsEvictionRisk(t *testing.T) {
	tests := []struct {
		name       string
		level      int
		headroom   uint64
		reserve    uint64
		wantAtRisk bool
	}{
		{"ample headroom with 42 percent available", 42, 8 << 30, 1 << 30, false},
		{"headroom below reserve", 42, 1, 1 << 30, true},
		{"headroom equal to reserve is not at risk", 42, 1 << 30, 1 << 30, false},
		{"zero reserve defers to percentage only", 42, 0, 0, false},
		{"zero reserve with two percent available is at risk", 2, 1 << 30, 0, true},
		{"one percent available wins regardless of headroom", 1, 1 << 40, 1 << 30, true},
		{"three percent available meets the eviction floor", 3, 1 << 40, 0, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsEvictionRisk(tc.level, tc.headroom, tc.reserve); got != tc.wantAtRisk {
				t.Errorf("IsEvictionRisk(%d, %d, %d) = %v, want %v", tc.level, tc.headroom, tc.reserve, got, tc.wantAtRisk)
			}
		})
	}
}
