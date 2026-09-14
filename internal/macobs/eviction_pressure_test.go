package macobs

import (
	"math"
	"testing"
)

func TestBuildEvictionPressure(t *testing.T) {
	const gib = uint64(1) << 30

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
			level:        MemorystatusLevelNormal,
			wiredLimit:   16 * gib,
			resident:     8 * gib,
			reserve:      gib,
			wantHeadroom: 8 * gib,
			wantFraction: 0.5,
			wantAtRisk:   false,
		},
		{
			name:         "headroom below reserve is at risk",
			level:        MemorystatusLevelNormal,
			wiredLimit:   16 * gib,
			resident:     16*gib - gib/2,
			reserve:      gib,
			wantHeadroom: gib / 2,
			wantFraction: 0.03125,
			wantAtRisk:   true,
		},
		{
			name:          "critical level is at risk even with ample headroom",
			recoveryCount: 42,
			level:         MemorystatusLevelCritical,
			wiredLimit:    64 * gib,
			resident:      8 * gib,
			reserve:       gib,
			wantHeadroom:  56 * gib,
			wantFraction:  0.875,
			wantAtRisk:    true,
		},
		{
			name:         "warning level is at risk",
			level:        MemorystatusLevelWarning,
			wiredLimit:   16 * gib,
			resident:     4 * gib,
			reserve:      gib,
			wantHeadroom: 12 * gib,
			wantFraction: 0.75,
			wantAtRisk:   true,
		},
		{
			name:         "resident exceeds wired limit clamps headroom to zero without wraparound",
			level:        MemorystatusLevelNormal,
			wiredLimit:   4 * gib,
			resident:     8 * gib,
			reserve:      gib,
			wantHeadroom: 0,
			wantFraction: 0,
			wantAtRisk:   true,
		},
		{
			name:         "resident exactly at wired limit yields zero headroom",
			level:        MemorystatusLevelNormal,
			wiredLimit:   4 * gib,
			resident:     4 * gib,
			reserve:      gib,
			wantHeadroom: 0,
			wantFraction: 0,
			wantAtRisk:   true,
		},
		{
			name:         "zero wired limit has zero fraction and does not panic",
			level:        MemorystatusLevelNormal,
			wiredLimit:   0,
			resident:     0,
			reserve:      gib,
			wantHeadroom: 0,
			wantFraction: 0,
			wantAtRisk:   false,
		},
		{
			name:         "zero wired limit with zero reserve is not at risk",
			level:        MemorystatusLevelNormal,
			wiredLimit:   0,
			resident:     0,
			reserve:      0,
			wantHeadroom: 0,
			wantFraction: 0,
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
		{"ample headroom, normal", MemorystatusLevelNormal, 8 << 30, 1 << 30, false},
		{"headroom below reserve", MemorystatusLevelNormal, 1, 1 << 30, true},
		{"headroom equal to reserve is not at risk", MemorystatusLevelNormal, 1 << 30, 1 << 30, false},
		{"zero reserve defers to level only", MemorystatusLevelNormal, 0, 0, false},
		{"zero reserve with warning level is at risk", MemorystatusLevelWarning, 1 << 30, 0, true},
		{"critical wins regardless of headroom", MemorystatusLevelCritical, 1 << 40, 1 << 30, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsEvictionRisk(tc.level, tc.headroom, tc.reserve); got != tc.wantAtRisk {
				t.Errorf("IsEvictionRisk(%d, %d, %d) = %v, want %v", tc.level, tc.headroom, tc.reserve, got, tc.wantAtRisk)
			}
		})
	}
}
