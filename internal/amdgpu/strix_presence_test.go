package amdgpu

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestStrixPresenceCacheRoundtrip(t *testing.T) {
	defer func() {
		// restore
		_ = os.Remove(StrixPresenceFile)
	}()

	target := &StrixTarget{
		Mode:           "ssh",
		Host:           "test-strix",
		Reachable:      true,
		CPUModel:       "AMD Ryzen AI MAX+ 395",
		GPUName:        "AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
		TargetISA:      "gfx1151",
		ComputeUnits:   40,
		TotalRAMBytes:  68719476736,
		UMABufferBytes: 60129542144,
		DPMLevel:       "high",
		LockupTimeout:  -1,
		LatencyMS:      1.5,
		DiscoveredAt:   time.Now().UTC().Format(time.RFC3339),
	}

	savePresenceCache(target)
	loaded, ok := loadPresenceCache("test-strix")
	if !ok || loaded == nil {
		t.Fatal("expected loadPresenceCache to find saved cache")
	}

	if loaded.Host != "test-strix" {
		t.Errorf("loaded host = %q, want 'test-strix'", loaded.Host)
	}
	if loaded.ComputeUnits != 40 {
		t.Errorf("loaded CUs = %d, want 40", loaded.ComputeUnits)
	}
	if loaded.TargetISA != "gfx1151" {
		t.Errorf("loaded ISA = %q, want 'gfx1151'", loaded.TargetISA)
	}
}

func TestDiscoverStrixTargetSimulatedUnreachable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// An unreachable non-existent host should return reachable=false without crashing
	target, err := DiscoverStrixTarget(ctx, "nonexistent-strix-host-xyz-12345.invalid")
	if err == nil && target != nil && target.Reachable {
		t.Errorf("expected unreachable target for invalid host, got %v", target)
	}
}
