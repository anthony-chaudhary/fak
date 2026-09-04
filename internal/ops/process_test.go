package ops

import (
	"context"
	"testing"
)

func TestProcessManagerSweep(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxThreads = 100000 // high enough not to trigger on host
	cfg.OrphanReapEnabled = true

	reaped := make(map[int]bool)
	pm := NewProcessManager(cfg)
	pm.Killer = func(pid int) (bool, string) {
		reaped[pid] = true
		return true, "killed"
	}

	res, err := pm.SweepProcessRunaways(context.Background(), true)
	if err != nil {
		t.Fatalf("SweepProcessRunaways: %v", err)
	}
	if len(reaped) != 0 {
		t.Errorf("expected 0 reaped in dry run, got %d", len(reaped))
	}
	_ = res
}
