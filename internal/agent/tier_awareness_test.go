package agent

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

func aegisInklingSnapshot(masterAvail, workerAvail, microAvail int) fakclient.CapacityTiersSnapshot {
	return fakclient.CapacityTiersSnapshot{
		Enabled:     true,
		TotalTokens: 393216, // 1*128k + 4*32k + 32*4k
		FreeTokens:  masterAvail*131072 + workerAvail*32768 + microAvail*4096,
		Tiers: []fakclient.TierStatus{
			{
				Name:         "master",
				Level:        1,
				MaxContext:   131072,
				MinSlots:     1,
				ActiveSlots:  1 - masterAvail,
				AvailableNow: masterAvail,
				Preemptible:  false,
			},
			{
				Name:         "worker",
				Level:        2,
				MaxContext:   32768,
				MinSlots:     4,
				ActiveSlots:  4 - workerAvail,
				AvailableNow: workerAvail,
				Preemptible:  false,
			},
			{
				Name:         "micro",
				Level:        3,
				MaxContext:   4096,
				MinSlots:     32,
				ActiveSlots:  32 - microAvail,
				AvailableNow: microAvail,
				Preemptible:  true,
			},
		},
		RecommendedFanout: fakclient.RecommendedFanout{
			MasterMaxCtx: 131072,
			WorkerSlots:  workerAvail,
			WorkerMaxCtx: 32768,
			MicroSlots:   microAvail,
			MicroMaxCtx:  4096,
		},
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestTierTopologyPlanningAegisInkling(t *testing.T) {
	snap := aegisInklingSnapshot(1, 4, 32)

	masterPlan := PlanAgentTopology(snap, RoleMaster)
	if masterPlan.Degraded {
		t.Fatalf("expected healthy master plan, got Degraded=true (%s)", masterPlan.Advice)
	}
	if masterPlan.AssignedTier != "master" || masterPlan.ContextCap != 131072 {
		t.Fatalf("unexpected master assignment: %+v", masterPlan)
	}
	if masterPlan.CompactionWatermark != (131072*80)/100 {
		t.Fatalf("CompactionWatermark = %d, want %d", masterPlan.CompactionWatermark, (131072*80)/100)
	}
	if masterPlan.MaxParallelWorkers != 4 || masterPlan.WorkerMaxCtx != 32768 {
		t.Fatalf("unexpected worker fanout: %+v", masterPlan)
	}
	if masterPlan.MaxParallelMicros != 32 || masterPlan.MicroMaxCtx != 4096 {
		t.Fatalf("unexpected micro fanout: %+v", masterPlan)
	}

	workerPlan := PlanAgentTopology(snap, RoleWorker)
	if workerPlan.Degraded || workerPlan.AssignedTier != "worker" || workerPlan.ContextCap != 32768 {
		t.Fatalf("unexpected worker plan: %+v", workerPlan)
	}
	if workerPlan.CompactionWatermark != (32768*80)/100 {
		t.Fatalf("worker CompactionWatermark = %d, want %d", workerPlan.CompactionWatermark, (32768*80)/100)
	}

	microPlan := PlanAgentTopology(snap, RoleMicro)
	if microPlan.Degraded || microPlan.AssignedTier != "micro" || microPlan.ContextCap != 4096 {
		t.Fatalf("unexpected micro plan: %+v", microPlan)
	}
	if microPlan.CompactionWatermark != (4096*80)/100 {
		t.Fatalf("micro CompactionWatermark = %d, want %d", microPlan.CompactionWatermark, (4096*80)/100)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestTierDegradedMasterFallback(t *testing.T) {
	snap := aegisInklingSnapshot(0, 4, 32)
	plan := PlanAgentTopology(snap, RoleMaster)
	if !plan.Degraded {
		t.Fatalf("expected Degraded=true when master has 0 available slots")
	}
	if !strings.Contains(plan.Advice, "L2") && !strings.Contains(plan.Advice, "worker") {
		t.Fatalf("expected L2 worker fallback advice, got: %q", plan.Advice)
	}
	if plan.MaxParallelWorkers != 4 || plan.MaxParallelMicros != 32 {
		t.Fatalf("expected remaining worker/micro slots preserved: %+v", plan)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestTierSelectForSubtask(t *testing.T) {
	snap := aegisInklingSnapshot(1, 4, 32)

	tier, capTok, _ := SelectTierForSubtask(snap, 2000, RoleMaster)
	if tier != "micro" || capTok != 4096 {
		t.Fatalf("2000 tok -> (%q, %d), want (micro, 4096)", tier, capTok)
	}

	tier, capTok, _ = SelectTierForSubtask(snap, 16000, RoleMaster)
	if tier != "worker" || capTok != 32768 {
		t.Fatalf("16000 tok -> (%q, %d), want (worker, 32768)", tier, capTok)
	}

	tier, capTok, _ = SelectTierForSubtask(snap, 64000, RoleMaster)
	if tier != "master" || capTok != 131072 {
		t.Fatalf("64000 tok -> (%q, %d), want (master, 131072)", tier, capTok)
	}

	// When micro has 0 slots, 2000 tok escalates to worker.
	busyMicro := aegisInklingSnapshot(1, 4, 0)
	tier, capTok, advice := SelectTierForSubtask(busyMicro, 2000, RoleMaster)
	if tier != "worker" || capTok != 32768 {
		t.Fatalf("busy micro 2000 tok -> (%q, %d), want (worker, 32768)", tier, capTok)
	}
	if !strings.Contains(advice, "micro") {
		t.Fatalf("expected escalation advice mentioning micro, got %q", advice)
	}

	// When master has 0 slots and task exceeds worker, advise splitting to worker.
	busyMaster := aegisInklingSnapshot(0, 4, 32)
	tier, capTok, advice = SelectTierForSubtask(busyMaster, 64000, RoleMaster)
	if tier != "worker" || capTok != 32768 {
		t.Fatalf("busy master 64000 tok -> (%q, %d), want (worker, 32768)", tier, capTok)
	}
	if !strings.Contains(advice, "split") {
		t.Fatalf("expected split advice, got %q", advice)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestTierFormatAwarenessPrompt(t *testing.T) {
	snap := aegisInklingSnapshot(0, 4, 32)
	prompt := FormatTierAwarenessPrompt(snap, RoleMaster)
	lines := strings.Split(strings.TrimSpace(prompt), "\n")
	if len(lines) > 8 {
		t.Fatalf("prompt has %d lines, want <= 8:\n%s", len(lines), prompt)
	}
	for _, want := range []string{"131072", "32768", "4096", "4 worker", "32 micro", "DEGRADED"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, prompt)
		}
	}
}
