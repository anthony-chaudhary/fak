package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestVulkanTransferQueueSelectionRejectsUnavailableFamilies(t *testing.T) {
	tests := []struct {
		name     string
		compute  int
		families []VulkanQueueFamily
		reason   string
	}{
		{
			name:    "zero queue count",
			compute: 0,
			families: []VulkanQueueFamily{
				{Index: 0, QueueCount: 1, Compute: true, Transfer: true},
				{Index: 1, QueueCount: 0, Transfer: true},
			},
			reason: "no_distinct_transfer_family",
		},
		{
			name:    "negative family index",
			compute: 0,
			families: []VulkanQueueFamily{
				{Index: 0, QueueCount: 1, Compute: true, Transfer: true},
				{Index: -1, QueueCount: 1, Transfer: true},
			},
			reason: "invalid_queue_family",
		},
		{
			name:    "duplicate family index",
			compute: 0,
			families: []VulkanQueueFamily{
				{Index: 0, QueueCount: 1, Compute: true, Transfer: true},
				{Index: 0, QueueCount: 1, Transfer: true},
			},
			reason: "duplicate_queue_family",
		},
		{
			name:    "negative compute family",
			compute: -1,
			families: []VulkanQueueFamily{
				{Index: 1, QueueCount: 1, Transfer: true},
			},
			reason: "invalid_compute_family",
		},
		{
			name:    "dedicated candidate is compute family",
			compute: 0,
			families: []VulkanQueueFamily{
				{Index: 0, QueueCount: 1, Compute: true, Transfer: true},
			},
			reason: "no_distinct_transfer_family",
		},
		{
			name:    "compute family absent",
			compute: 99,
			families: []VulkanQueueFamily{
				{Index: 1, QueueCount: 1, Transfer: true},
			},
			reason: "invalid_compute_family",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan := SelectTransferQueueFamily("gfx1102", tt.compute, tt.families)
			if plan.Selected || plan.TransferFamily != -1 || plan.FallbackReason != tt.reason {
				t.Fatalf("plan = %+v, want unselected family -1 reason %q", plan, tt.reason)
			}
		})
	}

	selected := SelectTransferQueueFamily("gfx1102", 0, []VulkanQueueFamily{
		{Index: 0, QueueCount: 1, Compute: true, Transfer: true},
		{Index: 1, QueueCount: 1, Transfer: true},
	})
	if !selected.Selected || !selected.Dedicated || selected.TransferFamily != 1 || selected.FallbackReason != "" {
		t.Fatalf("available dedicated selection = %+v", selected)
	}

	unavailableCompute := SelectTransferQueueFamily("gfx1102", 0, []VulkanQueueFamily{
		{Index: 0, QueueCount: 0, Compute: true, Transfer: true},
		{Index: 1, QueueCount: -1, Transfer: true},
	})
	if unavailableCompute.Selected || unavailableCompute.FallbackReason != "invalid_compute_family" {
		t.Fatalf("unavailable compute-family selection = %+v", unavailableCompute)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestVulkanTransferQueueBindingRejectsMalformedPlan(t *testing.T) {
	pool := newVulkanTransferQueueTestPool(t)
	malformed := VulkanTransferQueuePlan{
		GFX:            "gfx1102",
		ComputeFamily:  2,
		TransferFamily: 3,
		Selected:       true,
		Dedicated:      true,
	}

	got := pool.BindTransferQueue(malformed)
	if got.Selected || got.TransferFamily != -1 || got.FallbackReason != "invalid_transfer_queue_plan" {
		t.Fatalf("BindTransferQueue(malformed) = %+v, want closed invalid-plan fallback", got)
	}
	if pool.TransferOverlapped() {
		t.Fatal("fabricated selected plan armed transfer overlap")
	}
	if _, ok := pool.TransferQueuePlan(); ok {
		t.Fatal("malformed plan became the active binding")
	}

	mutated := validVulkanTransferQueuePlan(2, 3)
	mutated.TransferFamily = 4
	got = pool.BindTransferQueue(mutated)
	if got.Selected || got.TransferFamily != -1 || got.FallbackReason != "invalid_transfer_queue_plan" {
		t.Fatalf("BindTransferQueue(mutated selection) = %+v, want closed invalid-plan fallback", got)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestVulkanStagingPoolTransferBindingEpochs(t *testing.T) {
	pool := newVulkanTransferQueueTestPool(t)
	planA := validVulkanTransferQueuePlan(0, 1)
	if got := pool.BindTransferQueue(planA); !got.Selected {
		t.Fatalf("BindTransferQueue(planA) = %+v, want selected", got)
	}

	completeVulkanTransferQueueCycle(t, pool, 7)
	if got := pool.BindTransferQueue(planA); !got.Selected {
		t.Fatalf("same-plan rebind = %+v, want selected", got)
	}
	slot, err := pool.AcquireSlot(1, 64)
	if err != nil {
		t.Fatalf("AcquireSlot after same-plan rebind: %v", err)
	}
	if err := pool.MarkTransferComplete(slot.ID, 1); !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("same-plan timeline reset error = %v, want ErrVulkanInvalidGeometry", err)
	}
	if err := pool.MarkTransferComplete(slot.ID, 8); err != nil {
		t.Fatalf("MarkTransferComplete(8): %v", err)
	}
	releaseVulkanTransferQueueSlot(t, pool, slot.ID)

	planB := validVulkanTransferQueuePlan(0, 2)
	if got := pool.BindTransferQueue(planB); !got.Selected {
		t.Fatalf("idle plan change = %+v, want selected", got)
	}
	completeVulkanTransferQueueCycle(t, pool, 1)

	unbound := pool.BindTransferQueue(VulkanTransferQueuePlan{GFX: "gfx1102", ComputeFamily: 0, TransferFamily: -1})
	if unbound.Selected || unbound.FallbackReason != "unified_queue" {
		t.Fatalf("idle unbind = %+v, want unified-queue fallback", unbound)
	}
	if _, ok := pool.TransferQueuePlan(); ok {
		t.Fatal("idle unbind retained an active plan")
	}
	if got := pool.BindTransferQueue(planA); !got.Selected {
		t.Fatalf("bind after idle unbind = %+v, want selected", got)
	}
	completeVulkanTransferQueueCycle(t, pool, 1)
}

// fak-test:runtime fast est=1s lane=default
func TestVulkanStagingPoolRejectsBusyTransferRebind(t *testing.T) {
	pool := newVulkanTransferQueueTestPool(t)
	planA := validVulkanTransferQueuePlan(0, 1)
	pool.BindTransferQueue(planA)
	slot, err := pool.AcquireSlot(0, 64)
	if err != nil {
		t.Fatalf("AcquireSlot: %v", err)
	}

	planB := validVulkanTransferQueuePlan(0, 2)
	got := pool.BindTransferQueue(planB)
	if !got.Selected || got.TransferFamily != planA.TransferFamily || got.FallbackReason != "transfer_queue_rebind_busy" {
		t.Fatalf("busy rebind = %+v, want retained planA with closed busy fallback", got)
	}
	bound, ok := pool.TransferQueuePlan()
	if !ok || bound != planA {
		t.Fatalf("binding after refused rebind = (%+v, %v), want planA unchanged", bound, ok)
	}
	unbound := pool.BindTransferQueue(VulkanTransferQueuePlan{GFX: "gfx1102", ComputeFamily: 0, TransferFamily: -1})
	if !unbound.Selected || unbound.TransferFamily != planA.TransferFamily || unbound.FallbackReason != "transfer_queue_rebind_busy" {
		t.Fatalf("busy unbind = %+v, want retained planA with closed busy fallback", unbound)
	}
	if err := pool.MarkTransferComplete(slot.ID, 1); err != nil {
		t.Fatalf("MarkTransferComplete: %v", err)
	}
	releaseVulkanTransferQueueSlot(t, pool, slot.ID)
}

func newVulkanTransferQueueTestPool(t *testing.T) *VulkanStagingPool {
	t.Helper()
	pool, err := NewVulkanStagingPool(2, 1024, 64)
	if err != nil {
		t.Fatalf("NewVulkanStagingPool: %v", err)
	}
	return pool
}

func validVulkanTransferQueuePlan(computeFamily, transferFamily int) VulkanTransferQueuePlan {
	return SelectTransferQueueFamily("gfx1102", computeFamily, []VulkanQueueFamily{
		{Index: computeFamily, QueueCount: 1, Compute: true, Transfer: true},
		{Index: transferFamily, QueueCount: 1, Transfer: true},
	})
}

func completeVulkanTransferQueueCycle(t *testing.T, pool *VulkanStagingPool, timeline uint64) *VulkanStagingSlot {
	t.Helper()
	slot, err := pool.AcquireSlot(int(timeline), 64)
	if err != nil {
		t.Fatalf("AcquireSlot: %v", err)
	}
	if err := pool.MarkTransferComplete(slot.ID, timeline); err != nil {
		t.Fatalf("MarkTransferComplete(%d): %v", timeline, err)
	}
	releaseVulkanTransferQueueSlot(t, pool, slot.ID)
	return slot
}

func releaseVulkanTransferQueueSlot(t *testing.T, pool *VulkanStagingPool, slotID int) {
	t.Helper()
	if err := pool.AcquireForCompute(slotID); err != nil {
		t.Fatalf("AcquireForCompute(%d): %v", slotID, err)
	}
	if err := pool.ReleaseCompute(slotID); err != nil {
		t.Fatalf("ReleaseCompute(%d): %v", slotID, err)
	}
}
