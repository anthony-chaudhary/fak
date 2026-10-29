package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestRequestDevicePressureBudgetScalesWithLiveFree pins the source of the request-time KV
// capacity pre-check budget (the "budget=" field of the inkernel_chat pressure-trim line): it is
// ApplyByteHeadroom(free, inKernelRequestDeviceHeadroom) over the backend's LIVE device free
// reading, not a constant and not FAK_GPU_BUDGET_MB. The raider-fak witness budget=33813299
// (fak-private docs/tickets/raider-kv-pool-sizing-20261028) is exactly 0.85 x 39780352 bytes free
// (~37.9 MiB left after the resident Q4_K_M weights), so it can only move by freeing VRAM.
func TestRequestDevicePressureBudgetScalesWithLiveFree(t *testing.T) {
	plan := compute.MemoryPlan{{Class: compute.MemoryKVCache, Bytes: 450217984, Scope: compute.MemoryScopeDevice}}
	const total = int64(24463) << 20
	cases := []struct {
		free, wantBudget int64
	}{
		{39780352, 33813299},
		{2 * 39780352, 67626598},
		{1 << 30, ApplyByteHeadroom(1<<30, inKernelRequestDeviceHeadroom)},
	}
	prev := int64(-1)
	for _, c := range cases {
		p := &InKernelPlanner{backend: capacityProbeBackend{total: total, free: c.free, known: true}}
		fit, ok := p.requestDevicePressureFit(plan)
		if !ok {
			t.Fatalf("free=%d: pressure fit unavailable on a probing backend", c.free)
		}
		if fit.budget != c.wantBudget {
			t.Fatalf("free=%d: budget=%d, want %d", c.free, fit.budget, c.wantBudget)
		}
		if fit.budget <= prev {
			t.Fatalf("free=%d: budget %d does not grow with free (prev %d)", c.free, fit.budget, prev)
		}
		prev = fit.budget
		if fit.margin != fit.budget-450217984 {
			t.Fatalf("free=%d: margin=%d, want budget-want", c.free, fit.margin)
		}
	}
}
