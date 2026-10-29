package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// deviceCohortPlanner builds a Qwen3.5-hybrid Q4_K planner bound to a registered
// device backend (the cpu-ref HAL behind a capacity-probe wrapper), the shape
// `fak serve --gguf <qwen> --backend cuda` constructs, with no physical GPU.
func deviceCohortPlanner(t *testing.T, batch bool) *InKernelPlanner {
	t.Helper()
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend unavailable")
	}
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	cfg.LayerTypes = []string{"linear_attention"}
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = 8
	cfg.LinearNumKeyHeads = 2
	cfg.LinearValueHeadDim = 8
	cfg.LinearNumValueHeads = 4
	m := model.NewSynthetic(cfg)
	m.Quantize()
	const free = int64(1) << 30
	backend := capacityProbeBackend{Backend: base, total: free, free: free, known: true}
	p := NewInKernelPlannerWithConfig(m, loadProbeTok(t), "synthetic-qwen-device-cohort", true, backend, false, InKernelPlannerConfig{BatchDecode: batch})
	p.maxNew = 6
	return p
}

// fak-test:runtime fast est=50ms lane=default
func TestInKernelQwenDeviceBackendSelectsDecodeCohort(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	p := deviceCohortPlanner(t, true)
	if !p.requiresDeviceSerialization() {
		t.Fatal("fixture must be a device-serialized planner")
	}
	if !p.coalescesQwenDecode() {
		t.Fatal("device-backed Qwen3.5 Q4_K planner with BatchDecode does not select the decode cohort")
	}
	if off := deviceCohortPlanner(t, false); off.coalescesQwenDecode() {
		t.Fatal("BatchDecode=false must keep the serialized device route")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelQwenDeviceCohortChargesInflightMemoryAdmission(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	p := deviceCohortPlanner(t, true)
	// An admitted cohort member already holds the whole reported device budget, so a
	// concurrent device turn must be refused by the coalesced in-flight reservation
	// before any session or forward exists, not oversubscribe the device.
	free := p.backend.(capacityProbeBackend).free
	p.deviceBudgetMu.Lock()
	p.deviceReservedBytes = free
	p.deviceBudgetMu.Unlock()
	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "alpha"}}, nil)
	var capacityErr *InKernelCapacityError
	if !errors.As(err, &capacityErr) || capacityErr.Scope != compute.MemoryScopeDevice || capacityErr.Site != inKernelCoalescedDeviceReservationSite {
		t.Fatalf("concurrent device turn over a held cohort reservation = %T (%v), want closed coalesced reservation refusal", err, err)
	}
	p.deviceBudgetMu.Lock()
	held := p.deviceReservedBytes
	p.deviceBudgetMu.Unlock()
	if held != free {
		t.Fatalf("refused admission mutated reservation to %d, want %d", held, free)
	}
}
