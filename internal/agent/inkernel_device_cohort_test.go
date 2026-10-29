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

// cohortCapabilityBackend advertises only admission capability. Tests refuse
// before a forward; this fixture must never be mistaken for a GPU witness.
type cohortCapabilityBackend struct {
	capacityProbeBackend
	path string
}

func (b *cohortCapabilityBackend) Qwen35IndependentDecodeBatchPath() string { return b.path }

func (*cohortCapabilityBackend) Qwen35IndependentDecodeBatch(compute.Qwen35IndependentDecodeBatchRequest) (compute.Qwen35IndependentDecodeBatchResult, error) {
	panic("admission-only fixture must not execute a batch")
}

// fak-test:runtime fast est=50ms lane=default
func TestInKernelQwenDeviceCohortRequiresBatchCapability(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	p := deviceCohortPlanner(t, true)
	if !p.requiresDeviceSerialization() {
		t.Fatal("fixture must be a device-serialized planner")
	}
	base := p.backend.(capacityProbeBackend)
	valid := &cohortCapabilityBackend{capacityProbeBackend: base, path: compute.Qwen35IndependentDecodeBatchPath}
	var typedNil *cohortCapabilityBackend
	for _, tc := range []struct {
		name    string
		backend compute.Backend
		metal   bool
		batch   bool
		want    bool
	}{
		{"unsupported", base, false, true, false},
		{"unsupported-with-metal-flag", base, true, true, false},
		{"wrong-marker", &cohortCapabilityBackend{capacityProbeBackend: base, path: "wrong"}, false, true, false},
		{"typed-nil", typedNil, false, true, false},
		{"exact-capability", valid, false, true, true},
		{"disabled", valid, false, false, false},
		{"metal-host", nil, true, true, true},
		{"cpu-host", nil, false, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p.backend, p.metal, p.batchDecode = tc.backend, tc.metal, tc.batch
			if got := p.coalescesQwenDecode(); got != tc.want {
				t.Fatalf("cohort admission = %t, want %t", got, tc.want)
			}
		})
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
	p.backend = &cohortCapabilityBackend{capacityProbeBackend: p.backend.(capacityProbeBackend), path: compute.Qwen35IndependentDecodeBatchPath}
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

// fak-test:runtime fast est=100ms lane=default
func TestInKernelQwenDeviceWithoutBatchUsesSerialAdmission(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	p := deviceCohortPlanner(t, true)
	backend := p.backend.(capacityProbeBackend)
	backend.free = 1
	p.backend = backend
	// Distinguish the production Complete routes: serial admission prices the
	// request against device free memory; cohort admission additionally charges
	// this held reservation and reports coalesced-device-reservation instead.
	p.deviceReservedBytes = 1
	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "alpha"}}, nil)
	var capacityErr *InKernelCapacityError
	if !errors.As(err, &capacityErr) || capacityErr.Scope != compute.MemoryScopeDevice || capacityErr.Site != "capacity-precheck" {
		t.Fatalf("unsupported backend Complete = %T (%v), want serial capacity-precheck", err, err)
	}
	if p.deviceReservedBytes != 1 {
		t.Fatal("serial admission changed a cohort reservation")
	}
}
