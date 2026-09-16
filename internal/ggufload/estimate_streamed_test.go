package ggufload

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// estimate_streamed_test.go — fak#13121. The MoE offload arm charged every routed-expert payload
// HOST-resident, so a 183.25 GiB DeepSeek-V4.1 Flash Q2_K routed set refused against a 62 GiB Halo
// before reading a byte. The streamed policy charges only a bounded resident working set and faults
// the remaining expert strides from the staged shards on demand.

// streamedTestWeightSource mirrors TestEstimateCPUOffloadExpertsMemoryPlanSplitsDeviceAndHost's
// fixture, scaled so the routed-expert payload is unmistakably larger than any resident bound a
// 62 GiB host could take while the dense side stays small.
func streamedTestWeightSource(t *testing.T, routedExpertBytes uint64) *WeightSource {
	t.Helper()
	f := &File{
		Metadata: map[string]Value{
			"general.architecture": {Type: TypeString, Value: "glm-dsa"},
		},
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256}, Type: TensorF32},                            // device dense, 1024 B
			{Name: "blk.0.ffn_gate_inp.weight", Dims: []uint64{128}, Type: TensorF32},                    // router device, 512 B
			{Name: "blk.0.ffn_gate_shexp.weight", Dims: []uint64{512}, Type: TensorF32},                  // shared expert device, 2048 B
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{routedExpertBytes / 4}, Type: TensorF32}, // routed blob host
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// TestEstimateCPUOffloadExpertsStreamed witnesses the leaf's core charge claim: the streamed policy
// charges a BOUNDED resident set, not the routed payload, while the device dense side is byte-equal
// to the full-charge arm. The negative half is asserted beside it: a streamed plan whose resident
// bound is itself negative is refused by name, and the full-charge arm still refuses when the whole
// routed set is charged.
func TestEstimateCPUOffloadExpertsStreamed(t *testing.T) {
	const dense = int64(1024 + 512 + 2048)
	const routed = int64(32 << 20) // 32 MiB of routed experts, far above any small resident bound
	ws := streamedTestWeightSource(t, uint64(routed))

	full, err := ws.EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsMemoryPlan: %v", err)
	}
	if got := full.ByClass()[compute.MemoryOffload]; got != routed {
		t.Fatalf("full-charge host offload = %d, want the whole routed set %d", got, routed)
	}

	const resident = int64(4 << 20) // 4 MiB bounded working set
	plan, err := ws.EstimateCPUOffloadExpertsStreamedMemoryPlan(resident)
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsStreamedMemoryPlan: %v", err)
	}
	by := plan.ByClass()
	if got := by[compute.MemoryOffload]; got != resident {
		t.Fatalf("streamed host offload = %d, want the bounded resident set %d (not the routed payload %d)", got, resident, routed)
	}
	if got := plan.HostTotal(); got != resident {
		t.Fatalf("streamed HostTotal = %d, want %d", got, resident)
	}
	if got := plan.DeviceTotal(); got != dense {
		t.Fatalf("streamed DeviceTotal = %d, want the unchanged dense side %d", got, dense)
	}
	byDetail := memoryPlanBytesByDetail(plan)
	if got := byDetail["gguf-host-expert-offload-streamed"]; got != resident {
		t.Fatalf("streamed resident detail = %d, want %d; plan=%+v", got, resident, plan)
	}
	if got := byDetail["gguf-host-expert-offload"]; got != 0 {
		t.Fatalf("streamed plan still emits the full-charge host row (%d); the policy did not take", got)
	}

	// Stream-through: a zero resident bound charges NO host offload and keeps the dense side.
	through, err := ws.EstimateCPUOffloadExpertsStreamedMemoryPlan(0)
	if err != nil {
		t.Fatalf("stream-through plan: %v", err)
	}
	if got := through.HostTotal(); got != 0 {
		t.Fatalf("stream-through HostTotal = %d, want 0", got)
	}
	if got := through.DeviceTotal(); got != dense {
		t.Fatalf("stream-through DeviceTotal = %d, want %d", got, dense)
	}

	// Negative bound: refused by name, never silently clamped into an unrepresentable policy.
	_, err = ws.EstimateCPUOffloadExpertsStreamedMemoryPlan(-1)
	if err == nil || !strings.Contains(err.Error(), "negative") {
		t.Fatalf("negative resident budget error = %v, want a named refusal", err)
	}

	// The comparison that names the leaf: the host-scoped demand collapses from the full routed
	// payload to the bounded resident set. This mirrors the physical 183.25 GiB vs 62 GiB case the
	// issue frames, where the full charge refused and the streamed charge fits.
	if plan.HostTotal() >= full.HostTotal() {
		t.Fatalf("streamed host total %d did not fall below the full route-charged %d", plan.HostTotal(), full.HostTotal())
	}
	if plan.HostTotal() > resident {
		t.Fatalf("streamed host total %d exceeds the declared resident bound %d", plan.HostTotal(), resident)
	}
}
