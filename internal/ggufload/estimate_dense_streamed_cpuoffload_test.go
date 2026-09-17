package ggufload

import (
	"strings"
	"testing"
)

// estimate_dense_streamed_cpuoffload_test.go - fak#13209. The device --cpu-offload-experts arm's
// sizing path is serveGGUFCPUOffloadPathMemoryPlan -> EstimateCPUOffloadExpertsExpertParallelMemoryPlan,
// whose estimator only ever emitted "gguf-device-dense-load" (device) and "gguf-host-expert-offload"
// (host). It could NEVER emit the bounded "gguf-host-dense-streamed" row the serve staging guard
// (cmd/fak serveDeviceDenseStagingBoundCharge) looks for, so the bounded transit charge was DEAD on
// the real arm and a physical strix3 serve still refused with FitTooBig on the full 63.22 GiB dense
// transit.
//
// EstimateCPUOffloadExpertsBoundedDenseMemoryPlan threads the bounded streamed-dense fold into the
// CPU-offload estimator: the eligible dense bounded k-quant side is moved OUT of the device
// "gguf-device-dense-load" charge and folded into ONE "gguf-host-dense-streamed" host row charged
// min(eligible, bound). The fixture is the SAME qwen3moe MoE header estimate_dense_streamed_moe_test.go
// uses: one ELIGIBLE dense Q4_K matmul (blk.0.ffn_down.weight), one non-Q4_K F32 device tensor
// (token_embd.weight), and the batched routed-expert blob (blk.0.ffn_gate_exps.weight, host-scoped,
// no canonical mapping => non-eligible). It lives in package ggufload because the fixture builds a
// WeightSource from the unexported File/TensorInfo.

// TestEstimateCPUOffloadExpertsBoundedDenseWorkingSet is the RED->GREEN witness: with a declared
// bound the eligible dense k-quant side is charged min(eligible, bound) as a host row and REMOVED
// from the device dense charge; every non-eligible byte stays fully charged (fail-closed).
func TestEstimateCPUOffloadExpertsBoundedDenseWorkingSet(t *testing.T) {
	const bound = int64(8 * 1024)
	ws := moeStreamedTestWeightSource(t, moeDenseEligibleBytes)

	// The historical full-charge plan is the baseline: it charges the eligible dense side to device
	// and has NO bounded row.
	full, err := ws.EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsMemoryPlan: %v", err)
	}
	if got := full.DeviceTotal(); got <= 0 {
		t.Fatalf("baseline DeviceTotal = %d, want a non-zero device dense charge", got)
	}
	byFull := memoryPlanBytesByDetail(full)
	if _, ok := byFull["gguf-host-dense-streamed"]; ok {
		t.Fatalf("baseline plan already carries a bounded dense row; the RED premise is broken")
	}

	plan, err := ws.EstimateCPUOffloadExpertsBoundedDenseMemoryPlan(1, bound)
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsBoundedDenseMemoryPlan: %v", err)
	}
	byDetail := memoryPlanBytesByDetail(plan)
	if got := byDetail["gguf-host-dense-streamed"]; got != bound {
		t.Fatalf("gguf-host-dense-streamed = %d, want the declared bound %d; plan=%+v", got, bound, plan)
	}
	// The eligible dense side moved OUT of the device dense charge: DeviceTotal falls by exactly
	// the eligible bytes.
	if got, want := plan.DeviceTotal(), full.DeviceTotal()-moeDenseEligibleBytes; got != want {
		t.Fatalf("DeviceTotal = %d, want baseline %d less the eligible dense side %d = %d",
			got, full.DeviceTotal(), moeDenseEligibleBytes, want)
	}
	// The bounded dense row is a HOST row, so it is ADDED to the pre-existing host routed-expert
	// row (HostTotal = routedHost + bound). The routed host bytes are unchanged by the dense policy.
	if got, want := plan.HostTotal(), full.HostTotal()+bound; got != want {
		t.Fatalf("HostTotal = %d, want the routed host row %d plus the bounded dense row %d = %d", got, full.HostTotal(), bound, want)
	}

	// Stream-through (bound 0): no dense host residency is charged, and the device remainder is the
	// same as the bounded case.
	through, err := ws.EstimateCPUOffloadExpertsBoundedDenseMemoryPlan(1, 0)
	if err != nil {
		t.Fatalf("stream-through bounded dense plan: %v", err)
	}
	// Stream-through charges NO dense host residency: the host total is the unchanged routed
	// host row, no bounded dense row appears, and the device remainder matches the bounded case.
	if got, want := through.HostTotal(), full.HostTotal(); got != want {
		t.Fatalf("stream-through HostTotal = %d, want the unchanged routed host row %d", got, want)
	}
	if got := through.DeviceTotal(); got != plan.DeviceTotal() {
		t.Fatalf("stream-through DeviceTotal = %d, want the non-eligible remainder %d", got, plan.DeviceTotal())
	}
	if _, ok := memoryPlanBytesByDetail(through)["gguf-host-dense-streamed"]; ok {
		t.Fatalf("stream-through plan emitted a bounded dense row; a zero bound must charge nothing")
	}

	// A bound above the eligible side is capped by the actual eligible bytes, never inflated.
	capped, err := ws.EstimateCPUOffloadExpertsBoundedDenseMemoryPlan(1, moeDenseEligibleBytes*4)
	if err != nil {
		t.Fatalf("over-bound bounded dense plan: %v", err)
	}
	if got := memoryPlanBytesByDetail(capped)["gguf-host-dense-streamed"]; got != moeDenseEligibleBytes {
		t.Fatalf("capped bounded dense row = %d, want the eligible dense side %d", got, moeDenseEligibleBytes)
	}
}

// TestEstimateCPUOffloadExpertsBoundedDenseRefusesNegative fails-CLOSED on a negative bound: an
// unrepresentable policy is refused by name, never silently clamped.
func TestEstimateCPUOffloadExpertsBoundedDenseRefusesNegative(t *testing.T) {
	ws := moeStreamedTestWeightSource(t, moeDenseEligibleBytes)
	_, err := ws.EstimateCPUOffloadExpertsBoundedDenseMemoryPlan(1, -1)
	if err == nil {
		t.Fatal("negative bounded dense working set: want a named refusal, got nil")
	}
	if !strings.Contains(err.Error(), "gguf: streamed-dense host working set -1 is negative") {
		t.Fatalf("negative-bound refusal = %q, want the named message", err.Error())
	}
}

// TestEstimateCPUOffloadExpertsBoundedDenseUnboundedCallersByteIdentical pins the preserve: the
// historical full-charge plan (denseResident = -1) is byte-for-byte unchanged, so every existing
// caller of EstimateCPUOffloadExpertsMemoryPlan and the streamed/ep siblings is unaffected.
func TestEstimateCPUOffloadExpertsBoundedDenseUnboundedCallersByteIdentical(t *testing.T) {
	ws := moeStreamedTestWeightSource(t, moeDenseEligibleBytes)
	full, err := ws.EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsMemoryPlan: %v", err)
	}
	ep, err := ws.EstimateCPUOffloadExpertsExpertParallelMemoryPlan(1)
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsExpertParallelMemoryPlan: %v", err)
	}
	if len(full) != len(ep) {
		t.Fatalf("full plan has %d rows, EP plan has %d; ranks<=1 must be byte-identical", len(full), len(ep))
	}
	for i := range full {
		if full[i] != ep[i] {
			t.Fatalf("row %d: full=%+v ep=%+v; ranks<=1 must be byte-identical", i, full[i], ep[i])
		}
	}
	for _, d := range full {
		if d.Detail == "gguf-host-dense-streamed" {
			t.Fatalf("unbounded plan emitted a bounded dense row: %+v", d)
		}
	}
}
