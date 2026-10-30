//go:build vulkan && (windows || linux) && cgo

// Run on a Vulkan host with libfakvulkan and the SPIR-V bundle built:
// go test -tags vulkan -run 'VulkanV41RouterGate' ./internal/compute/

package compute

import (
	"errors"
	"math"
	"os"
	"strconv"
	"testing"
	"time"
)

func vkRouterGate(t *testing.T) *vulkanBackend {
	t.Helper()
	v := vk(t)
	if !v.V41RouterTopKAvailable() {
		t.Fatal("v41_router_topk.spv is absent from the runtime SPIR-V directory or failed to build")
	}
	if !v.V41RouterGateTopKAvailable() {
		t.Fatalf("fused router gate unavailable: module=%d verdict=%d", vulkanV41RouterGateModule(), vulkanV41RouterGateVerdict())
	}
	return v
}

type routerGateRun struct {
	picks  V41RouterPicks
	logits []float32
	err    error
}

// routerGateReference is the production split: MatMul's one-token F32 path
// followed by the router top-k on the resident logits.
func routerGateReference(t *testing.T, v *vulkanBackend, w, x, bias Tensor, experts, topK int, full, hasBias bool) routerGateRun {
	t.Helper()
	y := v.MatMul(w, x)
	defer v.Free(y)
	logits := v.Read(y)
	view := y
	view.Shape = []int{1, experts}
	picks, err := v.V41RouterTopK(view, bias, 1, experts, topK, 1.5, full, hasBias)
	return routerGateRun{picks: picks, logits: logits, err: err}
}

func routerGateFused(v *vulkanBackend, w, x, bias Tensor, experts, hidden, topK int, full, hasBias bool) routerGateRun {
	picks, logits, err := v.v41RouterGateTopK(w, x, bias, experts, hidden, topK, 1.5, full, hasBias, true)
	return routerGateRun{picks: picks, logits: logits, err: err}
}

func requireRouterGateEqual(t *testing.T, label string, ref, got routerGateRun) {
	t.Helper()
	var refFault, gotFault *V41RouterTopKFault
	refFaulted, gotFaulted := errors.As(ref.err, &refFault), errors.As(got.err, &gotFault)
	if refFaulted || gotFaulted {
		if !refFaulted || !gotFaulted || *refFault != *gotFault {
			t.Fatalf("%s: fault reference=%v fused=%v", label, ref.err, got.err)
		}
		return
	}
	if ref.err != nil || got.err != nil {
		t.Fatalf("%s: reference err=%v fused err=%v", label, ref.err, got.err)
	}
	if len(got.logits) != len(ref.logits) {
		t.Fatalf("%s: %d fused logits, want %d", label, len(got.logits), len(ref.logits))
	}
	for i := range ref.logits {
		if math.Float32bits(got.logits[i]) != math.Float32bits(ref.logits[i]) {
			t.Fatalf("%s: logit %d fused %#08x != matmul %#08x", label, i, math.Float32bits(got.logits[i]), math.Float32bits(ref.logits[i]))
		}
	}
	if len(got.picks.Experts) != len(ref.picks.Experts) || len(got.picks.Weights) != len(ref.picks.Weights) {
		t.Fatalf("%s: pick extent fused=%d/%d reference=%d/%d", label, len(got.picks.Experts), len(got.picks.Weights), len(ref.picks.Experts), len(ref.picks.Weights))
	}
	for slot := range ref.picks.Experts {
		if got.picks.Experts[slot] != ref.picks.Experts[slot] || math.Float32bits(got.picks.Weights[slot]) != math.Float32bits(ref.picks.Weights[slot]) {
			t.Fatalf("%s: slot %d fused (%d, %#08x) != reference (%d, %#08x)", label, slot,
				got.picks.Experts[slot], math.Float32bits(got.picks.Weights[slot]), ref.picks.Experts[slot], math.Float32bits(ref.picks.Weights[slot]))
		}
	}
}

func routerGateUpload(v *vulkanBackend, shape []int, values []float32, class MemoryClass) Tensor {
	return v.UploadClass(NewF32(Default(), shape, values), F32, class, "router gate witness")
}

// Full routing publishes softplus as F32 before sqrt: softplus(-110) rounds to
// +0, so all selected weights must be +0. Reduced routing retains the tiny
// positive score, and one selected score normalizes to the exact route scale.
// Check each path against those host-contract results, not only against its
// twin shader: the same missing boundary in both modules would pass equality.
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no witness.
// Runtime estimate only; unmeasured.
// fak-test:runtime fast est=1s lane=default
func TestVulkanV41RouterGateSoftplusPublication(t *testing.T) {
	v := vkRouterGate(t)
	const experts, hidden = 384, 64
	const logit float32 = -110
	w, x := make([]float32, experts*hidden), make([]float32, hidden)
	x[0] = 1
	for expert := 0; expert < experts; expert++ {
		w[expert*hidden] = logit
	}
	wt := v.Upload(NewF32(Default(), []int{experts, hidden}, w), F32)
	defer v.Free(wt)
	xt := routerGateUpload(v, []int{hidden}, x, MemoryActivation)
	defer v.Free(xt)
	dummy := routerGateUpload(v, []int{1}, []float32{0}, MemoryActivation)
	defer v.Free(dummy)
	for _, tc := range []struct {
		name   string
		full   bool
		topK   int
		weight float32
	}{
		{"full_f32_zero", true, 3, 0},
		{"reduced_preserves_tiny_score", false, 1, 1.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ref := routerGateReference(t, v, wt, xt, dummy, experts, tc.topK, tc.full, false)
			got := routerGateFused(v, wt, xt, dummy, experts, hidden, tc.topK, tc.full, false)
			for _, path := range []struct {
				name string
				run  routerGateRun
			}{{"split", ref}, {"fused", got}} {
				if path.run.err != nil {
					t.Fatalf("%s: route faulted: %v", path.name, path.run.err)
				}
				if len(path.run.logits) != experts || len(path.run.picks.Experts) != tc.topK || len(path.run.picks.Weights) != tc.topK {
					t.Fatalf("%s: unexpected logit/pick extents %d/%d/%d", path.name, len(path.run.logits), len(path.run.picks.Experts), len(path.run.picks.Weights))
				}
				for expert, z := range path.run.logits {
					if math.Float32bits(z) != math.Float32bits(logit) {
						t.Fatalf("%s: expert %d logit = %g, want %g", path.name, expert, z, logit)
					}
				}
				for slot := 0; slot < tc.topK; slot++ {
					if path.run.picks.Experts[slot] != int32(slot) || math.Float32bits(path.run.picks.Weights[slot]) != math.Float32bits(tc.weight) {
						t.Fatalf("%s: slot %d got (%d, %#08x), want (%d, %#08x)", path.name, slot,
							path.run.picks.Experts[slot], math.Float32bits(path.run.picks.Weights[slot]), slot, math.Float32bits(tc.weight))
					}
				}
			}
			requireRouterGateEqual(t, tc.name, ref, got)
		})
	}
}

// The fused module must equal MatMul + router top-k bit for bit on the V4.1
// geometry over randomized rows, every bias/full combination, and a row whose
// leading twelve experts tie exactly.
// fak-test:runtime fast est=3s lane=default
func TestVulkanV41RouterGateBitIdenticalV41Rows(t *testing.T) {
	v := vkRouterGate(t)
	const experts, hidden, topK = 384, 5120, 6
	rows := 64
	if n, err := strconv.Atoi(os.Getenv("FAK_V41_ROUTER_GATE_ROWS")); err == nil && n > 0 {
		rows = n
	}
	w, aligned, bias := v41RouterGateProbeFixture(experts, hidden)
	wt := v.Upload(NewF32(Default(), []int{experts, hidden}, w), F32)
	defer v.Free(wt)
	bt := routerGateUpload(v, []int{experts}, bias, MemoryActivation)
	defer v.Free(bt)
	dummy := routerGateUpload(v, []int{1}, []float32{0}, MemoryActivation)
	defer v.Free(dummy)
	ties := 0
	for row := 0; row < rows; row++ {
		x := aligned
		if row > 0 {
			x = make([]float32, hidden)
			for i := range x {
				x[i] = decodeSplitValue(i, 1000003+row)
			}
		}
		hasBias, full := row%4 != 3, row%5 != 4
		b := bt
		if !hasBias {
			b = dummy
		}
		// Odd rows record inside the decode step window, as production does.
		stepped := row%2 == 1
		if stepped && !v.BeginV41StepBatch() {
			t.Fatal("V4.1 step batch refused")
		}
		xt := routerGateUpload(v, []int{hidden}, x, MemoryActivation)
		ref := routerGateReference(t, v, wt, xt, b, experts, topK, full, hasBias)
		got := routerGateFused(v, wt, xt, b, experts, hidden, topK, full, hasBias)
		v.Free(xt)
		if stepped {
			if err := v.EndV41StepBatch(); err != nil {
				t.Fatal(err)
			}
		}
		requireRouterGateEqual(t, "row "+strconv.Itoa(row), ref, got)
		if row == 0 {
			for slot, want := range []int32{5, 37, 69, 101, 133, 165} {
				if got.picks.Experts[slot] != want {
					t.Fatalf("tied row slot %d picked %d, want %d (lower expert first)", slot, got.picks.Experts[slot], want)
				}
			}
			ties++
		}
	}
	t.Logf("V41_ROUTER_GATE rows=%d bit_identical=%d tied_rows=%d", rows, rows, ties)
}

// Partial chunks, a single chunk, the widest supported input and the
// largest expert count keep MatMul's chunk order.
// fak-test:runtime fast est=2s lane=default
func TestVulkanV41RouterGateBitIdenticalShapes(t *testing.T) {
	v := vkRouterGate(t)
	for _, sh := range []struct {
		name                  string
		experts, hidden, topK int
	}{
		{"partial_chunk_tail", 72, 3*1024 + 40, 6},
		{"tail_only", 9, 7, 3},
		{"one_full_chunk_max_experts", 1024, 1024, 8},
		{"widest_input", 16, 64 * 1024, 4},
		{"two_chunks_one_expert_per_lane", 65, 2048, 1},
	} {
		t.Run(sh.name, func(t *testing.T) {
			w, x, bias := v41RouterGateProbeFixture(sh.experts, sh.hidden)
			wt := v.Upload(NewF32(Default(), []int{sh.experts, sh.hidden}, w), F32)
			defer v.Free(wt)
			xt := routerGateUpload(v, []int{sh.hidden}, x, MemoryActivation)
			defer v.Free(xt)
			bt := routerGateUpload(v, []int{sh.experts}, bias, MemoryActivation)
			defer v.Free(bt)
			ref := routerGateReference(t, v, wt, xt, bt, sh.experts, sh.topK, true, true)
			got := routerGateFused(v, wt, xt, bt, sh.experts, sh.hidden, sh.topK, true, true)
			requireRouterGateEqual(t, sh.name, ref, got)
		})
	}
}

// Faults report the router's stage, expert and value bits: the lowest
// faulting expert wins across stages, and an all-zero selection faults only
// when the route is not full.
// fak-test:runtime fast est=2s lane=default
func TestVulkanV41RouterGateFaultsMatchRouter(t *testing.T) {
	v := vkRouterGate(t)
	const experts, hidden, topK = 96, 2048, 6
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	for _, tc := range []struct {
		name  string
		edit  func(w, x, bias []float32)
		full  bool
		stage V41RouterTopKStage
	}{
		{"nan_logit", func(w, _, _ []float32) { w[7*hidden+3] = nan }, true, V41RouterTopKStageLogit},
		{"lower_bias_beats_later_logit", func(w, _, bias []float32) { w[7*hidden+3] = nan; bias[3] = inf }, true, V41RouterTopKStageBias},
		{"max_bias_selects", func(_, _, bias []float32) { bias[70] = math.MaxFloat32 }, true, V41RouterTopKStageSuccess},
		{"zero_scores_not_full", func(w, x, _ []float32) {
			for i := range w {
				w[i] = -1
			}
			for i := range x {
				x[i] = 1
			}
		}, false, V41RouterTopKStageNormalization},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w, x, bias := v41RouterGateProbeFixture(experts, hidden)
			tc.edit(w, x, bias)
			wt := v.Upload(NewF32(Default(), []int{experts, hidden}, w), F32)
			defer v.Free(wt)
			xt := routerGateUpload(v, []int{hidden}, x, MemoryActivation)
			defer v.Free(xt)
			bt := routerGateUpload(v, []int{experts}, bias, MemoryActivation)
			defer v.Free(bt)
			ref := routerGateReference(t, v, wt, xt, bt, experts, topK, tc.full, true)
			got := routerGateFused(v, wt, xt, bt, experts, hidden, topK, tc.full, true)
			var fault *V41RouterTopKFault
			if tc.stage == V41RouterTopKStageSuccess {
				if got.err != nil {
					t.Fatalf("fused route faulted: %v", got.err)
				}
			} else if !errors.As(got.err, &fault) || fault.Stage != tc.stage {
				t.Fatalf("fused fault=%v, want stage %s", got.err, tc.stage)
			}
			requireRouterGateEqual(t, tc.name, ref, got)
		})
	}
}

// Rejected geometry and foreign operands fail closed before any dispatch.
// fak-test:runtime fast est=1s lane=default
func TestVulkanV41RouterGateRefusesMalformedOperands(t *testing.T) {
	v := vkRouterGate(t)
	const experts, hidden = 16, 64
	w, x, bias := v41RouterGateProbeFixture(experts, hidden)
	wt := v.Upload(NewF32(Default(), []int{experts, hidden}, w), F32)
	defer v.Free(wt)
	xt := routerGateUpload(v, []int{hidden}, x, MemoryActivation)
	defer v.Free(xt)
	bt := routerGateUpload(v, []int{experts}, bias, MemoryActivation)
	defer v.Free(bt)
	host := NewF32(Default(), []int{hidden}, x)
	for name, call := range map[string]func() error{
		"host_input": func() error {
			_, err := v.V41RouterGateTopK(wt, host, bt, experts, hidden, 4, 1.5, true, true)
			return err
		},
		"shape_mismatch": func() error {
			_, err := v.V41RouterGateTopK(wt, xt, bt, experts, hidden+1, 4, 1.5, true, true)
			return err
		},
		"topk_over_bound": func() error {
			_, err := v.V41RouterGateTopK(wt, xt, bt, experts, hidden, V41RouterTopKMaxK+1, 1.5, true, true)
			return err
		},
	} {
		err := call()
		var backend *BackendError
		if !errors.As(err, &backend) || !errors.Is(err, ErrVulkanInvalidGeometry) {
			t.Errorf("%s: err=%v, want invalid geometry", name, err)
		}
	}
	if status := vulkanSubmissionStatusForTest(); status != 0 {
		t.Fatalf("refused calls poisoned the context: status %d", status)
	}
}

// Opt-in comparison of one routed decode row inside the V4.1 step window on
// this device; each side ends in its single picks readback. With
// FAK_VULKAN_TIMESTAMP_PROFILE=1 and FAK_VULKAN_TIMESTAMP_PROFILE_EVERY=1 set
// before process start it also logs per-kernel GPU time. Not a Halo number.
// fak-test:runtime fast est=2s lane=optin
func TestVulkanV41RouterGateTiming(t *testing.T) {
	if os.Getenv("FAK_V41_ROUTER_GATE_TIMING") != "1" {
		t.Skip("set FAK_V41_ROUTER_GATE_TIMING=1")
	}
	v := vkRouterGate(t)
	const experts, hidden, topK, iters = 384, 5120, 6, 400
	w, x, bias := v41RouterGateProbeFixture(experts, hidden)
	wt := v.Upload(NewF32(Default(), []int{experts, hidden}, w), F32)
	defer v.Free(wt)
	xt := routerGateUpload(v, []int{hidden}, x, MemoryActivation)
	defer v.Free(xt)
	bt := routerGateUpload(v, []int{experts}, bias, MemoryActivation)
	defer v.Free(bt)
	measure := func(name string, run func()) {
		for i := 0; i < 20; i++ {
			run()
		}
		before := KernelProfileSnapshot()
		if !v.BeginV41StepBatch() {
			t.Fatal("V4.1 step batch refused")
		}
		start := time.Now()
		for i := 0; i < iters; i++ {
			run()
		}
		wall := time.Since(start)
		if err := v.EndV41StepBatch(); err != nil {
			t.Fatal(err)
		}
		t.Logf("V41_ROUTER_GATE_TIMING device=%q path=%s wall_us_per_row=%.1f", v.Tier(), name, float64(wall.Nanoseconds())/1e3/iters)
		if d := KernelProfileDelta(before, KernelProfileSnapshot()); d != nil && d.Ops > 0 {
			t.Logf("V41_ROUTER_GATE_TIMING path=%s gpu_us_per_row=%.1f batches=%d ops=%d", name, d.GPUNS/1e3/iters, d.Batches, d.Ops)
			for _, r := range d.Rows {
				t.Logf("V41_ROUTER_GATE_TIMING path=%s kernel=%s grid=%dx%d calls=%d us_per_call=%.2f", name, r.Kernel, r.GX, r.GY, r.Calls, r.NS/1e3/float64(r.Calls))
			}
		}
	}
	measure("split", func() {
		y := v.MatMul(wt, xt)
		view := y
		view.Shape = []int{1, experts}
		if _, err := v.V41RouterTopK(view, bt, 1, experts, topK, 1.5, true, true); err != nil {
			t.Fatal(err)
		}
		v.Free(y)
	})
	measure("fused", func() {
		if _, err := v.V41RouterGateTopK(wt, xt, bt, experts, hidden, topK, 1.5, true, true); err != nil {
			t.Fatal(err)
		}
	})
}

// The admission self-check's regions are equal on this device. On a device
// that disagrees, this names the first differing logit, the usual signature
// of a contraction difference between the fused and MatMul projection sums.
// fak-test:runtime fast est=1s lane=default
func TestVulkanV41RouterGateSelfCheckRegionsEqual(t *testing.T) {
	v := vk(t)
	if vulkanV41RouterGateModule() == 0 {
		t.Fatal("v41_router_gate_topk.spv is absent from the runtime SPIR-V directory or failed to build")
	}
	image, err := v.v41RouterGateSelfCheckImage()
	if err != nil {
		t.Fatal(err)
	}
	ref, got := vulkanLEFloat32s(image[0]), vulkanLEFloat32s(image[4])
	for i := range ref {
		if math.Float32bits(ref[i]) != math.Float32bits(got[i]) {
			t.Fatalf("logit %d matmul %#08x fused %#08x", i, math.Float32bits(ref[i]), math.Float32bits(got[i]))
		}
	}
	for i, name := range []string{"experts", "weights", "status"} {
		if string(image[1+i]) != string(image[5+i]) {
			t.Fatalf("%s matmul+router %x fused %x", name, image[1+i], image[5+i])
		}
	}
	if string(image[3]) != string(make([]byte, 16)) {
		t.Fatalf("self-check reference faulted: status %x", image[3])
	}
}
