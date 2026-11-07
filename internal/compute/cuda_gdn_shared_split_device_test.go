//go:build cuda && cgo

package compute

import (
	"context"
	"math"
	"os"
	"os/exec"
	"runtime"
	"testing"
	"time"
)

// Required-device qualification of the existing whole-GDN sequence API, not a
// benchmark or full-model witness. Native compensated scratch is per operation
// here; the model-wide sequence-loan route needs its own qualification.
// fak-test:runtime slow est=30s lane=optin
func TestCUDAGDNSharedSplitDevice(t *testing.T) {
	if os.Getenv("FAK_CUDA_GDN_SHARED_SPLIT_REQUIRED") != "1" {
		t.Skip("explicit shared-split device qualification not requested")
	}
	t.Setenv(cudaGDNRequiredEnv, "1")
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread) // All later GPU cleanups run before unpinning.
	be := cudaGDNBackend(t)
	if !KernelProfileReaderRegistered() || os.Getenv("FAK_CUDA_TIMESTAMP_PROFILE_EVERY") != "1" {
		if !KernelProfileReaderRegistered() && os.Getenv(cudaTimestampProfileEnv) == "1" {
			t.Fatal("CUDA timestamp profile requested but no reader registered")
		}
		ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run", "^TestCUDAGDNSharedSplitDevice$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), cudaTimestampProfileEnv+"=1", "FAK_CUDA_TIMESTAMP_PROFILE_EVERY=1", cudaGDNRequiredEnv+"=1")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("profiled projection witness: %v\n%s", err, out)
		}
		return
	}
	t.Cleanup(be.Recycle)
	t.Setenv("FAK_CUDA_GDN_RECURRENCE", "serial")
	t.Setenv("FAK_CUDA_GDN_POST_CONV", "")
	t.Setenv("FAK_CUDA_Q8_SEQUENCE_PREFILL", "")
	t.Setenv("FAK_CUDA_Q8_PREFILL", "")
	g := cudaGDNGeometry{hidden: 256, nK: 2, nV: 6, kHd: 128, vHd: 128, kernel: 4, eps: 1e-6}
	rng := cudaGDNLCG(0x4a6e35)
	q8 := func(rows, cols int) Tensor {
		codes := make([]int8, rows*cols)
		for i := range codes {
			rng = rng*6364136223846793005 + 1442695040888963407
			codes[i] = int8(int(uint32(rng>>32)%255) - 127)
		}
		scales := rng.vector(len(codes)/32, 0.0005)
		for i := range scales {
			scales[i] += 0.001
		}
		w := be.Upload(NewQ8(Default(), []int{rows, cols}, codes, scales, 32), Q8_0)
		t.Cleanup(func() { be.Free(w) })
		return w
	}
	norm := make([]float32, g.vHd)
	for i := range norm {
		norm[i] = 1
	}
	aLog := rng.vector(g.nV, 0.2)
	for i := range aLog {
		aLog[i] -= 0.6
	}
	o := cudaGDNOperands{
		inQKV: q8(g.convDim(), g.hidden), inZ: q8(g.valueDim(), g.hidden),
		inB: q8(g.nV, g.hidden), inA: q8(g.nV, g.hidden), outW: q8(g.hidden, g.valueDim()),
		convW:  uploadCUDAGDN(t, be, []int{g.convDim(), 1, g.kernel}, rng.vector(g.convDim()*g.kernel, 0.3), MemoryWeights, "shared-split-conv"),
		aLog:   uploadCUDAGDN(t, be, []int{g.nV}, aLog, MemoryWeights, "shared-split-a-log"),
		dtBias: uploadCUDAGDN(t, be, []int{g.nV}, rng.vector(g.nV, 0.2), MemoryWeights, "shared-split-dt-bias"),
		norm:   uploadCUDAGDN(t, be, []int{g.vHd}, norm, MemoryWeights, "shared-split-norm"),
	}
	f32B := uploadCUDAGDN(t, be, []int{g.nV, g.hidden}, rng.vector(g.nV*g.hidden, 0.1), MemoryWeights, "shared-split-f32-b")
	panel := uploadCUDAGDN(t, be, []int{512, g.hidden}, rng.vector(512*g.hidden, 1), MemoryActivation, "shared-split-input")
	next := uploadCUDAGDN(t, be, []int{g.hidden}, rng.vector(g.hidden, 1), MemoryActivation, "shared-split-next")
	convInit := rng.vector((g.kernel-1)*g.convDim(), 0.5)
	stateInit := rng.vector(g.nV*g.kHd*g.vHd, 0.1)
	type snapshot struct {
		values       [][]float32
		scratchBytes int
	}
	run := func(t *testing.T, mode string, tokens int, mixed, wantShared bool) snapshot {
		t.Helper()
		t.Setenv("FAK_CUDA_GDN_INPUT_PROJECTION", mode)
		be.Recycle()
		defer be.Recycle()
		arm := o
		if mixed {
			arm.inB = f32B
		}
		arm.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "shared-split-conv-state")
		arm.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, stateInit, MemoryKVCache, "shared-split-recurrent-state")
		x := panel
		x.Shape = []int{tokens, g.hidden}
		before := KernelProfileSnapshot()
		cudaMu.Lock()
		start := len(be.transient)
		cudaMu.Unlock()
		be.ResetHostXfer()
		be.ResetH2DXfer()
		out, conv, state, err := be.Qwen35GDNSequence(x, arm.inQKV, arm.inZ, arm.inB, arm.inA,
			arm.convW, arm.aLog, arm.dtBias, arm.norm, arm.outW, arm.convState, arm.recurrentState,
			g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps)
		if err != nil {
			t.Fatal(err)
		}
		if conv.buf != arm.convState.buf || state.buf != arm.recurrentState.buf {
			t.Fatal("sequence replaced durable state")
		}
		if be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
			t.Fatalf("sequence transferred host bytes: d2h=%d h2d=%d", be.HostXferBytes(), be.H2DXferBytes())
		}
		d := KernelProfileDelta(before, KernelProfileSnapshot())
		if d == nil || d.Backend != "cuda" || d.Truncated != 0 {
			t.Fatalf("incomplete CUDA dispatch profile: %+v", d)
		}
		var splitCalls, weightCalls, gemmCalls, scalarCalls uint64
		for _, row := range d.Rows {
			switch row.Kernel {
			case "k_fp16_split":
				splitCalls += row.Calls
			case "k_q8_dequant_split":
				weightCalls += row.Calls
			case "cublas_gemm_ex":
				gemmCalls += row.Calls
			case "k_qwen35_gdn_fused_in_proj", "k_qwen35_gdn_fused_in_proj_warp":
				scalarCalls += row.Calls
			}
		}
		var wantSplits, wantWeights, wantScalar uint64
		if tokens >= 128 {
			wantSplits, wantWeights = 1, 1 // Unchanged output projection.
			if !mixed {
				wantWeights += 4
				if wantShared {
					wantSplits++
				} else {
					wantSplits += 4
				}
			}
		}
		if tokens < 128 || mixed {
			wantScalar = 1
		}
		if splitCalls != wantSplits || weightCalls != wantWeights || gemmCalls != 3*wantWeights || scalarCalls != wantScalar {
			t.Fatalf("mode=%q P=%d mixed=%v: split/weight/gemm/scalar=%d/%d/%d/%d want=%d/%d/%d/%d",
				mode, tokens, mixed, splitCalls, weightCalls, gemmCalls, scalarCalls, wantSplits, wantWeights, 3*wantWeights, wantScalar)
		}
		cudaMu.Lock()
		buffers := append([]*cudaBuf(nil), be.transient[start:]...)
		cudaMu.Unlock()
		layout := qwen35GDNAllocations("", "_", tokens, tokens, g.hidden, g.keyDim(), g.valueDim(), g.nV, g.convDim())
		if len(buffers) != len(layout) || out.buf != buffers[len(buffers)-1] {
			t.Fatal("sequence scratch/output allocation contract changed")
		}
		var result snapshot
		// mixed and z remain projection intermediates; b/a now hold beta/decay.
		// Observe the existing owned scratch before Recycle, without a new ABI.
		for i, allocation := range layout {
			result.scratchBytes += buffers[i].n
			result.values = append(result.values, be.Read(makeTensor(be, F32, RowMajor, allocation.shape, nil, buffers[i])))
		}
		requireFiniteNonzeroCUDA(t, "sequence output", result.values[len(layout)-1])
		result.values = append(result.values, be.Read(conv), be.Read(state))
		be.Recycle()
		arm.x = next
		continued, _, _, err := arm.decode(be, g)
		if err != nil {
			t.Fatalf("next scalar decode: %v", err)
		}
		result.values = append(result.values, be.Read(continued), be.Read(conv), be.Read(state))
		return result
	}
	for _, tc := range []struct {
		name, mode        string
		tokens            int
		mixed, wantShared bool
	}{
		{"first-large-panel", "shared-split", 128, false, true},
		{"tail-panel", "shared-split", 129, false, true},
		{"normal-chunk", "shared-split", 512, false, true},
		{"short-control", "shared-split", 127, false, false},
		{"mixed-control", "shared-split", 128, true, false},
		{"unknown-control", "shared-splitx", 128, false, false},
	} {
		// Keep uploads, operation, reads and continuation on this pinned goroutine.
		// A child t.Run would move the operation off its thread affinity.
		t.Logf("case=%s P=%d", tc.name, tc.tokens)
		want := run(t, "", tc.tokens, tc.mixed, false)
		got := run(t, tc.mode, tc.tokens, tc.mixed, tc.wantShared)
		if got.scratchBytes != want.scratchBytes || len(got.values) != len(want.values) {
			t.Fatal("shared split changed Go scratch/output footprint")
		}
		for i := range want.values {
			if !equalF32Bits(got.values[i], want.values[i]) {
				t.Fatalf("%s scratch/state/continuation buffer %d differs bitwise", tc.name, i)
			}
			for j, value := range got.values[i] {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("%s buffer %d[%d] nonfinite", tc.name, i, j)
				}
			}
		}
	}
}
