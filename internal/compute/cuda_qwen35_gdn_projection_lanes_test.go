//go:build cuda && cgo

package compute

import (
	"os"
	"os/exec"
	"testing"
)

// The exact admitted geometry uses real Q8 block-32 weights, not a reduced shape
// that silently takes the old dispatch. Profiling proves selection only: its
// instrumented timings are not a production performance comparison.
// fak-test:runtime slow est=20s
func TestCUDAQwen35GDNFourLaneProjectionMatchesSequence(t *testing.T) {
	be := cudaGDNBackend(t)
	if !KernelProfileReaderRegistered() {
		if os.Getenv(cudaTimestampProfileEnv) == "1" {
			t.Fatal("CUDA timestamp profile requested but no reader registered")
		}
		cmd := exec.Command(os.Args[0], "-test.run", "^TestCUDAQwen35GDNFourLaneProjectionMatchesSequence$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), cudaTimestampProfileEnv+"=1", "FAK_CUDA_TIMESTAMP_PROFILE_EVERY=1", cudaGDNRequiredEnv+"=1")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("profiled projection witness: %v\n%s", err, out)
		}
		return
	}
	t.Cleanup(be.Recycle)
	t.Setenv("FAK_CUDA_GDN_RECURRENCE", "serial")
	g := cudaGDNGeometry{hidden: 5120, nK: 16, nV: 48, kHd: 128, vHd: 128, kernel: 4, eps: 1e-6}
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
		convW:  uploadCUDAGDN(t, be, []int{g.convDim(), 1, g.kernel}, rng.vector(g.convDim()*g.kernel, 0.3), MemoryWeights, "lanes4-conv"),
		aLog:   uploadCUDAGDN(t, be, []int{g.nV}, aLog, MemoryWeights, "lanes4-a-log"),
		dtBias: uploadCUDAGDN(t, be, []int{g.nV}, rng.vector(g.nV, 0.2), MemoryWeights, "lanes4-dt-bias"),
		norm:   uploadCUDAGDN(t, be, []int{g.vHd}, norm, MemoryWeights, "lanes4-norm"),
	}
	f32B := uploadCUDAGDN(t, be, []int{g.nV, g.hidden}, rng.vector(g.nV*g.hidden, 0.1), MemoryWeights, "lanes4-f32-b")
	panel := uploadCUDAGDN(t, be, []int{128, g.hidden}, rng.vector(128*g.hidden, 1), MemoryActivation, "lanes4-input")
	next := uploadCUDAGDN(t, be, []int{g.hidden}, rng.vector(g.hidden, 1), MemoryActivation, "lanes4-next")
	convInit := rng.vector((g.kernel-1)*g.convDim(), 0.5)
	stateInit := rng.vector(g.nV*g.kHd*g.vHd, 0.1)
	type snapshot struct {
		values       [][]float32
		scratchBytes int
	}
	run := func(t *testing.T, mode string, tokens int, mixed, wantLanes bool) snapshot {
		t.Helper()
		t.Setenv("FAK_CUDA_GDN_INPUT_PROJECTION", mode)
		be.Recycle()
		defer be.Recycle()
		arm := o
		if mixed {
			arm.inB = f32B
		}
		arm.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "lanes4-conv-state")
		arm.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, stateInit, MemoryKVCache, "lanes4-recurrent-state")
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
		var laneCalls, scalarCalls uint64
		for _, row := range d.Rows {
			switch row.Kernel {
			case "k_qwen35_gdn_fused_in_proj_lanes":
				laneCalls += row.Calls
				if row.GX != uint32(g.convDim()+g.valueDim()+2*g.nV) || row.GY != 1 {
					t.Fatalf("lane projection grid: %+v", row)
				}
			case "k_qwen35_gdn_fused_in_proj":
				scalarCalls += row.Calls
			}
		}
		var wantLaneCalls, wantScalarCalls uint64
		if wantLanes {
			wantLaneCalls = 1
		} else if tokens < 128 || mixed {
			wantScalarCalls = 1
		}
		if laneCalls != wantLaneCalls || scalarCalls != wantScalarCalls {
			t.Fatalf("projection dispatch lanes=%d scalar=%d, want %d/%d", laneCalls, scalarCalls, wantLaneCalls, wantScalarCalls)
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
		name, mode string
		tokens     int
		mixed      bool
		wantLanes  bool
	}{
		{"exact-four", "lanes4", 4, false, true},
		{"short-fallback", "lanes4", 3, false, false},
		{"mixed-weight-fallback", "lanes4", 4, true, false},
		{"unknown-mode-fallback", "lanes4x", 4, false, false},
		{"gemm-unchanged", "lanes4", 128, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := run(t, "", tc.tokens, tc.mixed, false)
			got := run(t, tc.mode, tc.tokens, tc.mixed, tc.wantLanes)
			if got.scratchBytes != want.scratchBytes || len(got.values) != len(want.values) {
				t.Fatal("projection changed scratch/output footprint")
			}
			for i := range want.values {
				if !equalF32Bits(got.values[i], want.values[i]) {
					t.Fatalf("scratch/state/continuation buffer %d differs bitwise", i)
				}
			}
		})
	}
}
