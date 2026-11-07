//go:build cuda && cgo

package compute

import (
	"os"
	"os/exec"
	"testing"
)

// The warp-per-row GDN in/out projections keep each old block thread's fmaf order
// and reduction tree, so decode and sequence outputs, conv state and recurrent state
// must be bitwise the one-block-per-row kernels'. The profile proves which kernel ran.
// fak-test:runtime slow est=20s
func TestCUDAQwen35GDNRowDotWarpMatchesBlock(t *testing.T) {
	be := cudaGDNBackend(t)
	if !KernelProfileReaderRegistered() {
		if os.Getenv(cudaTimestampProfileEnv) == "1" {
			t.Fatal("CUDA timestamp profile requested but no reader registered")
		}
		cmd := exec.Command(os.Args[0], "-test.run", "^TestCUDAQwen35GDNRowDotWarpMatchesBlock$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), cudaTimestampProfileEnv+"=1", "FAK_CUDA_TIMESTAMP_PROFILE_EVERY=1", cudaGDNRequiredEnv+"=1")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("profiled row-dot witness: %v\n%s", err, out)
		}
		return
	}
	t.Cleanup(be.Recycle)
	t.Cleanup(func() { qwen35GDNRowDotForceForTest(-1) })
	t.Setenv("FAK_CUDA_GDN_RECURRENCE", "serial")
	rng := cudaGDNLCG(0x77a3d1)
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
	type fixture struct {
		name   string
		g      cudaGDNGeometry
		q8     bool // Q8 qkv/z/a/out; b stays F32 to cover a mixed row set
		tokens int
	}
	for _, fx := range []fixture{
		{"qwen38-q8-mixed", cudaGDNGeometry{hidden: 5120, nK: 16, nV: 48, kHd: 128, vHd: 128, kernel: 4, eps: 1e-6}, true, 3},
		{"small-f32", cudaGDNGeometry{hidden: 96, nK: 2, nV: 4, kHd: 16, vHd: 24, kernel: 4, eps: 1e-6}, false, 5},
	} {
		t.Run(fx.name, func(t *testing.T) {
			g := fx.g
			o := newCUDAGDNOperands(t, be, g)
			if fx.q8 {
				o.inQKV, o.inZ, o.inA, o.outW = q8(g.convDim(), g.hidden), q8(g.valueDim(), g.hidden), q8(g.nV, g.hidden), q8(g.hidden, g.valueDim())
			}
			panel := uploadCUDAGDN(t, be, []int{fx.tokens, g.hidden}, rng.vector(fx.tokens*g.hidden, 1), MemoryActivation, "rowdot-panel")
			next := uploadCUDAGDN(t, be, []int{g.hidden}, rng.vector(g.hidden, 1), MemoryActivation, "rowdot-next")
			convInit := rng.vector((g.kernel-1)*g.convDim(), 0.5)
			stateInit := rng.vector(g.nV*g.kHd*g.vHd, 0.1)
			run := func(t *testing.T, mode int) ([][]float32, map[string]uint64) {
				t.Helper()
				qwen35GDNRowDotForceForTest(mode)
				be.Recycle()
				defer be.Recycle()
				arm := o
				arm.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "rowdot-conv-state")
				arm.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, stateInit, MemoryKVCache, "rowdot-recurrent-state")
				before := KernelProfileSnapshot()
				x := panel
				x.Shape = []int{fx.tokens, g.hidden}
				seq, _, _, err := be.Qwen35GDNSequence(x, arm.inQKV, arm.inZ, arm.inB, arm.inA,
					arm.convW, arm.aLog, arm.dtBias, arm.norm, arm.outW, arm.convState, arm.recurrentState,
					g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps)
				if err != nil {
					t.Fatal(err)
				}
				var values [][]float32
				values = append(values, be.Read(seq), be.Read(arm.convState), be.Read(arm.recurrentState))
				for step := 0; step < 2; step++ {
					arm.x = next
					out, conv, state, err := arm.decode(be, g)
					if err != nil {
						t.Fatalf("decode step %d: %v", step, err)
					}
					values = append(values, be.Read(out), be.Read(conv), be.Read(state))
				}
				requireFiniteNonzeroCUDA(t, "decode output", values[len(values)-3])
				d := KernelProfileDelta(before, KernelProfileSnapshot())
				if d == nil || d.Backend != "cuda" || d.Truncated != 0 {
					t.Fatalf("incomplete CUDA dispatch profile: %+v", d)
				}
				calls := map[string]uint64{}
				for _, row := range d.Rows {
					calls[row.Kernel] += row.Calls
				}
				return values, calls
			}
			want, blockCalls := run(t, 0)
			got, warpCalls := run(t, 1)
			if blockCalls["k_qwen35_gdn_fused_in_proj_warp"] != 0 || blockCalls["k_qwen35_gdn_out_proj_warp"] != 0 ||
				blockCalls["k_qwen35_gdn_fused_in_proj"] != 3 || blockCalls["k_qwen35_gdn_out_proj"] != 3 {
				t.Fatalf("block arm dispatch: %v", blockCalls)
			}
			if warpCalls["k_qwen35_gdn_fused_in_proj"] != 0 || warpCalls["k_qwen35_gdn_out_proj"] != 0 ||
				warpCalls["k_qwen35_gdn_fused_in_proj_warp"] != 3 || warpCalls["k_qwen35_gdn_out_proj_warp"] != 3 {
				t.Fatalf("warp arm dispatch: %v", warpCalls)
			}
			if len(got) != len(want) {
				t.Fatal("row-dot arms returned different buffer sets")
			}
			for i := range want {
				if !equalF32Bits(got[i], want[i]) {
					t.Fatalf("buffer %d differs bitwise between block and warp row-dot kernels", i)
				}
			}
		})
	}
}
