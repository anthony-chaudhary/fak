//go:build cuda && cgo

package compute

import (
	"os"
	"os/exec"
	"testing"
)

// Profiling witnesses selection, not performance. This runs real whole-operation
// entries and carried state on CUDA; absence is a failure in required-device mode.
// fak-test:runtime medium est=5s
func TestCUDAGDNVLLMPostConvDevice(t *testing.T) {
	be := cudaGDNBackend(t)
	if !KernelProfileReaderRegistered() {
		if os.Getenv(cudaTimestampProfileEnv) == "1" {
			t.Fatal("requested CUDA dispatch profile has no reader")
		}
		cmd := exec.Command(os.Args[0], "-test.run", "^TestCUDAGDNVLLMPostConvDevice$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), cudaTimestampProfileEnv+"=1", "FAK_CUDA_TIMESTAMP_PROFILE_EVERY=1", cudaGDNRequiredEnv+"=1")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("profiled required-device GDN witness: %v\n%s", err, out)
		}
		return
	}
	t.Cleanup(be.Recycle)
	t.Setenv("FAK_CUDA_GDN_RECURRENCE", "serial")
	for _, tc := range []struct {
		name, mode string
		g          cudaGDNGeometry
		tokens     int
		selected   bool
	}{
		{"decode", "vllm", cudaGDNGeometry{32, 2, 6, 128, 128, 4, 1e-6}, 1, true},
		{"qwen38-eight-token-axes", "vllm", cudaGDNGeometry{32, 16, 48, 128, 128, 4, 1e-6}, 8, true},
		{"longer-panel-fallback", "vllm", cudaGDNGeometry{16, 1, 3, 128, 128, 4, 1e-6}, 9, false},
		{"head-ratio-fallback", "vllm", cudaGDNGeometry{16, 1, 5, 128, 128, 4, 1e-6}, 2, false},
		{"dimension-fallback", "vllm", cudaGDNGeometry{16, 2, 4, 64, 128, 4, 1e-6}, 2, false},
		{"unknown-mode-fallback", "vllmx", cudaGDNGeometry{16, 1, 3, 128, 128, 1, 1e-6}, 1, false},
		{"default-fallback", "", cudaGDNGeometry{16, 1, 3, 128, 128, 4, 1e-6}, 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := tc.g
			o := newCUDAGDNOperands(t, be, g)
			rng := cudaGDNLCG(0x6d6e7631)
			convInit := rng.vector((g.kernel-1)*g.convDim(), 0.5)
			stateInit := rng.vector(g.nV*g.kHd*g.vHd, 0.2)
			inputs := rng.vector(tc.tokens*g.hidden, 1)
			shape := []int{tc.tokens, g.hidden}
			if tc.tokens == 1 {
				shape = []int{g.hidden}
			}
			input := uploadCUDAGDN(t, be, shape, inputs, MemoryActivation, "vllm-gdn-input")
			continuedInput := rng.vector(g.hidden, 1)
			continued := uploadCUDAGDN(t, be, []int{g.hidden}, continuedInput, MemoryActivation, "vllm-gdn-continuation")
			oracle := qwen35GDNDecodeHostOracle{
				g: g, inQKV: be.Read(o.inQKV), inZ: be.Read(o.inZ), inB: be.Read(o.inB), inA: be.Read(o.inA),
				convW: be.Read(o.convW), aLog: be.Read(o.aLog), dtBias: be.Read(o.dtBias), norm: be.Read(o.norm), outW: be.Read(o.outW),
			}
			run := func(mode string, selected bool) {
				t.Helper()
				t.Setenv("FAK_CUDA_GDN_POST_CONV", mode)
				arm := o
				arm.x = input
				arm.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "vllm-gdn-conv-state")
				arm.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, stateInit, MemoryKVCache, "vllm-gdn-recurrent-state")
				host := oracle
				host.conv, host.recurrent = qwen35GDNFloat64s(convInit), qwen35GDNFloat64s(stateInit)
				var want []float64
				for token := 0; token < tc.tokens; token++ {
					want = append(want, host.step(inputs[token*g.hidden:(token+1)*g.hidden])...)
				}
				before := KernelProfileSnapshot()
				be.ResetHostXfer()
				be.ResetH2DXfer()
				var out, conv, state Tensor
				var err error
				if tc.tokens == 1 {
					out, conv, state, err = arm.decode(be, g)
				} else {
					out, conv, state, err = be.Qwen35GDNSequence(arm.x, arm.inQKV, arm.inZ, arm.inB, arm.inA,
						arm.convW, arm.aLog, arm.dtBias, arm.norm, arm.outW, arm.convState, arm.recurrentState,
						g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps)
				}
				if err != nil {
					t.Fatal(err)
				}
				if conv.Buf() != arm.convState.Buf() || state.Buf() != arm.recurrentState.Buf() || !out.Ready() || !state.Ready() {
					t.Fatal("post-conv path changed durable state identity or completion")
				}
				if be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
					t.Fatal("post-conv operation transferred operands or outputs through the host")
				}
				d := KernelProfileDelta(before, KernelProfileSnapshot())
				if d == nil || d.Backend != "cuda" || d.Truncated != 0 {
					t.Fatalf("missing complete dispatch profile: %+v", d)
				}
				var calls uint64
				for _, row := range d.Rows {
					if row.Kernel == "k_qwen35_gdn_post_conv_vllm" {
						calls += row.Calls
						if row.GX != uint32(g.nV) || row.GY != 1 {
							t.Fatalf("candidate launch geometry: %+v", row)
						}
					}
				}
				wantCalls := uint64(0)
				if selected {
					wantCalls = 1
				}
				if calls != wantCalls {
					t.Fatalf("candidate calls %d, want %d", calls, wantCalls)
				}
				requireQwen35GDNClose(t, "output", tc.tokens, be.Read(out), want)
				requireQwen35GDNClose(t, "conv state", tc.tokens, be.Read(conv), host.conv)
				requireQwen35GDNClose(t, "canonical recurrent state", tc.tokens, be.Read(state), host.recurrent)
				arm.x = continued
				be.ResetHostXfer()
				be.ResetH2DXfer()
				out, _, _, err = arm.decode(be, g)
				if err != nil {
					t.Fatal(err)
				}
				if be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
					t.Fatal("continuation transferred through host")
				}
				requireQwen35GDNClose(t, "continued output", tc.tokens+1, be.Read(out), host.step(continuedInput))
				requireQwen35GDNClose(t, "continued canonical state", tc.tokens+1, be.Read(state), host.recurrent)
			}
			run("", false)
			run(tc.mode, tc.selected)
		})
	}
}
