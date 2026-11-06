//go:build cuda && cgo

package compute

import (
	"fmt"
	"os"
	"runtime"
	"testing"
)

// This closes the composition gap between the shipped packed-input, MMQ and
// vLLM GDN witnesses. P=8 is the only panel size that selects both MMQ (P>=8)
// and vLLM post-conv (P<=8). P=7/P=9 witness the neighboring fallback edges.
//
// Run alone on an admitted CUDA node with an archive built from this source:
// FAK_CUDA_GDN_MMQ_REQUIRED=1 FAK_CUDA_TIMESTAMP_PROFILE=1 FAK_CUDA_TIMESTAMP_PROFILE_EVERY=1 go test -tags cuda ./internal/compute -run '^TestCUDAGDNMMQVLLMComposition$' -count=1 -v
// H200 qualification additionally requires an external H200 UUID/model receipt;
// sm_90 alone also admits H100 and must not be presented as H200 evidence.
// This is not a full-model quality, throughput, graph or FP8-KV qualification.
// The Stream-K case also requires at least 30 visible SMs to witness the exact
// useful-work capped grid; a smaller partition is not this qualification cohort.
// fak-test:runtime slow est=65s lane=optin
func TestCUDAGDNMMQVLLMComposition(t *testing.T) {
	if os.Getenv("FAK_CUDA_GDN_MMQ_REQUIRED") != "1" {
		t.Skip("set FAK_CUDA_GDN_MMQ_REQUIRED=1 on an admitted CUDA node")
	}
	backend, ok := Lookup("cuda")
	if !ok || backend == nil || backend.Name() != "cuda" {
		t.Fatal("required composition witness: exact CUDA backend is absent")
	}
	be, ok := backend.(*cudaBackend)
	if !ok {
		t.Fatalf("required composition witness: CUDA backend type %T", backend)
	}
	if err := CheckCUDAQ4KPrefillArm(CUDAQ4KPrefillMMQ, cudaBackendSM(be)); err != nil {
		t.Fatal(err)
	}
	if !KernelProfileReaderRegistered() || os.Getenv("FAK_CUDA_TIMESTAMP_PROFILE_EVERY") != "1" {
		t.Fatal("required composition witness needs complete CUDA timestamp profiling enabled before startup")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	t.Cleanup(be.Recycle)
	prefill, err := CUDAQ4KPrefillArmInUse()
	if err != nil {
		t.Fatal(err)
	}
	decode := CUDAQ4KDecodeArmInUse()
	t.Cleanup(func() {
		if err := SetCUDAQ4KPrefillArm(prefill); err != nil {
			t.Error(err)
		}
		if err := SetCUDAQ4KDecodeArm(decode); err != nil {
			t.Error(err)
		}
	})
	if err := SetCUDAQ4KPrefillArm(CUDAQ4KPrefillMMQ); err != nil {
		t.Fatal(err)
	}
	if err := SetCUDAQ4KDecodeArm(CUDAQ4KDecodeMMVQ); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_CUDA_GDN_RECURRENCE", "serial")
	t.Setenv("FAK_CUDA_GDN_INPUT_PROJECTION", "")
	t.Logf("composition qualification tier=%s sm=%d; external device identity required", be.Tier(), cudaBackendSM(be))

	for _, tc := range []struct {
		name           string
		tokens, nK, nV int
		qkv, z         Dtype
		rounded        bool
		streamK        bool
	}{
		{"decode", 1, 2, 6, Q4_K, Q4_K, false, false},
		{"below-mmq", 7, 2, 6, Q4_K, Q4_K, false, false},
		{"joint-exact", 8, 2, 6, Q4_K, Q4_K, false, false},
		{"joint-rounded", 8, 2, 6, Q4_K, Q4_K, true, false},
		{"above-vllm", 9, 2, 6, Q4_K, Q4_K, true, false},
		{"long-panel", 128, 2, 6, Q4_K, Q4_K, true, false},
		{"q4-q6", 8, 2, 6, Q4_K, Q6_K, false, false},
		{"q6-q4", 8, 2, 6, Q6_K, Q4_K, false, false},
		{"qwen38-head-axes", 8, 16, 48, Q4_K, Q4_K, true, false},
		{"ratio1", 8, 1, 1, Q4_K, Q4_K, false, false},
		{"ratio2", 8, 1, 2, Q4_K, Q4_K, false, false},
		{"ratio4", 8, 1, 4, Q4_K, Q4_K, false, false},
		{"ratio8", 8, 1, 8, Q4_K, Q4_K, false, false},
		{"joint-stream-k-three-blocks", 8, 2, 6, Q4_K, Q4_K, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Neither width is Qwen's full hidden width. Three K superblocks
			// exercise row strides and partial-tile fixup before GDN consumes Y.
			hidden := 256
			if tc.streamK {
				hidden = 768
			}
			g := cudaGDNGeometry{hidden, tc.nK, tc.nV, 128, 128, 4, 1e-6}
			o := newCUDAGDNOperands(t, be, g)
			var qkv, z []float32
			var qkvRaw, zRaw []byte
			o.inQKV, qkv, qkvRaw = cudaGDNPackedWeight(t, be, tc.qkv, g.convDim(), g.hidden)
			o.inZ, z, zRaw = cudaGDNPackedWeight(t, be, tc.z, g.valueDim(), g.hidden)
			rng := cudaGDNLCG(0x4d4d5147444e)
			convInit := rng.vector((g.kernel-1)*g.convDim(), 0.5)
			stateInit := rng.vector(g.nV*g.kHd*g.vHd, 0.2)
			inputs := cudaGDNMMQInputs(&rng, tc.tokens, g.hidden, tc.rounded)
			shape := []int{tc.tokens, g.hidden}
			if tc.tokens == 1 {
				shape = []int{g.hidden}
			}
			input := uploadCUDAGDN(t, be, shape, inputs, MemoryActivation, "mmq-vllm-input")
			oracle := qwen35GDNDecodeHostOracle{g: g, inQKV: qkv, inZ: z,
				inB: be.Read(o.inB), inA: be.Read(o.inA), convW: be.Read(o.convW),
				aLog: be.Read(o.aLog), dtBias: be.Read(o.dtBias), norm: be.Read(o.norm), outW: be.Read(o.outW)}
			var mixedMMQ, zMMQ []float32
			wantMMQ := uint64(0)
			if tc.tokens >= cudaQ4KMMQMinCols {
				if tc.qkv == Q4_K {
					mixedMMQ = q4kMMQInt8Model(qkvRaw, inputs, g.convDim(), g.hidden, tc.tokens)
					wantMMQ++
				}
				if tc.z == Q4_K {
					zMMQ = q4kMMQInt8Model(zRaw, inputs, g.valueDim(), g.hidden, tc.tokens)
					wantMMQ++
				}
			}
			for _, mode := range []string{"", "vllm"} {
				name := mode
				if name == "" {
					name = "baseline"
				}
				t.Run(name, func(t *testing.T) {
					t.Setenv("FAK_CUDA_GDN_POST_CONV", mode)
					arm := o
					arm.x = input
					arm.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "mmq-vllm-conv")
					arm.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, stateInit, MemoryKVCache, "mmq-vllm-state")
					host := oracle
					host.conv, host.recurrent = qwen35GDNFloat64s(convInit), qwen35GDNFloat64s(stateInit)
					var want []float64
					for token := 0; token < tc.tokens; token++ {
						x := qwen35GDNFloat64s(inputs[token*g.hidden : (token+1)*g.hidden])
						mixed := qwen35GDNHostMatvec(qkv, g.convDim(), g.hidden, x)
						gateZ := qwen35GDNHostMatvec(z, g.valueDim(), g.hidden, x)
						if mixedMMQ != nil {
							mixed = qwen35GDNFloat64s(mixedMMQ[token*g.convDim() : (token+1)*g.convDim()])
						}
						if zMMQ != nil {
							gateZ = qwen35GDNFloat64s(zMMQ[token*g.valueDim() : (token+1)*g.valueDim()])
						}
						want = append(want, host.stepProjected(mixed, gateZ,
							qwen35GDNHostMatvec(host.inB, g.nV, g.hidden, x),
							qwen35GDNHostMatvec(host.inA, g.nV, g.hidden, x))...)
					}
					cudaGDNMMQRun(t, be, arm, g, tc.tokens, wantMMQ, tc.streamK, mode == "vllm" && tc.tokens <= 8, want, &host)
					if !equalF32Bits(be.Read(input), inputs) {
						t.Fatal("whole operation changed immutable device input")
					}
					// Three additional decode steps switch implementations while
					// carrying exactly the same canonical buffers. Exact Q8 inputs
					// isolate recurrence/interchange from MMVQ activation rounding.
					for step, nextMode := range []string{"vllm", "", "vllm"} {
						t.Setenv("FAK_CUDA_GDN_POST_CONV", nextMode)
						x := cudaGDNMMQInputs(&rng, 1, g.hidden, false)
						// Vary consecutive inputs even though each stays on the Q8 grid.
						for i := range x {
							if (i+step)%3 == 0 {
								x[i] = -x[i]
							}
						}
						arm.x = uploadCUDAGDN(t, be, []int{g.hidden}, x, MemoryActivation, "mmq-vllm-continued-input")
						want := host.step(x)
						cudaGDNMMQRun(t, be, arm, g, 1, 0, false, nextMode == "vllm", want, &host)
					}
				})
			}
		})
	}
}

// Arbitrary activations exercise MMQ's half-rounded scale AND unquantized block
// sum; they must be checked against the independent integer model, not an F32
// GEMM with a widened tolerance. Dyadic inputs keep MMVQ/Q6 fallback exact.
func cudaGDNMMQInputs(rng *cudaGDNLCG, tokens, hidden int, rounded bool) []float32 {
	if rounded {
		x := rng.vector(tokens*hidden, 1.75)
		for row := 0; row < tokens; row++ {
			for i := 0; i < 32; i++ {
				// Zero blocks test the quantizer's amax==0 path; other
				// blocks retain nonzero positive and negative activations.
				x[row*hidden+i] = 0
			}
		}
		return x
	}
	x := make([]float32, tokens*hidden)
	for i := range x {
		x[i] = 127.0 / 128
		if (i+i/hidden)%5 < 2 {
			x[i] = -x[i]
		}
	}
	return x
}

func cudaGDNMMQRun(t *testing.T, be *cudaBackend, o cudaGDNOperands, g cudaGDNGeometry,
	tokens int, wantMMQ uint64, wantStreamK, wantVLLM bool, want []float64, host *qwen35GDNDecodeHostOracle) {
	t.Helper()
	// t.Run uses its own goroutine; pin the goroutine executing the operation.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	before := KernelProfileSnapshot()
	mmqBefore, operationsBefore := be.q4kMMQLaunchesForTest(), be.Qwen35GDNOperationCount()
	be.ResetHostXfer()
	be.ResetH2DXfer()
	var out, conv, state Tensor
	var err error
	if tokens == 1 {
		out, conv, state, err = o.decode(be, g)
	} else {
		out, conv, state, err = be.Qwen35GDNSequence(o.x, o.inQKV, o.inZ, o.inB, o.inA,
			o.convW, o.aLog, o.dtBias, o.norm, o.outW, o.convState, o.recurrentState,
			g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps)
	}
	if err != nil {
		t.Fatalf("whole GDN composition: %v", err)
	}
	defer be.Free(out)
	if conv.Buf() != o.convState.Buf() || state.Buf() != o.recurrentState.Buf() || !out.Ready() || !conv.Ready() || !state.Ready() {
		t.Fatal("composition lost durable canonical state identity or completion")
	}
	if be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
		t.Fatal("composition transferred operands/results through host")
	}
	if got := be.q4kMMQLaunchesForTest() - mmqBefore; got != wantMMQ {
		t.Fatalf("packed MMQ projections=%d want=%d; fallback cannot certify composition", got, wantMMQ)
	}
	if got := be.Qwen35GDNOperationCount() - operationsBefore; got != uint64(tokens) {
		t.Fatalf("completed GDN tokens=%d want=%d", got, tokens)
	}
	delta := KernelProfileDelta(before, KernelProfileSnapshot())
	if delta == nil || delta.Backend != "cuda" || delta.Truncated != 0 {
		t.Fatalf("missing complete CUDA dispatch profile: %+v", delta)
	}
	var vllmCalls, quantizeCalls, mmqCalls, fixupCalls uint64
	mainGrids, fixupGrids := make(map[uint32]uint64), make(map[uint32]uint64)
	for _, row := range delta.Rows {
		switch row.Kernel {
		case "k_qwen35_gdn_post_conv_vllm":
			vllmCalls += row.Calls
			if row.GX != uint32(g.nV) || row.GY != 1 {
				t.Fatalf("vLLM post-conv launch geometry: %+v", row)
			}
		case "quantize_mmq_q8_1_ds4":
			quantizeCalls += row.Calls
		case "mul_mat_q":
			mmqCalls += row.Calls
			if wantStreamK && row.GY != 1 {
				t.Fatalf("MMQ main launch geometry: %+v", row)
			}
			mainGrids[row.GX] += row.Calls
		case "mul_mat_q_stream_k_fixup":
			fixupCalls += row.Calls
			if wantStreamK && row.GY != 4 {
				t.Fatalf("MMQ fixup launch geometry: %+v", row)
			}
			fixupGrids[row.GX] += row.Calls
		}
	}
	wantCalls := uint64(0)
	if wantVLLM {
		wantCalls = 1
	}
	if vllmCalls != wantCalls || quantizeCalls != wantMMQ || mmqCalls != wantMMQ {
		t.Fatalf("dispatch vllm=%d want=%d MMQ-quantize=%d main=%d want=%d", vllmCalls, wantCalls, quantizeCalls, mmqCalls, wantMMQ)
	}
	if wantStreamK {
		// P=8 selects J=8; I=128. QKV has 10 row tiles and Z has 6.
		// With three K blocks and >=30 visible SMs, the useful-work cap
		// launches 30/18 CTAs, each owning one K block. Both projections
		// require fixup (10%30 and 6%18 are nonzero), with grid.y=I/32=4.
		if fixupCalls != 2 || mainGrids[30] != 1 || mainGrids[18] != 1 ||
			fixupGrids[30] != 1 || fixupGrids[18] != 1 {
			t.Fatalf("required three-block Stream-K path absent: main=%v fixup=%v calls=%d; need >=30 visible SMs", mainGrids, fixupGrids, fixupCalls)
		}
	}
	got := be.Read(out)
	requireFiniteNonzeroCUDA(t, "composed output", got)
	requireQwen35GDNClose(t, "composed output", tokens, got, want)
	requireQwen35GDNClose(t, "composed convolution state", tokens, be.Read(conv), host.conv)
	requireQwen35GDNClose(t, "composed canonical recurrent state", tokens, be.Read(state), host.recurrent)
	t.Log(fmt.Sprintf("P=%d hidden=%d MMQ=%d fixup=%d vLLM=%d output+conv+state match independent model", tokens, g.hidden, wantMMQ, fixupCalls, wantCalls))
}
