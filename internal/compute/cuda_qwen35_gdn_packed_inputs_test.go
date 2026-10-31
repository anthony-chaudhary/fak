//go:build cuda && cgo

package compute

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
)

// Run alone on a separately admitted CUDA node with a source-matched archive:
// FAK_CUDA_GDN_PACKED_INPUTS_REQUIRED=1 FAK_CUDA_GRAPH=1 go test -tags cuda ./internal/compute -run '^TestCUDAGDNPackedInputs$' -count=1 -v
// This is a small real Upload -> whole-GDN witness, not a served-model or memory
// performance receipt. It selects/restores the existing MMVQ/compensated arms.
// MMQ/FP8LT precision, full-model loader admission, lane-batch execution, failed
// stream drains and captured packed execution remain separate qualification.
// fak-test:runtime slow est=30s lane=optin
func TestCUDAGDNPackedInputs(t *testing.T) {
	if os.Getenv("FAK_CUDA_GDN_PACKED_INPUTS_REQUIRED") != "1" {
		t.Skip("set FAK_CUDA_GDN_PACKED_INPUTS_REQUIRED=1 on an admitted CUDA acceptance node")
	}
	backend, ok := Lookup("cuda")
	if !ok || backend == nil || backend.Name() != "cuda" {
		t.Fatal("required witness: exact CUDA backend is not registered")
	}
	be, ok := backend.(*cudaBackend)
	if !ok {
		t.Fatalf("required witness: CUDA backend type %T", backend)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	t.Cleanup(be.Recycle)
	beforePrefill, err := CUDAQ4KPrefillArmInUse()
	if err != nil {
		t.Fatal(err)
	}
	beforeDecode := CUDAQ4KDecodeArmInUse()
	if err := SetCUDAQ4KPrefillArm(CUDAQ4KPrefillCompensated); err != nil {
		t.Fatal(err)
	}
	if err := SetCUDAQ4KDecodeArm(CUDAQ4KDecodeMMVQ); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := SetCUDAQ4KPrefillArm(beforePrefill); err != nil {
			t.Error(err)
		}
		if err := SetCUDAQ4KDecodeArm(beforeDecode); err != nil {
			t.Error(err)
		}
	})
	t.Setenv("FAK_CUDA_GDN_RECURRENCE", "serial")
	t.Setenv("FAK_CUDA_GDN_INPUT_PROJECTION", "")
	g := cudaGDNGeometry{hidden: 256, nK: 1, nV: 2, kHd: 4, vHd: 4, kernel: 4, eps: 1e-5}

	t.Run("private-loan-plan", func(t *testing.T) {
		for _, tc := range []struct {
			qkv, z Dtype
			arm    CUDAQ4KPrefillArm
			wantW  int
		}{
			{Q4_K, Q6_K, CUDAQ4KPrefillCompensated, 16 * 256 * 2},
			{Q6_K, Q4_K, CUDAQ4KPrefillCompensated, 8 * 256 * 2},
			{Q4_K, Q4_K, CUDAQ4KPrefillMMQ, 0},
			{Q4_K, Q4_K, CUDAQ4KPrefillFP8LT, 0},
			{Q6_K, Q6_K, CUDAQ4KPrefillCompensated, 0},
		} {
			layer := Qwen35SequenceLayer{Linear: true,
				GDNInQKV: Tensor{Dtype: tc.qkv, Shape: []int{16, 256}},
				GDNInZ:   Tensor{Dtype: tc.z, Shape: []int{8, 256}}}
			req := Qwen35SequencePrefillRequest{TokenIDs: make([]int, 128), Layers: []Qwen35SequenceLayer{layer}}
			plan, err := planCUDASequenceSplitScratch(req, tc.arm, nil)
			wantX := 0
			if tc.wantW != 0 {
				wantX = 128 * 256 * 2
			}
			if err != nil || plan.weightHalfBytes != tc.wantW || plan.activationHalfBytes != wantX {
				t.Fatalf("%v/%v/%s plan=%+v err=%v", tc.qkv, tc.z, tc.arm, plan, err)
			}
			if tc.qkv == Q4_K || tc.z == Q4_K {
				bad := errors.New("latched arm refusal")
				if _, err := planCUDASequenceSplitScratch(req, tc.arm, bad); !errors.Is(err, bad) {
					t.Fatalf("Q4 planner lost configured-arm refusal: %v", err)
				}
			}
		}
	})

	// Open a real empty capture before large fixture pools have been created.
	// No forged capture flag, pointer or owner can make this refusal witness pass.
	t.Run("capture-preflight", func(t *testing.T) {
		o := newCUDAGDNOperands(t, be, g)
		o.inQKV, _, _ = cudaGDNPackedWeight(t, be, Q4_K, g.convDim(), g.hidden)
		if !be.GraphBegin() {
			t.Fatal("required capture refusal needs FAK_CUDA_GRAPH=1 and a real capture-capable backend")
		}
		defer be.GraphAbort()
		beforeTransient, beforeLive := be.cudaAllocationCountsForTest()
		beforeFence := atomic.LoadUint64(&be.fenceGen)
		be.ResetHostXfer()
		be.ResetH2DXfer()
		out, conv, state, err := o.decode(be, g)
		var refusal *Qwen35GDNResidencyError
		if !errors.As(err, &refusal) || out.Backend() != nil || conv.Backend() != nil || state.Backend() != nil {
			t.Fatalf("captured packed operation was not preflight-refused: %v", err)
		}
		afterTransient, afterLive := be.cudaAllocationCountsForTest()
		if afterTransient != beforeTransient || afterLive != beforeLive || atomic.LoadUint64(&be.fenceGen) != beforeFence ||
			be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 || !be.IsCapturing() || !o.convState.Ready() || !o.recurrentState.Ready() {
			t.Fatal("capture refusal allocated, transferred, fenced, invalidated state or ended the caller's capture")
		}
	})

	for _, kinds := range [][2]Dtype{{Q4_K, Q4_K}, {Q6_K, Q4_K}, {Q4_K, Q6_K}, {Q6_K, Q6_K}, {F32, Q4_K}, {Q6_K, F32}, {F32, F32}, {Q8_0, Q8_0}} {
		for _, tokens := range []int{1, 4, 128} {
			t.Run(fmt.Sprintf("%s-%s-P%d", kinds[0], kinds[1], tokens), func(t *testing.T) {
				o := newCUDAGDNOperands(t, be, g)
				var qkv, z []float32
				var qkvRaw, zRaw []byte
				o.inQKV, qkv, qkvRaw = cudaGDNPackedWeight(t, be, kinds[0], g.convDim(), g.hidden)
				o.inZ, z, zRaw = cudaGDNPackedWeight(t, be, kinds[1], g.valueDim(), g.hidden)
				keepQKV, keepZ := bytes.Clone(qkvRaw), bytes.Clone(zRaw)
				inB, inA := be.Read(o.inB), be.Read(o.inA)
				if kinds[0] == Q8_0 && kinds[1] == Q8_0 {
					o.inB, inB, _ = cudaGDNPackedWeight(t, be, Q8_0, g.nV, g.hidden)
					o.inA, inA, _ = cudaGDNPackedWeight(t, be, Q8_0, g.nV, g.hidden)
				}
				oracle := qwen35GDNDecodeHostOracle{g: g, inQKV: qkv, inZ: z, inB: inB, inA: inA,
					convW: be.Read(o.convW), aLog: be.Read(o.aLog), dtBias: be.Read(o.dtBias), norm: be.Read(o.norm), outW: be.Read(o.outW),
					conv: make([]float64, (g.kernel-1)*g.convDim()), recurrent: make([]float64, g.nV*g.kHd*g.vHd)}
				input := make([]float32, tokens*g.hidden)
				for i := range input {
					// +/-127/128 gives exactly representable Q8_1 d=1/128,
					// codes +/-127, and an exact half block sum. This isolates
					// new dispatch/layout from activation-rounding policy.
					input[i] = 127.0 / 128
					if (i+i/g.hidden)%5 < 2 {
						input[i] = -input[i]
					}
				}
				shape := []int{tokens, g.hidden}
				if tokens == 1 {
					shape = []int{g.hidden}
				}
				o.x = uploadCUDAGDN(t, be, shape, input, MemoryActivation, "packed-gdn-input")
				var want []float64
				for token := 0; token < tokens; token++ {
					want = append(want, oracle.step(input[token*g.hidden:(token+1)*g.hidden])...)
				}
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
					t.Fatalf("real packed whole operation: %v", err)
				}
				t.Cleanup(func() { be.Free(out) })
				if conv.Buf() != o.convState.Buf() || state.Buf() != o.recurrentState.Buf() || !out.Ready() || !conv.Ready() || !state.Ready() {
					t.Fatal("whole operation lost caller state identity or completed readiness")
				}
				if be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
					t.Fatal("whole operation transferred operands or results through the host")
				}
				got := be.Read(out)
				requireFiniteNonzeroCUDA(t, "packed GDN output", got)
				requireQwen35GDNClose(t, "packed GDN output", tokens, got, want)
				requireQwen35GDNClose(t, "packed GDN conv", tokens, be.Read(conv), oracle.conv)
				requireQwen35GDNClose(t, "packed GDN recurrent", tokens, be.Read(state), oracle.recurrent)
				if !equalF32Bits(be.Read(o.x), input) || !bytes.Equal(qkvRaw, keepQKV) || !bytes.Equal(zRaw, keepZ) {
					t.Fatal("immutable input or packed host weight changed")
				}
			})
		}
	}

	t.Run("owned-preflight-refusals", func(t *testing.T) {
		o := newCUDAGDNOperands(t, be, g)
		o.inQKV, _, _ = cudaGDNPackedWeight(t, be, Q4_K, g.convDim(), g.hidden)
		bPacked, _, _ := cudaGDNPackedWeight(t, be, Q4_K, g.nV, g.hidden)
		badBlock := o.inQKV
		q := *badBlock.Quant
		q.Block = 32
		badBlock.Quant = &q // independent metadata; original owned operand stays unchanged
		cases := []cudaGDNOperands{o, o, o}
		cases[0].inQKV = badBlock
		cases[1].inB = bPacked
		cases[2].x = NewF32(Default(), []int{g.hidden}, make([]float32, g.hidden))
		beforeConv, beforeState := be.Read(o.convState), be.Read(o.recurrentState)
		for i, bad := range cases {
			beforeTransient, beforeLive := be.cudaAllocationCountsForTest()
			beforeFence := atomic.LoadUint64(&be.fenceGen)
			be.ResetHostXfer()
			be.ResetH2DXfer()
			out, conv, state, err := bad.decode(be, g)
			var geometry *Qwen35GDNGeometryError
			var residency *Qwen35GDNResidencyError
			if (!errors.As(err, &geometry) && !errors.As(err, &residency)) || out.Backend() != nil || conv.Backend() != nil || state.Backend() != nil {
				t.Fatalf("case%d not typed preflight refusal: %v", i, err)
			}
			afterTransient, afterLive := be.cudaAllocationCountsForTest()
			if beforeTransient != afterTransient || beforeLive != afterLive || atomic.LoadUint64(&be.fenceGen) != beforeFence || be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
				t.Fatalf("case%d preflight refusal had allocation/transfer/fence effects", i)
			}
		}
		if !equalF32Bits(be.Read(o.convState), beforeConv) || !equalF32Bits(be.Read(o.recurrentState), beforeState) {
			t.Fatal("refusal mutated caller state")
		}
	})

	// The existing deterministic launch-status seam runs after the first packed
	// projection; it is not an injected asynchronous device-fault proof.
	t.Run("first-projection-error", func(t *testing.T) {
		o := newCUDAGDNOperands(t, be, g)
		o.inQKV, _, _ = cudaGDNPackedWeight(t, be, Q4_K, g.convDim(), g.hidden)
		o.inZ, _, _ = cudaGDNPackedWeight(t, be, Q6_K, g.valueDim(), g.hidden)
		originalLatch := be.faultLatch
		be.faultLatch = NewDeviceFaultLatch("cuda", cudaFaultReconstructBudget)
		t.Cleanup(func() { be.faultLatch = originalLatch })
		t.Cleanup(func() { qwen35GDNInjectFaultForTest(0) })
		qwen35GDNInjectFaultForTest(2)
		out, conv, state, err := o.decode(be, g)
		var failed *Qwen35GDNKernelError
		if !errors.As(err, &failed) || failed.Stage != "fused-input-projections" || out.Backend() != nil || conv.Backend() != nil || state.Backend() != nil {
			t.Fatalf("packed projection error was not preserved: %v", err)
		}
		if o.convState.Ready() || o.recurrentState.Ready() || !be.faultLatch.Snapshot().Refusing() {
			t.Fatal("failed operation did not invalidate state and poison its session latch")
		}
	})
}

// Encode independent small signed dyadic values directly into canonical GGUF
// blocks. The float oracle is built from those values, not a device dequantizer.
func cudaGDNPackedWeight(t *testing.T, be *cudaBackend, dtype Dtype, out, in int) (Tensor, []float32, []byte) {
	t.Helper()
	values := make([]float32, out*in)
	codes := make([]int8, len(values))
	for row := 0; row < out; row++ {
		for col := 0; col < in; col++ {
			v := int8((col*5+row*3+col/32)%16 - 8)
			codes[row*in+col] = v
			values[row*in+col] = float32(v) / 128
		}
	}
	var host Tensor
	var raw []byte
	switch dtype {
	case F32:
		host = NewF32(Default(), []int{out, in}, values)
	case Q8_0:
		scales := make([]float32, len(values)/32)
		for i := range scales {
			scales[i] = 1.0 / 128
		}
		host = NewQ8(Default(), []int{out, in}, codes, scales, 32)
	case Q4_K, Q6_K:
		blockBytes := 144
		if dtype == Q6_K {
			blockBytes = 210
		}
		raw = make([]byte, len(values)/256*blockBytes)
		for block := 0; block < len(values)/256; block++ {
			b := raw[block*blockBytes : (block+1)*blockBytes]
			v := codes[block*256 : (block+1)*256]
			if dtype == Q4_K {
				binary.LittleEndian.PutUint16(b, 0x2000)     // d=1/128
				binary.LittleEndian.PutUint16(b[2:], 0x2000) // dmin=1/128
				for i := 0; i < 4; i++ {
					b[4+i], b[8+i], b[12+i] = 1, 8, 0x81 // scale1, min8
				}
				for group := 0; group < 4; group++ {
					for lane := 0; lane < 32; lane++ {
						b[16+group*32+lane] = byte(v[group*64+lane]+8) | byte(v[group*64+32+lane]+8)<<4
					}
				}
			} else {
				binary.LittleEndian.PutUint16(b[208:], 0x2000)
				for i := 192; i < 208; i++ {
					b[i] = 1
				}
				for half := 0; half < 2; half++ {
					for lane := 0; lane < 32; lane++ {
						a, c := byte(v[half*128+lane]+32), byte(v[half*128+64+lane]+32)
						d, e := byte(v[half*128+32+lane]+32), byte(v[half*128+96+lane]+32)
						b[half*64+lane] = a&15 | (c&15)<<4
						b[half*64+32+lane] = d&15 | (e&15)<<4
						b[128+half*32+lane] = a>>4 | (d>>4)<<2 | (c>>4)<<4 | (e>>4)<<6
					}
				}
			}
		}
		if dtype == Q4_K {
			host = NewQ4K(Default(), []int{out, in}, raw)
		} else {
			host = NewQ6K(Default(), []int{out, in}, raw)
		}
	default:
		t.Fatalf("unsupported fixture dtype %v", dtype)
	}
	resident := be.Upload(host, dtype)
	t.Cleanup(func() { be.Free(resident) })
	if resident.Dtype != dtype {
		t.Fatalf("Upload changed requested dtype %v to %v", dtype, resident.Dtype)
	}
	if raw != nil {
		buf, ok := resident.buf.(*cudaBuf)
		if !ok || buf.ptr == nil || buf.n != len(raw) || buf.scales != nil || buf.managed || buf.device != 0 {
			t.Fatal("packed Upload did not retain exact narrow, device-only storage")
		}
	}
	return resident, values, raw
}
