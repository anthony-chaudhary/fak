package compute

import (
	"math"
	"os"
	"strings"
	"testing"
)

func runSerialPrefillLoop(t *testing.T, be Backend, X, Wq, Wk, Wv, Wo Tensor, kv KVStore, layer, startPos, nH, nKV, hd int, theta float64, scale float32, P, D int) []float32 {
	grp := nH / nKV
	if scale <= 0 {
		scale = float32(1.0 / math.Sqrt(float64(hd)))
	}
	Xf := be.Read(X)
	var serialOut []float32

	for tok := 0; tok < P; tok++ {
		pos := startPos + tok
		xt := Xf[tok*D : (tok+1)*D]
		xTen := NewF32(Default(), []int{D}, xt)
		if be.Name() != "cpu-ref" {
			xTen = be.Upload(xTen, F32)
		}

		q := be.MatMul(Wq, xTen)
		kRaw := be.MatMul(Wk, xTen)
		v := be.MatMul(Wv, xTen)

		qRoped := be.RoPE(q, pos, nH, hd, theta)
		kRoped := be.RoPE(kRaw, pos, nKV, hd, theta)

		if kv != nil {
			kv.AppendKV(layer, kRaw, kRoped, v, pos)
		}

		attnOut := be.Attention(qRoped, kv, layer, true, grp, scale)

		var outToken Tensor
		if Wo.buf != nil {
			outToken = be.MatMul(Wo, attnOut)
		} else {
			outToken = attnOut
		}
		serialOut = append(serialOut, be.Read(outToken)...)
	}
	return serialOut
}

func TestBatchedPrefillMatchesSerialLoop(t *testing.T) {
	be := Default() // cpu-ref
	bpb, ok := be.(BatchedPrefillBackend)
	if !ok {
		t.Fatalf("Default backend %s does not implement BatchedPrefillBackend", be.Name())
	}
	if !be.Caps().BatchedPrefill {
		t.Fatalf("Default backend %s does not advertise BatchedPrefill in Caps", be.Name())
	}

	testCases := []struct {
		name      string
		P         int
		D         int
		nH, nKV   int
		hd        int
		startPos  int
		withWo    bool
		prefillKV int
	}{
		{name: "P=1 single-token", P: 1, D: 32, nH: 4, nKV: 2, hd: 8, startPos: 0, withWo: true},
		{name: "P=4 MHA with Wo", P: 4, D: 32, nH: 4, nKV: 4, hd: 8, startPos: 0, withWo: true},
		{name: "P=8 GQA with Wo", P: 8, D: 64, nH: 4, nKV: 2, hd: 16, startPos: 0, withWo: true},
		{name: "P=16 MQA with Wo", P: 16, D: 64, nH: 4, nKV: 1, hd: 16, startPos: 0, withWo: true},
		{name: "P=8 GQA without Wo", P: 8, D: 64, nH: 4, nKV: 2, hd: 16, startPos: 0, withWo: false},
		{name: "P=8 with non-zero startPos", P: 8, D: 64, nH: 4, nKV: 2, hd: 16, startPos: 10, withWo: true, prefillKV: 10},
		{name: "P=32 large prompt panel", P: 32, D: 64, nH: 8, nKV: 2, hd: 16, startPos: 0, withWo: true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var rng lcg = lcg(123456789 + uint64(tc.P)*31 + uint64(tc.D))
			P := tc.P
			D := tc.D
			nH := tc.nH
			nKV := tc.nKV
			hd := tc.hd
			qOut := nH * hd
			kvOut := nKV * hd
			theta := 10000.0
			scale := float32(1.0 / math.Sqrt(float64(hd)))

			xData := randVec(&rng, P*D)
			wqData := randVec(&rng, qOut*D)
			wkData := randVec(&rng, kvOut*D)
			wvData := randVec(&rng, kvOut*D)
			var woData []float32
			if tc.withWo {
				woData = randVec(&rng, D*qOut)
			}

			xTen := NewF32(be, []int{P, D}, xData)
			wqTen := NewF32(be, []int{qOut, D}, wqData)
			wkTen := NewF32(be, []int{kvOut, D}, wkData)
			wvTen := NewF32(be, []int{kvOut, D}, wvData)
			var woTen Tensor
			if tc.withWo {
				woTen = NewF32(be, []int{D, qOut}, woData)
			}

			kvCfg := KVConfig{
				NumLayers:  1,
				NumKVHeads: nKV,
				HeadDim:    hd,
				RopeTheta:  theta,
			}
			kvSerial := be.NewKV(kvCfg)
			kvBatched := be.NewKV(kvCfg)

			// If test needs pre-filled KV positions before prefill:
			if tc.prefillKV > 0 {
				for p := 0; p < tc.prefillKV; p++ {
					kRaw := randVec(&rng, kvOut)
					kRoPE := randVec(&rng, kvOut)
					v := randVec(&rng, kvOut)
					kvSerial.AppendKV(0, NewF32(be, []int{kvOut}, kRaw), NewF32(be, []int{kvOut}, kRoPE), NewF32(be, []int{kvOut}, v), p)
					kvBatched.AppendKV(0, NewF32(be, []int{kvOut}, kRaw), NewF32(be, []int{kvOut}, kRoPE), NewF32(be, []int{kvOut}, v), p)
				}
			}

			// 1. Run serial single-token loop
			serialOut := runSerialPrefillLoop(t, be, xTen, wqTen, wkTen, wvTen, woTen, kvSerial, 0, tc.startPos, nH, nKV, hd, theta, scale, P, D)

			// 2. Run batched prefill pass
			batchArgs := PrefillBatchArgs{
				X:          xTen,
				Wq:         wqTen,
				Wk:         wkTen,
				Wv:         wvTen,
				Wo:         woTen,
				KV:         kvBatched,
				Layer:      0,
				StartPos:   tc.startPos,
				NumHeads:   nH,
				NumKVHeads: nKV,
				HeadDim:    hd,
				RopeTheta:  theta,
				Scale:      scale,
			}
			batchRes, err := bpb.PrefillBatch(batchArgs)
			if err != nil {
				t.Fatalf("PrefillBatch failed: %v", err)
			}
			if batchRes.Tokens != P {
				t.Fatalf("batchRes.Tokens = %d, want %d", batchRes.Tokens, P)
			}

			batchedOut := be.Read(batchRes.Output)
			if len(batchedOut) != len(serialOut) {
				t.Fatalf("output length mismatch: batched=%d serial=%d", len(batchedOut), len(serialOut))
			}

			// Numerical equivalence: cosine similarity >= 0.999, maxDelta < 1e-5
			cos := cosine(serialOut, batchedOut)
			maxD := maxAbsDelta(serialOut, batchedOut)
			if cos < 0.999 {
				t.Fatalf("cosine similarity %.8f < 0.999 (maxDelta=%.2e)", cos, maxD)
			}
			if maxD >= 1e-5 {
				t.Fatalf("maxDelta %.2e >= 1e-5 (cosine=%.8f)", maxD, cos)
			}

			// KV store equivalence: verify both K and V caches match exactly
			if kvSerial.Len() != kvBatched.Len() {
				t.Fatalf("KV cache length mismatch: serial=%d batched=%d", kvSerial.Len(), kvBatched.Len())
			}
			kSerial := be.Read(kvSerial.KeysView(0))
			kBatched := be.Read(kvBatched.KeysView(0))
			if cosK := cosine(kSerial, kBatched); cosK < 0.999 {
				t.Fatalf("KV KeysView cosine %.8f < 0.999", cosK)
			}
			if maxDK := maxAbsDelta(kSerial, kBatched); maxDK >= 1e-5 {
				t.Fatalf("KV KeysView maxDelta %.2e >= 1e-5", maxDK)
			}

			vSerial := be.Read(kvSerial.ValuesView(0))
			vBatched := be.Read(kvBatched.ValuesView(0))
			if cosV := cosine(vSerial, vBatched); cosV < 0.999 {
				t.Fatalf("KV ValuesView cosine %.8f < 0.999", cosV)
			}
			if maxDV := maxAbsDelta(vSerial, vBatched); maxDV >= 1e-5 {
				t.Fatalf("KV ValuesView maxDelta %.2e >= 1e-5", maxDV)
			}

			t.Logf("PASS %-28s: P=%2d D=%2d cos=%.9f maxDelta=%.2e", tc.name, P, D, cos, maxD)
		})
	}
}

type spyCountingBackend struct {
	Backend
	serialMatMulCount   int
	serialAttnCount     int
	serialRoPECount     int
	batchedPrefillCount int
}

func (s *spyCountingBackend) MatMul(w, x Tensor) Tensor {
	s.serialMatMulCount++
	return s.Backend.MatMul(w, x)
}

func (s *spyCountingBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	s.serialAttnCount++
	return s.Backend.Attention(q, kv, layer, causal, grp, scale)
}

func (s *spyCountingBackend) RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	s.serialRoPECount++
	return s.Backend.RoPE(x, pos, nHeads, headDim, theta)
}

func (s *spyCountingBackend) PrefillBatch(args PrefillBatchArgs) (PrefillBatchResult, error) {
	s.batchedPrefillCount++
	bpb := s.Backend.(BatchedPrefillBackend)
	return bpb.PrefillBatch(args)
}

// TestBatchedPrefillDispatchCount proves that batched prefill performs an O(1)
// single-dispatch pass instead of P serial single-token loops.
func TestBatchedPrefillDispatchCount(t *testing.T) {
	ref := Default()
	promptLengths := []int{4, 8, 16, 32}

	for _, P := range promptLengths {
		var rng lcg = lcg(998877 + uint64(P))
		D := 32
		nH := 4
		nKV := 2
		hd := 8
		qOut := nH * hd
		kvOut := nKV * hd
		theta := 10000.0
		scale := float32(0.25)

		xData := randVec(&rng, P*D)
		wqData := randVec(&rng, qOut*D)
		wkData := randVec(&rng, kvOut*D)
		wvData := randVec(&rng, kvOut*D)
		woData := randVec(&rng, D*qOut)

		xTen := NewF32(ref, []int{P, D}, xData)
		wqTen := NewF32(ref, []int{qOut, D}, wqData)
		wkTen := NewF32(ref, []int{kvOut, D}, wkData)
		wvTen := NewF32(ref, []int{kvOut, D}, wvData)
		woTen := NewF32(ref, []int{D, qOut}, woData)

		kvCfg := KVConfig{NumLayers: 1, NumKVHeads: nKV, HeadDim: hd, RopeTheta: theta}

		// 1. Serial 1-token loop execution
		serialSpy := &spyCountingBackend{Backend: ref}
		kvSerial := serialSpy.NewKV(kvCfg)
		_ = runSerialPrefillLoop(t, serialSpy, xTen, wqTen, wkTen, wvTen, woTen, kvSerial, 0, 0, nH, nKV, hd, theta, scale, P, D)

		totalSerialOps := serialSpy.serialMatMulCount + serialSpy.serialAttnCount + serialSpy.serialRoPECount
		expectedSerialOps := P * 7 // 4 MatMul + 2 RoPE + 1 Attention per token
		if totalSerialOps != expectedSerialOps {
			t.Fatalf("P=%d: serial ops = %d, want %d", P, totalSerialOps, expectedSerialOps)
		}

		// 2. Batched prefill execution
		batchedSpy := &spyCountingBackend{Backend: ref}
		kvBatched := batchedSpy.NewKV(kvCfg)
		batchArgs := PrefillBatchArgs{
			X:          xTen,
			Wq:         wqTen,
			Wk:         wkTen,
			Wv:         wvTen,
			Wo:         woTen,
			KV:         kvBatched,
			Layer:      0,
			StartPos:   0,
			NumHeads:   nH,
			NumKVHeads: nKV,
			HeadDim:    hd,
			RopeTheta:  theta,
			Scale:      scale,
		}
		_, err := batchedSpy.PrefillBatch(batchArgs)
		if err != nil {
			t.Fatalf("P=%d: PrefillBatch failed: %v", P, err)
		}

		// Batched dispatch count must be O(1): exactly 1 invocation of PrefillBatch
		if batchedSpy.batchedPrefillCount != 1 {
			t.Fatalf("P=%d: batched dispatch count = %d, want 1 (O(1) dispatch pass)", P, batchedSpy.batchedPrefillCount)
		}

		t.Logf("PROVE O(1) vs O(P): P=%2d => Serial ops = %3d (%d*P) | Batched prefill dispatches = %d (O(1))",
			P, totalSerialOps, totalSerialOps/P, batchedSpy.batchedPrefillCount)
	}
}

// TestMetalBatchedPrefillParity tests the Metal GPU backend batched prefill implementation
// when Metal is available, holding it to cosine >= 0.999 vs cpuref.
func TestMetalBatchedPrefillParity(t *testing.T) {
	metalDev, ok := Lookup("metal")
	if !ok || metalDev == nil {
		t.Skip("Metal backend unavailable on this machine; skipping")
	}
	bpb, ok := metalDev.(BatchedPrefillBackend)
	if !ok {
		t.Fatalf("Metal backend does not implement BatchedPrefillBackend")
	}
	if !metalDev.Caps().BatchedPrefill {
		t.Fatalf("Metal backend does not advertise BatchedPrefill in Caps")
	}

	ref := Default()
	P := 8
	D := 32
	nH := 4
	nKV := 2
	hd := 8
	qOut := nH * hd
	kvOut := nKV * hd
	theta := 10000.0
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	var rng lcg = 42
	xData := randVec(&rng, P*D)
	wqData := randVec(&rng, qOut*D)
	wkData := randVec(&rng, kvOut*D)
	wvData := randVec(&rng, kvOut*D)
	woData := randVec(&rng, D*qOut)

	xTen := NewF32(ref, []int{P, D}, xData)
	wqTen := NewF32(ref, []int{qOut, D}, wqData)
	wkTen := NewF32(ref, []int{kvOut, D}, wkData)
	wvTen := NewF32(ref, []int{kvOut, D}, wvData)
	woTen := NewF32(ref, []int{D, qOut}, woData)

	kvCfg := KVConfig{NumLayers: 1, NumKVHeads: nKV, HeadDim: hd, RopeTheta: theta}
	kvRef := ref.NewKV(kvCfg)
	kvMetal := metalDev.NewKV(kvCfg)

	refArgs := PrefillBatchArgs{
		X: xTen, Wq: wqTen, Wk: wkTen, Wv: wvTen, Wo: woTen,
		KV: kvRef, Layer: 0, StartPos: 0, NumHeads: nH, NumKVHeads: nKV, HeadDim: hd,
		RopeTheta: theta, Scale: scale,
	}
	refRes, err := PrefillBatch(ref, refArgs)
	if err != nil {
		t.Fatalf("ref PrefillBatch failed: %v", err)
	}

	metalArgs := PrefillBatchArgs{
		X: xTen, Wq: wqTen, Wk: wkTen, Wv: wvTen, Wo: woTen,
		KV: kvMetal, Layer: 0, StartPos: 0, NumHeads: nH, NumKVHeads: nKV, HeadDim: hd,
		RopeTheta: theta, Scale: scale,
	}
	metalRes, err := bpb.PrefillBatch(metalArgs)
	if err != nil {
		t.Fatalf("metal PrefillBatch failed: %v", err)
	}

	refOut := ref.Read(refRes.Output)
	metalOut := metalDev.Read(metalRes.Output)

	cos := cosine(refOut, metalOut)
	maxD := maxAbsDelta(refOut, metalOut)
	if cos < 0.999 {
		t.Fatalf("Metal vs CPU ref prefill cosine %.6f < 0.999 (maxDelta=%.2e)", cos, maxD)
	}
	t.Logf("PASS Metal BatchedPrefill: cosine=%.8f maxDelta=%.2e", cos, maxD)
}

// TestCUDAPrefillBatchSourceContract verifies source and symbol alignment for CUDA
// batched prompt prefill without requiring an active CUDA device.
func TestCUDAPrefillBatchSourceContract(t *testing.T) {
	cudaGo, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatalf("failed to read cuda.go: %v", err)
	}
	if !strings.Contains(string(cudaGo), "func (c *cudaBackend) PrefillBatch") {
		t.Errorf("cuda.go missing PrefillBatch method implementation")
	}

	prefillCUDA, err := os.ReadFile("prefill_cuda.go")
	if err != nil {
		t.Fatalf("failed to read prefill_cuda.go: %v", err)
	}
	if !strings.Contains(string(prefillCUDA), "var _ BatchedPrefillBackend = (*cudaBackend)(nil)") {
		t.Errorf("prefill_cuda.go missing BatchedPrefillBackend interface assertion")
	}

	cudaState, err := os.ReadFile("cuda_backend_state.go")
	if err != nil {
		t.Fatalf("failed to read cuda_backend_state.go: %v", err)
	}
	if !strings.Contains(string(cudaState), "BatchedPrefill: true") {
		t.Errorf("cuda_backend_state.go Caps() missing BatchedPrefill: true")
	}
}

// TestPrefillBatchValidation verifies bounds checking, dimension validation, and fail-closed errors.
func TestPrefillBatchValidation(t *testing.T) {
	ref := Default()
	validX := NewF32(ref, []int{4, 16}, make([]float32, 64))
	validWq := NewF32(ref, []int{16, 16}, make([]float32, 256))
	validWk := NewF32(ref, []int{8, 16}, make([]float32, 128))
	validWv := NewF32(ref, []int{8, 16}, make([]float32, 128))

	baseArgs := func() PrefillBatchArgs {
		return PrefillBatchArgs{
			X:          validX,
			Wq:         validWq,
			Wk:         validWk,
			Wv:         validWv,
			NumHeads:   4,
			NumKVHeads: 2,
			HeadDim:    4,
			StartPos:   0,
		}
	}

	if _, err := PrefillBatch(ref, PrefillBatchArgs{}); err == nil {
		t.Error("expected error for empty args")
	}

	// Unallocated X
	args := baseArgs()
	args.X = Tensor{}
	if _, err := PrefillBatch(ref, args); err == nil {
		t.Error("expected error for empty X tensor")
	}

	// Missing Wq
	args = baseArgs()
	args.Wq = Tensor{}
	if _, err := PrefillBatch(ref, args); err == nil {
		t.Error("expected error for empty Wq tensor")
	}

	// Negative StartPos
	args = baseArgs()
	args.StartPos = -1
	if _, err := PrefillBatch(ref, args); err == nil {
		t.Error("expected error for negative StartPos")
	}

	// Indivisible heads
	args = baseArgs()
	args.NumHeads = 5
	args.NumKVHeads = 2
	if _, err := PrefillBatch(ref, args); err == nil {
		t.Error("expected error for indivisible NumHeads % NumKVHeads")
	}

	// Invalid dimension mismatch
	args = baseArgs()
	args.Wq = NewF32(ref, []int{15, 16}, make([]float32, 240)) // wrong shape
	if _, err := PrefillBatch(ref, args); err == nil {
		t.Error("expected error for Wq shape mismatch")
	}
}
