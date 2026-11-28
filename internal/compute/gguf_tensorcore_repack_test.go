// fak-test:runtime fast est=1s lane=default
package compute

import (
	"bytes"
	"math"
	"testing"
)

func TestCudaTensorcoreRepackIdempotentRecoveryAndCPURefParity(t *testing.T) {
	be := Default()

	t.Run("Q4_K_repack_idempotent_lossless_and_matmul_parity", func(t *testing.T) {
		const (
			outDim = 22 // Intentional non-multiple of TileOut=16 to exercise tail padding cleanly.
			inDim  = 512
			rows   = 3
		)
		raw := makeDeterministicQ4KRaw(outDim, inDim, 31337)
		origSnapshot := append([]byte(nil), raw...)

		repacked, err := RepackQ4KTensorOffline(raw, outDim, inDim)
		if err != nil {
			t.Fatalf("RepackQ4KTensorOffline: %v", err)
		}
		if !bytes.Equal(raw, origSnapshot) {
			t.Fatal("on-disk GGUF Q4_K bytes were modified by offline repack")
		}

		// Idempotent guard: repeated RepackOnce leaves tiled bytes and RepackCount untouched.
		tiledSnapshot := append([]byte(nil), repacked.TiledBlocks...)
		if err := repacked.RepackOnce(); err != nil {
			t.Fatalf("RepackOnce #1: %v", err)
		}
		if err := repacked.RepackOnce(); err != nil {
			t.Fatalf("RepackOnce #2: %v", err)
		}
		if repacked.RepackCount != 1 || !bytes.Equal(repacked.TiledBlocks, tiledSnapshot) {
			t.Fatalf("RepackOnce was not idempotent: count=%d", repacked.RepackCount)
		}

		// Exact recovery of original packed GGUF bytes.
		recovered, err := repacked.UnpackRaw()
		if err != nil {
			t.Fatalf("UnpackRaw: %v", err)
		}
		if !bytes.Equal(recovered, origSnapshot) {
			t.Fatal("UnpackRaw did not recover exact original Q4_K bytes")
		}

		// Direct repacked matmul parity vs un-repacked cpuref Q4_K path.
		rng := lcg(9999)
		xData := randVec(&rng, rows*inDim)
		got, err := repacked.MatMul(xData, rows)
		if err != nil {
			t.Fatalf("repacked.MatMul: %v", err)
		}
		wTensor := NewQ4K(be, []int{outDim, inDim}, raw)
		xTensor := NewF32(be, []int{rows, inDim}, xData)
		want := be.Read(be.BatchedMatMul(wTensor, xTensor, rows))

		var maxDiff float64
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			if d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-4 {
			t.Fatalf("repacked Q4_K matmul vs cpuref maxDiff=%g (> 1e-4)", maxDiff)
		}
		if cos := cosine(got, want); cos < 0.99999 {
			t.Fatalf("repacked Q4_K cosine=%f", cos)
		}
	})

	t.Run("AWQ_repack_idempotent_lossless_and_matmul_parity", func(t *testing.T) {
		const (
			outDim = 20
			inDim  = 128
			rows   = 4
		)
		packed, scales, _, _ := makeAWQTestFixture(outDim, inDim, 1, 2026)
		packedSnapshot := append([]byte(nil), packed...)

		repacked, err := RepackAWQTensorOffline(packed, scales, outDim, inDim)
		if err != nil {
			t.Fatalf("RepackAWQTensorOffline: %v", err)
		}
		if !bytes.Equal(packed, packedSnapshot) {
			t.Fatal("original AWQ packed slice mutated by offline repack")
		}

		if err := repacked.RepackOnce(); err != nil {
			t.Fatalf("AWQ RepackOnce: %v", err)
		}
		recPacked, recScales, err := repacked.UnpackRaw()
		if err != nil {
			t.Fatalf("AWQ UnpackRaw: %v", err)
		}
		if !bytes.Equal(recPacked, packedSnapshot) {
			t.Fatal("AWQ UnpackRaw packed mismatch")
		}
		for i := range scales {
			if recScales[i] != scales[i] {
				t.Fatalf("AWQ UnpackRaw scale[%d]=%g want %g", i, recScales[i], scales[i])
			}
		}

		rng := lcg(4242)
		xData := randVec(&rng, rows*inDim)
		got, err := repacked.MatMul(xData, rows)
		if err != nil {
			t.Fatalf("repacked AWQ MatMul: %v", err)
		}
		denseW := dequantAWQWithGIdxOracle(packed, scales, nil, outDim, inDim)
		want := be.Read(be.BatchedMatMul(NewF32(be, []int{outDim, inDim}, denseW), NewF32(be, []int{rows, inDim}, xData), rows))

		var maxDiff float64
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			if d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-4 {
			t.Fatalf("repacked AWQ matmul vs cpuref maxDiff=%g (> 1e-4)", maxDiff)
		}
	})
}
