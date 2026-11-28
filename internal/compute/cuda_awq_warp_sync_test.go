// fak-test:runtime fast est=1s lane=default
package compute

import (
	"bytes"
	"math"
	"testing"
)

// dequantAWQWithGIdxOracle decodes packed AWQ [out, in/2] nibbles (zero-point=8)
// into dense float32 [out, in] matching internal/compute AWQ conventions plus optional g_idx.
func dequantAWQWithGIdxOracle(packed []byte, scales []float32, gIdx []int32, outDim, inDim int) []float32 {
	out := make([]float32, outDim*inDim)
	perChannel := len(scales) == outDim
	nGroups := 1
	if !perChannel {
		nGroups = len(scales) / outDim
	}
	half := inDim / 2
	for r := 0; r < outDim; r++ {
		for col := 0; col < inDim; col++ {
			b := packed[r*half+col/2]
			var q byte
			if col%2 == 0 {
				q = b & 0x0f
			} else {
				q = (b >> 4) & 0x0f
			}
			var sc float32
			if perChannel {
				sc = scales[r]
			} else {
				sc = scales[r*nGroups+int(gIdx[col])]
			}
			out[r*inDim+col] = float32(int(q)-8) * sc
		}
	}
	return out
}

func makeAWQTestFixture(outDim, inDim, nGroups int, seed uint64) ([]byte, []float32, []float32, []int32) {
	rng := lcg(seed)
	packed := make([]byte, outDim*(inDim/2))
	for r := 0; r < outDim; r++ {
		for k := 0; k < inDim/2; k++ {
			lo := byte(((r * 3) + k) % 16)
			hi := byte(((r * 5) + k*7 + 3) % 16)
			packed[r*(inDim/2)+k] = lo | (hi << 4)
		}
	}
	scales := make([]float32, outDim*nGroups)
	for i := range scales {
		scales[i] = 0.02 + 0.04*float32((i*11+3)%7)
	}
	gIdx := make([]int32, inDim)
	for c := 0; c < inDim; c++ {
		// Scrambled act-order group mapping across columns.
		gIdx[c] = int32((c*5 + 3) % nGroups)
	}
	_ = rng
	return packed, scales, randVec(&rng, 1), gIdx
}

func TestCudaAwqWarpSyncSemaphoreLifecycleAndBounds(t *testing.T) {
	if _, err := NewAWQTileSemaphore(1, 32); err == nil {
		t.Fatal("expected error when stages < 2")
	}
	if _, err := NewAWQTileSemaphore(3, 31); err == nil {
		t.Fatal("expected error when tileK is odd")
	}
	sem, err := NewAWQTileSemaphore(3, 32)
	if err != nil {
		t.Fatalf("NewAWQTileSemaphore: %v", err)
	}
	if sem.EmptyCount() != 3 || sem.FilledCount() != 0 {
		t.Fatalf("unexpected initial counts empty=%d filled=%d", sem.EmptyCount(), sem.FilledCount())
	}

	// Fill all 3 stages without consumer drain -> 4th producer acquire refuses rather than overflowing.
	slots := make([]int, 3)
	for i := 0; i < 3; i++ {
		slot, err := sem.AcquireEmpty()
		if err != nil {
			t.Fatalf("AcquireEmpty[%d]: %v", i, err)
		}
		slots[i] = slot
		if err := sem.ReleaseFilled(slot); err != nil {
			t.Fatalf("ReleaseFilled[%d]: %v", i, err)
		}
		if err := sem.CheckInvariants(); err != nil {
			t.Fatalf("CheckInvariants[%d]: %v", i, err)
		}
	}
	if _, err := sem.AcquireEmpty(); err == nil {
		t.Fatal("expected producer stall error when all stages are filled")
	}

	// Drain in FIFO slot order.
	for i := 0; i < 3; i++ {
		slot, err := sem.AcquireFilled()
		if err != nil {
			t.Fatalf("AcquireFilled[%d]: %v", i, err)
		}
		if slot != slots[i] {
			t.Fatalf("consumer slot mismatch got %d want %d", slot, slots[i])
		}
		if err := sem.ReleaseEmpty(slot); err != nil {
			t.Fatalf("ReleaseEmpty[%d]: %v", i, err)
		}
		if err := sem.CheckInvariants(); err != nil {
			t.Fatalf("CheckInvariants after release[%d]: %v", i, err)
		}
	}
	if _, err := sem.AcquireFilled(); err == nil {
		t.Fatal("expected consumer stall error when pipeline is empty")
	}
	if sem.TotalProduced() != 3 || sem.TotalConsumed() != 3 {
		t.Fatalf("unexpected totals produced=%d consumed=%d", sem.TotalProduced(), sem.TotalConsumed())
	}
}

func TestCudaAwqWarpSyncMatchesCPUOracleIdentityAndActOrder(t *testing.T) {
	be := Default()
	const (
		outDim = 24
		inDim  = 64
		tileK  = 16
		stages = 3
	)

	cases := []struct {
		name    string
		rows    int
		nGroups int
		actOrd  bool
	}{
		{name: "identity_gemv_per_channel", rows: 1, nGroups: 1, actOrd: false},
		{name: "identity_gemm_per_channel", rows: 4, nGroups: 1, actOrd: false},
		{name: "actorder_gemv_4groups", rows: 1, nGroups: 4, actOrd: true},
		{name: "actorder_gemm_8groups", rows: 5, nGroups: 8, actOrd: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := lcg(42 + uint64(tc.rows*17+tc.nGroups))
			packed, scales, _, gIdx := makeAWQTestFixture(outDim, inDim, tc.nGroups, 9001)
			if !tc.actOrd {
				gIdx = nil
			}
			xData := randVec(&rng, tc.rows*inDim)

			remap, err := BuildAWQColumnRemap(inDim, gIdx)
			if err != nil {
				t.Fatalf("BuildAWQColumnRemap: %v", err)
			}
			if tc.actOrd && remap.IsIdentityPerm {
				t.Fatal("expected non-identity column_remap for scrambled act-order g_idx")
			}

			// Verify packed bytes remain completely recoverable through remap -> unremap.
			remapped, err := RemapAWQPackedWeights(packed, outDim, remap)
			if err != nil {
				t.Fatalf("RemapAWQPackedWeights: %v", err)
			}
			recovered, err := UnremapAWQPackedWeights(remapped, outDim, remap)
			if err != nil {
				t.Fatalf("UnremapAWQPackedWeights: %v", err)
			}
			if !bytes.Equal(recovered, packed) {
				t.Fatal("unremap did not recover exact original AWQ packed bytes")
			}

			// Verify PermutedGIdx is non-decreasing (group contiguous).
			for i := 1; i < len(remap.PermutedGIdx); i++ {
				if remap.PermutedGIdx[i] < remap.PermutedGIdx[i-1] {
					t.Fatalf("permuted g_idx not monotonic at %d: %d < %d", i, remap.PermutedGIdx[i], remap.PermutedGIdx[i-1])
				}
			}

			got, sem, err := AWQWarpSyncGemm(packed, scales, xData, tc.rows, outDim, remap, stages, tileK)
			if err != nil {
				t.Fatalf("AWQWarpSyncGemm: %v", err)
			}
			if err := sem.CheckInvariants(); err != nil {
				t.Fatalf("final semaphore invariant check: %v", err)
			}
			if sem.TotalProduced() != sem.TotalConsumed() || sem.EmptyCount() != stages {
				t.Fatalf("pipeline not cleanly drained: produced=%d consumed=%d empty=%d",
					sem.TotalProduced(), sem.TotalConsumed(), sem.EmptyCount())
			}

			// Host oracle via existing cpuref MatMul / BatchedMatMul over un-permuted dense AWQ dequant.
			denseW := dequantAWQWithGIdxOracle(packed, scales, remap.OrigGIdx, outDim, inDim)
			wTensor := NewF32(be, []int{outDim, inDim}, denseW)
			var want []float32
			if tc.rows == 1 {
				xTensor := NewF32(be, []int{inDim}, xData)
				want = be.Read(be.MatMul(wTensor, xTensor))
			} else {
				xTensor := NewF32(be, []int{tc.rows, inDim}, xData)
				want = be.Read(be.BatchedMatMul(wTensor, xTensor, tc.rows))
			}

			if len(got) != len(want) {
				t.Fatalf("len mismatch got %d want %d", len(got), len(want))
			}
			var maxDiff float64
			for i := range want {
				d := math.Abs(float64(got[i] - want[i]))
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff > 1e-4 {
				t.Fatalf("cpuref parity mismatch: maxDiff=%g (> 1e-4)", maxDiff)
			}
			if cos := cosine(got, want); cos < 0.99999 {
				t.Fatalf("cosine=%f (< 0.99999)", cos)
			}
		})
	}
}
