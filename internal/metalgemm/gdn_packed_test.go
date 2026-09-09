//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
	"time"
)

// TestQwenGDNPackedBTreeKernel_Parity verifies parity between the packed 8-row
// B-tree SIMDgroup kernel and the CPU reference oracleGDNRun for D_k=128, D_v=128.
// Enforces cosine similarity >= 0.999999 and max abs difference < 0.0001 for
// both output and recurrent state across multi-step sequences.
func TestQwenGDNPackedBTreeKernel_Parity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}

	g := oracleGDNGeometry{nK: 2, nV: 8, kHd: 128, vHd: 128, kernel: 3}
	if !IsGDNPackedBTreeEligible(g.kHd, g.vHd) {
		t.Fatalf("geometry %+v not recognized as eligible for packed B-tree kernel", g)
	}

	native, err := NewGDNState(nativeGDNGeometry(g))
	if err != nil {
		t.Fatalf("NewGDNState failed: %v", err)
	}
	defer native.Close()

	oracleState := newOracleGDNState(g)

	steps := []struct {
		tokens int
		phase  float32
	}{
		{tokens: 4, phase: 0.1},
		{tokens: 2, phase: 0.7},
		{tokens: 1, phase: 1.3},
	}

	for i, step := range steps {
		panel := oracleGDNFixture(g, step.tokens, step.phase)
		wantOut := oracleGDNRun(g, panel, oracleState)

		gotOut, accounting, accepted, runErr := native.Run(nativeGDNPanel(panel))
		if !accepted || runErr != nil {
			t.Fatalf("step %d: Run failed: accepted=%v, err=%v", i, accepted, runErr)
		}
		requireGDNAccounting(t, accounting, 1)

		outCos, outMaxAbs := gdnCosineMaxAbs(wantOut, gotOut)
		if outCos < gdnCosineFloor || outMaxAbs > gdnMaxAbsLimit {
			t.Fatalf("step %d: output parity failed: cosine=%.9f (want >= %.6f), maxAbs=%g (want <= %g)",
				i, outCos, gdnCosineFloor, outMaxAbs, gdnMaxAbsLimit)
		}
		t.Logf("step %d (tokens=%d): output parity cosine=%.9f, maxAbs=%g", i, step.tokens, outCos, outMaxAbs)

		_, recurrent, snapErr := native.Snapshot()
		if snapErr != nil {
			t.Fatalf("step %d: Snapshot failed: %v", i, snapErr)
		}

		recCos, recMaxAbs := gdnCosineMaxAbs(oracleState.recurrent, recurrent)
		if recCos < gdnCosineFloor || recMaxAbs > gdnMaxAbsLimit {
			t.Fatalf("step %d: recurrent state parity failed: cosine=%.9f (want >= %.6f), maxAbs=%g (want <= %g)",
				i, recCos, gdnCosineFloor, recMaxAbs, gdnMaxAbsLimit)
		}
		t.Logf("step %d: recurrent state parity cosine=%.9f, maxAbs=%g", i, recCos, recMaxAbs)
	}

	// Verify reset zeroes state cleanly.
	if err := native.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}
	_, resetRecurrent, err := native.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot after reset failed: %v", err)
	}
	for idx, val := range resetRecurrent {
		if val != 0 {
			t.Fatalf("recurrent[%d] = %g after Reset, want 0", idx, val)
		}
	}
}

// TestQwenGDNPackedButterflyShuffleCorrectness verifies that 4-lane intra-row
// butterfly shuffles correctly reduce 128 elements without lane leakage across rows.
// Tests both algorithmic invariants and physical on-device SIMDgroup execution.
func TestQwenGDNPackedButterflyShuffleCorrectness(t *testing.T) {
	const numRows = 8
	const lanesPerRow = 4
	const elemsPerLane = 32
	const elemsPerRow = lanesPerRow * elemsPerLane // 128 = D_k

	// 1. Algorithmic verification: 8 rows x 128 elements.
	// Lane l holds 32 elements. Local reduction gives laneSum[l].
	// Butterfly shuffle with XOR 1 and XOR 2 must yield exactly the sum of row l/4 in all 4 lanes.
	rowCoeffs := []float64{10.0, 25.0, 42.0, 77.0, 105.0, 150.0, 210.0, 300.0}
	data := make([][elemsPerRow]float64, numRows)
	rowTrueSums := make([]float64, numRows)

	for r := 0; r < numRows; r++ {
		for k := 0; k < elemsPerRow; k++ {
			val := rowCoeffs[r] + math.Sin(float64(k+1)*0.43)*2.5
			data[r][k] = val
			rowTrueSums[r] += val
		}
	}

	// Local reduction for each lane
	laneValues := make([]float64, 32)
	for l := 0; l < 32; l++ {
		row := l / lanesPerRow
		sub := l % lanesPerRow
		laneSum := 0.0
		for c := 0; c < elemsPerLane; c++ {
			laneSum += data[row][sub*elemsPerLane+c]
		}
		laneValues[l] = laneSum
	}

	// 2-step butterfly shuffle: XOR 1, then XOR 2
	shuffled := make([]float64, 32)
	copy(shuffled, laneValues)
	// Step 1: XOR 1
	step1 := make([]float64, 32)
	for l := 0; l < 32; l++ {
		step1[l] = shuffled[l] + shuffled[l^1]
	}
	// Step 2: XOR 2
	step2 := make([]float64, 32)
	for l := 0; l < 32; l++ {
		step2[l] = step1[l] + step1[l^2]
	}

	// Verify all lanes in row r have exact rowTrueSums[row] and zero cross-row leakage
	for l := 0; l < 32; l++ {
		row := l / lanesPerRow
		sub := l % lanesPerRow
		diff := math.Abs(step2[l] - rowTrueSums[row])
		if diff > 1e-10 {
			t.Fatalf("algorithmic shuffle: lane %d (row %d, sub %d) = %g, want %g (diff %g)",
				l, row, sub, step2[l], rowTrueSums[row], diff)
		}
	}

	// Cross-row perturbation check: perturbing row targetRow MUST NOT change any other row
	for targetRow := 0; targetRow < numRows; targetRow++ {
		perturbedLaneValues := make([]float64, 32)
		copy(perturbedLaneValues, laneValues)
		for s := 0; s < lanesPerRow; s++ {
			perturbedLaneValues[targetRow*lanesPerRow+s] += 999999.0
		}
		pStep1 := make([]float64, 32)
		for l := 0; l < 32; l++ {
			pStep1[l] = perturbedLaneValues[l] + perturbedLaneValues[l^1]
		}
		pStep2 := make([]float64, 32)
		for l := 0; l < 32; l++ {
			pStep2[l] = pStep1[l] + pStep1[l^2]
		}
		for otherRow := 0; otherRow < numRows; otherRow++ {
			if otherRow == targetRow {
				continue
			}
			for s := 0; s < lanesPerRow; s++ {
				lane := otherRow*lanesPerRow + s
				if math.Abs(pStep2[lane]-rowTrueSums[otherRow]) > 1e-10 {
					t.Fatalf("cross-row leakage detected! Perturbing row %d leaked into lane %d (row %d): got %g, want %g",
						targetRow, lane, otherRow, pStep2[lane], rowTrueSums[otherRow])
				}
			}
		}
	}

	// 2. Physical Metal execution: run actual Metal kernel with simd_shuffle_xor on Apple Silicon GPU.
	if !Available() {
		t.Log("Metal unavailable; skipped physical on-device kernel verification")
		return
	}

	inVec := make([]float32, 32)
	for l := 0; l < 32; l++ {
		inVec[l] = float32(laneValues[l])
	}
	outVec := make([]float32, 32)

	if !RunTestShuffle(inVec, outVec) {
		t.Fatal("physical Metal butterfly shuffle kernel execution failed")
	}

	for l := 0; l < 32; l++ {
		row := l / lanesPerRow
		sub := l % lanesPerRow
		// Exact float32 sum of the 4 lanes of this row
		want := inVec[row*lanesPerRow+0] + inVec[row*lanesPerRow+1] + inVec[row*lanesPerRow+2] + inVec[row*lanesPerRow+3]
		got := outVec[l]
		diff := float32(math.Abs(float64(want - got)))
		if diff > 1e-5 {
			t.Fatalf("on-device Metal shuffle: lane %d (row %d, sub %d) got %g, want %g (diff %g)",
				l, row, sub, got, want, diff)
		}
	}

	// Test on-device one-hot row isolation: activate exactly one row at a time.
	for activeRow := 0; activeRow < numRows; activeRow++ {
		oneHotIn := make([]float32, 32)
		for s := 0; s < lanesPerRow; s++ {
			oneHotIn[activeRow*lanesPerRow+s] = float32(s + 1)
		}
		oneHotOut := make([]float32, 32)
		if !RunTestShuffle(oneHotIn, oneHotOut) {
			t.Fatalf("on-device one-hot test for row %d failed", activeRow)
		}
		expectedSum := float32(1 + 2 + 3 + 4) // 10.0
		for l := 0; l < 32; l++ {
			row := l / lanesPerRow
			got := oneHotOut[l]
			if row == activeRow {
				if math.Abs(float64(got-expectedSum)) > 1e-5 {
					t.Fatalf("active row %d lane %d got %g, want %g", activeRow, l, got, expectedSum)
				}
			} else {
				if got != 0.0 {
					t.Fatalf("cross-row leakage on GPU! Active row %d leaked into inactive row %d lane %d: value=%g",
						activeRow, row, l, got)
				}
			}
		}
	}

	t.Log("Verified 4-lane intra-row butterfly shuffle: 128 elements reduced per row with zero cross-row lane leakage.")
}

// BenchmarkQwenGDNPackedBTreeKernel measures and logs the execution latency of the
// packed 8-row B-tree kernel vs the baseline recurrent kernel on Apple Silicon Metal.
func BenchmarkQwenGDNPackedBTreeKernel(b *testing.B) {
	if !Available() {
		b.Skip("Metal unavailable")
	}

	g := oracleGDNGeometry{nK: 2, nV: 8, kHd: 128, vHd: 128, kernel: 3}
	panel := oracleGDNFixture(g, 1, 0.2) // single token decode step

	native, err := NewGDNState(nativeGDNGeometry(g))
	if err != nil {
		b.Fatalf("NewGDNState failed: %v", err)
	}
	defer native.Close()

	// Warmup
	SetGDNForceBaseline(false)
	for i := 0; i < 5; i++ {
		_, _, _, _ = native.Run(nativeGDNPanel(panel))
	}

	var packedDuration time.Duration
	var packedN int
	b.Run("Packed_8Row_Dk128_Dv128", func(b *testing.B) {
		SetGDNForceBaseline(false)
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			_, _, accepted, err := native.Run(nativeGDNPanel(panel))
			if !accepted || err != nil {
				b.Fatalf("Run failed: accepted=%v, err=%v", accepted, err)
			}
		}
		packedDuration = time.Since(start)
		packedN = b.N
	})

	var baselineDuration time.Duration
	var baselineN int
	b.Run("Baseline_Recurrent_Dk128_Dv128", func(b *testing.B) {
		SetGDNForceBaseline(true)
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			_, _, accepted, err := native.Run(nativeGDNPanel(panel))
			if !accepted || err != nil {
				b.Fatalf("Run failed: accepted=%v, err=%v", accepted, err)
			}
		}
		baselineDuration = time.Since(start)
		baselineN = b.N
	})

	// Restore default
	SetGDNForceBaseline(false)

	if packedN > 0 && baselineN > 0 && packedDuration > 0 && baselineDuration > 0 {
		packedAvgNs := float64(packedDuration.Nanoseconds()) / float64(packedN)
		baseAvgNs := float64(baselineDuration.Nanoseconds()) / float64(baselineN)
		speedup := baseAvgNs / packedAvgNs
		b.Logf("=== Qwen GatedDeltaNet Linear Attention Benchmark (Dk=128, Dv=128, nK=2, nV=8) ===")
		b.Logf("Packed 8-row B-tree: %.2f µs/step (%.2f ns/op)", packedAvgNs/1000.0, packedAvgNs)
		b.Logf("Baseline Recurrent:  %.2f µs/step (%.2f ns/op)", baseAvgNs/1000.0, baseAvgNs)
		b.Logf("Speedup Ratio:       %.2fx", speedup)
	}
}
