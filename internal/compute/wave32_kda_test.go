package compute

import (
	"math"
	"math/rand"
	"testing"
)

// TestWave32RegisterContract verifies that the register allocation contract
// strictly adheres to AMD RDNA 3.5 APU (gfx1151 / AMD Strix Halo) hardware limits.
func TestWave32RegisterContract(t *testing.T) {
	contract, err := NewWave32RegisterContract(Wave32DefaultNumHeads)
	if err != nil {
		t.Fatalf("failed to create default Wave32 register contract: %v", err)
	}

	if contract.TargetArch != "gfx1151" {
		t.Errorf("TargetArch = %q, want %q", contract.TargetArch, "gfx1151")
	}
	if contract.WavefrontSize != 32 {
		t.Errorf("WavefrontSize = %d, want 32", contract.WavefrontSize)
	}
	if contract.NumWavefronts != 4 {
		t.Errorf("NumWavefronts = %d, want 4", contract.NumWavefronts)
	}
	if contract.TotalLanes != 128 {
		t.Errorf("TotalLanes = %d, want 128", contract.TotalLanes)
	}
	if contract.HeadDim != 128 {
		t.Errorf("HeadDim = %d, want 128", contract.HeadDim)
	}
	if contract.StateDimsPerThread != 4 {
		t.Errorf("StateDimsPerThread = %d, want 4", contract.StateDimsPerThread)
	}
	if contract.LDSBytesPerBlock != 0 {
		t.Errorf("LDSBytesPerBlock = %d, want 0 (recurrent state must be register-resident)", contract.LDSBytesPerBlock)
	}
	if contract.InnerLoopDRAMBytes != 0 {
		t.Errorf("InnerLoopDRAMBytes = %d, want 0", contract.InnerLoopDRAMBytes)
	}
	if contract.VGPRsAllocated > contract.MaxVGPRsPerThread {
		t.Errorf("VGPRsAllocated %d exceeds hardware limit %d", contract.VGPRsAllocated, contract.MaxVGPRsPerThread)
	}

	// Negative cases: contract must fail closed on mismatched geometry.
	badContract := contract
	badContract.LDSBytesPerBlock = 1024
	if err := badContract.Validate(); err == nil {
		t.Errorf("expected validation failure for non-zero LDS bytes")
	}

	badContract = contract
	badContract.InnerLoopDRAMBytes = 512
	if err := badContract.Validate(); err == nil {
		t.Errorf("expected validation failure for non-zero inner-loop DRAM bytes")
	}

	badContract = contract
	badContract.WavefrontSize = 64
	if err := badContract.Validate(); err == nil {
		t.Errorf("expected validation failure for Wave64 on RDNA 3.5 target")
	}
}

// TestWave32IntraWaveShuffle verifies that the simulated 32-lane wavefront __shfl_down
// tree reduction accurately reduces across lanes without LDS or memory buffers.
func TestWave32IntraWaveShuffle(t *testing.T) {
	wave := &Wave32Wavefront{WaveID: 0}
	var vals [Wave32WavefrontSize]float32
	var expectedSum float32
	for l := 0; l < Wave32WavefrontSize; l++ {
		val := float32(l + 1)
		vals[l] = val
		expectedSum += val
	}

	var audit Wave32KDAMemoryAudit
	total := wave.IntraWaveAllReduceSum(vals, &audit)

	if math.Abs(float64(total-expectedSum)) > 1e-5 {
		t.Errorf("IntraWaveAllReduceSum = %g, want %g", total, expectedSum)
	}
	if audit.IntraWaveShuffles < 5 {
		t.Errorf("IntraWaveShuffles = %d, want at least 5 (log2(32))", audit.IntraWaveShuffles)
	}
	if err := audit.AssertZeroDRAMTraffic(); err != nil {
		t.Errorf("audit failed: %v", err)
	}
}

// TestWave32KDASingleStepEquivalence verifies that Wave32KDAStep matches ReferenceKDAStep
// for a single token state transition within floating-point tolerance.
func TestWave32KDASingleStepEquivalence(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	var stateInit [Wave32HeadDim][Wave32HeadDim]float32
	for i := 0; i < Wave32HeadDim; i++ {
		for j := 0; j < Wave32HeadDim; j++ {
			stateInit[i][j] = rng.Float32()*0.2 - 0.1
		}
	}

	q := make([]float32, Wave32HeadDim)
	k := make([]float32, Wave32HeadDim)
	v := make([]float32, Wave32HeadDim)
	for i := 0; i < Wave32HeadDim; i++ {
		q[i] = rng.Float32()*0.2 - 0.1
		k[i] = rng.Float32()*0.2 - 0.1
		v[i] = rng.Float32()*0.2 - 0.1
	}
	beta := float32(0.5)
	decay := float32(math.Exp(-0.1))

	// 1. Reference step
	refOut, refNextState := ReferenceKDAStep(stateInit, q, k, v, beta, decay)

	// 2. Wave32 workgroup step
	contract, _ := NewWave32RegisterContract(Wave32DefaultNumHeads)
	wg := NewWave32Workgroup(contract)
	wg.LoadMatrixState(stateInit)

	waveOut := wg.Wave32KDAStep(q, k, v, beta, decay)
	waveNextState := wg.ReadMatrixState()

	// Assert output parity
	outDelta := MaxAbsDelta(refOut, waveOut)
	if outDelta > 1e-6 {
		t.Errorf("output max abs delta %g exceeds tolerance 1e-6", outDelta)
	}
	outCosine := CosineSimilarity(refOut, waveOut)
	if outCosine < 0.999999 {
		t.Errorf("output cosine similarity %g < 0.999999", outCosine)
	}

	// Assert next state matrix parity
	for i := 0; i < Wave32HeadDim; i++ {
		rowDelta := MaxAbsDelta(refNextState[i][:], waveNextState[i][:])
		if rowDelta > 1e-6 {
			t.Fatalf("row %d state delta %g exceeds tolerance 1e-6", i, rowDelta)
		}
	}
}

// TestWave32KDASequenceParityAndZeroDRAMTraffic runs multi-token autoregressive sequences
// across sequence lengths up to 4096 tokens, asserting bit/float parity against reference
// and strictly ZERO DRAM traffic in the inner loop.
func TestWave32KDASequenceParityAndZeroDRAMTraffic(t *testing.T) {
	seqLengths := []int{1, 4, 16, 64, 256, 1024, 2048, 4096}

	for _, T := range seqLengths {
		t.Run(testing.Benchmark(func(b *testing.B) {}).String(), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(1000 + T)))

			var initialState [Wave32HeadDim][Wave32HeadDim]float32
			for i := 0; i < Wave32HeadDim; i++ {
				for j := 0; j < Wave32HeadDim; j++ {
					initialState[i][j] = rng.Float32()*0.02 - 0.01
				}
			}

			qSeq := make([][]float32, T)
			kSeq := make([][]float32, T)
			vSeq := make([][]float32, T)
			betaSeq := make([]float32, T)
			fProjSeq := make([]float32, T)

			for step := 0; step < T; step++ {
				q := make([]float32, Wave32HeadDim)
				k := make([]float32, Wave32HeadDim)
				v := make([]float32, Wave32HeadDim)
				for d := 0; d < Wave32HeadDim; d++ {
					q[d] = rng.Float32()*0.1 - 0.05
					k[d] = rng.Float32()*0.1 - 0.05
					v[d] = rng.Float32()*0.1 - 0.05
				}
				qSeq[step] = q
				kSeq[step] = k
				vSeq[step] = v
				betaSeq[step] = float32(1.0 / (1.0 + math.Exp(-float64(rng.Float32()*2.0-1.0))))
				fProjSeq[step] = rng.Float32()*2.0 - 1.0
			}

			aLog := float32(-0.5)
			dtBias := float32(0.1)

			// Execute reference formulation
			refOutputs, refFinalState, err := ReferenceKDASequence(
				initialState, qSeq, kSeq, vSeq, betaSeq, aLog, fProjSeq, dtBias,
			)
			if err != nil {
				t.Fatalf("ReferenceKDASequence failed at T=%d: %v", T, err)
			}

			// Execute Wave32 register-resident kernel
			contract, _ := NewWave32RegisterContract(Wave32DefaultNumHeads)
			wg := NewWave32Workgroup(contract)

			waveOutputs, waveFinalState, err := wg.Wave32KDASequence(
				initialState, qSeq, kSeq, vSeq, betaSeq, aLog, fProjSeq, dtBias,
			)
			if err != nil {
				t.Fatalf("Wave32KDASequence failed at T=%d: %v", T, err)
			}

			// 1. Verify ZERO DRAM and ZERO LDS traffic in inner loop
			if err := wg.Audit.AssertZeroDRAMTraffic(); err != nil {
				t.Fatalf("T=%d DRAM audit failed: %v", T, err)
			}
			if wg.Audit.TokensProcessed != T {
				t.Errorf("TokensProcessed = %d, want %d", wg.Audit.TokensProcessed, T)
			}

			// 2. Verify output parity at every token step
			var maxOutputDelta float64
			for step := 0; step < T; step++ {
				delta := MaxAbsDelta(refOutputs[step], waveOutputs[step])
				if delta > maxOutputDelta {
					maxOutputDelta = delta
				}
				cosine := CosineSimilarity(refOutputs[step], waveOutputs[step])
				if cosine < 0.99999 {
					t.Errorf("T=%d token %d cosine similarity %g < 0.99999", T, step, cosine)
				}
			}

			// 3. Verify final recurrent state matrix parity
			var maxStateDelta float64
			for i := 0; i < Wave32HeadDim; i++ {
				d := MaxAbsDelta(refFinalState[i][:], waveFinalState[i][:])
				if d > maxStateDelta {
					maxStateDelta = d
				}
			}

			if maxOutputDelta > 1e-5 {
				t.Errorf("T=%d max output delta %g exceeds 1e-5", T, maxOutputDelta)
			}
			if maxStateDelta > 1e-5 {
				t.Errorf("T=%d max state delta %g exceeds 1e-5", T, maxStateDelta)
			}

			t.Logf("T=%4d tokens PASSED: maxOutputDelta=%.3e maxStateDelta=%.3e innerLoopDRAMBytes=%d",
				T, maxOutputDelta, maxStateDelta, wg.Audit.InnerLoopDRAMBytes)
		})
	}
}

// TestWave32VectorKDAParity verifies that 128-dim vector linear recurrence mapped
// across 32 lanes holding 4 state dimensions matches the reference vector formulation.
func TestWave32VectorKDAParity(t *testing.T) {
	rng := rand.New(rand.NewSource(99))

	var stateInit [Wave32HeadDim]float32
	for i := 0; i < Wave32HeadDim; i++ {
		stateInit[i] = rng.Float32()*0.2 - 0.1
	}

	wave := &Wave32Wavefront{WaveID: 0}
	for l := 0; l < Wave32WavefrontSize; l++ {
		for m := 0; m < Wave32StateDimsPerThread; m++ {
			wave.Threads[l].StateRegs[m] = stateInit[l*Wave32StateDimsPerThread+m]
		}
	}

	q := make([]float32, Wave32HeadDim)
	k := make([]float32, Wave32HeadDim)
	v := make([]float32, Wave32HeadDim)
	for i := 0; i < Wave32HeadDim; i++ {
		q[i] = rng.Float32()*0.1 - 0.05
		k[i] = rng.Float32()*0.1 - 0.05
		v[i] = rng.Float32()*0.1 - 0.05
	}
	beta := float32(0.8)
	decay := float32(math.Exp(-0.05))

	// Reference vector step
	refOut, refNextState := ReferenceVectorKDAStep(stateInit, q, k, v, beta, decay)

	// Wave32 vector step with intra-wave shuffles
	var audit Wave32KDAMemoryAudit
	waveOut := wave.Wave32VectorKDAStep(q, k, v, beta, decay, &audit)

	// Compare outputs
	if math.Abs(float64(refOut-waveOut)) > 1e-5 {
		t.Errorf("vector readout delta got=%g want=%g", waveOut, refOut)
	}

	// Compare next states
	var nextStateFromWave [Wave32HeadDim]float32
	for l := 0; l < Wave32WavefrontSize; l++ {
		for m := 0; m < Wave32StateDimsPerThread; m++ {
			nextStateFromWave[l*Wave32StateDimsPerThread+m] = wave.Threads[l].StateRegs[m]
		}
	}

	delta := MaxAbsDelta(refNextState[:], nextStateFromWave[:])
	if delta > 1e-5 {
		t.Errorf("vector state max abs delta %g > 1e-5", delta)
	}
	if err := audit.AssertZeroDRAMTraffic(); err != nil {
		t.Errorf("vector audit error: %v", err)
	}
}

// TestWave32KDABoundedForgetStability verifies numerical stability over 4096 tokens
// under extreme inputs, confirming that the bounded forget gate [-5.0, 0.0] prevents
// state explosion and underflow.
func TestWave32KDABoundedForgetStability(t *testing.T) {
	const T = 4096
	rng := rand.New(rand.NewSource(777))

	var initialState [Wave32HeadDim][Wave32HeadDim]float32
	qSeq := make([][]float32, T)
	kSeq := make([][]float32, T)
	vSeq := make([][]float32, T)
	betaSeq := make([]float32, T)
	fProjSeq := make([]float32, T)

	for step := 0; step < T; step++ {
		rawQ := make([]float32, Wave32HeadDim)
		rawK := make([]float32, Wave32HeadDim)
		v := make([]float32, Wave32HeadDim)
		for d := 0; d < Wave32HeadDim; d++ {
			rawQ[d] = rng.Float32()*2.0 - 1.0
			rawK[d] = rng.Float32()*2.0 - 1.0
			v[d] = rng.Float32()*0.2 - 0.1
		}
		// In KDA / DeltaNet, keys and queries are L2-normalized per head.
		qSeq[step] = L2Normalize(rawQ)
		kSeq[step] = L2Normalize(rawK)
		vSeq[step] = v
		betaSeq[step] = 0.5
		// Alternate between saturating positive and negative projections
		if step%2 == 0 {
			fProjSeq[step] = 1e4
		} else {
			fProjSeq[step] = -1e4
		}
	}

	contract, _ := NewWave32RegisterContract(Wave32DefaultNumHeads)
	wg := NewWave32Workgroup(contract)

	outputs, finalState, err := wg.Wave32KDASequence(
		initialState, qSeq, kSeq, vSeq, betaSeq, 5.0, fProjSeq, 0.0,
	)
	if err != nil {
		t.Fatalf("sequence execution failed: %v", err)
	}

	// Verify no NaN or Inf in outputs or state
	for step := 0; step < T; step++ {
		for d := 0; d < Wave32HeadDim; d++ {
			val := outputs[step][d]
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Fatalf("token %d output %d is NaN/Inf: %g", step, d, val)
			}
		}
	}

	for i := 0; i < Wave32HeadDim; i++ {
		for j := 0; j < Wave32HeadDim; j++ {
			val := finalState[i][j]
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Fatalf("final state [%d][%d] is NaN/Inf: %g", i, j, val)
			}
		}
	}
}
