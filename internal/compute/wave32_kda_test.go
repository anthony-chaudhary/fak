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

// TestWave32GDNRegisterContract verifies that the Qwen 3.8 GDN register allocation contract
// adheres strictly to AMD RDNA 3.5 APU (gfx1151 / AMD Strix Halo) hardware limits.
func TestWave32GDNRegisterContract(t *testing.T) {
	contract, err := NewWave32GDNRegisterContract(Wave32GDNDefaultNumKeyHeads, Wave32GDNDefaultNumValueHeads)
	if err != nil {
		t.Fatalf("failed to create default Wave32 GDN register contract: %v", err)
	}

	if contract.TargetArch != "gfx1151" {
		t.Errorf("TargetArch = %q, want %q", contract.TargetArch, "gfx1151")
	}
	if contract.ModelArch != Wave32ModelArchQwen38GDN {
		t.Errorf("ModelArch = %q, want %q", contract.ModelArch, Wave32ModelArchQwen38GDN)
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
	if contract.NumKeyHeads != 16 {
		t.Errorf("NumKeyHeads = %d, want 16", contract.NumKeyHeads)
	}
	if contract.NumValueHeads != 48 {
		t.Errorf("NumValueHeads = %d, want 48", contract.NumValueHeads)
	}
	if contract.ConvKernelSize != 4 {
		t.Errorf("ConvKernelSize = %d, want 4", contract.ConvKernelSize)
	}
	if !contract.HasRMSNormOutput {
		t.Errorf("HasRMSNormOutput = false, want true")
	}
	if !contract.HasSiluOutputGate {
		t.Errorf("HasSiluOutputGate = false, want true")
	}
	if contract.StateDimsPerThread != 4 {
		t.Errorf("StateDimsPerThread = %d, want 4", contract.StateDimsPerThread)
	}
	if contract.LDSBytesPerBlock != 0 {
		t.Errorf("LDSBytesPerBlock = %d, want 0", contract.LDSBytesPerBlock)
	}
	if contract.InnerLoopDRAMBytes != 0 {
		t.Errorf("InnerLoopDRAMBytes = %d, want 0", contract.InnerLoopDRAMBytes)
	}
	if contract.VGPRsAllocated > contract.MaxVGPRsPerThread {
		t.Errorf("VGPRsAllocated %d exceeds limit %d", contract.VGPRsAllocated, contract.MaxVGPRsPerThread)
	}

	// Negative cases: contract must fail closed on mismatched parameters
	bad := contract
	bad.VGPRsAllocated = 257
	if err := bad.Validate(); err == nil {
		t.Errorf("expected validation failure for VGPRs > 256")
	}

	bad = contract
	bad.LDSBytesPerBlock = 128
	if err := bad.Validate(); err == nil {
		t.Errorf("expected validation failure for non-zero LDS bytes")
	}

	bad = contract
	bad.InnerLoopDRAMBytes = 64
	if err := bad.Validate(); err == nil {
		t.Errorf("expected validation failure for non-zero DRAM bytes")
	}

	bad = contract
	bad.WavefrontSize = 64
	if err := bad.Validate(); err == nil {
		t.Errorf("expected validation failure for Wave64")
	}

	bad = contract
	bad.TargetArch = "gfx90a"
	if err := bad.Validate(); err == nil {
		t.Errorf("expected validation failure for non-gfx1151 target arch")
	}

	bad = contract
	bad.ConvKernelSize = 0
	if err := bad.Validate(); err == nil {
		t.Errorf("expected validation failure for ConvKernelSize < 1")
	}

	bad = contract
	bad.HasRMSNormOutput = false
	if err := bad.Validate(); err == nil {
		t.Errorf("expected validation failure for HasRMSNormOutput=false")
	}

	bad = contract
	bad.HasSiluOutputGate = false
	if err := bad.Validate(); err == nil {
		t.Errorf("expected validation failure for HasSiluOutputGate=false")
	}

	bad = contract
	bad.NumKeyHeads = 0
	if err := bad.Validate(); err == nil {
		t.Errorf("expected validation failure for NumKeyHeads <= 0")
	}
}

// TestWave32GDNSingleStepEquivalence verifies that Wave32GDNStep matches ReferenceGDNStep
// for a single token state transition within floating-point tolerance.
func TestWave32GDNSingleStepEquivalence(t *testing.T) {
	rng := rand.New(rand.NewSource(1234))

	var stateInit [Wave32HeadDim][Wave32HeadDim]float32
	for i := 0; i < Wave32HeadDim; i++ {
		for j := 0; j < Wave32HeadDim; j++ {
			stateInit[i][j] = rng.Float32()*0.2 - 0.1
		}
	}

	q := make([]float32, Wave32HeadDim)
	k := make([]float32, Wave32HeadDim)
	v := make([]float32, Wave32HeadDim)
	z := make([]float32, Wave32HeadDim)
	norm := make([]float32, Wave32HeadDim)
	for i := 0; i < Wave32HeadDim; i++ {
		q[i] = rng.Float32()*0.2 - 0.1
		k[i] = rng.Float32()*0.2 - 0.1
		v[i] = rng.Float32()*0.2 - 0.1
		z[i] = rng.Float32()*0.4 - 0.2
		norm[i] = rng.Float32()*0.5 + 0.5
	}
	beta := ComputeQwenGDNBeta(rng.Float32()*2.0 - 1.0)
	decay := ComputeQwenGDNDecay(-0.5, rng.Float32()*2.0-1.0, 0.1)
	eps := float32(1e-6)

	// 1. Reference GDN step
	refOut, refNextState := ReferenceGDNStep(stateInit, q, k, v, z, norm, beta, decay, eps)

	// 2. Wave32 workgroup GDN step
	contract, _ := NewWave32GDNRegisterContract(Wave32GDNDefaultNumKeyHeads, Wave32GDNDefaultNumValueHeads)
	wg := NewWave32Workgroup(contract)
	wg.LoadMatrixState(stateInit)

	waveOut := wg.Wave32GDNStep(q, k, v, z, norm, beta, decay, eps)
	waveNextState := wg.ReadMatrixState()

	// Assert output parity
	outDelta := MaxAbsDelta(refOut, waveOut)
	if outDelta > 1e-6 {
		t.Errorf("GDN output max abs delta %g exceeds tolerance 1e-6", outDelta)
	}
	outCosine := CosineSimilarity(refOut, waveOut)
	if outCosine < 0.999999 {
		t.Errorf("GDN output cosine similarity %g < 0.999999", outCosine)
	}

	// Assert next state matrix parity
	for i := 0; i < Wave32HeadDim; i++ {
		rowDelta := MaxAbsDelta(refNextState[i][:], waveNextState[i][:])
		if rowDelta > 1e-6 {
			t.Fatalf("row %d state delta %g exceeds tolerance 1e-6", i, rowDelta)
		}
	}
}

// TestWave32GDNSequenceParityAndZeroDRAMTraffic runs multi-token GDN autoregressive sequences
// across sequence lengths up to 4096 tokens, asserting bit/float parity against reference
// and strictly ZERO DRAM traffic in the inner loop.
func TestWave32GDNSequenceParityAndZeroDRAMTraffic(t *testing.T) {
	seqLengths := []int{1, 4, 16, 64, 256, 1024, 2048, 4096}

	for _, T := range seqLengths {
		t.Run(testing.Benchmark(func(b *testing.B) {}).String(), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(2000 + T)))

			var initialState [Wave32HeadDim][Wave32HeadDim]float32
			for i := 0; i < Wave32HeadDim; i++ {
				for j := 0; j < Wave32HeadDim; j++ {
					initialState[i][j] = rng.Float32()*0.02 - 0.01
				}
			}

			qSeq := make([][]float32, T)
			kSeq := make([][]float32, T)
			vSeq := make([][]float32, T)
			zSeq := make([][]float32, T)
			bProjSeq := make([]float32, T)
			aProjSeq := make([]float32, T)

			for step := 0; step < T; step++ {
				q := make([]float32, Wave32HeadDim)
				k := make([]float32, Wave32HeadDim)
				v := make([]float32, Wave32HeadDim)
				z := make([]float32, Wave32HeadDim)
				for d := 0; d < Wave32HeadDim; d++ {
					q[d] = rng.Float32()*0.1 - 0.05
					k[d] = rng.Float32()*0.1 - 0.05
					v[d] = rng.Float32()*0.1 - 0.05
					z[d] = rng.Float32()*0.2 - 0.1
				}
				qSeq[step] = q
				kSeq[step] = k
				vSeq[step] = v
				zSeq[step] = z
				bProjSeq[step] = rng.Float32()*2.0 - 1.0
				aProjSeq[step] = rng.Float32()*2.0 - 1.0
			}

			norm := make([]float32, Wave32HeadDim)
			for d := 0; d < Wave32HeadDim; d++ {
				norm[d] = rng.Float32()*0.4 + 0.8
			}

			aLog := float32(-0.5)
			dtBias := float32(0.1)
			eps := float32(1e-6)

			// Execute reference formulation
			refOutputs, refFinalState, err := ReferenceGDNSequence(
				initialState, qSeq, kSeq, vSeq, zSeq, bProjSeq, aProjSeq, aLog, dtBias, norm, eps,
			)
			if err != nil {
				t.Fatalf("ReferenceGDNSequence failed at T=%d: %v", T, err)
			}

			// Execute Wave32 register-resident GDN kernel
			contract, _ := NewWave32GDNRegisterContract(Wave32GDNDefaultNumKeyHeads, Wave32GDNDefaultNumValueHeads)
			wg := NewWave32Workgroup(contract)

			waveOutputs, waveFinalState, err := wg.Wave32GDNSequence(
				initialState, qSeq, kSeq, vSeq, zSeq, bProjSeq, aProjSeq, aLog, dtBias, norm, eps,
			)
			if err != nil {
				t.Fatalf("Wave32GDNSequence failed at T=%d: %v", T, err)
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

			t.Logf("GDN T=%4d tokens PASSED: maxOutputDelta=%.3e maxStateDelta=%.3e innerLoopDRAMBytes=%d",
				T, maxOutputDelta, maxStateDelta, wg.Audit.InnerLoopDRAMBytes)
		})
	}
}

// TestWave32GDNBoundedForgetStability verifies numerical stability over 4096 tokens
// under extreme inputs, confirming that the GDN decay exp(-exp(aLog)*softplus(aProj+dtBias))
// prevents state explosion and underflow.
func TestWave32GDNBoundedForgetStability(t *testing.T) {
	const T = 4096
	rng := rand.New(rand.NewSource(888))

	var initialState [Wave32HeadDim][Wave32HeadDim]float32
	qSeq := make([][]float32, T)
	kSeq := make([][]float32, T)
	vSeq := make([][]float32, T)
	zSeq := make([][]float32, T)
	bProjSeq := make([]float32, T)
	aProjSeq := make([]float32, T)

	for step := 0; step < T; step++ {
		rawQ := make([]float32, Wave32HeadDim)
		rawK := make([]float32, Wave32HeadDim)
		v := make([]float32, Wave32HeadDim)
		z := make([]float32, Wave32HeadDim)
		for d := 0; d < Wave32HeadDim; d++ {
			rawQ[d] = rng.Float32()*2.0 - 1.0
			rawK[d] = rng.Float32()*2.0 - 1.0
			v[d] = rng.Float32()*0.2 - 0.1
			z[d] = rng.Float32()*4.0 - 2.0
		}
		qSeq[step] = L2Normalize(rawQ)
		kSeq[step] = L2Normalize(rawK)
		vSeq[step] = v
		zSeq[step] = z
		// Extreme alternating projections
		if step%2 == 0 {
			aProjSeq[step] = 1e4
			bProjSeq[step] = 1e4
		} else {
			aProjSeq[step] = -1e4
			bProjSeq[step] = -1e4
		}
	}

	norm := make([]float32, Wave32HeadDim)
	for d := 0; d < Wave32HeadDim; d++ {
		norm[d] = 1.0
	}

	contract, _ := NewWave32GDNRegisterContract(Wave32GDNDefaultNumKeyHeads, Wave32GDNDefaultNumValueHeads)
	wg := NewWave32Workgroup(contract)

	outputs, finalState, err := wg.Wave32GDNSequence(
		initialState, qSeq, kSeq, vSeq, zSeq, bProjSeq, aProjSeq, 5.0, 0.0, norm, 1e-6,
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

// TestWave32Conv1DState verifies causal depthwise 1D convolution rolling buffer logic,
// error checking, and step-wise history evolution with SiLU activation.
func TestWave32Conv1DState(t *testing.T) {
	// 1. Validation tests
	if _, err := NewWave32Conv1DState(0, 4); err == nil {
		t.Errorf("expected error for kernelSize < 1")
	}
	if _, err := NewWave32Conv1DState(4, 0); err == nil {
		t.Errorf("expected error for channels <= 0")
	}

	// 2. Step with kernelSize = 3, channels = 2
	// hist = 2. weights has length 2 * 3 = 6.
	// Channel 0 weights: [w0, w1, w2] = [0.5, 0.3, 0.2]
	// Channel 1 weights: [w0, w1, w2] = [0.1, 0.4, 0.6]
	cs, err := NewWave32Conv1DState(3, 2)
	if err != nil {
		t.Fatalf("failed to create conv1d state: %v", err)
	}

	weights := []float32{
		0.5, 0.3, 0.2, // channel 0
		0.1, 0.4, 0.6, // channel 1
	}

	// Mismatched input/weight length checks
	if _, err := cs.Step([]float32{1.0}, weights); err == nil {
		t.Errorf("expected error for input length mismatch")
	}
	if _, err := cs.Step([]float32{1.0, 2.0}, []float32{0.5}); err == nil {
		t.Errorf("expected error for weights length mismatch")
	}

	// Step 1: initial history is [[0, 0], [0, 0]]
	// input = [1.0, 2.0]
	// ch0: acc = 0.5*0 + 0.3*0 + 0.2*1.0 = 0.2; out = Silu(0.2)
	// ch1: acc = 0.1*0 + 0.4*0 + 0.6*2.0 = 1.2; out = Silu(1.2)
	// buffer after step 1: [[0, 0], [1.0, 2.0]]
	out1, err := cs.Step([]float32{1.0, 2.0}, weights)
	if err != nil {
		t.Fatalf("Step 1 failed: %v", err)
	}
	want1_0 := Silu(0.2)
	want1_1 := Silu(1.2)
	if math.Abs(float64(out1[0]-want1_0)) > 1e-6 || math.Abs(float64(out1[1]-want1_1)) > 1e-6 {
		t.Errorf("Step 1 out = [%g, %g], want [%g, %g]", out1[0], out1[1], want1_0, want1_1)
	}

	// Step 2: history is [[0, 0], [1.0, 2.0]]
	// input = [3.0, 4.0]
	// ch0: acc = 0.5*0 + 0.3*1.0 + 0.2*3.0 = 0.3 + 0.6 = 0.9
	// ch1: acc = 0.1*0 + 0.4*2.0 + 0.6*4.0 = 0.8 + 2.4 = 3.2
	// buffer after step 2: [[1.0, 2.0], [3.0, 4.0]]
	out2, err := cs.Step([]float32{3.0, 4.0}, weights)
	if err != nil {
		t.Fatalf("Step 2 failed: %v", err)
	}
	want2_0 := Silu(0.9)
	want2_1 := Silu(3.2)
	if math.Abs(float64(out2[0]-want2_0)) > 1e-6 || math.Abs(float64(out2[1]-want2_1)) > 1e-6 {
		t.Errorf("Step 2 out = [%g, %g], want [%g, %g]", out2[0], out2[1], want2_0, want2_1)
	}

	// Step 3: history is [[1.0, 2.0], [3.0, 4.0]]
	// input = [5.0, 6.0]
	// ch0: acc = 0.5*1.0 + 0.3*3.0 + 0.2*5.0 = 0.5 + 0.9 + 1.0 = 2.4
	// ch1: acc = 0.1*2.0 + 0.4*4.0 + 0.6*6.0 = 0.2 + 1.6 + 3.6 = 5.4
	// buffer after step 3: [[3.0, 4.0], [5.0, 6.0]]
	out3, err := cs.Step([]float32{5.0, 6.0}, weights)
	if err != nil {
		t.Fatalf("Step 3 failed: %v", err)
	}
	want3_0 := Silu(2.4)
	want3_1 := Silu(5.4)
	if math.Abs(float64(out3[0]-want3_0)) > 1e-6 || math.Abs(float64(out3[1]-want3_1)) > 1e-6 {
		t.Errorf("Step 3 out = [%g, %g], want [%g, %g]", out3[0], out3[1], want3_0, want3_1)
	}

	// Check final buffer contents
	if cs.Buffer[0][0] != 3.0 || cs.Buffer[0][1] != 4.0 || cs.Buffer[1][0] != 5.0 || cs.Buffer[1][1] != 6.0 {
		t.Errorf("final buffer mismatch: %v", cs.Buffer)
	}
}

// TestWave32GatedDeltaNetStepVectorizedParity tests Wave32GatedDeltaNetStep across multiple head dimensions
// (d=128, d=64, d=8) against the reference formulation, asserting cosine similarity > 0.99999 and state parity.
func TestWave32GatedDeltaNetStepVectorizedParity(t *testing.T) {
	for _, dim := range []int{128, 64, 8} {
		rng := rand.New(rand.NewSource(int64(42 + dim)))
		kHd, vHd := dim, dim
		stRef := make([]float32, kHd*vHd)
		stOpt := make([]float32, kHd*vHd)
		for i := range stRef {
			val := rng.Float32()*0.2 - 0.1
			stRef[i] = val
			stOpt[i] = val
		}
		qn := make([]float32, kHd)
		kn := make([]float32, kHd)
		vh := make([]float32, vHd)
		for i := 0; i < kHd; i++ {
			qn[i] = rng.Float32()*0.2 - 0.1
			kn[i] = rng.Float32()*0.2 - 0.1
		}
		for i := 0; i < vHd; i++ {
			vh[i] = rng.Float32()*0.2 - 0.1
		}
		bt := float32(0.6)
		g := float32(math.Exp(-0.05))

		// 1. Reference calculation
		odRef := make([]float32, vHd)
		kvmemRef := make([]float32, vHd)
		deltaRef := make([]float32, vHd)
		for i := range stRef {
			stRef[i] *= g
		}
		for i := 0; i < kHd; i++ {
			ki := kn[i]
			base := i * vHd
			for d := 0; d < vHd; d++ {
				kvmemRef[d] += stRef[base+d] * ki
			}
		}
		for d := 0; d < vHd; d++ {
			deltaRef[d] = (vh[d] - kvmemRef[d]) * bt
		}
		for i := 0; i < kHd; i++ {
			ki := kn[i]
			qi := qn[i]
			base := i * vHd
			for d := 0; d < vHd; d++ {
				stRef[base+d] += ki * deltaRef[d]
				odRef[d] += stRef[base+d] * qi
			}
		}

		// 2. Vectorized Wave32GatedDeltaNetStep
		odOpt := make([]float32, vHd)
		kvmemOpt := make([]float32, vHd)
		deltaOpt := make([]float32, vHd)
		Wave32GatedDeltaNetStep(stOpt, qn, kn, vh, bt, g, odOpt, kvmemOpt, deltaOpt)

		// Parity checks
		outCosine := CosineSimilarity(odRef, odOpt)
		if outCosine < 0.99999 {
			t.Errorf("dim=%d: output cosine %g < 0.99999", dim, outCosine)
		}
		outDelta := MaxAbsDelta(odRef, odOpt)
		if outDelta > 1e-5 {
			t.Errorf("dim=%d: output max delta %g > 1e-5", dim, outDelta)
		}
		stDelta := MaxAbsDelta(stRef, stOpt)
		if stDelta > 1e-5 {
			t.Errorf("dim=%d: state max delta %g > 1e-5", dim, stDelta)
		}
	}
}

// BenchmarkWave32GatedDeltaNetStep benchmarks single-head Gated-DeltaNet step latency.
func BenchmarkWave32GatedDeltaNetStep(b *testing.B) {
	const d = 128
	st := make([]float32, d*d)
	qn := make([]float32, d)
	kn := make([]float32, d)
	vh := make([]float32, d)
	od := make([]float32, d)
	kvmem := make([]float32, d)
	delta := make([]float32, d)
	for i := range st {
		st[i] = 0.01
	}
	for i := 0; i < d; i++ {
		qn[i] = 0.1
		kn[i] = 0.1
		vh[i] = 0.1
	}
	bt := float32(0.5)
	g := float32(0.99)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Wave32GatedDeltaNetStep(st, qn, kn, vh, bt, g, od, kvmem, delta)
	}
}
