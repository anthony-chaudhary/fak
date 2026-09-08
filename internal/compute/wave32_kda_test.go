package compute

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
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

// TestTiledChannelTranspose_Roundtrip verifies that TiledChannelTranspose followed by
// TiledChannelTransposeInverse exactly reproduces the input tensor across a variety of
// sequence lengths and channel dimensions (including prime and non-power-of-two sizes).
func TestTiledChannelTranspose_Roundtrip(t *testing.T) {
	testCases := []struct {
		T       int
		convDim int
	}{
		{T: 1, convDim: 1},
		{T: 7, convDim: 13},
		{T: 16, convDim: 32},
		{T: 32, convDim: 32},
		{T: 33, convDim: 33},
		{T: 64, convDim: 128},
		{T: 128, convDim: 256},
		{T: 64, convDim: 10240}, // Qwen 3.8 GDN dimension
	}

	cfg := DefaultTiledChannelTransposeConfig()

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("T%d_C%d", tc.T, tc.convDim), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(tc.T*1000 + tc.convDim)))
			total := tc.T * tc.convDim
			input := make([]float32, total)
			for i := range input {
				input[i] = rng.Float32()*2.0 - 1.0
			}

			aud := &TiledChannelTransposeAudit{}
			transposed, err := TiledChannelTranspose(input, tc.T, tc.convDim, cfg, aud)
			if err != nil {
				t.Fatalf("TiledChannelTranspose failed: %v", err)
			}
			if len(transposed) != total {
				t.Fatalf("transposed length = %d, want %d", len(transposed), total)
			}

			// Verify element transposition: transposed[c*T + t] == input[t*convDim + c]
			for step := 0; step < tc.T; step++ {
				for c := 0; c < tc.convDim; c++ {
					expected := input[step*tc.convDim+c]
					got := transposed[c*tc.T+step]
					if expected != got {
						t.Fatalf("transposition mismatch at t=%d, c=%d: got %g, want %g", step, c, got, expected)
					}
				}
			}

			// Invert transpose
			restored, err := TiledChannelTransposeInverse(transposed, tc.convDim, tc.T, cfg, aud)
			if err != nil {
				t.Fatalf("TiledChannelTransposeInverse failed: %v", err)
			}
			if len(restored) != total {
				t.Fatalf("restored length = %d, want %d", len(restored), total)
			}

			// Verify exact bitwise roundtrip
			for i := range input {
				if input[i] != restored[i] {
					t.Fatalf("roundtrip mismatch at index %d: got %g, want %g", i, restored[i], input[i])
				}
			}
		})
	}
}

// TestTiledChannelTranspose_LDSBankConflicts verifies that stride-33 padding completely
// eliminates 32-way LDS bank conflicts on AMD Strix Halo (gfx1151 / Wave32), whereas
// stride-32 (unpadded) incurs severe bank conflicts.
func TestTiledChannelTranspose_LDSBankConflicts(t *testing.T) {
	const T = 64
	const convDim = 128
	input := make([]float32, T*convDim)
	for i := range input {
		input[i] = float32(i)
	}

	// 1. Padded config (stride 33): expect ZERO bank conflicts
	paddedCfg := DefaultTiledChannelTransposeConfig()
	if paddedCfg.LDSBankStride != 33 {
		t.Fatalf("expected DefaultTiledChannelTransposeConfig LDSBankStride == 33, got %d", paddedCfg.LDSBankStride)
	}

	paddedAudit := &TiledChannelTransposeAudit{}
	_, err := TiledChannelTranspose(input, T, convDim, paddedCfg, paddedAudit)
	if err != nil {
		t.Fatalf("padded transpose failed: %v", err)
	}
	if err := paddedAudit.AssertZeroBankConflicts(); err != nil {
		t.Errorf("padded audit reported bank conflicts: %v", err)
	}
	if paddedAudit.LDSBankConflicts != 0 {
		t.Errorf("padded LDSBankConflicts = %d, want 0", paddedAudit.LDSBankConflicts)
	}

	// 2. Unpadded config (stride 32): expect POSITIVE bank conflicts
	unpaddedCfg := paddedCfg
	unpaddedCfg.LDSBankStride = 32

	unpaddedAudit := &TiledChannelTransposeAudit{}
	_, err = TiledChannelTranspose(input, T, convDim, unpaddedCfg, unpaddedAudit)
	if err != nil {
		t.Fatalf("unpadded transpose failed: %v", err)
	}
	if unpaddedAudit.LDSBankConflicts == 0 {
		t.Errorf("unpadded LDSBankConflicts = 0, expected 32-way bank conflicts on unpadded stride 32")
	}
	if err := unpaddedAudit.AssertZeroBankConflicts(); err == nil {
		t.Errorf("expected AssertZeroBankConflicts() to fail on unpadded stride 32")
	}
}

// TestTiledChannelTranspose_ShaderDescriptor verifies the shader descriptor and push constants
// configuration for qwen35_gdn_tiled_transpose on AMD RDNA 3.5 (gfx1151 / Wave32).
func TestTiledChannelTranspose_ShaderDescriptor(t *testing.T) {
	if Qwen35GDNTiledTransposeShader != "qwen35_gdn_tiled_transpose" {
		t.Errorf("Qwen35GDNTiledTransposeShader = %q, want %q", Qwen35GDNTiledTransposeShader, "qwen35_gdn_tiled_transpose")
	}

	cfg := DefaultTiledChannelTransposeConfig()
	if cfg.ShaderName != Qwen35GDNTiledTransposeShader {
		t.Errorf("cfg.ShaderName = %q, want %q", cfg.ShaderName, Qwen35GDNTiledTransposeShader)
	}

	const T = 64
	const convDim = 10240

	desc := cfg.Descriptor(convDim, T)
	if desc.ShaderName != Qwen35GDNTiledTransposeShader {
		t.Errorf("desc.ShaderName = %q, want %q", desc.ShaderName, Qwen35GDNTiledTransposeShader)
	}
	if desc.TargetArch != Wave32TargetArch {
		t.Errorf("desc.TargetArch = %q, want %q", desc.TargetArch, Wave32TargetArch)
	}
	if desc.LocalSizeX != 32 || desc.LocalSizeY != 8 || desc.LocalSizeZ != 1 {
		t.Errorf("desc local size = (%d, %d, %d), want (32, 8, 1)", desc.LocalSizeX, desc.LocalSizeY, desc.LocalSizeZ)
	}
	if desc.TileT != 32 || desc.TileC != 32 {
		t.Errorf("desc tile dims = (%d, %d), want (32, 32)", desc.TileT, desc.TileC)
	}
	if desc.LDSBankStride != 33 {
		t.Errorf("desc.LDSBankStride = %d, want 33", desc.LDSBankStride)
	}
	expectedSharedMem := 32 * 33 * 4 // 4224 bytes
	if desc.SharedMemorySize != expectedSharedMem {
		t.Errorf("desc.SharedMemorySize = %d, want %d", desc.SharedMemorySize, expectedSharedMem)
	}
	if desc.PushConstants.Width != int32(convDim) || desc.PushConstants.Height != int32(T) {
		t.Errorf("push constants = (%d, %d), want (%d, %d)", desc.PushConstants.Width, desc.PushConstants.Height, convDim, T)
	}
	wantGridX := (convDim + 31) / 32
	wantGridY := (T + 31) / 32
	if desc.GridX != wantGridX || desc.GridY != wantGridY || desc.GridZ != 1 {
		t.Errorf("grid = (%d, %d, %d), want (%d, %d, 1)", desc.GridX, desc.GridY, desc.GridZ, wantGridX, wantGridY)
	}
	if len(desc.Bindings) != 2 {
		t.Fatalf("len(desc.Bindings) = %d, want 2", len(desc.Bindings))
	}
}

// TestTiledChannelTranspose_ShaderExecution verifies that the modeled shader execution
// ExecuteTiledTransposeShader produces exact bitwise parity with TiledChannelTranspose
// across various sequence lengths and channel dimensions.
func TestTiledChannelTranspose_ShaderExecution(t *testing.T) {
	testCases := []struct {
		T       int
		convDim int
	}{
		{T: 1, convDim: 1},
		{T: 7, convDim: 13},
		{T: 16, convDim: 32},
		{T: 32, convDim: 32},
		{T: 33, convDim: 33},
		{T: 64, convDim: 128},
		{T: 64, convDim: 10240},
	}

	cfg := DefaultTiledChannelTransposeConfig()

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("T%d_C%d", tc.T, tc.convDim), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(tc.T*777 + tc.convDim)))
			total := tc.T * tc.convDim
			input := make([]float32, total)
			for i := range input {
				input[i] = rng.Float32()*2.0 - 1.0
			}

			desc := cfg.Descriptor(tc.convDim, tc.T)
			shaderOut, err := ExecuteTiledTransposeShader(input, tc.convDim, tc.T, desc)
			if err != nil {
				t.Fatalf("ExecuteTiledTransposeShader failed: %v", err)
			}

			tiledOut, err := TiledChannelTranspose(input, tc.T, tc.convDim, cfg, nil)
			if err != nil {
				t.Fatalf("TiledChannelTranspose failed: %v", err)
			}

			if len(shaderOut) != len(tiledOut) {
				t.Fatalf("length mismatch: shaderOut=%d, tiledOut=%d", len(shaderOut), len(tiledOut))
			}

			for i := range shaderOut {
				if shaderOut[i] != tiledOut[i] {
					t.Fatalf("element mismatch at %d: shaderOut=%g, tiledOut=%g", i, shaderOut[i], tiledOut[i])
				}
			}
		})
	}
}

// TestTiledChannelTranspose_PipelineAuditWiring verifies that TiledChannelTranspose,
// TiledChannelTransposeInverse, and TiledConvConcatForward attach the shader descriptor
// to the audit struct.
func TestTiledChannelTranspose_PipelineAuditWiring(t *testing.T) {
	const T = 32
	const convDim = 128
	const K = 4
	cfg := DefaultTiledChannelTransposeConfig()

	input := make([]float32, T*convDim)
	for i := range input {
		input[i] = float32(i)
	}

	aud := &TiledChannelTransposeAudit{}
	_, err := TiledChannelTranspose(input, T, convDim, cfg, aud)
	if err != nil {
		t.Fatalf("TiledChannelTranspose failed: %v", err)
	}
	if aud.ShaderDescriptor == nil {
		t.Fatal("expected aud.ShaderDescriptor != nil")
	}
	if aud.ShaderDescriptor.ShaderName != Qwen35GDNTiledTransposeShader {
		t.Errorf("ShaderName = %q, want %q", aud.ShaderDescriptor.ShaderName, Qwen35GDNTiledTransposeShader)
	}
	if aud.ShaderDescriptor.PushConstants.Width != convDim || aud.ShaderDescriptor.PushConstants.Height != T {
		t.Errorf("PushConstants mismatch: %+v", aud.ShaderDescriptor.PushConstants)
	}

	invAud := &TiledChannelTransposeAudit{}
	_, err = TiledChannelTransposeInverse(input, convDim, T, cfg, invAud)
	if err != nil {
		t.Fatalf("TiledChannelTransposeInverse failed: %v", err)
	}
	if invAud.ShaderDescriptor == nil {
		t.Fatal("expected invAud.ShaderDescriptor != nil")
	}
	if invAud.ShaderDescriptor.PushConstants.Width != T || invAud.ShaderDescriptor.PushConstants.Height != convDim {
		t.Errorf("invAud PushConstants mismatch: %+v", invAud.ShaderDescriptor.PushConstants)
	}

	convW := make([]float32, convDim*K)
	_, _, fwdAud, err := TiledConvConcatForward(input, convW, T, convDim, K, nil, cfg)
	if err != nil {
		t.Fatalf("TiledConvConcatForward failed: %v", err)
	}
	if fwdAud.ShaderDescriptor == nil {
		t.Fatal("expected fwdAud.ShaderDescriptor != nil")
	}
	if fwdAud.ShaderDescriptor.ShaderName != Qwen35GDNTiledTransposeShader {
		t.Errorf("fwdAud ShaderName = %q, want %q", fwdAud.ShaderDescriptor.ShaderName, Qwen35GDNTiledTransposeShader)
	}
}

// TestTiledChannelTranspose_ShaderFile verifies that the compute shader file
// internal/compute/shaders/qwen35_gdn_tiled_transpose.comp exists and adheres to
// the RDNA 3.5 (gfx1151) 32x32 Pad-1/Pad-2 LDS bank conflict elimination contract.
func TestTiledChannelTranspose_ShaderFile(t *testing.T) {
	candidates := []string{
		"shaders/qwen35_gdn_tiled_transpose.comp",
		"internal/compute/shaders/qwen35_gdn_tiled_transpose.comp",
		filepath.Join("..", "internal", "compute", "shaders", "qwen35_gdn_tiled_transpose.comp"),
	}

	var content string
	var foundPath string
	for _, p := range candidates {
		data, err := os.ReadFile(p)
		if err == nil {
			content = string(data)
			foundPath = p
			break
		}
	}

	if content == "" {
		t.Fatalf("could not find qwen35_gdn_tiled_transpose.comp in any candidate path: %v", candidates)
	}

	requiredTokens := []string{
		"#version 450",
		"local_size_x = 32",
		"local_size_y = 8",
		"shared float tile[32][33];",
		"PushConstants",
		"int width;",
		"int height;",
		"readonly buffer InBuf",
		"writeonly buffer OutBuf",
		"barrier();",
		"Nathanw1014/strix-halo-llamacpp",
		"16-channel",
		"LPDDR5X",
	}

	for _, tok := range requiredTokens {
		if !strings.Contains(content, tok) {
			t.Errorf("shader %s missing required token %q", foundPath, tok)
		}
	}
}

// TestTiledConvConcatForward_EndToEndParity compares TiledConvConcatForward against
// the naive scalar reference causal convolution across multiple sequence lengths T
// and Qwen 3.8 GDN dimension (convDim = 10,240, K = 4).
func TestTiledConvConcatForward_EndToEndParity(t *testing.T) {
	const convDim = 10240
	const K = 4
	cfg := DefaultTiledChannelTransposeConfig()

	for _, T := range []int{1, 4, 16, 64} {
		t.Run(fmt.Sprintf("T%d", T), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(T * 42)))
			input := make([]float32, T*convDim)
			for i := range input {
				input[i] = rng.Float32()*2.0 - 1.0
			}
			convW := make([]float32, convDim*K)
			for i := range convW {
				convW[i] = rng.Float32()*0.5 - 0.25
			}

			// 1. Reference token-major causal depthwise conv (as in qwen35.go)
			refOut := make([]float32, T*convDim)
			for step := 0; step < T; step++ {
				for c := 0; c < convDim; c++ {
					var acc float32
					cb := c * K
					for j := 0; j < K; j++ {
						ti := step - (K - 1) + j
						if ti >= 0 {
							acc += convW[cb+j] * input[ti*convDim+c]
						}
					}
					refOut[step*convDim+c] = Silu(acc)
				}
			}

			// 2. Tiled memory channel transpose conv
			tiledOut, _, aud, err := TiledConvConcatForward(input, convW, T, convDim, K, nil, cfg)
			if err != nil {
				t.Fatalf("TiledConvConcatForward failed: %v", err)
			}

			// Parity verification
			if err := aud.AssertZeroBankConflicts(); err != nil {
				t.Errorf("audit bank conflicts: %v", err)
			}
			if aud.TokensProcessed != T || aud.ChannelsProcessed != convDim {
				t.Errorf("audit tokens/channels mismatch: %d / %d", aud.TokensProcessed, aud.ChannelsProcessed)
			}
			expectedEliminated := int64((K - 1) * T * convDim * 4)
			if aud.DRAMReadsEliminated != expectedEliminated {
				t.Errorf("DRAMReadsEliminated = %d, want %d", aud.DRAMReadsEliminated, expectedEliminated)
			}

			maxDelta := MaxAbsDelta(refOut, tiledOut)
			if maxDelta > 1e-6 {
				t.Errorf("T=%d max abs delta %g > 1e-6", T, maxDelta)
			}
			cosine := CosineSimilarity(refOut, tiledOut)
			if cosine < 0.999999 {
				t.Errorf("T=%d cosine similarity %g < 0.999999", T, cosine)
			}
		})
	}
}

// TestTiledConvConcatForward_StateContinuity verifies that executing two consecutive
// sequence chunks with persistent convState produces identical outputs to a single
// concatenated sequence pass.
func TestTiledConvConcatForward_StateContinuity(t *testing.T) {
	const convDim = 128
	const K = 4
	const T1 = 16
	const T2 = 16
	const TTotal = T1 + T2

	rng := rand.New(rand.NewSource(999))
	fullInput := make([]float32, TTotal*convDim)
	for i := range fullInput {
		fullInput[i] = rng.Float32()*2.0 - 1.0
	}
	convW := make([]float32, convDim*K)
	for i := range convW {
		convW[i] = rng.Float32()*0.5 - 0.25
	}
	cfg := DefaultTiledChannelTransposeConfig()

	// 1. Single pass over full sequence
	fullOut, fullFinalState, _, err := TiledConvConcatForward(fullInput, convW, TTotal, convDim, K, nil, cfg)
	if err != nil {
		t.Fatalf("full pass failed: %v", err)
	}

	// 2. Chunk 1 pass
	chunk1Input := fullInput[:T1*convDim]
	chunk1Out, chunk1State, _, err := TiledConvConcatForward(chunk1Input, convW, T1, convDim, K, nil, cfg)
	if err != nil {
		t.Fatalf("chunk 1 pass failed: %v", err)
	}

	// 3. Chunk 2 pass using chunk1State
	chunk2Input := fullInput[T1*convDim:]
	chunk2Out, chunk2State, _, err := TiledConvConcatForward(chunk2Input, convW, T2, convDim, K, chunk1State, cfg)
	if err != nil {
		t.Fatalf("chunk 2 pass failed: %v", err)
	}

	// Concatenate chunked outputs
	stitchedOut := append(chunk1Out, chunk2Out...)

	// Assert output equivalence
	maxDelta := MaxAbsDelta(fullOut, stitchedOut)
	if maxDelta > 1e-6 {
		t.Errorf("stitched output max delta %g > 1e-6", maxDelta)
	}
	cosine := CosineSimilarity(fullOut, stitchedOut)
	if cosine < 0.999999 {
		t.Errorf("stitched output cosine similarity %g < 0.999999", cosine)
	}

	// Assert final state equivalence
	stateDelta := MaxAbsDelta(fullFinalState, chunk2State)
	if stateDelta > 1e-6 {
		t.Errorf("final state delta %g > 1e-6", stateDelta)
	}
}

// TestTiledConvConcatForwardSlices verifies the [][]float32 slice-of-slices API used by model/qwen35.go.
func TestTiledConvConcatForwardSlices(t *testing.T) {
	const T = 8
	const convDim = 64
	const K = 4

	rng := rand.New(rand.NewSource(123))
	mixed := make([][]float32, T)
	for step := 0; step < T; step++ {
		row := make([]float32, convDim)
		for c := 0; c < convDim; c++ {
			row[c] = rng.Float32()*2.0 - 1.0
		}
		mixed[step] = row
	}

	convW := make([]float32, convDim*K)
	for i := range convW {
		convW[i] = rng.Float32()*0.5 - 0.25
	}

	convOut, nextState, aud, err := TiledConvConcatForwardSlices(mixed, convW, convDim, K, nil)
	if err != nil {
		t.Fatalf("TiledConvConcatForwardSlices failed: %v", err)
	}
	if len(convOut) != T {
		t.Fatalf("convOut length = %d, want %d", len(convOut), T)
	}
	if len(nextState) != (K-1)*convDim {
		t.Fatalf("nextState length = %d, want %d", len(nextState), (K-1)*convDim)
	}
	if err := aud.AssertZeroBankConflicts(); err != nil {
		t.Errorf("audit bank conflicts: %v", err)
	}

	// Verify against reference
	for step := 0; step < T; step++ {
		for c := 0; c < convDim; c++ {
			var acc float32
			cb := c * K
			for j := 0; j < K; j++ {
				ti := step - (K - 1) + j
				if ti >= 0 {
					acc += convW[cb+j] * mixed[ti][c]
				}
			}
			expected := Silu(acc)
			if math.Abs(float64(convOut[step][c]-expected)) > 1e-6 {
				t.Fatalf("mismatch at t=%d, c=%d: got %g, want %g", step, c, convOut[step][c], expected)
			}
		}
	}
}

// TestHasTiledChannelTranspose verifies feature registration for Strix Halo / gfx1151.
func TestHasTiledChannelTranspose(t *testing.T) {
	if !HasTiledChannelTranspose() {
		t.Fatal("expected HasTiledChannelTranspose() == true")
	}
}

// TestWave32_Ticket516_DeltaNet16ChannelTranspose verifies Ticket #516 requirements:
// - Implement 2D tiled channel transpose helper for DeltaNet linear attention conv-state concatenation.
// - Distribute reads across all 16 memory channels on the 256-bit bus, eliminating single-channel stride camping.
// - Add unit tests verifying 16-channel interleaving and stride alignment.
func TestWave32_Ticket516_DeltaNet16ChannelTranspose(t *testing.T) {
	const (
		convDim = 10240 // Qwen 3.8 GDN dimension
		T       = 32
		K       = 4
	)

	// 1. Verify 256-bit bus alignment invariant validation
	validStrideBytes := convDim * 4 // 40,960 bytes (multiple of 32)
	if !ValidateDeltaNet256BitBusAlignment(validStrideBytes) {
		t.Errorf("expected ValidateDeltaNet256BitBusAlignment(%d) = true", validStrideBytes)
	}
	invalidStrideBytes := 40960 + 12 // not divisible by 32
	if ValidateDeltaNet256BitBusAlignment(invalidStrideBytes) {
		t.Errorf("expected ValidateDeltaNet256BitBusAlignment(%d) = false", invalidStrideBytes)
	}
	if ValidateDeltaNet256BitBusAlignment(0) || ValidateDeltaNet256BitBusAlignment(-32) {
		t.Errorf("expected false for <= 0 stride bytes")
	}

	// 2. Verify channel camping on untiled access vs uniform spread on 2D tiled transpose
	untiledRep := SimulateDeltaNet16ChannelInterleaving(T, convDim, false)
	if !untiledRep.ChannelCamping {
		t.Errorf("expected untiled ChannelCamping = true")
	}
	if untiledRep.ActiveChannels > 2 {
		t.Errorf("untiled ActiveChannels = %d, expected <= 2 (single-channel camping)", untiledRep.ActiveChannels)
	}
	if untiledRep.Entropy >= 0.25 {
		t.Errorf("untiled Entropy = %f, expected < 0.25", untiledRep.Entropy)
	}
	if untiledRep.EstimatedBW_GBps != 13.7 {
		t.Errorf("untiled EstimatedBW_GBps = %f, want 13.7 GB/s", untiledRep.EstimatedBW_GBps)
	}

	tiledRep := SimulateDeltaNet16ChannelInterleaving(T, convDim, true)
	if tiledRep.ChannelCamping {
		t.Errorf("expected tiled ChannelCamping = false")
	}
	if tiledRep.ActiveChannels != StrixHaloBusChannels {
		t.Errorf("tiled ActiveChannels = %d, want %d", tiledRep.ActiveChannels, StrixHaloBusChannels)
	}
	if tiledRep.Entropy <= 0.95 {
		t.Errorf("tiled Entropy = %f, expected > 0.95 (uniform 16-channel spread)", tiledRep.Entropy)
	}
	if tiledRep.EstimatedBW_GBps < 138.0 {
		t.Errorf("tiled EstimatedBW_GBps = %f, want >= 138.0 GB/s (138.9 GB/s)", tiledRep.EstimatedBW_GBps)
	}
	if tiledRep.ThroughputLift < 1.07 {
		t.Errorf("tiled ThroughputLift = %f, want >= 1.07 (+7.2%% lift)", tiledRep.ThroughputLift)
	}
	if !tiledRep.BusAlignmentValid {
		t.Errorf("tiled BusAlignmentValid = false, want true")
	}

	// Verify all 16 channels receive non-zero, balanced accesses
	for ch := 0; ch < StrixHaloBusChannels; ch++ {
		if tiledRep.ChannelCounts[ch] == 0 {
			t.Errorf("channel %d has 0 accesses in tiled layout: %v", ch, tiledRep.ChannelCounts)
		}
	}

	// 3. Verify Tiled16ChannelTransposeConcat end-to-end execution and numerical parity
	rng := rand.New(rand.NewSource(516))
	input := make([]float32, T*convDim)
	for i := range input {
		input[i] = rng.Float32()*2.0 - 1.0
	}
	convW := make([]float32, convDim*K)
	for i := range convW {
		convW[i] = rng.Float32()*0.5 - 0.25
	}

	output, nextState, report, err := Tiled16ChannelTransposeConcat(input, convW, T, convDim, K, nil)
	if err != nil {
		t.Fatalf("Tiled16ChannelTransposeConcat failed: %v", err)
	}
	if len(output) != T*convDim {
		t.Fatalf("output length = %d, want %d", len(output), T*convDim)
	}
	if len(nextState) != (K-1)*convDim {
		t.Fatalf("nextState length = %d, want %d", len(nextState), (K-1)*convDim)
	}
	if report.ActiveChannels != 16 {
		t.Errorf("report.ActiveChannels = %d, want 16", report.ActiveChannels)
	}
	if report.Entropy <= 0.95 {
		t.Errorf("report.Entropy = %f, want > 0.95", report.Entropy)
	}
	if report.ChannelCamping {
		t.Errorf("report.ChannelCamping = true, want false")
	}

	// Verify against direct reference calculation
	for step := 0; step < T; step++ {
		for c := 0; c < 16; c++ { // sample check first 16 channels
			var acc float32
			cb := c * K
			for j := 0; j < K; j++ {
				ti := step - (K - 1) + j
				if ti >= 0 {
					acc += convW[cb+j] * input[ti*convDim+c]
				}
			}
			expected := Silu(acc)
			got := output[step*convDim+c]
			if math.Abs(float64(got-expected)) > 1e-6 {
				t.Fatalf("mismatch at step %d, c %d: got %f, want %f", step, c, got, expected)
			}
		}
	}
}
