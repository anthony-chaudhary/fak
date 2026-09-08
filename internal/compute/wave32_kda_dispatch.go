package compute

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
)

// ComputeQwenGDNDecay computes the state decay factor for Qwen 3.8 GDN:
// decay = exp(-exp(aLog) * softplus(aProj + dtBias))
func ComputeQwenGDNDecay(aLog, aProj, dtBias float32) float32 {
	if aLog > 50.0 {
		aLog = 50.0
	} else if aLog < -50.0 {
		aLog = -50.0
	}
	aa := float32(math.Exp(float64(aLog)))
	dt := Softplus(aProj + dtBias)
	decayArg := -aa * dt
	val := float32(math.Exp(float64(decayArg)))
	if math.IsNaN(float64(val)) || val < 1e-5 {
		return 1e-5
	}
	if val > 1.0 {
		return 1.0
	}
	return val
}

// ComputeQwenGDNBeta computes the beta gating factor for Qwen 3.8 GDN:
// beta = sigmoid(bProj) = 1 / (1 + exp(-bProj))
func ComputeQwenGDNBeta(bProj float32) float32 {
	if math.IsNaN(float64(bProj)) {
		return float32(math.NaN())
	}
	return float32(1.0 / (1.0 + math.Exp(-float64(bProj))))
}

// Wave32Conv1DState maintains the rolling history buffer for causal depthwise 1D convolution.
type Wave32Conv1DState struct {
	KernelSize int         `json:"kernel_size"`
	Channels   int         `json:"channels"`
	Buffer     [][]float32 `json:"buffer"`
}

// NewWave32Conv1DState allocates a new rolling history buffer for kernelSize and channels.
func NewWave32Conv1DState(kernelSize, channels int) (*Wave32Conv1DState, error) {
	if kernelSize < 1 {
		return nil, fmt.Errorf("compute: invalid conv1d kernel size %d, must be >= 1", kernelSize)
	}
	if channels <= 0 {
		return nil, fmt.Errorf("compute: invalid conv1d channels %d, must be > 0", channels)
	}
	hist := kernelSize - 1
	buf := make([][]float32, hist)
	for i := range buf {
		buf[i] = make([]float32, channels)
	}
	return &Wave32Conv1DState{
		KernelSize: kernelSize,
		Channels:   channels,
		Buffer:     buf,
	}, nil
}

// Step computes one causal 1D depthwise convolution step followed by SiLU activation,
// and updates the rolling history buffer.
// weights has length channels * kernelSize, laid out as weights[c*kernelSize + k].
func (cs *Wave32Conv1DState) Step(input []float32, weights []float32) ([]float32, error) {
	return cs.StepWithBias(input, weights, nil)
}

// StepWithBias computes one causal 1D depthwise convolution step with optional bias
// followed by SiLU activation, and updates the rolling history buffer.
func (cs *Wave32Conv1DState) StepWithBias(input []float32, weights []float32, bias []float32) ([]float32, error) {
	if len(input) != cs.Channels {
		return nil, fmt.Errorf("compute: conv1d input length %d != channels %d", len(input), cs.Channels)
	}
	expectedWeights := cs.Channels * cs.KernelSize
	if len(weights) != expectedWeights {
		return nil, fmt.Errorf("compute: conv1d weights length %d != expected %d", len(weights), expectedWeights)
	}
	if bias != nil && len(bias) != cs.Channels {
		return nil, fmt.Errorf("compute: conv1d bias length %d != channels %d", len(bias), cs.Channels)
	}
	hist := cs.KernelSize - 1
	out := make([]float32, cs.Channels)
	for c := 0; c < cs.Channels; c++ {
		var acc float32
		if bias != nil {
			acc = bias[c]
		}
		wBase := c * cs.KernelSize
		for j := 0; j < hist; j++ {
			acc += weights[wBase+j] * cs.Buffer[j][c]
		}
		acc += weights[wBase+hist] * input[c]
		out[c] = Silu(acc)
	}
	if hist > 0 {
		for j := 0; j < hist-1; j++ {
			copy(cs.Buffer[j], cs.Buffer[j+1])
		}
		copy(cs.Buffer[hist-1], input)
	}
	return out, nil
}

// ReferenceGDNStep computes a single token state transition using the mathematical
// reference formulation for Qwen 3.8 GDN (Gated Delta Net).
//
// Formulation:
//  1. Forget decay: S'_{i, j} = S_{i, j} * decay
//  2. Memory retrieval: kvmem_j = sum_{i=0}^{127} S'_{i, j} * k_i
//  3. Delta update: delta_j = (v_j - kvmem_j) * beta
//  4. State transition: S_{next}[i, j] = S'_{i, j} + k_i * delta_j
//  5. Readout: readout_j = sum_{i=0}^{127} S_{next}[i, j] * q_i
//  6. RMSNorm: invRMS = 1 / sqrt(sum(readout^2) / 128 + eps)
//  7. Output gate: output_j = norm_j * (readout_j * invRMS) * silu(z_j)
func ReferenceGDNStep(
	state [Wave32HeadDim][Wave32HeadDim]float32,
	q, k, v, z []float32,
	norm []float32,
	beta, decay, eps float32,
) (output []float32, nextState [Wave32HeadDim][Wave32HeadDim]float32) {
	if eps <= 0 {
		eps = Wave32GDNDefaultRMSNormEps
	}
	output = make([]float32, Wave32HeadDim)

	// Step 1 & 2: Decay state and compute kvmem_j = sum_i S'[i, j] * k[i]
	var sPrime [Wave32HeadDim][Wave32HeadDim]float32
	var kvmem [Wave32HeadDim]float32
	for i := 0; i < Wave32HeadDim; i++ {
		ki := k[i]
		for j := 0; j < Wave32HeadDim; j++ {
			sp := state[i][j] * decay
			sPrime[i][j] = sp
			kvmem[j] += sp * ki
		}
	}

	// Step 3: Compute delta error per value dimension
	var delta [Wave32HeadDim]float32
	for j := 0; j < Wave32HeadDim; j++ {
		delta[j] = (v[j] - kvmem[j]) * beta
	}

	// Step 4: Update state S_next = S' + k (x) delta, and compute readout = sum_i S_next[i, j] * q[i]
	var readout [Wave32HeadDim]float32
	for i := 0; i < Wave32HeadDim; i++ {
		ki := k[i]
		qi := q[i]
		for j := 0; j < Wave32HeadDim; j++ {
			sn := sPrime[i][j] + ki*delta[j]
			nextState[i][j] = sn
			readout[j] += sn * qi
		}
	}

	// Step 5: RMSNorm calculation
	var sumSquares float64
	for j := 0; j < Wave32HeadDim; j++ {
		sumSquares += float64(readout[j]) * float64(readout[j])
	}
	invRMS := float32(1.0 / math.Sqrt(sumSquares/float64(Wave32HeadDim)+float64(eps)))

	// Step 6: Output gate with RMSNorm and SiLU
	for j := 0; j < Wave32HeadDim; j++ {
		normVal := float32(1.0)
		if j < len(norm) {
			normVal = norm[j]
		}
		zVal := float32(0.0)
		if j < len(z) {
			zVal = z[j]
		}
		output[j] = normVal * (readout[j] * invRMS) * Silu(zVal)
	}

	return output, nextState
}

// ReferenceGDNSequence runs the mathematical reference GDN formulation across T tokens.
func ReferenceGDNSequence(
	initialState [Wave32HeadDim][Wave32HeadDim]float32,
	qSeq, kSeq, vSeq, zSeq [][]float32,
	bProjSeq, aProjSeq []float32,
	aLog, dtBias float32,
	norm []float32,
	eps float32,
) (outputs [][]float32, finalState [Wave32HeadDim][Wave32HeadDim]float32, err error) {
	tokens := len(qSeq)
	if tokens == 0 {
		return nil, initialState, errors.New("compute: empty sequence for reference GDN")
	}
	if len(kSeq) != tokens || len(vSeq) != tokens || len(zSeq) != tokens || len(bProjSeq) != tokens || len(aProjSeq) != tokens {
		return nil, initialState, errors.New("compute: sequence length mismatch among operands")
	}

	currentState := initialState
	outputs = make([][]float32, tokens)

	for t := 0; t < tokens; t++ {
		if len(qSeq[t]) != Wave32HeadDim || len(kSeq[t]) != Wave32HeadDim || len(vSeq[t]) != Wave32HeadDim || len(zSeq[t]) != Wave32HeadDim {
			return nil, currentState, fmt.Errorf("compute: token %d dimension mismatch, want %d", t, Wave32HeadDim)
		}

		beta := ComputeQwenGDNBeta(bProjSeq[t])
		decay := ComputeQwenGDNDecay(aLog, aProjSeq[t], dtBias)

		out, next := ReferenceGDNStep(currentState, qSeq[t], kSeq[t], vSeq[t], zSeq[t], norm, beta, decay, eps)
		outputs[t] = out
		currentState = next
	}

	return outputs, currentState, nil
}

// Wave32GDNStep executes a single token state transition on the simulated Wave32 workgroup
// for Qwen 3.8 GDN. The recurrent state resides strictly in thread VGPR registers across
// the entire step, with zero reads or writes to DRAM or LDS.
// RMSNorm is computed using 4 Wave32 wavefront tree reductions (4 * 5 = 20 shuffles total).
func (wg *Wave32Workgroup) Wave32GDNStep(
	q, k, v, z []float32,
	norm []float32,
	beta, decay, eps float32,
) []float32 {
	if eps <= 0 {
		eps = Wave32GDNDefaultRMSNormEps
	}
	output := make([]float32, Wave32HeadDim)

	// In the Wave32 kernel, the 128 key dimensions map to the 128 threads across 4 wavefronts.
	// Each thread i holds row i of the recurrent state in its MatrixStateRegs [128]float32.

	// Step 1: Thread i applies decay to its register-resident row and computes partial kvmem
	var kvmem [Wave32HeadDim]float32
	for i := 0; i < Wave32TotalLanes; i++ {
		th := wg.GetThread(i)
		ki := k[i]
		for j := 0; j < Wave32HeadDim; j++ {
			val := th.MatrixStateRegs[j] * decay
			th.MatrixStateRegs[j] = val
			wg.Audit.VGPRReads++
			wg.Audit.VGPRWrites++
			kvmem[j] += val * ki
		}
	}

	// Step 2: Compute delta error per value dimension
	var delta [Wave32HeadDim]float32
	for j := 0; j < Wave32HeadDim; j++ {
		delta[j] = (v[j] - kvmem[j]) * beta
	}

	// Step 3: Thread i updates state registers with ki * delta[j] and computes readout
	var readout [Wave32HeadDim]float32
	for i := 0; i < Wave32TotalLanes; i++ {
		th := wg.GetThread(i)
		ki := k[i]
		qi := q[i]
		for j := 0; j < Wave32HeadDim; j++ {
			sn := th.MatrixStateRegs[j] + ki*delta[j]
			th.MatrixStateRegs[j] = sn
			wg.Audit.VGPRReads++
			wg.Audit.VGPRWrites++
			readout[j] += sn * qi
		}
	}

	// Step 4: RMSNorm calculation across 4 Wave32 wavefronts using IntraWaveReduceSum.
	// 4 wavefronts * 5 shuffles = 20 shuffles per step without touching LDS or DRAM.
	var sumSquares float32
	for w := 0; w < Wave32NumWavefronts; w++ {
		var vals [Wave32WavefrontSize]float32
		base := w * Wave32WavefrontSize
		for l := 0; l < Wave32WavefrontSize; l++ {
			r := readout[base+l]
			vals[l] = r * r
		}
		sumSquares += wg.Waves[w].IntraWaveReduceSum(vals, &wg.Audit)
	}
	invRMS := float32(1.0 / math.Sqrt(float64(sumSquares)/float64(Wave32HeadDim)+float64(eps)))

	// Step 5: Output gate: norm[j] * (readout[j] * invRMS) * silu(z[j])
	for j := 0; j < Wave32HeadDim; j++ {
		normVal := float32(1.0)
		if j < len(norm) {
			normVal = norm[j]
		}
		zVal := float32(0.0)
		if j < len(z) {
			zVal = z[j]
		}
		output[j] = normVal * (readout[j] * invRMS) * Silu(zVal)
	}

	// Explicit witness: Inner loop state updates touched ZERO DRAM or LDS bytes.
	wg.Audit.TokensProcessed++
	return output
}

// Wave32GDNSequence executes the autoregressive sequence generation loop across T tokens for Qwen 3.8 GDN.
// State is uploaded once to VGPRs before the loop, maintained in registers for all T tokens
// (with strictly zero DRAM traffic in the inner loop), and downloaded once at the end.
func (wg *Wave32Workgroup) Wave32GDNSequence(
	initialState [Wave32HeadDim][Wave32HeadDim]float32,
	qSeq, kSeq, vSeq, zSeq [][]float32,
	bProjSeq, aProjSeq []float32,
	aLog, dtBias float32,
	norm []float32,
	eps float32,
) (outputs [][]float32, finalState [Wave32HeadDim][Wave32HeadDim]float32, err error) {
	tokens := len(qSeq)
	if tokens == 0 {
		return nil, initialState, errors.New("compute: empty sequence for Wave32 GDN")
	}
	if len(kSeq) != tokens || len(vSeq) != tokens || len(zSeq) != tokens || len(bProjSeq) != tokens || len(aProjSeq) != tokens {
		return nil, initialState, errors.New("compute: sequence length mismatch among operands")
	}

	// Reset memory audit tracker for this sequence.
	wg.Audit = Wave32KDAMemoryAudit{}

	// Load initial state into VGPRs (one-time setup before loop; 0 inner-loop DRAM bytes).
	wg.LoadMatrixState(initialState)

	outputs = make([][]float32, tokens)

	// Autoregressive token sequence loop: all state mutations occur in VGPR registers.
	for t := 0; t < tokens; t++ {
		if len(qSeq[t]) != Wave32HeadDim || len(kSeq[t]) != Wave32HeadDim || len(vSeq[t]) != Wave32HeadDim || len(zSeq[t]) != Wave32HeadDim {
			return nil, initialState, fmt.Errorf("compute: token %d dimension mismatch, want %d", t, Wave32HeadDim)
		}
		beta := ComputeQwenGDNBeta(bProjSeq[t])
		decay := ComputeQwenGDNDecay(aLog, aProjSeq[t], dtBias)
		outputs[t] = wg.Wave32GDNStep(qSeq[t], kSeq[t], vSeq[t], zSeq[t], norm, beta, decay, eps)
	}

	// Download final state from VGPRs (one-time readback after loop).
	finalState = wg.ReadMatrixState()

	// Assert that inner loop incurred strictly zero DRAM and LDS traffic.
	if err := wg.Audit.AssertZeroDRAMTraffic(); err != nil {
		return nil, finalState, err
	}

	return outputs, finalState, nil
}

// Wave32GatedDeltaNetStep executes a fused Gated-DeltaNet step on this simulated RDNA 3.5 workgroup.
func (wg *Wave32Workgroup) Wave32GatedDeltaNetStep(
	q, k, v, z []float32,
	norm []float32,
	beta, decay, eps float32,
) []float32 {
	return wg.Wave32GDNStep(q, k, v, z, norm, beta, decay, eps)
}

// HasVectorizedDeltaNet reports whether this process can execute an optimized
// DeltaNet kernel. The environment can disable a detected kernel, but cannot
// force one on when the CPU/OS does not support its instruction set.
func HasVectorizedDeltaNet() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_VECTORIZED_DELTANET")))
	switch v {
	case "0", "false", "no", "off":
		return false
	default:
		return hasDeltaNetSIMD()
	}
}

// HasVectorizedDeltaNetFor reports whether the optimized kernel supports the
// requested key/value head geometry. The AVX-512 implementation is deliberately
// limited to Qwen's canonical 128x128 recurrence; other shapes use the Go path.
func HasVectorizedDeltaNetFor(kHd, vHd int) bool {
	return kHd == 128 && vHd == 128 && HasVectorizedDeltaNet()
}

// HasTiledChannelTranspose reports whether the tiled memory channel transpose
// kernel for DeltaNet linear attention conv concat is available.
func HasTiledChannelTranspose() bool {
	return true
}

// Wave32GatedDeltaNetStep performs a fused Gated-DeltaNet recurrent step across head dimension d=128
// (or general head dimensions). It fuses:
//  1. Decay scaling: st[i, d] *= g
//  2. Memory retrieval: kvmem[d] = sum_i st[i, d] * kn[i]
//  3. Delta computation: delta[d] = (vh[d] - kvmem[d]) * bt
//  4. Recurrent state update: st[i, d] += kn[i] * delta[d]
//  5. Readout projection: od[d] += st[i, d] * qn[i]
//
// Recurrent state updates and memory retrievals are unrolled across the head dimension (d=128).
func Wave32GatedDeltaNetStep(
	st []float32,
	qn, kn, vh []float32,
	bt, g float32,
	od, kvmem, delta []float32,
) {
	kHd := len(qn)
	vHd := len(vh)
	if kHd == 0 || vHd == 0 || len(kn) < kHd || len(st) < kHd*vHd {
		return
	}
	if len(kvmem) < vHd || len(delta) < vHd || len(od) < vHd {
		return
	}

	if HasVectorizedDeltaNetFor(kHd, vHd) && tryDeltaNetSIMD(st, qn, kn, vh, bt, g, od, kvmem, delta) {
		return
	}
	wave32GatedDeltaNetStepGo(st, qn, kn, vh, bt, g, od, kvmem, delta)
}

func wave32GatedDeltaNetStepGo(
	st []float32,
	qn, kn, vh []float32,
	bt, g float32,
	od, kvmem, delta []float32,
) {
	kHd := len(qn)
	vHd := len(vh)

	clear(kvmem[:vHd])

	// Fast path for canonical head dimension vHd == 128 (8 chunks of 16 float32s)
	if vHd == 128 {
		for i := 0; i < kHd; i++ {
			ki := kn[i]
			base := i * 128
			for d := 0; d < 128; d += 16 {
				st[base+d] *= g
				kvmem[d] += st[base+d] * ki
				st[base+d+1] *= g
				kvmem[d+1] += st[base+d+1] * ki
				st[base+d+2] *= g
				kvmem[d+2] += st[base+d+2] * ki
				st[base+d+3] *= g
				kvmem[d+3] += st[base+d+3] * ki
				st[base+d+4] *= g
				kvmem[d+4] += st[base+d+4] * ki
				st[base+d+5] *= g
				kvmem[d+5] += st[base+d+5] * ki
				st[base+d+6] *= g
				kvmem[d+6] += st[base+d+6] * ki
				st[base+d+7] *= g
				kvmem[d+7] += st[base+d+7] * ki
				st[base+d+8] *= g
				kvmem[d+8] += st[base+d+8] * ki
				st[base+d+9] *= g
				kvmem[d+9] += st[base+d+9] * ki
				st[base+d+10] *= g
				kvmem[d+10] += st[base+d+10] * ki
				st[base+d+11] *= g
				kvmem[d+11] += st[base+d+11] * ki
				st[base+d+12] *= g
				kvmem[d+12] += st[base+d+12] * ki
				st[base+d+13] *= g
				kvmem[d+13] += st[base+d+13] * ki
				st[base+d+14] *= g
				kvmem[d+14] += st[base+d+14] * ki
				st[base+d+15] *= g
				kvmem[d+15] += st[base+d+15] * ki
			}
		}

		for d := 0; d < 128; d += 16 {
			delta[d] = (vh[d] - kvmem[d]) * bt
			delta[d+1] = (vh[d+1] - kvmem[d+1]) * bt
			delta[d+2] = (vh[d+2] - kvmem[d+2]) * bt
			delta[d+3] = (vh[d+3] - kvmem[d+3]) * bt
			delta[d+4] = (vh[d+4] - kvmem[d+4]) * bt
			delta[d+5] = (vh[d+5] - kvmem[d+5]) * bt
			delta[d+6] = (vh[d+6] - kvmem[d+6]) * bt
			delta[d+7] = (vh[d+7] - kvmem[d+7]) * bt
			delta[d+8] = (vh[d+8] - kvmem[d+8]) * bt
			delta[d+9] = (vh[d+9] - kvmem[d+9]) * bt
			delta[d+10] = (vh[d+10] - kvmem[d+10]) * bt
			delta[d+11] = (vh[d+11] - kvmem[d+11]) * bt
			delta[d+12] = (vh[d+12] - kvmem[d+12]) * bt
			delta[d+13] = (vh[d+13] - kvmem[d+13]) * bt
			delta[d+14] = (vh[d+14] - kvmem[d+14]) * bt
			delta[d+15] = (vh[d+15] - kvmem[d+15]) * bt
		}

		for i := 0; i < kHd; i++ {
			ki := kn[i]
			qi := qn[i]
			base := i * 128
			for d := 0; d < 128; d += 16 {
				st[base+d] += ki * delta[d]
				od[d] += st[base+d] * qi
				st[base+d+1] += ki * delta[d+1]
				od[d+1] += st[base+d+1] * qi
				st[base+d+2] += ki * delta[d+2]
				od[d+2] += st[base+d+2] * qi
				st[base+d+3] += ki * delta[d+3]
				od[d+3] += st[base+d+3] * qi
				st[base+d+4] += ki * delta[d+4]
				od[d+4] += st[base+d+4] * qi
				st[base+d+5] += ki * delta[d+5]
				od[d+5] += st[base+d+5] * qi
				st[base+d+6] += ki * delta[d+6]
				od[d+6] += st[base+d+6] * qi
				st[base+d+7] += ki * delta[d+7]
				od[d+7] += st[base+d+7] * qi
				st[base+d+8] += ki * delta[d+8]
				od[d+8] += st[base+d+8] * qi
				st[base+d+9] += ki * delta[d+9]
				od[d+9] += st[base+d+9] * qi
				st[base+d+10] += ki * delta[d+10]
				od[d+10] += st[base+d+10] * qi
				st[base+d+11] += ki * delta[d+11]
				od[d+11] += st[base+d+11] * qi
				st[base+d+12] += ki * delta[d+12]
				od[d+12] += st[base+d+12] * qi
				st[base+d+13] += ki * delta[d+13]
				od[d+13] += st[base+d+13] * qi
				st[base+d+14] += ki * delta[d+14]
				od[d+14] += st[base+d+14] * qi
				st[base+d+15] += ki * delta[d+15]
				od[d+15] += st[base+d+15] * qi
			}
		}
		return
	}

	// General head dimension path with unrolling by 8 / 4 / 1
	for i := 0; i < kHd; i++ {
		ki := kn[i]
		base := i * vHd
		d := 0
		for ; d+8 <= vHd; d += 8 {
			st[base+d] *= g
			kvmem[d] += st[base+d] * ki
			st[base+d+1] *= g
			kvmem[d+1] += st[base+d+1] * ki
			st[base+d+2] *= g
			kvmem[d+2] += st[base+d+2] * ki
			st[base+d+3] *= g
			kvmem[d+3] += st[base+d+3] * ki
			st[base+d+4] *= g
			kvmem[d+4] += st[base+d+4] * ki
			st[base+d+5] *= g
			kvmem[d+5] += st[base+d+5] * ki
			st[base+d+6] *= g
			kvmem[d+6] += st[base+d+6] * ki
			st[base+d+7] *= g
			kvmem[d+7] += st[base+d+7] * ki
		}
		for ; d < vHd; d++ {
			st[base+d] *= g
			kvmem[d] += st[base+d] * ki
		}
	}

	d := 0
	for ; d+8 <= vHd; d += 8 {
		delta[d] = (vh[d] - kvmem[d]) * bt
		delta[d+1] = (vh[d+1] - kvmem[d+1]) * bt
		delta[d+2] = (vh[d+2] - kvmem[d+2]) * bt
		delta[d+3] = (vh[d+3] - kvmem[d+3]) * bt
		delta[d+4] = (vh[d+4] - kvmem[d+4]) * bt
		delta[d+5] = (vh[d+5] - kvmem[d+5]) * bt
		delta[d+6] = (vh[d+6] - kvmem[d+6]) * bt
		delta[d+7] = (vh[d+7] - kvmem[d+7]) * bt
	}
	for ; d < vHd; d++ {
		delta[d] = (vh[d] - kvmem[d]) * bt
	}

	for i := 0; i < kHd; i++ {
		ki := kn[i]
		qi := qn[i]
		base := i * vHd
		d := 0
		for ; d+8 <= vHd; d += 8 {
			st[base+d] += ki * delta[d]
			od[d] += st[base+d] * qi
			st[base+d+1] += ki * delta[d+1]
			od[d+1] += st[base+d+1] * qi
			st[base+d+2] += ki * delta[d+2]
			od[d+2] += st[base+d+2] * qi
			st[base+d+3] += ki * delta[d+3]
			od[d+3] += st[base+d+3] * qi
			st[base+d+4] += ki * delta[d+4]
			od[d+4] += st[base+d+4] * qi
			st[base+d+5] += ki * delta[d+5]
			od[d+5] += st[base+d+5] * qi
			st[base+d+6] += ki * delta[d+6]
			od[d+6] += st[base+d+6] * qi
			st[base+d+7] += ki * delta[d+7]
			od[d+7] += st[base+d+7] * qi
		}
		for ; d < vHd; d++ {
			st[base+d] += ki * delta[d]
			od[d] += st[base+d] * qi
		}
	}
}
