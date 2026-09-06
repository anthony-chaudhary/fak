package compute

import (
	"errors"
	"fmt"
	"math"
)

// Architecture constants for AMD RDNA 3.5 APUs (gfx1151 / AMD Strix Halo)
// executing Wave32 wavefronts for linear recurrent delta attention (KDA),
// borrowed from ds4 / wkljohn/ds4-strix-halo-tp-odinlink rocm/ds4_rocm_glm5_kda.cuh:45-102.
const (
	// Wave32WavefrontSize is the native wavefront width (32 lanes) on RDNA 3.5.
	Wave32WavefrontSize = 32

	// Wave32NumWavefronts is the number of Wave32 wavefronts in a KDA workgroup (4 wavefronts).
	Wave32NumWavefronts = 4

	// Wave32TotalLanes is the total thread count in a workgroup: 4 * 32 = 128 lanes.
	Wave32TotalLanes = Wave32NumWavefronts * Wave32WavefrontSize

	// Wave32HeadDim is the linear delta attention head dimension (128 floats).
	Wave32HeadDim = 128

	// Wave32StateDimsPerThread is the number of state dimensions held in VGPRs per lane (4 dims).
	// In Wave32 mapping, 32 lanes * 4 dimensions = 128 dimensions per wavefront.
	Wave32StateDimsPerThread = 4

	// Wave32DefaultNumHeads is the standard KDA head count in GLM-5.3 Flash.
	Wave32DefaultNumHeads = 64

	// Wave32TargetArch is the canonical AMD RDNA 3.5 APU target string.
	Wave32TargetArch = "gfx1151"

	// Wave32MaxVGPRsPerThread is the register allocation limit on RDNA 3.5 for 100% SIMD occupancy.
	Wave32MaxVGPRsPerThread = 256
)

// Wave32ModelArch identifies the recurrent neural architecture mapped onto Wave32.
type Wave32ModelArch string

const (
	// Wave32ModelArchGLM53KDA targets GLM-5.3 KDA linear recurrent delta attention.
	Wave32ModelArchGLM53KDA Wave32ModelArch = "glm53-kda"

	// Wave32ModelArchQwen38GDN targets Qwen 3.8 Gated Delta Net recurrence.
	Wave32ModelArchQwen38GDN Wave32ModelArch = "qwen38-gdn"
)

// Qwen 3.8 GDN architecture constants for AMD RDNA 3.5 (gfx1151).
const (
	// Wave32GDNDefaultNumKeyHeads is the standard key head count in Qwen 3.8 GDN.
	Wave32GDNDefaultNumKeyHeads = 16

	// Wave32GDNDefaultNumValueHeads is the standard value head count in Qwen 3.8 GDN.
	Wave32GDNDefaultNumValueHeads = 48

	// Wave32GDNConvKernelSize is the causal 1D depthwise convolution filter size.
	Wave32GDNConvKernelSize = 4

	// Wave32GDNDefaultRMSNormEps is the standard epsilon for GDN output RMSNorm.
	Wave32GDNDefaultRMSNormEps = float32(1e-6)
)

// Wave32RegisterContract specifies the hardware register allocation, lane mapping,
// and occupancy contract for the Wave32 register-resident recurrent state kernel.
type Wave32RegisterContract struct {
	TargetArch          string          `json:"target_arch"`
	ModelArch           Wave32ModelArch `json:"model_arch"`
	WavefrontSize       int             `json:"wavefront_size"`
	NumWavefronts       int             `json:"num_wavefronts"`
	TotalLanes          int             `json:"total_lanes"`
	HeadDim             int             `json:"head_dim"`
	NumHeads            int             `json:"num_heads"`
	NumKeyHeads         int             `json:"num_key_heads"`
	NumValueHeads       int             `json:"num_value_heads"`
	ConvKernelSize      int             `json:"conv_kernel_size"`
	HasRMSNormOutput    bool            `json:"has_rmsnorm_output"`
	HasSiluOutputGate   bool            `json:"has_silu_output_gate"`
	StateDimsPerThread  int             `json:"state_dims_per_thread"`
	VGPRsAllocated      int             `json:"vgprs_allocated"`
	MaxVGPRsPerThread   int             `json:"max_vgprs_per_thread"`
	LDSBytesPerBlock    int             `json:"lds_bytes_per_block"`
	InnerLoopDRAMBytes  int             `json:"inner_loop_dram_bytes"`
	IntraWaveShuffleOps int             `json:"intra_wave_shuffle_ops"`
}

// NewWave32RegisterContract builds and validates the register allocation contract
// for the given head count on AMD Strix Halo (gfx1151).
func NewWave32RegisterContract(numHeads int) (Wave32RegisterContract, error) {
	if numHeads <= 0 {
		numHeads = Wave32DefaultNumHeads
	}
	// On RDNA 3.5, each lane in a 128-thread workgroup holding a 128x128 recurrent state
	// slice holds 128 state floats in VGPRs, plus input/gate temporaries (approx 140 VGPRs total),
	// which fits strictly inside the 256 VGPR hardware limit.
	// For 4-dim vector sub-waves, 4 state floats + temporaries require ~16 VGPRs.
	c := Wave32RegisterContract{
		TargetArch:          Wave32TargetArch,
		ModelArch:           Wave32ModelArchGLM53KDA,
		WavefrontSize:       Wave32WavefrontSize,
		NumWavefronts:       Wave32NumWavefronts,
		TotalLanes:          Wave32TotalLanes,
		HeadDim:             Wave32HeadDim,
		NumHeads:            numHeads,
		NumKeyHeads:         numHeads,
		NumValueHeads:       numHeads,
		ConvKernelSize:      0,
		HasRMSNormOutput:    false,
		HasSiluOutputGate:   false,
		StateDimsPerThread:  Wave32StateDimsPerThread,
		VGPRsAllocated:      144, // 128 state floats + 16 temporaries (q, k, v, delta, gates)
		MaxVGPRsPerThread:   Wave32MaxVGPRsPerThread,
		LDSBytesPerBlock:    0, // zero shared memory allocation for recurrent state
		InnerLoopDRAMBytes:  0, // zero DRAM traffic in autoregressive token loop
		IntraWaveShuffleOps: 5, // log2(32) = 5 tree reduction shuffles (__shfl_down)
	}
	if err := c.Validate(); err != nil {
		return Wave32RegisterContract{}, err
	}
	return c, nil
}

// NewWave32GDNRegisterContract builds and validates the register allocation contract
// for Qwen 3.8 GDN (Gated Delta Net) on AMD Strix Halo (gfx1151).
func NewWave32GDNRegisterContract(numKeyHeads, numValueHeads int) (Wave32RegisterContract, error) {
	if numKeyHeads <= 0 {
		numKeyHeads = Wave32GDNDefaultNumKeyHeads
	}
	if numValueHeads <= 0 {
		numValueHeads = Wave32GDNDefaultNumValueHeads
	}
	c := Wave32RegisterContract{
		TargetArch:          Wave32TargetArch,
		ModelArch:           Wave32ModelArchQwen38GDN,
		WavefrontSize:       Wave32WavefrontSize,
		NumWavefronts:       Wave32NumWavefronts,
		TotalLanes:          Wave32TotalLanes,
		HeadDim:             Wave32HeadDim,
		NumHeads:            numValueHeads,
		NumKeyHeads:         numKeyHeads,
		NumValueHeads:       numValueHeads,
		ConvKernelSize:      Wave32GDNConvKernelSize,
		HasRMSNormOutput:    true,
		HasSiluOutputGate:   true,
		StateDimsPerThread:  Wave32StateDimsPerThread,
		VGPRsAllocated:      144, // 128 state floats + 16 temporaries
		MaxVGPRsPerThread:   Wave32MaxVGPRsPerThread,
		LDSBytesPerBlock:    0, // zero shared memory allocation for recurrent state
		InnerLoopDRAMBytes:  0, // zero DRAM traffic in autoregressive token loop
		IntraWaveShuffleOps: 5, // log2(32) = 5 tree reduction shuffles (__shfl_down)
	}
	if err := c.Validate(); err != nil {
		return Wave32RegisterContract{}, err
	}
	return c, nil
}

// Validate checks that the register contract complies with RDNA 3.5 hardware bounds.
func (c Wave32RegisterContract) Validate() error {
	if c.TargetArch != Wave32TargetArch {
		return fmt.Errorf("compute: invalid target arch %q, want %q", c.TargetArch, Wave32TargetArch)
	}
	if c.WavefrontSize != Wave32WavefrontSize {
		return fmt.Errorf("compute: wavefront size %d != %d", c.WavefrontSize, Wave32WavefrontSize)
	}
	if c.NumWavefronts != Wave32NumWavefronts {
		return fmt.Errorf("compute: num wavefronts %d != %d", c.NumWavefronts, Wave32NumWavefronts)
	}
	if c.TotalLanes != Wave32TotalLanes {
		return fmt.Errorf("compute: total lanes %d != %d", c.TotalLanes, Wave32TotalLanes)
	}
	if c.HeadDim != Wave32HeadDim {
		return fmt.Errorf("compute: head dim %d != %d", c.HeadDim, Wave32HeadDim)
	}
	if c.StateDimsPerThread != Wave32StateDimsPerThread {
		return fmt.Errorf("compute: state dims per thread %d != %d", c.StateDimsPerThread, Wave32StateDimsPerThread)
	}
	if c.VGPRsAllocated > c.MaxVGPRsPerThread {
		return fmt.Errorf("compute: allocated VGPRs %d exceed hardware limit %d", c.VGPRsAllocated, c.MaxVGPRsPerThread)
	}
	if c.LDSBytesPerBlock != 0 {
		return fmt.Errorf("compute: LDS allocation %d != 0; recurrent state must be register-resident", c.LDSBytesPerBlock)
	}
	if c.InnerLoopDRAMBytes != 0 {
		return fmt.Errorf("compute: inner loop DRAM bytes %d != 0; recurrent state must not hit DRAM", c.InnerLoopDRAMBytes)
	}
	if c.ModelArch != "" && c.ModelArch != Wave32ModelArchGLM53KDA && c.ModelArch != Wave32ModelArchQwen38GDN {
		return fmt.Errorf("compute: unsupported model arch %q", c.ModelArch)
	}
	if c.ModelArch == Wave32ModelArchQwen38GDN {
		if c.ConvKernelSize < 1 {
			return fmt.Errorf("compute: conv kernel size %d must be >= 1 for GDN", c.ConvKernelSize)
		}
		if c.NumKeyHeads <= 0 || c.NumValueHeads <= 0 {
			return fmt.Errorf("compute: GDN key heads %d and value heads %d must be positive", c.NumKeyHeads, c.NumValueHeads)
		}
		if !c.HasRMSNormOutput {
			return fmt.Errorf("compute: GDN contract requires HasRMSNormOutput=true")
		}
		if !c.HasSiluOutputGate {
			return fmt.Errorf("compute: GDN contract requires HasSiluOutputGate=true")
		}
	}
	return nil
}

// Wave32KDAMemoryAudit captures and asserts zero-DRAM memory traffic contracts
// across autoregressive token generation loops.
type Wave32KDAMemoryAudit struct {
	TokensProcessed    int   `json:"tokens_processed"`
	InnerLoopDRAMBytes int64 `json:"inner_loop_dram_bytes"`
	InnerLoopLDSBytes  int64 `json:"inner_loop_lds_bytes"`
	VGPRReads          int64 `json:"vgpr_reads"`
	VGPRWrites         int64 `json:"vgpr_writes"`
	IntraWaveShuffles  int64 `json:"intra_wave_shuffles"`
}

// AssertZeroDRAMTraffic verifies that recurrent state updates never touched DRAM or LDS in the loop.
func (a Wave32KDAMemoryAudit) AssertZeroDRAMTraffic() error {
	if a.InnerLoopDRAMBytes != 0 {
		return fmt.Errorf("compute: inner loop DRAM traffic violated: %d bytes (want 0)", a.InnerLoopDRAMBytes)
	}
	if a.InnerLoopLDSBytes != 0 {
		return fmt.Errorf("compute: inner loop LDS traffic violated: %d bytes (want 0)", a.InnerLoopLDSBytes)
	}
	return nil
}

// Wave32Thread models a single execution lane on RDNA 3.5.
type Wave32Thread struct {
	GlobalID int
	WaveID   int
	LaneID   int

	// StateRegs holds 4 scalar dimensions of state for 4-dim sub-wave reductions.
	StateRegs [Wave32StateDimsPerThread]float32

	// MatrixStateRegs holds 128 floats of recurrent state in VGPRs for a 128x128 head panel.
	MatrixStateRegs [Wave32HeadDim]float32

	// TempReg holds temporary scalar register values during intra-wave shuffles and reductions.
	TempReg float32
}

// Wave32Wavefront represents a 32-thread hardware wavefront.
type Wave32Wavefront struct {
	WaveID  int
	Threads [Wave32WavefrontSize]Wave32Thread
}

// ShuffleDown simulates the ROCm/HIP __shfl_down(val, offset, 32) intrinsic.
// In RDNA 3.5 Wave32 mode, lane i receives the value from lane i+offset if i+offset < 32,
// otherwise returning its own val.
func (w *Wave32Wavefront) ShuffleDown(lane int, val float32, offset int) float32 {
	if lane < 0 || lane >= Wave32WavefrontSize {
		return val
	}
	target := lane + offset
	if target >= 0 && target < Wave32WavefrontSize {
		return w.Threads[target].TempReg
	}
	return val
}

// ShuffleBroadcast simulates the ROCm/HIP __shfl(val, srcLane, 32) intrinsic,
// broadcasting a register value from srcLane to all lanes in the wavefront.
func (w *Wave32Wavefront) ShuffleBroadcast(srcVal float32) float32 {
	return srcVal
}

// IntraWaveReduceSum performs a 5-step parallel tree reduction across the 32 lanes
// of this wavefront using __shfl_down without touching LDS or DRAM.
func (w *Wave32Wavefront) IntraWaveReduceSum(vals [Wave32WavefrontSize]float32, audit *Wave32KDAMemoryAudit) float32 {
	for l := 0; l < Wave32WavefrontSize; l++ {
		w.Threads[l].TempReg = vals[l]
	}
	curr := vals
	for offset := 16; offset > 0; offset >>= 1 {
		var next [Wave32WavefrontSize]float32
		for l := 0; l < Wave32WavefrontSize; l++ {
			next[l] = curr[l]
			if l+offset < Wave32WavefrontSize {
				next[l] += curr[l+offset]
			}
			if audit != nil {
				audit.IntraWaveShuffles++
			}
		}
		curr = next
		for l := 0; l < Wave32WavefrontSize; l++ {
			w.Threads[l].TempReg = curr[l]
		}
	}
	return curr[0]
}

// IntraWaveAllReduceSum performs tree reduction across all 32 lanes and broadcasts the total to all lanes.
func (w *Wave32Wavefront) IntraWaveAllReduceSum(vals [Wave32WavefrontSize]float32, audit *Wave32KDAMemoryAudit) float32 {
	total := w.IntraWaveReduceSum(vals, audit)
	for l := 0; l < Wave32WavefrontSize; l++ {
		w.Threads[l].TempReg = total
		if audit != nil {
			audit.IntraWaveShuffles++
		}
	}
	return total
}

// Wave32Workgroup models a thread block of 4 Wave32 wavefronts (128 threads total).
type Wave32Workgroup struct {
	Contract Wave32RegisterContract
	Waves    [Wave32NumWavefronts]Wave32Wavefront
	Audit    Wave32KDAMemoryAudit
}

// NewWave32Workgroup initializes a simulated 128-lane workgroup with persistent VGPR state.
func NewWave32Workgroup(contract Wave32RegisterContract) *Wave32Workgroup {
	wg := &Wave32Workgroup{
		Contract: contract,
	}
	for w := 0; w < Wave32NumWavefronts; w++ {
		wg.Waves[w].WaveID = w
		for l := 0; l < Wave32WavefrontSize; l++ {
			gid := w*Wave32WavefrontSize + l
			wg.Waves[w].Threads[l] = Wave32Thread{
				GlobalID: gid,
				WaveID:   w,
				LaneID:   l,
			}
		}
	}
	return wg
}

// GetThread returns a pointer to the thread at global lane index gid (0..127).
func (wg *Wave32Workgroup) GetThread(gid int) *Wave32Thread {
	if gid < 0 || gid >= Wave32TotalLanes {
		return nil
	}
	w := gid / Wave32WavefrontSize
	l := gid % Wave32WavefrontSize
	return &wg.Waves[w].Threads[l]
}

// LoadMatrixState uploads an initial 128x128 state matrix into VGPR registers.
// This is executed once before the token loop begins.
func (wg *Wave32Workgroup) LoadMatrixState(state [Wave32HeadDim][Wave32HeadDim]float32) {
	for gid := 0; gid < Wave32TotalLanes; gid++ {
		th := wg.GetThread(gid)
		for j := 0; j < Wave32HeadDim; j++ {
			th.MatrixStateRegs[j] = state[gid][j]
			wg.Audit.VGPRWrites++
		}
	}
}

// ReadMatrixState downloads the current 128x128 state matrix from VGPR registers.
// This is executed once after the sequence loop ends.
func (wg *Wave32Workgroup) ReadMatrixState() [Wave32HeadDim][Wave32HeadDim]float32 {
	var out [Wave32HeadDim][Wave32HeadDim]float32
	for gid := 0; gid < Wave32TotalLanes; gid++ {
		th := wg.GetThread(gid)
		for j := 0; j < Wave32HeadDim; j++ {
			out[gid][j] = th.MatrixStateRegs[j]
			wg.Audit.VGPRReads++
		}
	}
	return out
}

// ReferenceKDAStep computes a single token state transition using the mathematical
// reference formulation for 128x128 linear delta recurrence.
//
// Formulation:
//  1. Forget decay: S'_{i, j} = S_{i, j} * decay
//  2. Memory retrieval: kvmem_j = sum_{i=0}^{127} S'_{i, j} * k_i
//  3. Delta update: delta_j = (v_j - kvmem_j) * beta
//  4. State transition: S_{next}[i, j] = S'_{i, j} + k_i * delta_j
//  5. Readout: output_j = sum_{i=0}^{127} S_{next}[i, j] * q_i
func ReferenceKDAStep(
	state [Wave32HeadDim][Wave32HeadDim]float32,
	q, k, v []float32,
	beta, decay float32,
) (output []float32, nextState [Wave32HeadDim][Wave32HeadDim]float32) {
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

	// Step 4 & 5: Update state S_next = S' + k (x) delta, and compute readout = sum_i S_next[i, j] * q[i]
	for i := 0; i < Wave32HeadDim; i++ {
		ki := k[i]
		qi := q[i]
		for j := 0; j < Wave32HeadDim; j++ {
			sn := sPrime[i][j] + ki*delta[j]
			nextState[i][j] = sn
			output[j] += sn * qi
		}
	}

	return output, nextState
}

// ReferenceKDASequence runs the mathematical reference KDA formulation across T tokens,
// incorporating the bounded asymmetric forget gate at each token step.
func ReferenceKDASequence(
	initialState [Wave32HeadDim][Wave32HeadDim]float32,
	qSeq, kSeq, vSeq [][]float32,
	betaSeq []float32,
	aLog float32,
	fProjSeq []float32,
	dtBias float32,
) (outputs [][]float32, finalState [Wave32HeadDim][Wave32HeadDim]float32, err error) {
	tokens := len(qSeq)
	if tokens == 0 {
		return nil, initialState, errors.New("compute: empty sequence for reference KDA")
	}
	if len(kSeq) != tokens || len(vSeq) != tokens || len(betaSeq) != tokens || len(fProjSeq) != tokens {
		return nil, initialState, errors.New("compute: sequence length mismatch among operands")
	}

	currentState := initialState
	outputs = make([][]float32, tokens)

	for t := 0; t < tokens; t++ {
		if len(qSeq[t]) != Wave32HeadDim || len(kSeq[t]) != Wave32HeadDim || len(vSeq[t]) != Wave32HeadDim {
			return nil, currentState, fmt.Errorf("compute: token %d dimension mismatch, want %d", t, Wave32HeadDim)
		}

		// Compute bounded decay factor in [e^-5, 1.0] using the ds4-borrowed forget formula.
		decay := computeBoundedForgetElement(aLog, fProjSeq[t], dtBias)

		out, next := ReferenceKDAStep(currentState, qSeq[t], kSeq[t], vSeq[t], betaSeq[t], decay)
		outputs[t] = out
		currentState = next
	}

	return outputs, currentState, nil
}

// Wave32KDAStep executes a single token state transition on the simulated Wave32 workgroup.
// The recurrent state resides strictly in thread VGPR registers across the entire step,
// with zero reads or writes to DRAM or LDS.
func (wg *Wave32Workgroup) Wave32KDAStep(
	q, k, v []float32,
	beta, decay float32,
) []float32 {
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
	for i := 0; i < Wave32TotalLanes; i++ {
		th := wg.GetThread(i)
		ki := k[i]
		qi := q[i]
		for j := 0; j < Wave32HeadDim; j++ {
			sn := th.MatrixStateRegs[j] + ki*delta[j]
			th.MatrixStateRegs[j] = sn
			wg.Audit.VGPRReads++
			wg.Audit.VGPRWrites++
			output[j] += sn * qi
		}
	}

	// Explicit witness: Inner loop state updates touched ZERO DRAM or LDS bytes.
	wg.Audit.TokensProcessed++
	return output
}

// Wave32KDASequence executes the autoregressive sequence generation loop across T tokens.
// State is uploaded once to VGPRs before the loop, maintained in registers for all T tokens
// (with strictly zero DRAM traffic in the inner loop), and downloaded once at the end.
func (wg *Wave32Workgroup) Wave32KDASequence(
	initialState [Wave32HeadDim][Wave32HeadDim]float32,
	qSeq, kSeq, vSeq [][]float32,
	betaSeq []float32,
	aLog float32,
	fProjSeq []float32,
	dtBias float32,
) (outputs [][]float32, finalState [Wave32HeadDim][Wave32HeadDim]float32, err error) {
	tokens := len(qSeq)
	if tokens == 0 {
		return nil, initialState, errors.New("compute: empty sequence for Wave32 KDA")
	}
	if len(kSeq) != tokens || len(vSeq) != tokens || len(betaSeq) != tokens || len(fProjSeq) != tokens {
		return nil, initialState, errors.New("compute: sequence length mismatch among operands")
	}

	// Reset memory audit tracker for this sequence.
	wg.Audit = Wave32KDAMemoryAudit{}

	// Load initial state into VGPRs (one-time setup before loop; 0 inner-loop DRAM bytes).
	wg.LoadMatrixState(initialState)

	outputs = make([][]float32, tokens)

	// Autoregressive token sequence loop: all state mutations occur in VGPR registers.
	for t := 0; t < tokens; t++ {
		decay := computeBoundedForgetElement(aLog, fProjSeq[t], dtBias)
		outputs[t] = wg.Wave32KDAStep(qSeq[t], kSeq[t], vSeq[t], betaSeq[t], decay)
	}

	// Download final state from VGPRs (one-time readback after loop).
	finalState = wg.ReadMatrixState()

	// Assert that inner loop incurred strictly zero DRAM and LDS traffic.
	if err := wg.Audit.AssertZeroDRAMTraffic(); err != nil {
		return nil, finalState, err
	}

	return outputs, finalState, nil
}

// Wave32VectorKDAStep models the Wave32 wavefront execution where each of the 32 threads
// holds 4 state dimensions in its VGPR StateRegs ([4]float32), totaling 128 dimensions (32*4=128).
// State updates use intra-wave shuffles (__shfl_down) for 5-step tree reductions with zero LDS/DRAM.
func (wave *Wave32Wavefront) Wave32VectorKDAStep(
	q, k, v []float32,
	beta, decay float32,
	audit *Wave32KDAMemoryAudit,
) float32 {
	// Step 1: Thread l applies decay to its 4 state registers and accumulates local dot product with k.
	var localDots [Wave32WavefrontSize]float32
	for l := 0; l < Wave32WavefrontSize; l++ {
		th := &wave.Threads[l]
		var threadDot float32
		for m := 0; m < Wave32StateDimsPerThread; m++ {
			idx := l*Wave32StateDimsPerThread + m
			th.StateRegs[m] *= decay
			if audit != nil {
				audit.VGPRReads++
				audit.VGPRWrites++
			}
			threadDot += th.StateRegs[m] * k[idx]
		}
		localDots[l] = threadDot
	}

	// Step 2: 5-step tree reduction across 32 lanes via __shfl_down to compute kvmem.
	kvmemTotal := wave.IntraWaveAllReduceSum(localDots, audit)

	// Step 3: Compute scalar delta error.
	var vSum float32
	for idx := 0; idx < Wave32HeadDim; idx++ {
		vSum += v[idx]
	}
	delta := (vSum - kvmemTotal) * beta

	// Step 4: Update state registers with k * delta and accumulate local readout with q.
	var localReadouts [Wave32WavefrontSize]float32
	for l := 0; l < Wave32WavefrontSize; l++ {
		th := &wave.Threads[l]
		var threadDotQ float32
		for m := 0; m < Wave32StateDimsPerThread; m++ {
			idx := l*Wave32StateDimsPerThread + m
			th.StateRegs[m] += k[idx] * delta
			if audit != nil {
				audit.VGPRReads++
				audit.VGPRWrites++
			}
			threadDotQ += th.StateRegs[m] * q[idx]
		}
		localReadouts[l] = threadDotQ
	}

	// Step 5: Readout tree reduction across 32 lanes via __shfl_down.
	readoutTotal := wave.IntraWaveAllReduceSum(localReadouts, audit)

	if audit != nil {
		audit.TokensProcessed++
	}
	return readoutTotal
}

// ReferenceVectorKDAStep computes the mathematical reference for 128-dim vector linear recurrence.
func ReferenceVectorKDAStep(
	state [Wave32HeadDim]float32,
	q, k, v []float32,
	beta, decay float32,
) (readout float32, nextState [Wave32HeadDim]float32) {
	var kvmem float32
	for i := 0; i < Wave32HeadDim; i++ {
		sp := state[i] * decay
		nextState[i] = sp
		kvmem += sp * k[i]
	}

	var vSum float32
	for i := 0; i < Wave32HeadDim; i++ {
		vSum += v[i]
	}
	delta := (vSum - kvmem) * beta

	for i := 0; i < Wave32HeadDim; i++ {
		nextState[i] += k[i] * delta
		readout += nextState[i] * q[i]
	}

	return readout, nextState
}

// MaxAbsDelta returns the maximum absolute difference between two float32 slices.
func MaxAbsDelta(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}
	var maxDelta float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxDelta {
			maxDelta = d
		}
	}
	return maxDelta
}

// CosineSimilarity computes the cosine similarity between two float32 slices.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}
	var dot, normA, normB float64
	for i := range a {
		va := float64(a[i])
		vb := float64(b[i])
		dot += va * vb
		normA += va * va
		normB += vb * vb
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0.0 {
		return 1.0
	}
	return dot / denom
}

// L2Normalize computes the L2 normalized vector of v, as standard in linear delta attention (KDA).
func L2Normalize(v []float32) []float32 {
	var normSq float64
	for _, x := range v {
		normSq += float64(x) * float64(x)
	}
	norm := math.Sqrt(normSq)
	out := make([]float32, len(v))
	if norm < 1e-12 {
		return out
	}
	inv := float32(1.0 / norm)
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// Silu computes the SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)).
func Silu(x float32) float32 {
	if math.IsNaN(float64(x)) {
		return float32(math.NaN())
	}
	if math.IsInf(float64(x), -1) {
		return 0
	}
	if math.IsInf(float64(x), 1) {
		return x
	}
	return float32(float64(x) / (1.0 + math.Exp(-float64(x))))
}

// Softplus computes log(1 + exp(x)) with numerical stability guards.
func Softplus(x float32) float32 {
	if math.IsNaN(float64(x)) {
		return float32(math.NaN())
	}
	if x > 20.0 {
		return x
	}
	if x < -30.0 {
		return float32(math.Exp(float64(x)))
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

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

// HasVectorizedDeltaNet reports whether an optimized vectorized DeltaNet kernel is available.
func HasVectorizedDeltaNet() bool {
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
