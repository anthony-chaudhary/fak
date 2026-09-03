//go:build darwin && arm64 && cgo

package model

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// FusedLinearAttentionOperationOrder defines the exact sequential operation order
// encoded into the single unified Metal command buffer:
// QKV/Z/B/A projection -> convolution shift -> Q/K norm -> recurrent state update -> gated RMSNorm -> output projection.
var FusedLinearAttentionOperationOrder = []string{
	"QKV_Z_B_A_PROJECTION",
	"CONVOLUTION_SHIFT",
	"QK_NORM",
	"RECURRENT_STATE_UPDATE",
	"GATED_RMSNORM",
	"OUTPUT_PROJECTION",
}

// FusedLinearAttentionReceipt binds the accepted single-command-buffer operation
// to its transfer, encoder, and completion topology.
type FusedLinearAttentionReceipt struct {
	CommandBuffers        int
	Commits               int
	CompletionWaits       int
	ProjectionDispatches  int
	Quantizers            int
	GDNEncoders           int
	Encoders              int
	InputUploads          int
	FinalReadbacks        int
	IntermediateReadbacks int
	StateH2DTransfers     int
	StateD2HTransfers     int
	H2DTransfers          int
	D2HTransfers          int
	TransferCount         int
	IntermediateTransfers int
	Committed             bool
	CompletedWait         bool
	OperationOrder        []string
}

// FusedLinearAttentionMixerOptions configures custom state, weights, or test flags.
type FusedLinearAttentionMixerOptions struct {
	State                          *metalgemm.GDNState
	Weights                        *metalgemm.Qwen35DecodeWeights
	InjectPostSubmitFailureForTest bool
}

// FusedLinearAttentionMixer manages one session-owned linear-attention layer
// executing the 4 input projections, convolution shift, normalization,
// resident GDN recurrent update, gated RMSNorm, and output projection
// in a single Metal command buffer submission with zero intermediate transfers.
type FusedLinearAttentionMixer struct {
	mu                             sync.Mutex
	m                              *Model
	layer                          int
	geom                           metalgemm.GDNGeometry
	weights                        metalgemm.Qwen35DecodeWeights
	panel                          metalgemm.GDNPanel
	state                          *metalgemm.GDNState
	ownsState                      bool
	closed                         bool
	customWeights                  []*metalgemm.Q8Weight
	injectPostSubmitFailureForTest bool
}

// NewFusedLinearAttentionMixer constructs a new fused linear attention mixer
// for the given model and layer. It allocates a dedicated resident GDN state
// ensuring complete session isolation.
func NewFusedLinearAttentionMixer(m *Model, layer int) (*FusedLinearAttentionMixer, error) {
	return NewFusedLinearAttentionMixerWithOptions(m, layer, FusedLinearAttentionMixerOptions{})
}

// NewFusedLinearAttentionMixerWithState constructs a fused mixer using an existing GDN state.
func NewFusedLinearAttentionMixerWithState(m *Model, layer int, state *metalgemm.GDNState) (*FusedLinearAttentionMixer, error) {
	return NewFusedLinearAttentionMixerWithOptions(m, layer, FusedLinearAttentionMixerOptions{State: state})
}

// NewFusedLinearAttentionMixerForSession constructs a fused mixer for a session's model and layer.
func NewFusedLinearAttentionMixerForSession(s *Session, layer int) (*FusedLinearAttentionMixer, error) {
	if s == nil || s.M == nil {
		return nil, errors.New("model: nil session or model")
	}
	return NewFusedLinearAttentionMixer(s.M, layer)
}

// NewFusedLinearAttentionMixerWithOptions constructs a fused linear attention mixer with options.
func NewFusedLinearAttentionMixerWithOptions(m *Model, layer int, opts FusedLinearAttentionMixerOptions) (*FusedLinearAttentionMixer, error) {
	if m == nil {
		return nil, errors.New("model: nil Model")
	}
	if layer < 0 || layer >= m.Cfg.NumLayers {
		return nil, fmt.Errorf("model: layer %d out of bounds [0, %d)", layer, m.Cfg.NumLayers)
	}
	if !m.Cfg.isLinearAttnLayer(layer) {
		return nil, fmt.Errorf("model: layer %d is not a linear attention layer", layer)
	}
	if !metalgemm.Available() {
		return nil, errors.New("model: Metal unavailable")
	}

	cfg := m.Cfg
	geom := metalgemm.GDNGeometry{
		NumKeyHeads:   cfg.LinearNumKeyHeads,
		NumValueHeads: cfg.LinearNumValueHeads,
		KeyHeadDim:    cfg.LinearKeyHeadDim,
		ValueHeadDim:  cfg.LinearValueHeadDim,
		ConvKernel:    cfg.LinearConvKernelDim,
	}

	p := func(suffix string) string { return layerName(layer, suffix) }
	panel := metalgemm.GDNPanel{
		Tokens:         1,
		Conv1D:         m.tensor(p("linear_attn.conv1d.weight")),
		ALog:           m.tensor(p("linear_attn.A_log")),
		DTBias:         m.tensor(p("linear_attn.dt_bias")),
		Norm:           m.tensor(p("linear_attn.norm.weight")),
		RMSNormEpsilon: float32(cfg.RMSNormEps),
	}

	var weights metalgemm.Qwen35DecodeWeights
	var customWeights []*metalgemm.Q8Weight

	if opts.Weights != nil {
		weights = *opts.Weights
	} else {
		names := []string{
			p("linear_attn.in_proj_qkv.weight"),
			p("linear_attn.in_proj_z.weight"),
			p("linear_attn.in_proj_b.weight"),
			p("linear_attn.in_proj_a.weight"),
			p("linear_attn.out_proj.weight"),
		}
		metalQ4KMu.Lock()
		table := metalQ8KW[m]
		handles := make([]*metalgemm.Q8Weight, len(names))
		if table != nil {
			for i, name := range names {
				handles[i] = table[name]
			}
		}
		metalQ4KMu.Unlock()

		for i, name := range names {
			if handles[i] == nil {
				qt := m.q8(name)
				if qt == nil {
					for _, h := range customWeights {
						h.Release()
					}
					return nil, fmt.Errorf("model: missing Q8 tensor %s", name)
				}
				h := metalgemm.UploadQ8(qt.q, qt.d, qt.out, qt.in)
				if h == nil {
					for _, ch := range customWeights {
						ch.Release()
					}
					return nil, fmt.Errorf("model: failed to upload Q8 weight %s", name)
				}
				handles[i] = h
				customWeights = append(customWeights, h)
			}
		}
		weights = metalgemm.Qwen35DecodeWeights{
			InQKV: handles[0],
			InZ:   handles[1],
			InB:   handles[2],
			InA:   handles[3],
			Out:   handles[4],
		}
	}

	state := opts.State
	ownsState := false
	if state == nil {
		var err error
		state, err = metalgemm.NewGDNState(geom)
		if err != nil {
			for _, h := range customWeights {
				h.Release()
			}
			return nil, fmt.Errorf("model: allocate GDN state: %w", err)
		}
		ownsState = true
	}

	return &FusedLinearAttentionMixer{
		m:                              m,
		layer:                          layer,
		geom:                           geom,
		weights:                        weights,
		panel:                          panel,
		state:                          state,
		ownsState:                      ownsState,
		customWeights:                  customWeights,
		injectPostSubmitFailureForTest: opts.InjectPostSubmitFailureForTest,
	}, nil
}

// Step runs the complete fused linear attention mixer for a single token input.
// It executes the 4 input projections, convolution shift, normalization,
// resident GDN recurrent update, gated RMSNorm, and output projection in a single
// command buffer submission with exactly one completion wait.
func (m *FusedLinearAttentionMixer) Step(x []float32) ([]float32, FusedLinearAttentionReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, FusedLinearAttentionReceipt{}, errors.New("model: mixer is closed")
	}
	if len(x) != m.m.Cfg.HiddenSize {
		return nil, FusedLinearAttentionReceipt{}, fmt.Errorf("model: input size %d != hidden size %d", len(x), m.m.Cfg.HiddenSize)
	}

	req := metalgemm.Qwen35DecodeRequest{
		Input:                          x,
		Weights:                        m.weights,
		State:                          m.state,
		Panel:                          m.panel,
		InjectPostSubmitFailureForTest: m.injectPostSubmitFailureForTest,
	}

	out, nativeReceipt, accepted, err := metalgemm.RunQwen35Decode(req)
	receipt := makeFusedLinearAttentionReceipt(nativeReceipt)
	if err != nil {
		return nil, receipt, err
	}
	if !accepted {
		return nil, receipt, errors.New("model: decode mixer declined")
	}
	return out, receipt, nil
}

// Forward is an alias for Step to fulfill standard module forward signatures.
func (m *FusedLinearAttentionMixer) Forward(x []float32) ([]float32, FusedLinearAttentionReceipt, error) {
	return m.Step(x)
}

// Encode appends the fused linear-attention mixer operations to a caller-owned
// projection graph without submitting or reading it.
func (m *FusedLinearAttentionMixer) Encode(graph *metalgemm.ProjectionGraph, input *metalgemm.GraphResult) (*metalgemm.GraphResult, FusedLinearAttentionReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, FusedLinearAttentionReceipt{}, errors.New("model: mixer is closed")
	}
	req := metalgemm.Qwen35DecodeRequest{
		Weights: m.weights,
		State:   m.state,
		Panel:   m.panel,
	}
	res, nativeReceipt, err := metalgemm.EncodeQwen35Decode(graph, input, req)
	receipt := makeFusedLinearAttentionReceipt(nativeReceipt)
	if err != nil {
		return nil, receipt, err
	}
	return res, receipt, nil
}

func makeFusedLinearAttentionReceipt(native metalgemm.Qwen35DecodeReceipt) FusedLinearAttentionReceipt {
	h2d := native.InputUploads
	d2h := native.FinalReadbacks
	intermediate := native.IntermediateReadbacks + native.StateH2DTransfers + native.StateD2HTransfers
	return FusedLinearAttentionReceipt{
		CommandBuffers:        native.CommandBuffers,
		Commits:               native.Commits,
		CompletionWaits:       native.CompletionWaits,
		ProjectionDispatches:  native.ProjectionDispatches,
		Quantizers:            native.Quantizers,
		GDNEncoders:           native.GDNEncoders,
		Encoders:              native.Encoders,
		InputUploads:          native.InputUploads,
		FinalReadbacks:        native.FinalReadbacks,
		IntermediateReadbacks: native.IntermediateReadbacks,
		StateH2DTransfers:     native.StateH2DTransfers,
		StateD2HTransfers:     native.StateD2HTransfers,
		H2DTransfers:          h2d,
		D2HTransfers:          d2h,
		TransferCount:         h2d + d2h,
		IntermediateTransfers: intermediate,
		Committed:             native.Committed,
		CompletedWait:         native.CompletedWait,
		OperationOrder:        append([]string(nil), FusedLinearAttentionOperationOrder...),
	}
}

// Seed replaces the resident convolution and recurrent state.
func (m *FusedLinearAttentionMixer) Seed(conv, recurrent []float32) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return errors.New("model: mixer is closed")
	}
	return m.state.Seed(conv, recurrent)
}

// Snapshot copies the current persistent convolution and recurrent states to host memory.
func (m *FusedLinearAttentionMixer) Snapshot() ([]float32, []float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, nil, errors.New("model: mixer is closed")
	}
	return m.state.Snapshot()
}

// Reset zeroes the resident convolution and recurrent states.
func (m *FusedLinearAttentionMixer) Reset() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return errors.New("model: mixer is closed")
	}
	return m.state.Reset()
}

// Close releases any allocated state and custom weights cleanly.
func (m *FusedLinearAttentionMixer) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	if m.ownsState && m.state != nil {
		m.state.Close()
		m.state = nil
	}
	for _, h := range m.customWeights {
		h.Release()
	}
	m.customWeights = nil
	return nil
}

// State returns the underlying resident GDN state.
func (m *FusedLinearAttentionMixer) State() *metalgemm.GDNState {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.state
}

// Geometry returns the GDN geometry.
func (m *FusedLinearAttentionMixer) Geometry() metalgemm.GDNGeometry {
	return m.geom
}

// Layer returns the layer index.
func (m *FusedLinearAttentionMixer) Layer() int {
	return m.layer
}

// Model returns the parent model.
func (m *FusedLinearAttentionMixer) Model() *Model {
	return m.m
}

// Weights returns the decode projection weights.
func (m *FusedLinearAttentionMixer) Weights() metalgemm.Qwen35DecodeWeights {
	return m.weights
}

// CPUOracleParity holds the outcome of comparing a step against the CPU reference.
type CPUOracleParity struct {
	OutputCosine    float64
	OutputMaxAbs    float64
	ConvCosine      float64
	ConvMaxAbs      float64
	RecurrentCosine float64
	RecurrentMaxAbs float64
	Passed          bool
}

// CosineSimilarityAndMaxAbs computes the cosine similarity and max absolute difference
// between two vectors.
func CosineSimilarityAndMaxAbs(a, b []float32) (float64, float64, error) {
	if len(a) != len(b) {
		return 0, 0, fmt.Errorf("vector length mismatch: %d vs %d", len(a), len(b))
	}
	var dot, an, bn, maxDiff float64
	for i := range a {
		va, vb := float64(a[i]), float64(b[i])
		dot += va * vb
		an += va * va
		bn += vb * vb
		if d := math.Abs(va - vb); d > maxDiff {
			maxDiff = d
		}
	}
	denom := math.Sqrt(an * bn)
	if denom == 0 {
		if an == 0 && bn == 0 {
			return 1.0, 0, nil
		}
		return 0, maxDiff, nil
	}
	cos := dot / denom
	if cos > 1.0 {
		cos = 1.0
	}
	return cos, maxDiff, nil
}

// FlattenLinearConvState flattens the CPU convolution state rows to a contiguous slice,
// aligning with the GPU's (keep * convDim) buffer window.
func FlattenLinearConvState(state *linearAttnLayerState, keep, convDim int) []float32 {
	out := make([]float32, keep*convDim)
	if state == nil || len(state.conv) == 0 {
		return out
	}
	start := keep - len(state.conv)
	if start < 0 {
		start = 0
	}
	for i, row := range state.conv {
		if i+start >= keep {
			break
		}
		copy(out[(i+start)*convDim:(i+start+1)*convDim], row)
	}
	return out
}

// FlattenLinearRecurrentState flattens the CPU recurrent heads into a contiguous slice.
func FlattenLinearRecurrentState(state *linearAttnLayerState) []float32 {
	if state == nil {
		return nil
	}
	var out []float32
	for _, head := range state.recurrent {
		out = append(out, head...)
	}
	return out
}

// ValidateCPUOracle runs one step on both the CPU reference and this fused mixer,
// validating that output, convolution state, and recurrent state all meet:
// cosine similarity >= 0.999999 and max absolute difference < 0.0001.
func (m *FusedLinearAttentionMixer) ValidateCPUOracle(cpu *Session, input []float32) (CPUOracleParity, error) {
	if cpu == nil {
		return CPUOracleParity{}, errors.New("model: nil CPU session")
	}
	cfg := m.m.Cfg
	if cpu.Cache.linear == nil {
		cpu.Cache.linear = newLinearAttnCache(cfg)
	}
	_, _, _, _, _, _, convDim := cfg.linearAttnDims()
	keep := cfg.LinearConvKernelDim - 1

	want := cpu.linearAttnStep(m.layer, input, q8Kernel{m.m})
	got, _, err := m.Step(input)
	if err != nil {
		return CPUOracleParity{}, fmt.Errorf("fused mixer step failed: %w", err)
	}

	outCos, outMaxAbs, err := CosineSimilarityAndMaxAbs(want, got)
	if err != nil {
		return CPUOracleParity{}, fmt.Errorf("output parity comparison failed: %w", err)
	}

	cpuLayer := cpu.Cache.linear.layer(cfg, m.layer)
	cpuConv := FlattenLinearConvState(cpuLayer, keep, convDim)
	gpuConv, gpuRecurrent, err := m.Snapshot()
	if err != nil {
		return CPUOracleParity{}, fmt.Errorf("mixer snapshot failed: %w", err)
	}

	convCos, convMaxAbs, err := CosineSimilarityAndMaxAbs(cpuConv, gpuConv)
	if err != nil {
		return CPUOracleParity{}, fmt.Errorf("conv state parity comparison failed: %w", err)
	}

	cpuRecurrent := FlattenLinearRecurrentState(cpuLayer)
	recCos, recMaxAbs, err := CosineSimilarityAndMaxAbs(cpuRecurrent, gpuRecurrent)
	if err != nil {
		return CPUOracleParity{}, fmt.Errorf("recurrent state parity comparison failed: %w", err)
	}

	passed := outCos >= 0.999999 && outMaxAbs < 1e-4 &&
		convCos >= 0.999999 && convMaxAbs < 1e-4 &&
		recCos >= 0.999999 && recMaxAbs < 1e-4

	parity := CPUOracleParity{
		OutputCosine:    outCos,
		OutputMaxAbs:    outMaxAbs,
		ConvCosine:      convCos,
		ConvMaxAbs:      convMaxAbs,
		RecurrentCosine: recCos,
		RecurrentMaxAbs: recMaxAbs,
		Passed:          passed,
	}

	if !passed {
		return parity, fmt.Errorf("CPU oracle parity threshold violation: output(cos=%g, maxAbs=%g) conv(cos=%g, maxAbs=%g) rec(cos=%g, maxAbs=%g)",
			outCos, outMaxAbs, convCos, convMaxAbs, recCos, recMaxAbs)
	}
	return parity, nil
}

// ValidateCPUOracleMultiStep runs multiple steps on both CPU reference and this fused mixer,
// validating parity at each step.
func (m *FusedLinearAttentionMixer) ValidateCPUOracleMultiStep(cpu *Session, inputs [][]float32) ([]CPUOracleParity, error) {
	results := make([]CPUOracleParity, len(inputs))
	for i, input := range inputs {
		p, err := m.ValidateCPUOracle(cpu, input)
		if err != nil {
			return results, fmt.Errorf("step %d failed: %w", i, err)
		}
		results[i] = p
	}
	return results, nil
}
