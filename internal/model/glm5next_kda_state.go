package model

import (
	"fmt"
)

// GLM5NextKDALayerState holds the recurrent state matrix and causal convolution
// state buffer for one GLM-5.3-Flash KDA layer.
type GLM5NextKDALayerState struct {
	// NumHeads is 64.
	NumHeads int
	// HeadDim is 128 (D_k = D_v = 128).
	HeadDim int
	// S is the recurrent state per head: S[h] is [128*128] = 16384 float32.
	// Flattened as [NumHeads * HeadDim * HeadDim].
	S []float32
	// ConvQ holds the past (window - 1) tokens for Q short convolution: [convWindow-1][NumHeads*HeadDim].
	ConvQ []float32
	// ConvK holds the past (window - 1) tokens for K short convolution.
	ConvK []float32
	// ConvV holds the past (window - 1) tokens for V short convolution.
	ConvV []float32
	// ConvWindow is 4.
	ConvWindow int
}

// NewGLM5NextKDALayerState allocates a fresh zero-initialized state for one KDA layer.
func NewGLM5NextKDALayerState(numHeads, headDim, convWindow int) *GLM5NextKDALayerState {
	if numHeads <= 0 {
		numHeads = 64
	}
	if headDim <= 0 {
		headDim = 128
	}
	if convWindow <= 0 {
		convWindow = 4
	}
	matrixSize := numHeads * headDim * headDim
	convHistoryTokens := convWindow - 1
	featureDim := numHeads * headDim
	convBufSize := convHistoryTokens * featureDim

	return &GLM5NextKDALayerState{
		NumHeads:   numHeads,
		HeadDim:    headDim,
		S:          make([]float32, matrixSize),
		ConvQ:      make([]float32, convBufSize),
		ConvK:      make([]float32, convBufSize),
		ConvV:      make([]float32, convBufSize),
		ConvWindow: convWindow,
	}
}

// Reset zeroes out the recurrent state matrix and convolution history buffers.
func (st *GLM5NextKDALayerState) Reset() {
	if st == nil {
		return
	}
	clear(st.S)
	clear(st.ConvQ)
	clear(st.ConvK)
	clear(st.ConvV)
}

// Clone creates a deep copy of the layer state.
func (st *GLM5NextKDALayerState) Clone() *GLM5NextKDALayerState {
	if st == nil {
		return nil
	}
	dup := &GLM5NextKDALayerState{
		NumHeads:   st.NumHeads,
		HeadDim:    st.HeadDim,
		S:          append([]float32(nil), st.S...),
		ConvQ:      append([]float32(nil), st.ConvQ...),
		ConvK:      append([]float32(nil), st.ConvK...),
		ConvV:      append([]float32(nil), st.ConvV...),
		ConvWindow: st.ConvWindow,
	}
	return dup
}

// ByteSize reports the memory in bytes held by this layer state.
func (st *GLM5NextKDALayerState) ByteSize() int64 {
	if st == nil {
		return 0
	}
	elements := len(st.S) + len(st.ConvQ) + len(st.ConvK) + len(st.ConvV)
	return int64(elements * 4)
}

// HeadMatrix returns a subslice view of head h's recurrent matrix S[h].
func (st *GLM5NextKDALayerState) HeadMatrix(h int) []float32 {
	if h < 0 || h >= st.NumHeads {
		panic(fmt.Sprintf("model: KDA head index %d out of range [0, %d)", h, st.NumHeads))
	}
	dimSq := st.HeadDim * st.HeadDim
	off := h * dimSq
	return st.S[off : off+dimSq]
}

// GLM5NextKDAState holds the collective recurrent states across all KDA layers in GLM-5.3-Flash.
type GLM5NextKDAState struct {
	Cfg    GLM5NextConfig
	Layers map[int]*GLM5NextKDALayerState
}

// NewGLM5NextKDAState allocates state for all 34 KDA layers in GLM-5.3-Flash.
func NewGLM5NextKDAState(cfg GLM5NextConfig) *GLM5NextKDAState {
	m := make(map[int]*GLM5NextKDALayerState, len(cfg.KDALayers))
	for _, l := range cfg.KDALayers {
		m[l] = NewGLM5NextKDALayerState(cfg.KDANumHeads, cfg.KDAHeadDim, cfg.ConvWindowSize)
	}
	return &GLM5NextKDAState{
		Cfg:    cfg,
		Layers: m,
	}
}

// Reset zeroes all KDA layer states.
func (s *GLM5NextKDAState) Reset() {
	if s == nil {
		return
	}
	for _, l := range s.Layers {
		l.Reset()
	}
}

// Clone returns an independent deep copy of all layer states.
func (s *GLM5NextKDAState) Clone() *GLM5NextKDAState {
	if s == nil {
		return nil
	}
	m := make(map[int]*GLM5NextKDALayerState, len(s.Layers))
	for k, v := range s.Layers {
		m[k] = v.Clone()
	}
	return &GLM5NextKDAState{
		Cfg:    s.Cfg,
		Layers: m,
	}
}

// TotalBytes returns the aggregate memory in bytes used by all KDA states.
func (s *GLM5NextKDAState) TotalBytes() int64 {
	if s == nil {
		return 0
	}
	var total int64
	for _, l := range s.Layers {
		total += l.ByteSize()
	}
	return total
}
