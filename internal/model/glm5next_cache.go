package model

// GLM5NextDSALayerBuffer holds key, value, and downsampled index keys for one DSA layer.
type GLM5NextDSALayerBuffer struct {
	K      []float32
	V      []float32
	IndexK []float32
}

// Clone returns a deep copy of the DSA layer buffer.
func (c *GLM5NextDSALayerBuffer) Clone() *GLM5NextDSALayerBuffer {
	if c == nil {
		return nil
	}
	return &GLM5NextDSALayerBuffer{
		K:      append([]float32(nil), c.K...),
		V:      append([]float32(nil), c.V...),
		IndexK: append([]float32(nil), c.IndexK...),
	}
}

// GLM5NextHybridState manages the complete hybrid state across all 45 layers of GLM-5.3-Flash:
// - 34 KDA recurrent linear attention layers
// - 11 DSA sparse attention token layers
type GLM5NextHybridState struct {
	KDA        *GLM5NextKDAState
	DSABuffers map[int]*GLM5NextDSALayerBuffer
	NumTokens  int
}

// NewGLM5NextHybridState creates a fresh hybrid state for all 45 layers.
func NewGLM5NextHybridState(cfg GLM5NextConfig) *GLM5NextHybridState {
	dsa := make(map[int]*GLM5NextDSALayerBuffer, len(cfg.FullAttnLayers))
	for _, l := range cfg.FullAttnLayers {
		dsa[l] = &GLM5NextDSALayerBuffer{}
	}
	return &GLM5NextHybridState{
		KDA:        NewGLM5NextKDAState(cfg),
		DSABuffers: dsa,
	}
}

// Snapshot produces an atomic, independent deep copy of the hybrid state.
func (s *GLM5NextHybridState) Snapshot() *GLM5NextHybridState {
	if s == nil {
		return nil
	}
	dsa := make(map[int]*GLM5NextDSALayerBuffer, len(s.DSABuffers))
	for k, v := range s.DSABuffers {
		dsa[k] = v.Clone()
	}
	return &GLM5NextHybridState{
		KDA:        s.KDA.Clone(),
		DSABuffers: dsa,
		NumTokens:  s.NumTokens,
	}
}

// Restore overwrites the target state in place from snapshot without leaking references.
func (s *GLM5NextHybridState) Restore(snap *GLM5NextHybridState) {
	if s == nil || snap == nil {
		return
	}
	s.KDA = snap.KDA.Clone()
	s.DSABuffers = make(map[int]*GLM5NextDSALayerBuffer, len(snap.DSABuffers))
	for k, v := range snap.DSABuffers {
		s.DSABuffers[k] = v.Clone()
	}
	s.NumTokens = snap.NumTokens
}

// Reset zeroes out all KDA states and empties DSA token buffers.
func (s *GLM5NextHybridState) Reset() {
	if s == nil {
		return
	}
	s.KDA.Reset()
	for _, l := range s.DSABuffers {
		l.K = l.K[:0]
		l.V = l.V[:0]
		l.IndexK = l.IndexK[:0]
	}
	s.NumTokens = 0
}

// TotalByteSize accounts for all memory held across KDA recurrent matrices and DSA buffers.
func (s *GLM5NextHybridState) TotalByteSize() int64 {
	if s == nil {
		return 0
	}
	bytes := s.KDA.TotalBytes()
	for _, l := range s.DSABuffers {
		elements := len(l.K) + len(l.V) + len(l.IndexK)
		bytes += int64(elements * 4)
	}
	return bytes
}
