package model

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// qwen35EmbeddingRows gathers a bounded, ordered F32 embedding panel for a
// sequence request. It validates the complete source and destination contract
// before allocating or indexing either representation.
func (s *Session) qwen35EmbeddingRows(ids []int) ([]float32, error) {
	if s == nil || s.M == nil || s.Backend == nil {
		return nil, fmt.Errorf("model: Qwen sequence embedding rows require a session, model, and backend")
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("model: Qwen sequence embedding rows require at least one token ID")
	}

	m := s.M
	hidden := m.Cfg.HiddenSize
	vocab := m.Cfg.VocabSize
	if hidden <= 0 || vocab <= 0 {
		return nil, fmt.Errorf("model: invalid Qwen embedding configuration: vocab=%d hidden=%d", vocab, hidden)
	}

	packed := m.Q2KEmbedding
	if packed != nil {
		if packed.Vocab() != vocab || packed.Hidden() != hidden {
			return nil, fmt.Errorf("model: packed Q2_K embedding shape [%d,%d], want [%d,%d]", packed.Vocab(), packed.Hidden(), vocab, hidden)
		}
		if hidden%qkK != 0 {
			return nil, fmt.Errorf("model: packed Q2_K embedding hidden dimension %d is not divisible by %d", hidden, qkK)
		}
		wantBytes := int64(vocab) * int64(hidden/qkK) * int64(q2kBlockBytes)
		if int64(packed.Bytes()) != wantBytes {
			return nil, fmt.Errorf("model: packed Q2_K embedding payload is %d bytes, want %d", packed.Bytes(), wantBytes)
		}
	} else {
		const name = "model.embed_tokens.weight"
		meta, ok := m.manifest[name]
		if !ok {
			return nil, fmt.Errorf("model: missing tensor %s", name)
		}
		if len(meta.Shape) != 2 || meta.Shape[0] != vocab || meta.Shape[1] != hidden {
			return nil, fmt.Errorf("model: F32 embedding shape %v, want [%d,%d]", meta.Shape, vocab, hidden)
		}
		if !strings.EqualFold(meta.Dtype, "F32") {
			return nil, fmt.Errorf("model: embedding tensor dtype %q, want F32", meta.Dtype)
		}
		wantBytes, valid := f32TensorBytes(meta.Shape)
		if !valid || int64(meta.Nbytes) != wantBytes {
			return nil, fmt.Errorf("model: F32 embedding payload is %d bytes, want %d", meta.Nbytes, wantBytes)
		}
		if meta.Offset < 0 || meta.Nbytes < 0 || meta.Offset > len(m.raw) || meta.Nbytes > len(m.raw)-meta.Offset {
			return nil, fmt.Errorf("model: F32 embedding payload range [%d,%d) exceeds backing store of %d bytes", meta.Offset, meta.Offset+meta.Nbytes, len(m.raw))
		}
		if meta.Offset%compute.F32.Bytes() != 0 {
			return nil, fmt.Errorf("model: F32 embedding payload offset %d is not %d-byte aligned", meta.Offset, compute.F32.Bytes())
		}
	}

	for _, id := range ids {
		if id < 0 || id >= vocab {
			return nil, fmt.Errorf("model: token ID %d is outside embedding vocabulary [0,%d)", id, vocab)
		}
	}
	panelShape := []int{len(ids), hidden}
	panelBytes, valid := f32TensorBytes(panelShape)
	if !valid {
		return nil, fmt.Errorf("model: Qwen embedding row panel shape %v overflows", panelShape)
	}
	if capper, ok := s.Backend.(weightBufferCapBackend); ok && capper.MaxWeightBufferBytes() > 0 && panelBytes > capper.MaxWeightBufferBytes() {
		return nil, fmt.Errorf("model: Qwen embedding row panel requires %d bytes, exceeding backend buffer cap %d", panelBytes, capper.MaxWeightBufferBytes())
	}
	elements := panelBytes / int64(compute.F32.Bytes())
	if elements > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("model: Qwen embedding row panel element count %d overflows int", elements)
	}

	if packed != nil {
		return packed.GatherRows(ids, m.Cfg.embedScale())
	}
	rows := make([]float32, int(elements))
	m.embedRowsInto(rows, ids, hidden, m.Cfg)
	return rows, nil
}

// qwen35SequenceOutputHeadFits checks the representation lmHeadMatHAL will
// actually stage. Row-gathering the input table is only safe when request
// construction will not immediately replace it with an oversized F32 head.
func (s *Session) qwen35SequenceOutputHeadFits() bool {
	capper, ok := s.Backend.(weightBufferCapBackend)
	if !ok || capper.MaxWeightBufferBytes() <= 0 {
		return true
	}
	var bytes int64
	valid := true
	if s.useHALQ8Weights() {
		if qt := s.M.q8w[s.M.headName()]; qt != nil {
			bytes = q8ResidentBytes(qt)
			return bytes > 0 && bytes <= capper.MaxWeightBufferBytes()
		}
	}
	if s.useHALQ4KWeights() {
		if qt := s.M.q4kw[s.M.q4kHeadName()]; qt != nil {
			bytes = q4kResidentBytes(qt)
			return bytes > 0 && bytes <= capper.MaxWeightBufferBytes()
		}
	}
	if s.useHALKQuantWeights() {
		name := "lm_head.weight"
		if _, ok := s.M.kqw[name]; !ok {
			name = "model.embed_tokens.weight"
		}
		if qt := s.M.kqw[name]; qt != nil && SupportsHALKQuant(qt.kind) {
			bytes = kQuantResidentBytes(qt)
			return bytes > 0 && bytes <= capper.MaxWeightBufferBytes()
		}
	}

	name := "lm_head.weight"
	if !s.M.has(name) {
		name = "model.embed_tokens.weight"
	}
	meta, ok := s.M.manifest[name]
	if !ok || len(meta.Shape) != 2 || meta.Shape[0] != s.M.Cfg.VocabSize || meta.Shape[1] != s.M.Cfg.HiddenSize {
		return false
	}
	bytes, valid = f32TensorBytes(meta.Shape)
	if !valid {
		return false
	}
	if s.useHALF16Weights() {
		bytes /= int64(compute.F32.Bytes() / compute.F16.Bytes())
	}
	return bytes > 0 && bytes <= capper.MaxWeightBufferBytes()
}
