// Ported from github.com/mit-han-lab/Quest

package kv

import (
	"fmt"
	"math"
)

// QuestPageMinMax is an opt-in sidecar holding, for every KV page and
// kv-head, the elementwise per-channel max and min of the keys written to
// that page. Quest scores pages against a query with these bounds to pick
// the critical pages to attend to. It is maintained incrementally on append
// and is independent of KVStore; nothing uses it unless a caller builds one.
type QuestPageMinMax struct {
	pageSize   int
	numKVHeads int
	headDim    int
	seqLen     int
	// maxes and mins are laid out [page][kvHead][channel].
	maxes []float32
	mins  []float32
}

// NewQuestPageMinMax builds an empty sidecar for keys of shape
// [numKVHeads][headDim] per token, grouped into pages of pageSize tokens.
func NewQuestPageMinMax(pageSize, numKVHeads, headDim int) (*QuestPageMinMax, error) {
	if pageSize <= 0 || numKVHeads <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("kv: quest page meta: page_size=%d num_kv_heads=%d head_dim=%d must all be positive",
			pageSize, numKVHeads, headDim)
	}
	return &QuestPageMinMax{pageSize: pageSize, numKVHeads: numKVHeads, headDim: headDim}, nil
}

// SeqLen is the number of tokens appended so far.
func (m *QuestPageMinMax) SeqLen() int { return m.seqLen }

// NumPages is the number of pages holding at least one token.
func (m *QuestPageMinMax) NumPages() int { return (m.seqLen + m.pageSize - 1) / m.pageSize }

// LastPageLen is the token count of the final page (0 when empty).
func (m *QuestPageMinMax) LastPageLen() int {
	if m.seqLen == 0 {
		return 0
	}
	return (m.seqLen-1)%m.pageSize + 1
}

// AppendToken appends one decode-step key laid out [numKVHeads][headDim].
func (m *QuestPageMinMax) AppendToken(key []float32) error {
	return m.Append(key)
}

// Append appends a run of token keys laid out [token][numKVHeads][headDim],
// folding each into its page's bounds. A token that opens a new page resets
// that page's bounds instead of reading stale memory, so a continued partial
// page and a fresh page rollover both match a from-scratch recompute.
// On error the sidecar is unchanged.
func (m *QuestPageMinMax) Append(keys []float32) error {
	stride := m.numKVHeads * m.headDim
	if len(keys)%stride != 0 {
		return fmt.Errorf("kv: quest page meta: key length %d is not a multiple of num_kv_heads*head_dim=%d",
			len(keys), stride)
	}
	n := len(keys) / stride
	if n == 0 {
		return nil
	}
	newLen := m.seqLen + n
	needPages := (newLen + m.pageSize - 1) / m.pageSize
	if grow := needPages*stride - len(m.maxes); grow > 0 {
		m.maxes = append(m.maxes, make([]float32, grow)...)
		m.mins = append(m.mins, make([]float32, grow)...)
	}
	for t := 0; t < n; t++ {
		pos := m.seqLen + t
		base := (pos / m.pageSize) * stride
		pmax := m.maxes[base : base+stride]
		pmin := m.mins[base : base+stride]
		if pos%m.pageSize == 0 {
			fillQuestSentinels(pmax, pmin)
		}
		questReduce(pmax, pmin, keys[t*stride:(t+1)*stride])
	}
	m.seqLen = newLen
	return nil
}

// PageMax returns a copy of the per-channel key max for (page, kvHead).
func (m *QuestPageMinMax) PageMax(page, kvHead int) ([]float32, error) {
	return m.pageSlice(m.maxes, page, kvHead)
}

// PageMin returns a copy of the per-channel key min for (page, kvHead).
func (m *QuestPageMinMax) PageMin(page, kvHead int) ([]float32, error) {
	return m.pageSlice(m.mins, page, kvHead)
}

func (m *QuestPageMinMax) pageSlice(buf []float32, page, kvHead int) ([]float32, error) {
	if page < 0 || page >= m.NumPages() || kvHead < 0 || kvHead >= m.numKVHeads {
		return nil, fmt.Errorf("kv: quest page meta: page=%d kv_head=%d out of range (pages=%d kv_heads=%d)",
			page, kvHead, m.NumPages(), m.numKVHeads)
	}
	off := (page*m.numKVHeads + kvHead) * m.headDim
	out := make([]float32, m.headDim)
	copy(out, buf[off:off+m.headDim])
	return out, nil
}

// RecomputeQuestPageMinMax rebuilds the sidecar from scratch over keys laid
// out [token][numKVHeads][headDim]; it is the reference the incremental path
// must equal.
func RecomputeQuestPageMinMax(keys []float32, pageSize, numKVHeads, headDim int) (*QuestPageMinMax, error) {
	m, err := NewQuestPageMinMax(pageSize, numKVHeads, headDim)
	if err != nil {
		return nil, err
	}
	stride := numKVHeads * headDim
	if len(keys)%stride != 0 {
		return nil, fmt.Errorf("kv: quest page meta: key length %d is not a multiple of num_kv_heads*head_dim=%d",
			len(keys), stride)
	}
	n := len(keys) / stride
	pages := (n + pageSize - 1) / pageSize
	m.maxes = make([]float32, pages*stride)
	m.mins = make([]float32, pages*stride)
	for p := 0; p < pages; p++ {
		pmax := m.maxes[p*stride : (p+1)*stride]
		pmin := m.mins[p*stride : (p+1)*stride]
		fillQuestSentinels(pmax, pmin)
		end := min((p+1)*pageSize, n)
		for t := p * pageSize; t < end; t++ {
			questReduce(pmax, pmin, keys[t*stride:(t+1)*stride])
		}
	}
	m.seqLen = n
	return m, nil
}

// Upstream seeds with +-CUDART_MAX_NORMAL_FP16, the identity for finite fp16
// keys; keys here are float32, so infinities are the matching identity.
func fillQuestSentinels(pmax, pmin []float32) {
	negInf, posInf := float32(math.Inf(-1)), float32(math.Inf(1))
	for i := range pmax {
		pmax[i] = negInf
		pmin[i] = posInf
	}
}

// questReduce mirrors CUDA max/min (fmaxf/__hmax): a NaN operand is ignored
// rather than propagated, so a NaN key never poisons a page bound.
func questReduce(pmax, pmin, key []float32) {
	for i, v := range key {
		if v > pmax[i] {
			pmax[i] = v
		}
		if v < pmin[i] {
			pmin[i] = v
		}
	}
}
