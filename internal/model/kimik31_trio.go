package model

import (
	"errors"
	"fmt"
	"math"
)

// ErrKimiK31TrioGeometry marks an invalid paged-indexer host contract.
var ErrKimiK31TrioGeometry = errors.New("model: kimi-k31 trio indexer geometry")

// KimiK31TrioPagedTopK is the closed host oracle for the paged-indexer
// component of K3.1 bring-up. It is not a device backend or forward route.
// This general host oracle does not model device row/top-k capacity limits.
// Visible NaN scores are refused; the upstream score domain excludes NaNs.
// It adapts SGLang topk.py at 61a3c7f2a45f6cfd5fdcec90fd5019d2334c2f0e
// (Apache-2.0): a selected logical token i maps to
// pageTable[i/pageSize]*pageSize+i%pageSize. Only the causal seqLen prefix
// participates. Missing tokens produce -1 entries, including seqLen zero.
// Selection uses the existing DSA oracle's deterministic tie rule; selected
// tokens are returned in logical-index order rather than score order.
func KimiK31TrioPagedTopK(scores []float32, seqLen, topK, pageSize int, pageTable []int) ([]int, error) {
	if seqLen < 0 || seqLen > len(scores) || topK <= 0 || pageSize <= 0 {
		return nil, fmt.Errorf("%w: sequence, top-k or page size", ErrKimiK31TrioGeometry)
	}
	pages := seqLen / pageSize
	if seqLen%pageSize != 0 {
		pages++
	}
	if len(pageTable) < pages {
		return nil, fmt.Errorf("%w: missing page entries", ErrKimiK31TrioGeometry)
	}
	// Validate the entire visible page table before returning any output.
	maxInt := int(^uint(0) >> 1)
	for page := 0; page < pages; page++ {
		lastOffset := pageSize - 1
		if page == pages-1 && seqLen%pageSize != 0 {
			lastOffset = seqLen%pageSize - 1
		}
		physical := pageTable[page]
		if physical < 0 || physical > (maxInt-lastOffset)/pageSize {
			return nil, fmt.Errorf("%w: physical page %d", ErrKimiK31TrioGeometry, page)
		}
	}
	for _, score := range scores[:seqLen] {
		if math.IsNaN(float64(score)) {
			return nil, fmt.Errorf("%w: NaN score", ErrKimiK31TrioGeometry)
		}
	}
	selected := SelectKimiK31DSATopK(scores[:seqLen], topK)
	out := make([]int, topK)
	for i := range out {
		out[i] = -1
	}
	for i, logical := range selected {
		out[i] = pageTable[logical/pageSize]*pageSize + logical%pageSize
	}
	return out, nil
}

// KimiK31TrioRequirements describes the closed K3.1 host bring-up contract.
// It deliberately does not register or enable a serving forward path.
type KimiK31TrioRequirements struct {
	Architecture string
	SparseMLA    bool
	TopKIndexer  bool
}

// KimiK31TrioDescriptor uses the canonical internal architecture established
// by ggufload's K3.1 admission, not the older deepseek2 K2/K3 alias.
func KimiK31TrioDescriptor() KimiK31TrioRequirements {
	return KimiK31TrioRequirements{Architecture: "kimik31", SparseMLA: true, TopKIndexer: true}
}

// ErrKimiK31TrioBackend marks unavailable bring-up capabilities.
var ErrKimiK31TrioBackend = errors.New("model: kimi-k31 trio backend unavailable")

// SelectKimiK31TrioBackend validates the closed sparse-MLA/indexer capability
// pair before selecting the host contract. SGLang PR48415, merge61a3c7f2,
// couples K3.1 DSA with sparse MLA and its packed top-k indexer. This oracle
// describes that dependency; booleans are caller-supplied capabilities, not
// evidence of CUDA/ROCm kernel qualification or permission to serve a model.
// An unsupported pair never falls back to dense MLA.
func SelectKimiK31TrioBackend(architecture string, sparseMLA, topKIndexer bool) (KimiK31TrioRequirements, error) {
	required := KimiK31TrioDescriptor()
	if architecture != required.Architecture {
		return KimiK31TrioRequirements{}, fmt.Errorf("%w: architecture %q", ErrKimiK31TrioBackend, architecture)
	}
	if !sparseMLA {
		return KimiK31TrioRequirements{}, fmt.Errorf("%w: sparse MLA", ErrKimiK31TrioBackend)
	}
	if !topKIndexer {
		return KimiK31TrioRequirements{}, fmt.Errorf("%w: top-k indexer", ErrKimiK31TrioBackend)
	}
	return required, nil
}

// KimiK31TrioRawTopK is the raw-index output mode of the paged-indexer
// contract: upstream topk_v2 with page_tables omitted (INDICES mode) writes
// the selected logical indices themselves. It carries the same selection and
// domain rules as KimiK31TrioPagedTopK without the page transform: only the
// causal seqLen prefix participates, slots past min(topK, seqLen) are -1,
// visible NaN scores are refused, and the deterministic tie rule of
// SelectKimiK31DSATopK holds. It adapts SGLang topk.py / topk_v2.cuh at
// 61a3c7f2a45f6cfd5fdcec90fd5019d2334c2f0e (Apache-2.0).
func KimiK31TrioRawTopK(scores []float32, seqLen, topK int) ([]int, error) {
	if seqLen < 0 || seqLen > len(scores) || topK <= 0 {
		return nil, fmt.Errorf("%w: sequence or top-k", ErrKimiK31TrioGeometry)
	}
	for _, score := range scores[:seqLen] {
		if math.IsNaN(float64(score)) {
			return nil, fmt.Errorf("%w: NaN score", ErrKimiK31TrioGeometry)
		}
	}
	selected := SelectKimiK31DSATopK(scores[:seqLen], topK)
	out := make([]int, topK)
	for i := range out {
		out[i] = -1
	}
	for i, logical := range selected {
		out[i] = logical
	}
	return out, nil
}

// KimiK31TrioPagedTopKRows is the batched form of the paged-indexer contract:
// one call over a decode batch in which row b has its own causal length
// seqLens[b] and its own page table. Every row delegates to
// KimiK31TrioPagedTopK, so the per-row selection, page transform and refusal
// rules are unchanged; the batch adds the shape rule that all three row
// slices agree on the batch size. A zero-row batch returns an empty non-nil
// result; any row's geometry refusal refuses the whole call with no partial
// output. It adapts SGLang topk.py / topk_v2.cuh at
// 61a3c7f2a45f6cfd5fdcec90fd5019d2334c2f0e (Apache-2.0).
func KimiK31TrioPagedTopKRows(scores [][]float32, seqLens []int, pageTables [][]int, pageSize, topK int) ([][]int, error) {
	batch := len(scores)
	if batch != len(seqLens) || batch != len(pageTables) {
		return nil, fmt.Errorf("%w: batch shape %d/%d/%d", ErrKimiK31TrioGeometry, batch, len(seqLens), len(pageTables))
	}
	if topK <= 0 || pageSize <= 0 {
		return nil, fmt.Errorf("%w: top-k or page size", ErrKimiK31TrioGeometry)
	}
	out := make([][]int, batch)
	for b := range scores {
		row, err := KimiK31TrioPagedTopK(scores[b], seqLens[b], topK, pageSize, pageTables[b])
		if err != nil {
			return nil, fmt.Errorf("row %d: %w", b, err)
		}
		out[b] = row
	}
	return out, nil
}
