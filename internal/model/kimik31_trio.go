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
