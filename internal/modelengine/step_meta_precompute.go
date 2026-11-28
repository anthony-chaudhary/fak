// Structural adaptation of SGLang Kimi-K3.1 DSA decode step metadata precompute
// (dsa_backend_mtp_precompute.py, kimi_k3.py @ 188cf7da66938ca2e1ea9cbf7ebbf9a2c81edc2a, Apache-2.0).
// Pure-Go component contract; no hardware claim.
package modelengine

import (
	"errors"
	"fmt"
	"slices"
)

var (
	// ErrStepMetaImmutable is returned when a sealed step metadata snapshot is mutated mid-step.
	ErrStepMetaImmutable = errors.New("modelengine: step metadata snapshot is immutable")
	// ErrInvalidStepMeta is returned when step metadata inputs are malformed.
	ErrInvalidStepMeta = errors.New("modelengine: invalid step metadata input")
)

// StepRankSpec holds the per-rank decode attention and MTP inputs at step start.
type StepRankSpec struct {
	Rank           int     `json:"rank"`
	WorldSize      int     `json:"world_size"`
	NumDraftTokens int     `json:"num_draft_tokens"`
	KVLengths      []int   `json:"kv_lengths"`
	PageIndices    []int32 `json:"page_indices"`
}

// StepMetaSnapshot is a sealed, immutable per-step attention metadata view
// consumed by decode forward passes without re-derivation.
type StepMetaSnapshot struct {
	stepID         uint64
	rank           int
	worldSize      int
	numDraftTokens int
	mtpActive      bool
	totalKVTokens  int
	maxKVLen       int
	kvLengths      []int
	pageIndices    []int32
	sealed         bool
}

// PrecomputeStepMeta builds and seals an immutable StepMetaSnapshot for stepID.
func PrecomputeStepMeta(stepID uint64, spec StepRankSpec) (*StepMetaSnapshot, error) {
	if spec.WorldSize < 1 || spec.Rank < 0 || spec.Rank >= spec.WorldSize {
		return nil, fmt.Errorf("%w: rank=%d world_size=%d", ErrInvalidStepMeta, spec.Rank, spec.WorldSize)
	}
	if spec.NumDraftTokens < 0 {
		return nil, fmt.Errorf("%w: num_draft_tokens=%d", ErrInvalidStepMeta, spec.NumDraftTokens)
	}
	totalKV := 0
	maxKV := 0
	for i, l := range spec.KVLengths {
		if l < 0 {
			return nil, fmt.Errorf("%w: negative kv_lengths[%d]=%d", ErrInvalidStepMeta, i, l)
		}
		totalKV += l
		if l > maxKV {
			maxKV = l
		}
	}
	return &StepMetaSnapshot{
		stepID:         stepID,
		rank:           spec.Rank,
		worldSize:      spec.WorldSize,
		numDraftTokens: spec.NumDraftTokens,
		mtpActive:      spec.NumDraftTokens > 0,
		totalKVTokens:  totalKV,
		maxKVLen:       maxKV,
		kvLengths:      slices.Clone(spec.KVLengths),
		pageIndices:    slices.Clone(spec.PageIndices),
		sealed:         true,
	}, nil
}

func (s *StepMetaSnapshot) StepID() uint64       { return s.stepID }
func (s *StepMetaSnapshot) Rank() int            { return s.rank }
func (s *StepMetaSnapshot) WorldSize() int       { return s.worldSize }
func (s *StepMetaSnapshot) NumDraftTokens() int  { return s.numDraftTokens }
func (s *StepMetaSnapshot) MTPActive() bool      { return s.mtpActive }
func (s *StepMetaSnapshot) TotalKVTokens() int   { return s.totalKVTokens }
func (s *StepMetaSnapshot) MaxKVLen() int        { return s.maxKVLen }
func (s *StepMetaSnapshot) Sealed() bool         { return s.sealed }
func (s *StepMetaSnapshot) KVLengths() []int     { return slices.Clone(s.kvLengths) }
func (s *StepMetaSnapshot) PageIndices() []int32 { return slices.Clone(s.pageIndices) }

// TryMutateKVLengths refuses mid-step mutations on a sealed snapshot.
func (s *StepMetaSnapshot) TryMutateKVLengths(next []int) error {
	if s.sealed {
		return ErrStepMetaImmutable
	}
	s.kvLengths = slices.Clone(next)
	return nil
}
