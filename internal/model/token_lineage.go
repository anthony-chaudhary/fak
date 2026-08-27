package model

import (
	"errors"
	"fmt"
	"math"
)

// Token lineage stores the exact tokenizer id beside each logical resident KV
// position. Qwen token ids are non-negative uint32 values, so four bytes retain
// identity without the collision risk of a hash or digest.
const tokenLineageBytesPerPosition = 4

// ErrTokenLineageMismatch marks a read-only diagnostic refusal: resident KV
// positions and their recorded token identities no longer describe the same span.
var ErrTokenLineageMismatch = errors.New("model: token lineage mismatch")

// TokenLineageVerification is the compact receipt returned by explicit lineage
// verification. Verification is never consulted by cache admission.
type TokenLineageVerification struct {
	Positions     int   `json:"positions"`
	MetadataBytes int64 `json:"metadata_bytes"`
}

type tokenLineage struct {
	ids   []uint32
	fault string
}

func (l *tokenLineage) append(tokenID int) {
	if tokenID < 0 || uint64(tokenID) > math.MaxUint32 {
		if l.fault == "" {
			l.fault = fmt.Sprintf("token id %d is outside exact uint32 representation", tokenID)
		}
		l.ids = append(l.ids, 0)
		return
	}
	l.ids = append(l.ids, uint32(tokenID))
}

func (l *tokenLineage) appendSpan(tokenIDs []int) {
	for _, tokenID := range tokenIDs {
		l.append(tokenID)
	}
}

func (l tokenLineage) clone(extraPositions int) tokenLineage {
	if extraPositions < 0 {
		extraPositions = 0
	}
	return tokenLineage{
		ids:   cloneUint32WithReserve(l.ids, extraPositions),
		fault: l.fault,
	}
}

func (l *tokenLineage) reserve(extraPositions int) {
	if extraPositions <= 0 || cap(l.ids) >= len(l.ids)+extraPositions {
		return
	}
	l.ids = cloneUint32WithReserve(l.ids, extraPositions)
}

// evict compacts metadata by the same resident indices as its KV owner. A
// structurally corrupt lineage is kept safely inspectable rather than panicking
// or silently healing the mismatch.
func (l *tokenLineage) evict(from, end, residentLen int) {
	if from < 0 || end <= from || from >= len(l.ids) {
		return
	}
	if end > residentLen {
		end = residentLen
	}
	if end > len(l.ids) {
		end = len(l.ids)
	}
	l.ids = append(l.ids[:from], l.ids[end:]...)
}

func (l tokenLineage) metadataBytes() int64 {
	return int64(len(l.ids) * tokenLineageBytesPerPosition)
}

func (l tokenLineage) verify(residentPositions, expected []int) (TokenLineageVerification, error) {
	report := TokenLineageVerification{Positions: len(residentPositions), MetadataBytes: l.metadataBytes()}
	if l.fault != "" {
		return report, fmt.Errorf("%w: %s", ErrTokenLineageMismatch, l.fault)
	}
	if len(l.ids) != len(residentPositions) {
		return report, fmt.Errorf("%w: lineage positions=%d resident positions=%d", ErrTokenLineageMismatch, len(l.ids), len(residentPositions))
	}
	if len(expected) != len(residentPositions) {
		return report, fmt.Errorf("%w: expected tokens=%d resident positions=%d", ErrTokenLineageMismatch, len(expected), len(residentPositions))
	}
	for i, pos := range residentPositions {
		if pos != i {
			return report, fmt.Errorf("%w: resident index %d carries position %d", ErrTokenLineageMismatch, i, pos)
		}
		want := expected[i]
		if want < 0 || uint64(want) > math.MaxUint32 || l.ids[i] != uint32(want) {
			return report, fmt.Errorf("%w: position %d token=%d expected=%d", ErrTokenLineageMismatch, i, l.ids[i], want)
		}
	}
	return report, nil
}

func cloneUint32WithReserve(src []uint32, extra int) []uint32 {
	dst := make([]uint32, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

// appendPosition commits one legacy host position and its exact token identity
// together at the existing position-authority seam.
func (c *KVCache) appendPosition(pos, tokenID int) {
	c.pos = append(c.pos, pos)
	c.lineage.append(tokenID)
}

// beginHALTokenLineageWrite returns the completion half of one native HAL write.
// Callers defer it while holding the session's cache-geometry read lock. It records
// only positions the backend actually made resident, including a partial write on a
// failing operation, and never verifies or gates admission.
func (s *Session) beginHALTokenLineageWrite(tokenIDs []int) func() {
	if s == nil || s.halKV == nil {
		return func() {}
	}
	before := s.halKV.Len()
	return func() {
		after := s.halKV.Len()
		if after < before {
			if s.halLineage.fault == "" {
				s.halLineage.fault = fmt.Sprintf("HAL write reduced resident length from %d to %d", before, after)
			}
			return
		}
		added := after - before
		if added == 0 {
			return
		}
		if len(s.halLineage.ids) != before && s.halLineage.fault == "" {
			s.halLineage.fault = fmt.Sprintf("HAL write began at resident position %d with %d lineage positions", before, len(s.halLineage.ids))
		}
		if added > len(tokenIDs) {
			if s.halLineage.fault == "" {
				s.halLineage.fault = fmt.Sprintf("HAL write added %d positions for %d token ids", added, len(tokenIDs))
			}
			added = len(tokenIDs)
		}
		s.halLineage.appendSpan(tokenIDs[:added])
	}
}

// evictKV is the shared shift/rollback/eviction seam. The resident store mutates
// first and the exact number it reports removed drives the paired lineage compaction.
func (s *Session) evictKV(from, n int) int {
	if s == nil {
		return 0
	}
	if s.Backend == nil || s.halKV == nil {
		return s.Cache.Evict(from, n)
	}
	residentLen := s.halKV.Len()
	removed := s.halKV.Evict(from, n)
	s.halLineage.evict(from, from+removed, residentLen)
	return removed
}

// VerifyTokenLineage explicitly and read-only checks expected token ids against
// the session's native position authority. It is a diagnostic API, not an
// admission predicate.
func (s *Session) VerifyTokenLineage(expected []int) (TokenLineageVerification, error) {
	if s == nil {
		return TokenLineageVerification{}, fmt.Errorf("%w: nil session", ErrTokenLineageMismatch)
	}
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	if s.Backend != nil {
		if s.halKV == nil {
			return TokenLineageVerification{}, fmt.Errorf("%w: backend session has no KV store", ErrTokenLineageMismatch)
		}
		return s.halLineage.verify(s.halKV.Pos(), expected)
	}
	if s.Cache == nil {
		return TokenLineageVerification{}, fmt.Errorf("%w: session has no host KV cache", ErrTokenLineageMismatch)
	}
	return s.Cache.lineage.verify(s.Cache.pos, expected)
}

// TokenLineageMetadataBytes reports the exact compact lineage payload owned by
// this session without reading KV tensors.
func (s *Session) TokenLineageMetadataBytes() int64 {
	if s == nil {
		return 0
	}
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	if s.Backend != nil {
		return s.halLineage.metadataBytes()
	}
	if s.Cache == nil {
		return 0
	}
	return s.Cache.lineage.metadataBytes()
}
