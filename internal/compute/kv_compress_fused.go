package compute

// Adapted from sgl-project/sglang python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py
// at commit 5481134403e01b46891e091ec63db42a36e6626f (Apache-2.0); changes were made:
// the compressor's pool-then-finish contract is re-expressed as a single host pass
// over a prefill ring, with no device kernel.

import (
	"fmt"
	"math"
)

// KVCompressFusedRing is a prefill ring of projected compressor rows: absolute
// position p lives in slot p mod Capacity of KV (and of Scores when Ratio > 1),
// each slot Width float32 values. Ratio one has no gate, so Scores is unused.
type KVCompressFusedRing struct {
	Ratio    int
	Width    int
	Capacity int
	KV       []float32
	Scores   []float32
}

// Slot reports the ring slot holding absolute position pos.
func (r KVCompressFusedRing) Slot(pos int) int { return pos % r.Capacity }

// KVCompressFusedChunk is the single compressed buffer one chunk produces.
// Rows holds Groups finished rows of Width values for compressed indices
// FirstGroup, FirstGroup+1, ...; Pending counts trailing positions of an
// unfinished group that stay in the ring for the next chunk to complete.
type KVCompressFusedChunk struct {
	FirstGroup int
	Groups     int
	Pending    int
	Rows       []float32
}

// KVCompressFusedGroups reports which compressed groups a chunk of positions
// [start, start+length) completes. Groups align to absolute positions, so a
// chunk boundary inside a group defers that group to the chunk that closes it.
func KVCompressFusedGroups(ratio, start, length int) (first, groups, pending int) {
	end := start + length
	first = start / ratio
	return first, end/ratio - first, end % ratio
}

func validateKVCompressFused(ring KVCompressFusedRing, start, length int, gain []float32, eps float32) error {
	if ring.Ratio < 1 || ring.Ratio > V41CompressorPoolMaxRatio || ring.Capacity < 1 ||
		ring.Width < 1 || ring.Width > math.MaxInt/ring.Capacity {
		return fmt.Errorf("%w: fused compress ratio=%d width=%d capacity=%d", ErrV41CompressorRingGeometry, ring.Ratio, ring.Width, ring.Capacity)
	}
	if len(ring.KV) != ring.Capacity*ring.Width || (ring.Ratio > 1 && len(ring.Scores) != ring.Capacity*ring.Width) {
		return fmt.Errorf("%w: fused compress ring buffers must hold %d values", ErrV41CompressorRingGeometry, ring.Capacity*ring.Width)
	}
	if start < 0 || length < 0 || start > math.MaxInt-length {
		return fmt.Errorf("%w: fused compress chunk start=%d length=%d", ErrV41CompressorRingGeometry, start, length)
	}
	// The group straddling the chunk start began in an earlier chunk; its rows
	// are read back from the ring, so the ring must not have lapped them.
	if span := length + start%ring.Ratio; span > ring.Capacity {
		return fmt.Errorf("%w: fused compress span %d exceeds ring capacity %d", ErrV41CompressorRingGeometry, span, ring.Capacity)
	}
	if len(gain) != ring.Width || !V41CompressorNormDomain(ring.Width, eps) {
		return fmt.Errorf("%w: fused compress gain width=%d eps=%g", ErrV41CompressorNormDomain, len(gain), eps)
	}
	for i, g := range gain {
		if !v41CompressorFinite32(g) {
			return fmt.Errorf("%w: fused compress gain element %d is not finite", ErrV41CompressorNormDomain, i)
		}
	}
	return nil
}

// KVCompressFused produces a chunk's compressed rows in one pass over the
// prefill ring: each completed group is read straight from its (possibly
// wrapped) ring slots, softmax-gated per latent dimension when Ratio > 1, and
// finished by the BF16 RMSNorm of V41CompressorNorm, all written into the one
// output buffer with no staging copy. The pooling order matches
// V41CompressorPoolBackend, so the result equals gathering the chunk out of
// the ring, pooling, normalizing, and copying the rows into place.
func KVCompressFused(ring KVCompressFusedRing, start, length int, gain []float32, eps float32) (KVCompressFusedChunk, error) {
	if err := validateKVCompressFused(ring, start, length, gain, eps); err != nil {
		return KVCompressFusedChunk{}, err
	}
	first, groups, pending := KVCompressFusedGroups(ring.Ratio, start, length)
	out := KVCompressFusedChunk{FirstGroup: first, Groups: groups, Pending: pending, Rows: make([]float32, groups*ring.Width)}
	for g := 0; g < groups; g++ {
		row := out.Rows[g*ring.Width : (g+1)*ring.Width]
		if err := ring.poolGroup(row, (first+g)*ring.Ratio); err != nil {
			return KVCompressFusedChunk{}, err
		}
		v41CompressorNormInPlace(row, gain, eps)
	}
	return out, nil
}

// poolGroup writes the pooled row of the group starting at absolute position
// base into dst. The softmax weight is recomputed rather than staged so the
// pass needs no scratch beyond the output row.
func (r KVCompressFusedRing) poolGroup(dst []float32, base int) error {
	w := r.Width
	for t := 0; t < r.Ratio; t++ {
		s := r.Slot(base+t) * w
		for d := 0; d < w; d++ {
			if !v41CompressorFinite32(r.KV[s+d]) || (r.Ratio > 1 && !v41CompressorFinite32(r.Scores[s+d])) {
				return &V41CompressorRingNonFiniteError{Dim: d, Status: r.nonFiniteStatus(s + d)}
			}
		}
	}
	if r.Ratio == 1 {
		s := r.Slot(base) * w
		copy(dst, r.KV[s:s+w])
		return nil
	}
	for d := 0; d < w; d++ {
		m := r.Scores[r.Slot(base)*w+d]
		for t := 1; t < r.Ratio; t++ {
			m = max(m, r.Scores[r.Slot(base+t)*w+d])
		}
		var denom float32
		for t := 0; t < r.Ratio; t++ {
			denom += float32(math.Exp(float64(r.Scores[r.Slot(base+t)*w+d] - m)))
		}
		var acc float32
		for t := 0; t < r.Ratio; t++ {
			s := r.Slot(base+t)*w + d
			weight := float32(math.Exp(float64(r.Scores[s]-m))) / denom
			acc += float32(r.KV[s] * weight) // explicit rounding forbids an FMA the reference does not take
		}
		dst[d] = acc
	}
	return nil
}

func (r KVCompressFusedRing) nonFiniteStatus(i int) int {
	status := 0
	if !v41CompressorFinite32(r.KV[i]) {
		status |= V41CompressorRingKVNonFinite
	}
	if r.Ratio > 1 && !v41CompressorFinite32(r.Scores[i]) {
		status |= V41CompressorRingScoreNonFinite
	}
	return status
}

// v41CompressorNormInPlace is V41CompressorNorm's rounding sequence written
// over its input row, for callers that already validated gain and eps.
func v41CompressorNormInPlace(row, gain []float32, eps float32) {
	var sum float32
	for i, v := range row {
		x := v41CompressorBF16Round(v)
		row[i] = x
		sum += x * x
	}
	inv := float32(v41CompressorRecip64(v41CompressorSqrt64(float64(sum/float32(len(row)) + eps))))
	for i := range row {
		v := row[i] * inv
		v *= gain[i]
		row[i] = v41CompressorBF16Round(v)
	}
}
