// shared_attention_geometry.go — shared-attention geometry contract: GQA/RoPE shape
// validation, byte-mask attention-view checks, and KV-cache append slot/buffer rules.
//
// ADAPTED from ashhart/TensorFold zig/src/core/shared_attention.zig:17@9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// (Apache-2.0; see THIRD_PARTY_NOTICES.md). The closed contract is preserved and
// re-expressed in idiomatic Go over fak's float32 tensors:
//
//   - Shape: query heads must divide KV heads (GQA), the head dimension is bounded at
//     256 and 32-aligned, the rotary range fits the head and is even, and the key
//     capacity is positive, u32-bounded, and free of element-count overflow.
//   - View: the byte mask[row*stride+key] ALONE decides which keys each row sees — no
//     causal structure is inferred from positions; rows/keys are bounded, the stride
//     covers the keys, the scale is finite, and the output store is disjoint from every
//     input store.
//   - Append: cache writes go to unique physical slots (no repeat, none past capacity)
//     on pairwise-disjoint K/V/cache stores.
//
// ForwardTP consults the shape half before native dispatch when
// TPConfig.SharedAttentionGeometry is set; the view and append validators pin the same
// contract for native kernel dispatch and cache-append consumers that build explicit
// views.
package model

import (
	"errors"
	"fmt"
	"math"
)

// Contract bounds carried over from the pinned upstream shape/view/append checks.
const (
	maxSharedAttentionQueryHeads = 128 // qh upper bound
	maxSharedAttentionHeadDim    = 256 // hd upper bound
	sharedAttentionHeadDimAlign  = 32  // hd alignment
	maxSharedAttentionRows       = 16  // rows / append-batch upper bound
	sharedAttentionF32Bytes      = 4   // one float32 element
)

// Sentinel errors for the shared-attention geometry contract. Callers match these with
// errors.Is; the wrapped detail names the offending value.
var (
	// ErrSharedAttentionShape: the GQA/RoPE shape violates the shared-attention contract.
	ErrSharedAttentionShape = errors.New("model: shared-attention shape refused")
	// ErrSharedAttentionView: an attention view (rows/keys/stride/scale) is out of contract.
	ErrSharedAttentionView = errors.New("model: shared-attention view refused")
	// ErrSharedAttentionAppendView: an append batch size is out of contract.
	ErrSharedAttentionAppendView = errors.New("model: shared-attention append view refused")
	// ErrSharedAttentionAppendSlot: an append slot lies outside the cache capacity.
	ErrSharedAttentionAppendSlot = errors.New("model: shared-attention append slot out of capacity")
	// ErrSharedAttentionDuplicateSlot: an append batch repeats a physical slot.
	ErrSharedAttentionDuplicateSlot = errors.New("model: shared-attention duplicate append slot")
	// ErrSharedAttentionAliased: two read/write roles share one backing store.
	ErrSharedAttentionAliased = errors.New("model: shared-attention aliased buffer")
	// ErrSharedAttentionBufferSmall: a buffer window cannot cover its role.
	ErrSharedAttentionBufferSmall = errors.New("model: shared-attention buffer too small")
	// ErrSharedAttentionMisaligned: a buffer offset breaks its required alignment.
	ErrSharedAttentionMisaligned = errors.New("model: shared-attention buffer misaligned")
)

// SharedAttentionShape is the GQA/RoPE geometry one shared-attention lane serves.
type SharedAttentionShape struct {
	QueryHeads int // qh: query heads; must divide KVHeads
	KVHeads    int // kh: KV heads
	HeadDim    int // hd: per-head width; > 0, <= 256, % 32 == 0
	RotaryDim  int // rotated head prefix; 0 (no RoPE) or even and <= HeadDim
	Capacity   int // maximum cached keys; > 0 and u32-bounded
}

// Validate enforces the shared-attention shape contract. A zero RotaryDim is legal
// (no rotation); every other failure is ErrSharedAttentionShape with the offending
// value named.
func (s SharedAttentionShape) Validate() error {
	switch {
	case s.QueryHeads <= 0 || s.KVHeads <= 0 || s.QueryHeads%s.KVHeads != 0:
		return fmt.Errorf("%w: query_heads=%d kv_heads=%d (query heads must divide kv heads)", ErrSharedAttentionShape, s.QueryHeads, s.KVHeads)
	case s.QueryHeads > maxSharedAttentionQueryHeads:
		return fmt.Errorf("%w: query_heads=%d exceeds %d", ErrSharedAttentionShape, s.QueryHeads, maxSharedAttentionQueryHeads)
	case s.HeadDim <= 0 || s.HeadDim > maxSharedAttentionHeadDim || s.HeadDim%sharedAttentionHeadDimAlign != 0:
		return fmt.Errorf("%w: head_dim=%d must be in (0,%d] and %d-aligned", ErrSharedAttentionShape, s.HeadDim, maxSharedAttentionHeadDim, sharedAttentionHeadDimAlign)
	case s.RotaryDim < 0 || s.RotaryDim > s.HeadDim || s.RotaryDim%2 != 0:
		return fmt.Errorf("%w: rotary_dim=%d must be even and within head_dim=%d", ErrSharedAttentionShape, s.RotaryDim, s.HeadDim)
	case s.Capacity <= 0 || s.Capacity > math.MaxUint32:
		return fmt.Errorf("%w: capacity=%d must be in (0,%d]", ErrSharedAttentionShape, s.Capacity, math.MaxUint32)
	}
	// Element-count overflow guard: capacity * kv_heads * head_dim must stay addressable
	// in f32 bytes. Each factor is individually bounded above, so this only bites on a
	// future widened bound — kept because the upstream contract carries the same mul.
	if uint64(s.Capacity)*uint64(s.KVHeads)*uint64(s.HeadDim) > uint64(math.MaxInt64)/sharedAttentionF32Bytes {
		return fmt.Errorf("%w: capacity*kv_heads*head_dim element count overflows", ErrSharedAttentionShape)
	}
	return nil
}

// SharedAttentionBuffer names one contiguous backing store plus a byte window into it.
// ID identifies the store (device buffer or host slab); two views alias exactly when
// they share an ID — conservative, matching the upstream rule that read/write
// operations use separate resources even at disjoint offsets. An empty ID is treated
// as unidentified and therefore aliases every other empty-ID buffer (fail-closed).
type SharedAttentionBuffer struct {
	ID    string // backing-store handle
	Bytes int    // total window bytes available at Off
	Off   int    // window start offset within the store
	Align int    // required byte alignment of Off; 0 means none
}

// fits reports whether the window covers need bytes at the caller's alignment.
func (b SharedAttentionBuffer) fits(need, align int) error {
	if b.Off < 0 || b.Bytes < 0 || b.Off > b.Bytes {
		return fmt.Errorf("%w: buffer %q window off=%d bytes=%d invalid", ErrSharedAttentionBufferSmall, b.ID, b.Off, b.Bytes)
	}
	if align > 0 && b.Off%align != 0 {
		return fmt.Errorf("%w: buffer %q off=%d not %d-byte aligned", ErrSharedAttentionMisaligned, b.ID, b.Off, align)
	}
	if need > b.Bytes-b.Off {
		return fmt.Errorf("%w: buffer %q holds %d bytes, role needs %d", ErrSharedAttentionBufferSmall, b.ID, b.Bytes-b.Off, need)
	}
	return nil
}

// aliased is the disjointness refusal: one store may not serve two read/write roles.
func aliased(a, b SharedAttentionBuffer) error {
	if a.ID == b.ID {
		return fmt.Errorf("%w: buffer %q serves two roles", ErrSharedAttentionAliased, a.ID)
	}
	return nil
}

// SharedAttentionView is one shared-attention call over an explicit key set: Rows of
// queries attend to Keys keys under a byte mask whose ONLY visible-key authority is
// Mask[Row*MaskStride+Key]. No causal structure is inferred from positions.
type SharedAttentionView struct {
	Shape      SharedAttentionShape
	Rows       int     // query rows, 1..maxSharedAttentionRows
	Keys       int     // keys per row, 1..Shape.Capacity
	MaskStride int     // byte-mask row stride, >= Keys
	Scale      float32 // softmax scale; must be finite
	Q          SharedAttentionBuffer
	K          SharedAttentionBuffer
	V          SharedAttentionBuffer
	Out        SharedAttentionBuffer
	Mask       SharedAttentionBuffer // exactly Rows*MaskStride bytes
}

// Validate enforces the shared-attention view contract: bounds first, then buffer
// sizes/alignment, then output-vs-input disjointness.
func (v SharedAttentionView) Validate() error {
	s := v.Shape
	if err := s.Validate(); err != nil {
		return err
	}
	switch {
	case v.Rows <= 0 || v.Rows > maxSharedAttentionRows:
		return fmt.Errorf("%w: rows=%d outside [1,%d]", ErrSharedAttentionView, v.Rows, maxSharedAttentionRows)
	case v.Keys <= 0 || v.Keys > s.Capacity:
		return fmt.Errorf("%w: keys=%d outside [1,capacity=%d]", ErrSharedAttentionView, v.Keys, s.Capacity)
	case v.MaskStride < v.Keys:
		return fmt.Errorf("%w: mask_stride=%d < keys=%d", ErrSharedAttentionView, v.MaskStride, v.Keys)
	case v.MaskStride > math.MaxInt32:
		return fmt.Errorf("%w: mask_stride=%d exceeds u32", ErrSharedAttentionView, v.MaskStride)
	case !finiteSharedAttentionScale(v.Scale):
		return fmt.Errorf("%w: scale=%v not finite", ErrSharedAttentionView, v.Scale)
	}
	const f32 = sharedAttentionF32Bytes
	if err := v.Q.fits(v.Rows*s.QueryHeads*s.HeadDim*f32, f32); err != nil {
		return err
	}
	kvNeed := v.Keys * s.KVHeads * s.HeadDim * f32
	if err := v.K.fits(kvNeed, f32); err != nil {
		return err
	}
	if err := v.V.fits(kvNeed, f32); err != nil {
		return err
	}
	if err := v.Out.fits(v.Rows*s.QueryHeads*s.HeadDim*f32, f32); err != nil {
		return err
	}
	// The byte mask is 1-aligned: mask[row*stride+key] is the sole visibility decision.
	if err := v.Mask.fits(v.Rows*v.MaskStride, 1); err != nil {
		return err
	}
	for _, in := range []SharedAttentionBuffer{v.Q, v.K, v.V, v.Mask} {
		if err := aliased(v.Out, in); err != nil {
			return err
		}
	}
	return nil
}

func finiteSharedAttentionScale(x float32) bool {
	fx := float64(x)
	return !math.IsNaN(fx) && !math.IsInf(fx, 0)
}

// SharedAttentionAppend moves prepared K/V rows directly into unique physical cache
// slots; the caller orders any subsequent attention.
type SharedAttentionAppend struct {
	Shape  SharedAttentionShape
	Slots  []int // physical slots: 1..maxSharedAttentionRows entries, unique, each < Capacity
	K      SharedAttentionBuffer
	V      SharedAttentionBuffer
	CacheK SharedAttentionBuffer
	CacheV SharedAttentionBuffer
}

// Validate enforces the append contract: batch bounds, in-capacity unique slots,
// buffer sizes, and pairwise disjointness of the four stores.
func (a SharedAttentionAppend) Validate() error {
	s := a.Shape
	if err := s.Validate(); err != nil {
		return err
	}
	if len(a.Slots) == 0 || len(a.Slots) > maxSharedAttentionRows {
		return fmt.Errorf("%w: slots=%d outside [1,%d]", ErrSharedAttentionAppendView, len(a.Slots), maxSharedAttentionRows)
	}
	for i, slot := range a.Slots {
		if slot < 0 || slot >= s.Capacity {
			return fmt.Errorf("%w: slot %d outside [0,%d)", ErrSharedAttentionAppendSlot, slot, s.Capacity)
		}
		for j := 0; j < i; j++ {
			if a.Slots[j] == slot {
				return fmt.Errorf("%w: slot %d repeated in batch", ErrSharedAttentionDuplicateSlot, slot)
			}
		}
	}
	width := s.KVHeads * s.HeadDim * sharedAttentionF32Bytes
	if err := a.K.fits(len(a.Slots)*width, sharedAttentionF32Bytes); err != nil {
		return err
	}
	if err := a.V.fits(len(a.Slots)*width, sharedAttentionF32Bytes); err != nil {
		return err
	}
	cacheNeed := s.Capacity * width
	if err := a.CacheK.fits(cacheNeed, sharedAttentionF32Bytes); err != nil {
		return err
	}
	if err := a.CacheV.fits(cacheNeed, sharedAttentionF32Bytes); err != nil {
		return err
	}
	for _, pair := range [][2]SharedAttentionBuffer{{a.K, a.CacheK}, {a.V, a.CacheK}, {a.K, a.CacheV}, {a.V, a.CacheV}, {a.CacheK, a.CacheV}} {
		if err := aliased(pair[0], pair[1]); err != nil {
			return err
		}
	}
	return nil
}
