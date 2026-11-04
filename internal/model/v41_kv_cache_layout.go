// SPDX-License-Identifier: Apache-2.0
// Two-pool fp8 unified KV layout pair check, adapted from SGLang.
// Source: sgl-project/sglang,
// python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/layout.py at
// revision 3c6a6c8068658b88e40bcaa5a0bf48b9a94ebdba (Apache-2.0):
// https://github.com/sgl-project/sglang/blob/3c6a6c8068658b88e40bcaa5a0bf48b9a94ebdba/python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/layout.py
// Reuse decision: ADAPT. Upstream check_two_pool_pair guards the two separate
// pool allocations that one shared row index addresses: neither writer
// bounds-checks that index, so a scatter walks off the end of the shorter pool
// with no error channel. No upstream Python source or test text is copied; the
// published row constants and the pair contract are re-expressed in Go.
//
// This file owns only the pair contract. The paged V4.1 record codec (528 B
// fp8 / 288 B fp4 tokens) lives in internal/v41kv; the unified two-pool rows
// here are a different geometry and must not be conflated with it.

package model

import (
	"errors"
	"fmt"
	"math"
)

const (
	// V41TwoPoolNopeRowBytes is the fixed packed fp8 nope row width shared by
	// the two-pool writers: 448 B latent, then 14 B of e8m0 tile scales (7
	// tiles, each written twice), then 50 B nobody touches. The reader assumes
	// this stride across the language boundary, so a pair whose nope row
	// differs is refused before any scatter.
	V41TwoPoolNopeRowBytes = 512
	// V41TwoPoolQuantTile is the fp8 quantization tile edge in elements.
	V41TwoPoolQuantTile = 64
)

// ErrV41TwoPoolLayout is the typed fail-closed refusal for a two-pool fp8 KV
// pair that one shared row index cannot address safely.
var ErrV41TwoPoolLayout = errors.New("model: V4.1 two-pool fp8 KV layout pair is invalid")

// V41TwoPool describes one side of the two-pool fp8 unified KV cache: a
// separate allocation of Rows rows, each RowBytes wide, backed by exactly
// Bytes of contiguous row-major storage. One row index addresses the same
// logical token in both pools.
type V41TwoPool struct {
	Rows         int // row count
	RowBytes     int // per-row byte stride
	ElementBytes int // element size in bytes (1 for packed fp8)
	Bytes        int // total storage extent in bytes
}

// CheckV41TwoPoolPair refuses two pools that are not a pair, before anything is
// written. ropeWidth is what the caller believes the rope row is wide (rot_dim
// for the fused store, the source row width for the scatter); both writers take
// the rope row stride off the pool, so a wider row would still land in the right
// place while a width that disagrees with the caller means the wrong pool was
// fetched. A missing rope pool, unequal row counts, a nope row that is not the
// published packed 512 B fp8 row, a rope row that disagrees with the declared
// width, or storage that does not hold its declared rows is refused instead of
// letting a scatter walk off the end of the shorter pool. Valid pairs pass
// unchanged.
func CheckV41TwoPoolPair(nope V41TwoPool, rope *V41TwoPool, ropeWidth int) error {
	if rope == nil {
		return fmt.Errorf("%w: the fp8 layout needs a rope pool next to the nope pool", ErrV41TwoPoolLayout)
	}
	if err := v41TwoPoolExtent(nope, "nope"); err != nil {
		return err
	}
	if err := v41TwoPoolExtent(*rope, "rope"); err != nil {
		return err
	}
	if nope.Rows != rope.Rows {
		return fmt.Errorf("%w: pool rows differ: nope %d vs rope %d; a shared row index would walk off the end of the shorter pool", ErrV41TwoPoolLayout, nope.Rows, rope.Rows)
	}
	if nope.ElementBytes != 1 {
		return fmt.Errorf("%w: nope pool must be packed fp8 (element size 1), got element size %d", ErrV41TwoPoolLayout, nope.ElementBytes)
	}
	if nope.RowBytes != V41TwoPoolNopeRowBytes {
		return fmt.Errorf("%w: nope pool must be the packed %d B fp8 row, got %d B", ErrV41TwoPoolLayout, V41TwoPoolNopeRowBytes, nope.RowBytes)
	}
	if ropeWidth <= 0 {
		return fmt.Errorf("%w: declared rope row width %d must be positive", ErrV41TwoPoolLayout, ropeWidth)
	}
	if rope.RowBytes != ropeWidth {
		return fmt.Errorf("%w: rope pool row is %d B, expected the caller-declared %d B; the wrong pool was fetched", ErrV41TwoPoolLayout, rope.RowBytes, ropeWidth)
	}
	return nil
}

// v41TwoPoolExtent refuses negative geometry and storage that does not hold
// exactly Rows rows of RowBytes each: a short extent is the off-the-end case,
// and a long one means the descriptor does not describe the pool it names.
func v41TwoPoolExtent(p V41TwoPool, name string) error {
	if p.Rows < 0 || p.RowBytes < 0 || p.ElementBytes < 1 || p.Bytes < 0 {
		return fmt.Errorf("%w: %s pool has a non-positive geometry field", ErrV41TwoPoolLayout, name)
	}
	if p.RowBytes != 0 && int64(p.Rows) > math.MaxInt64/int64(p.RowBytes) {
		return fmt.Errorf("%w: %s pool rows x row bytes overflows the address space", ErrV41TwoPoolLayout, name)
	}
	if want := int64(p.Rows) * int64(p.RowBytes); want != int64(p.Bytes) {
		return fmt.Errorf("%w: %s pool storage holds %d B, want %d B for %d rows x %d B", ErrV41TwoPoolLayout, name, p.Bytes, want, p.Rows, p.RowBytes)
	}
	return nil
}
