package compute

import (
	"math"
	"math/rand"
	"testing"
)

// TestW4A16PackedZeroPointParity is the deterministic software witness for the packed
// uint4 per-row zero-point layout (#13155). It covers, with no external service and no
// device:
//
//  1. the bit layout: row n at word[n/8] bits 4*(n%8), with an independently built word
//     so the extractor cannot agree with the packer by construction;
//  2. signed extraction across the full [-8, 7] nibble range;
//  3. numerical parity of the packed path against the per-group []int32 reference;
//  4. the divisibility-by-8 invariant with a typed fallback (never a mis-computation);
//  5. the reduced zero-point byte traffic the packed layout implies.
func TestW4A16PackedZeroPointParity(t *testing.T) {
	t.Run("bit_layout_word_n_over_8_bits_4", func(t *testing.T) {
		// Independently construct one word for rows 0..7: nibbles 1,2,3,4,5,6,7,-8.
		// -8 is 0x8 in four-bit two's complement.
		nibbles := []int32{1, 2, 3, 4, 5, 6, 7, -8}
		var want uint32
		for i, n := range nibbles {
			want |= (uint32(n) & 0xF) << (4 * uint(i))
		}

		z, err := PackW4A16ZeroPoints(nibbles)
		if err != nil {
			t.Fatalf("PackW4A16ZeroPoints: %v", err)
		}
		if len(z.Words) != 1 {
			t.Fatalf("words = %d, want 1", len(z.Words))
		}
		if z.Words[0] != want {
			t.Fatalf("word = %#08x, want %#08x (row n must occupy bits 4*(n%%8))", z.Words[0], want)
		}
		// The top nibble (bits 28..31) is row 7.
		if got := (z.Words[0] >> 28) & 0xF; got != 0x8 {
			t.Fatalf("row 7 nibble = %#x, want 0x8", got)
		}
		for row, exp := range nibbles {
			got, err := z.ZPNibble(row, 8)
			if err != nil {
				t.Fatalf("ZPNibble(%d): %v", row, err)
			}
			if got != exp {
				t.Fatalf("row %d = %d, want %d", row, got, exp)
			}
		}
	})

	t.Run("signed_extraction_full_range", func(t *testing.T) {
		rows := make([]int32, 8)
		for i := range rows {
			rows[i] = int32(i - 8) // -8..-1
		}
		z, err := PackW4A16ZeroPoints(rows)
		if err != nil {
			t.Fatalf("pack: %v", err)
		}
		for i := 0; i < 8; i++ {
			if got := z.zpNibble(i); got != rows[i] {
				t.Fatalf("negative nibble row %d = %d, want %d", i, got, rows[i])
			}
		}
		pos := []int32{0, 1, 2, 3, 4, 5, 6, 7}
		z2, err := PackW4A16ZeroPoints(pos)
		if err != nil {
			t.Fatalf("pack: %v", err)
		}
		for i := 0; i < 8; i++ {
			if got := z2.zpNibble(i); got != pos[i] {
				t.Fatalf("positive nibble row %d = %d, want %d", i, got, pos[i])
			}
		}
	})

	t.Run("parity_vs_per_group_reference", func(t *testing.T) {
		rng := rand.New(rand.NewSource(13155))
		channels := 16
		inDim := 256
		cfg := W4A16PackedConfig{ScaleGroupSize: 32}

		weights := make([]float32, channels*inDim)
		for i := range weights {
			weights[i] = rng.Float32()*2 - 1
		}
		x := make([]float32, inDim)
		for i := range x {
			x[i] = rng.Float32()*0.5 + 0.1
		}

		w, err := BuildW4A16PackedWeights(weights, channels, inDim, cfg)
		if err != nil {
			t.Fatalf("build: %v", err)
		}

		packed, err := w.W4A16PackedMatVec(x)
		if err != nil {
			t.Fatalf("packed matvec: %v", err)
		}

		// The reference reads the per-group []int32 zero-points that the packed
		// builder derived (one per row, reused across groups here).
		unpacked := make([]int32, channels)
		for c := 0; c < channels; c++ {
			unpacked[c], err = w.ZeroPoints.ZPNibble(c, channels)
			if err != nil {
				t.Fatalf("extract zp %d: %v", c, err)
			}
		}
		ref, err := w.W4A16PackedReferenceMatVec(x, unpacked)
		if err != nil {
			t.Fatalf("reference matvec: %v", err)
		}

		// Requirement 3: the packed extraction and the per-group reference are the
		// same arithmetic, so they must agree bit-for-bit.
		for i := range packed {
			if math.IsNaN(float64(packed[i])) || math.IsInf(float64(packed[i]), 0) {
				t.Fatalf("packed output %d is non-finite: %v", i, packed[i])
			}
			if packed[i] != ref[i] {
				t.Fatalf("parity mismatch at %d: packed=%v ref=%v", i, packed[i], ref[i])
			}
		}

		// The packed path must also track the true float32 matvec within the error
		// implied by the 4-bit weight step: |recon - true| <= (step/2)*sum|x| where
		// step = maxAbs/7 over the row. Assert that bound directly rather than a
		// guessed relative threshold.
		for c := 0; c < channels; c++ {
			var want, sumAbsX float64
			for i := 0; i < inDim; i++ {
				want += float64(weights[c*inDim+i]) * float64(x[i])
			}
			for _, v := range x {
				sumAbsX += math.Abs(float64(v))
			}
			var maxStep float64
			groupsPerRow := inDim / cfg.ScaleGroupSize
			for g := 0; g < groupsPerRow; g++ {
				if s := math.Abs(float64(w.Scales[c*groupsPerRow+g])); s > maxStep {
					maxStep = s
				}
			}
			bound := 0.5*maxStep*sumAbsX + 1e-3
			if diff := math.Abs(float64(packed[c]) - want); diff > bound {
				t.Fatalf("row %d error %.6f exceeds quantization bound %.6f (packed=%v true=%v)", c, diff, bound, packed[c], want)
			}
		}
	})

	t.Run("divisibility_invariant_falls_back", func(t *testing.T) {
		_, err := PackW4A16ZeroPoints([]int32{1, 2, 3})
		if err == nil {
			t.Fatalf("expected typed refusal for row count 3")
		}
		if _, ok := err.(ErrW4A16PackedShapeHeld); !ok {
			t.Fatalf("error type = %T, want ErrW4A16PackedShapeHeld", err)
		}

		w := W4A16PackedWeights{
			Channels:       12, // not divisible by 8
			InDim:          64,
			ScaleGroupSize: 32,
			Scales:         make([]float32, 12*2),
			Data:           make([]byte, 12*64/2),
		}
		if _, err := w.W4A16PackedMatVec(make([]float32, 64)); err == nil {
			t.Fatalf("expected ErrW4A16PackedShapeHeld for non-divisible channels")
		} else if _, ok := err.(ErrW4A16PackedShapeHeld); !ok {
			t.Fatalf("error type = %T, want ErrW4A16PackedShapeHeld", err)
		}

		if _, _, ok := W4A16ZeroPointByteTraffic(12); ok {
			t.Fatalf("traffic helper should refuse row count 12")
		}
	})

	t.Run("reduced_zero_point_traffic", func(t *testing.T) {
		packedBytes, unpackedBytes, ok := W4A16ZeroPointByteTraffic(64)
		if !ok {
			t.Fatalf("traffic helper refused 64 rows")
		}
		// Packed: 4 bits per row in uint4 words = 64/2 bytes.
		// Unpacked: one int32 per row = 64*4 bytes.
		if packedBytes != 32 || unpackedBytes != 256 {
			t.Fatalf("traffic = (%d,%d), want (32,256)", packedBytes, unpackedBytes)
		}
		if unpackedBytes != packedBytes*8 {
			t.Fatalf("packed layout should be 8x smaller than int32-per-row: %d != %d*8", unpackedBytes, packedBytes)
		}
	})
}
