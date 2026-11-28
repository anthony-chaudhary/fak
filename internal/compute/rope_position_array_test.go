// fak-test:runtime fast est=1s lane=default
package compute

import (
	"math"
	"os"
	"strings"
	"testing"
)

// TestRopePositionArrayMatchesRowwiseOracle pins the vLLM persistent cos/sin cache table
// and position-id array RoPE rotation against per-row cpuref.RoPE:
// (a) gapped / non-contiguous position arrays match cpuref.RoPE bitwise for NeoX,
// (b) inverse rotation restores original activations within fp32 epsilon for NeoX + Interleaved + partial offset,
// (c) [max_position, rot_dim] cache matches on-the-fly trigonometric generation and refuses out-of-range positions,
// (d) rope_position_array.cu exposes the corresponding CUDA kernel entrypoints.
func TestRopePositionArrayMatchesRowwiseOracle(t *testing.T) {
	be := Default()
	const (
		maxPos  = 512
		theta   = 1000000.0
		headDim = 64
		nHeads  = 6
	)

	cache, err := NewRopeCosSinCache(maxPos, headDim, theta)
	if err != nil {
		t.Fatalf("NewRopeCosSinCache: %v", err)
	}

	t.Run("gapped_positions_bitwise_cpuref_parity", func(t *testing.T) {
		positions := []int32{17, 0, 511, 3, 17, 255, 42}
		tokens := len(positions)
		rng := lcg(6161)
		xData := randVec(&rng, tokens*nHeads*headDim)

		got, err := ApplyRopePositionArray(xData, positions, cache, RopePositionConfig{
			Tokens:        tokens,
			NumHeads:      nHeads,
			HeadDim:       headDim,
			RopeDimOffset: 0,
			IsNeox:        true,
			Inverse:       false,
		})
		if err != nil {
			t.Fatalf("ApplyRopePositionArray: %v", err)
		}

		stride := nHeads * headDim
		for tok, pos := range positions {
			rowSlice := xData[tok*stride : (tok+1)*stride]
			rowTensor := NewF32(be, []int{nHeads, headDim}, rowSlice)
			want := be.Read(be.RoPE(rowTensor, int(pos), nHeads, headDim, theta))
			gotRow := got[tok*stride : (tok+1)*stride]
			for i := range want {
				gb := math.Float32bits(gotRow[i])
				wb := math.Float32bits(want[i])
				if gb != wb {
					t.Fatalf("token %d (pos=%d) elem %d bitwise mismatch got=0x%08x want=0x%08x",
						tok, pos, i, gb, wb)
				}
			}
		}
	})

	t.Run("partial_offset_neox_interleaved_and_inverse", func(t *testing.T) {
		const (
			rotDim = 32
			offset = 16
		)
		partialCache, err := NewRopeCosSinCache(maxPos, rotDim, theta)
		if err != nil {
			t.Fatalf("NewRopeCosSinCache(partial): %v", err)
		}
		positions := []int32{128, 7, 300, 1}
		tokens := len(positions)

		for _, isNeox := range []bool{true, false} {
			rng := lcg(7272)
			xData := randVec(&rng, tokens*nHeads*headDim)
			cfg := RopePositionConfig{
				Tokens:        tokens,
				NumHeads:      nHeads,
				HeadDim:       headDim,
				RopeDimOffset: offset,
				IsNeox:        isNeox,
				Inverse:       false,
			}
			fwd, err := ApplyRopePositionArray(xData, positions, partialCache, cfg)
			if err != nil {
				t.Fatalf("forward neox=%v: %v", isNeox, err)
			}

			// Verify unrotated bands [0, offset) and [offset+rotDim, headDim) are bitwise untouched.
			for tok := 0; tok < tokens; tok++ {
				for h := 0; h < nHeads; h++ {
					base := (tok*nHeads + h) * headDim
					for d := 0; d < offset; d++ {
						if math.Float32bits(fwd[base+d]) != math.Float32bits(xData[base+d]) {
							t.Fatalf("pre-offset dim %d mutated", d)
						}
					}
					for d := offset + rotDim; d < headDim; d++ {
						if math.Float32bits(fwd[base+d]) != math.Float32bits(xData[base+d]) {
							t.Fatalf("post-rot dim %d mutated", d)
						}
					}
				}
			}

			invCfg := cfg
			invCfg.Inverse = true
			restored, err := ApplyRopePositionArray(fwd, positions, partialCache, invCfg)
			if err != nil {
				t.Fatalf("inverse neox=%v: %v", isNeox, err)
			}
			var maxErr float64
			for i := range xData {
				d := math.Abs(float64(restored[i] - xData[i]))
				if d > maxErr {
					maxErr = d
				}
			}
			if maxErr > 1e-6 {
				t.Fatalf("neox=%v inverse round-trip maxErr=%g (> 1e-6)", isNeox, maxErr)
			}
		}
	})

	t.Run("cache_table_matches_onthefly_and_bounds", func(t *testing.T) {
		for _, pos := range []int{0, 1, 63, 255, maxPos - 1} {
			cGot, sGot, err := cache.Row(pos)
			if err != nil {
				t.Fatalf("Row(%d): %v", pos, err)
			}
			cWant, sWant := ropeRow(theta, headDim, pos)
			for j := range cWant {
				if math.Float32bits(cGot[j]) != math.Float32bits(cWant[j]) ||
					math.Float32bits(sGot[j]) != math.Float32bits(sWant[j]) {
					t.Fatalf("pos=%d j=%d cos/sin cache mismatch vs ropeRow", pos, j)
				}
			}
		}
		// Out-of-range positions refuse cleanly without wrap or panic.
		x := make([]float32, 2*headDim)
		for _, badPos := range []int32{-1, maxPos, maxPos + 100} {
			if _, err := ApplyRopePositionArray(x, []int32{badPos}, cache, RopePositionConfig{
				Tokens: 1, NumHeads: 2, HeadDim: headDim, IsNeox: true,
			}); err == nil {
				t.Fatalf("expected error for out-of-range position %d", badPos)
			}
		}
	})

	t.Run("cuda_source_contract", func(t *testing.T) {
		cu, err := os.ReadFile("rope_position_array.cu")
		if err != nil {
			t.Fatalf("read rope_position_array.cu: %v", err)
		}
		src := string(cu)
		for _, needle := range []string{
			"k_rope_build_cos_sin_cache",
			"k_rope_position_array",
			"fcuda_rope_cos_sin_cache_build_f32",
			"fcuda_rope_position_array_f32",
			"rope_dim_offset",
			"is_neox",
		} {
			if !strings.Contains(src, needle) {
				t.Fatalf("rope_position_array.cu missing %q", needle)
			}
		}
	})
}
