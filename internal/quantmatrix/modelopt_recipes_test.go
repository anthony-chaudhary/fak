package quantmatrix

import (
	"errors"
	"math"
	"testing"
)

// TestModelOptRecipes pins the ModelOpt NVFP4 / W4A16 recipe semantics: the
// exact scale/exponent layout, the late-tensor-scale semantics, the W4A16
// recipe, and fail-closed refusal of an unsupported recipe.
func TestModelOptRecipes(t *testing.T) {
	t.Run("NVFP4 scale and exponent layout pins", func(t *testing.T) {
		lay := NVFP4LayoutSpec()
		if lay != (NVFP4Layout{
			WeightBits:         4,
			PackedPerByte:      2,
			LowNibbleIsEvenK:   true,
			BlockSize:          16,
			BlockScaleBits:     8,
			BlockScalePerBlock: 1,
			GlobalScaleBits:    32,
			GlobalScaleCount:   1,
			SwizzleTileRows:    128,
			SwizzleTileCols:    4,
		}) {
			t.Fatalf("NVFP4Layout() = %+v", lay)
		}

		// E2M1 value table: bit 3 sign, bits 2:0 magnitude.
		wantE2M1 := [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6,
			float32(math.Copysign(0, -1)), -0.5, -1, -1.5, -2, -3, -4, -6}
		for i := range e2m1Values {
			if e2m1Values[i] != wantE2M1[i] {
				t.Fatalf("e2m1Values[%d]=%v want %v", i, e2m1Values[i], wantE2M1[i])
			}
		}

		// E4M3FN block-scale decode pins, including the reserved NaN byte.
		for _, tc := range []struct {
			raw  byte
			want float32
		}{
			{0x7e, 448},
			{0x01, 1.0 / 512},
			{0x38, 1},
			{0x30, 0.5},
			{0x00, 0},
		} {
			if got := decodeE4M3FN(tc.raw); got != tc.want {
				t.Fatalf("decodeE4M3FN(%#x)=%v want %v", tc.raw, got, tc.want)
			}
		}
		if !math.IsNaN(float64(decodeE4M3FN(0x7f))) {
			t.Fatalf("decodeE4M3FN(0x7f) must be NaN")
		}

		// Packed layout: even K in the low nibble, one E4M3FN per 16 K.
		for _, tc := range []struct {
			rows, cols int
			packed, scale, swizzled int
		}{
			{1, 16, 8, 1, 512},
			{2, 32, 32, 4, 512},
			{128, 64, 4096, 512, 512},
			{129, 64, 4128, 516, 1024},
		} {
			packed, scale, swizzled, err := NVFP4PayloadShape(tc.rows, tc.cols)
			if err != nil {
				t.Fatalf("NVFP4PayloadShape(%d,%d): %v", tc.rows, tc.cols, err)
			}
			if packed != tc.packed || scale != tc.scale || swizzled != tc.swizzled {
				t.Fatalf("NVFP4PayloadShape(%d,%d)=(%d,%d,%d) want (%d,%d,%d)",
					tc.rows, tc.cols, packed, scale, swizzled, tc.packed, tc.scale, tc.swizzled)
			}
		}
	})

	t.Run("late tensor scale semantics", func(t *testing.T) {
		// Fixture from the NVFP4 CPU backend: one block, global scale 0.1.
		codes := []byte{10, 7, 14, 5, 6, 5, 3, 14, 13, 13, 10, 12, 5, 2, 1, 4}
		packed := make([]byte, 8)
		for i, code := range codes {
			packed[i/2] |= code << uint(4*(i%2))
		}
		x := []float32{0.125, 0.125, -1, -2, 0.125, 1.375, 1, 1.375, 123, 1.375, 123, -2, 0.75, 0.75, -1, 0.125}
		scales := []byte{0x38}

		late, err := W4A16Product(packed, scales, 0.1, x)
		if err != nil {
			t.Fatalf("W4A16Product: %v", err)
		}
		// Late: the global scale is applied once to the accumulated sum.
		if math.Float32bits(late) != 0xc2440ccd {
			t.Fatalf("late=%v (%08x) want -49.01250076293945", late, math.Float32bits(late))
		}

		early, err := W4A16ProductEarlyScale(packed, scales, 0.1, x)
		if err != nil {
			t.Fatalf("W4A16ProductEarlyScale: %v", err)
		}
		if math.Float32bits(early) != 0xc2440ccc {
			t.Fatalf("early=%v (%08x) want -49.012497", early, math.Float32bits(early))
		}

		late2, early2, diverges, err := W4A16ScaleDivergence(packed, scales, 0.1, x)
		if err != nil {
			t.Fatalf("W4A16ScaleDivergence: %v", err)
		}
		if late2 != late || early2 != early {
			t.Fatalf("divergence late=%v early=%v want %v %v", late2, early2, late, early)
		}
		if !diverges {
			t.Fatalf("fixture failed to distinguish late tensor scaling from scaling every weight")
		}

		// Finite late-scale cancellation: a zero local sum stays zero even at
		// the maximum finite global scale.
		p2 := make([]byte, 8)
		p2[0] = 0xf7 // codes 7 and 15: +6 and -6, a cancelling pair.
		x2 := make([]float32, 16)
		x2[0], x2[1] = 1, 1
		cancelled, err := W4A16Product(p2, scales, math.MaxFloat32, x2)
		if err != nil {
			t.Fatalf("cancellation: %v", err)
		}
		if cancelled != 0 {
			t.Fatalf("cancelled sum=%v want 0", cancelled)
		}
	})

	t.Run("W4A16 recipe", func(t *testing.T) {
		resolved, err := ResolveRecipe(RecipeRequest{
			ID:             RecipeNVFP4W4A16,
			Rows:           1,
			Cols:           16,
			ScalePlacement: ScaleLate,
		})
		if err != nil {
			t.Fatalf("ResolveRecipe: %v", err)
		}
		if resolved.ID != RecipeNVFP4W4A16 || resolved.WeightBits != 4 || resolved.ActBits != 16 {
			t.Fatalf("resolved=%+v", resolved)
		}
		if resolved.ScalePlacement != ScaleLate {
			t.Fatalf("scale placement=%s want %s", resolved.ScalePlacement, ScaleLate)
		}
		if resolved.PackedBytes != 8 || resolved.ScaleBytes != 1 || resolved.SwizzledBytes != 512 {
			t.Fatalf("payload=(%d,%d,%d) want (8,1,512)", resolved.PackedBytes, resolved.ScaleBytes, resolved.SwizzledBytes)
		}
		if resolved.Source != flashinferW4A16Source {
			t.Fatalf("source=%s", resolved.Source)
		}
	})

	t.Run("fail-closed refusal on unsupported recipe", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			req  RecipeRequest
			want error
		}{
			{
				name: "unknown recipe",
				req:  RecipeRequest{ID: "recipe.future.v9", Rows: 1, Cols: 16, ScalePlacement: ScaleLate},
				want: ErrRecipeUnknown,
			},
			{
				name: "unsupported scale placement",
				req:  RecipeRequest{ID: RecipeNVFP4W4A16, Rows: 1, Cols: 16, ScalePlacement: ScaleEarly},
				want: ErrRecipePlacement,
			},
			{
				name: "K not a multiple of the block",
				req:  RecipeRequest{ID: RecipeNVFP4W4A16, Rows: 1, Cols: 15, ScalePlacement: ScaleLate},
				want: ErrRecipeShape,
			},
			{
				name: "zero rows",
				req:  RecipeRequest{ID: RecipeNVFP4W4A16, Rows: 0, Cols: 16, ScalePlacement: ScaleLate},
				want: ErrRecipeShape,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, err := ResolveRecipe(tc.req)
				if !errors.Is(err, tc.want) {
					t.Fatalf("error=%v want errors.Is(_, %v)", err, tc.want)
				}
			})
		}

		// The product path also refuses fail-closed on a bad scale.
		packed := make([]byte, 8)
		packed[0] = 2
		if _, err := W4A16Product(packed, []byte{0x7f}, 1, make([]float32, 16)); !errors.Is(err, ErrRecipeScale) {
			t.Fatalf("NaN block scale: error=%v want %v", err, ErrRecipeScale)
		}
		if _, err := W4A16Product(packed, []byte{0x38}, -1, make([]float32, 16)); !errors.Is(err, ErrRecipeScale) {
			t.Fatalf("negative global scale: error=%v want %v", err, ErrRecipeScale)
		}
		if _, err := W4A16Product(packed, []byte{0x38}, 1, append([]float32{float32(math.NaN())}, make([]float32, 15)...)); !errors.Is(err, ErrRecipeInput) {
			t.Fatalf("NaN activation: error=%v want %v", err, ErrRecipeInput)
		}
	})
}
