package compute

import (
	"errors"
	"testing"
)

// TestWMMAGuard_GFX1151_ISAOnly16x16x16 pins the gfx11 ISA floor in the always-compiled
// arch taxonomy (#13542).
//
// gfx11 (RDNA 3 / 3.5, incl. Strix Halo gfx1151) exposes exactly ONE WMMA tile shape:
// 16x16x16, for f16/bf16 and iu8/iu4 operands. There is no 16x16x32 WMMA primitive on
// gfx11 (v_wmma_i32_16x16x32_iu4 is gfx12/RDNA4 only), no FP4 form, and no FP8 form.
// Consequently:
//
//   - SupportedWMMATiles advertises a single 16x16x16 entry covering fp16/bf16/iu8/iu4;
//   - int8/int4/q8/q4 GEMM tunes to that same primitive with TileK=16 (software K
//     accumulation in chunks of one hardware step);
//   - fp8/e4m3/e5m2 fail closed with the typed ErrFP8WMMAUnsupported sentinel instead of
//     pretending a native path exists.
//
// Red on the parent tree: two tiles were advertised, int8 tuned to 16x16x32/TileK=32,
// and fp8 "succeeded". Green once the taxonomy describes only what gfx11 actually issues.
func TestWMMAGuard_GFX1151_ISAOnly16x16x16(t *testing.T) {
	a, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("gfx1151 not found in ROCm arch taxonomy")
	}

	// 1. Exactly one hardware WMMA tile shape on gfx11.
	tiles := a.SupportedWMMATiles()
	if len(tiles) != 1 {
		t.Fatalf("SupportedWMMATiles() length = %d, want 1 (gfx11 has no 16x16x32 WMMA)", len(tiles))
	}
	tile := tiles[0]
	if tile.M != 16 || tile.N != 16 || tile.K != 16 || tile.Lanes != 32 || tile.SubgroupSize != 32 {
		t.Errorf("tile geometry = %+v, want M/N/K 16x16x16 lanes 32 subgroup 32", tile)
	}
	if tile.Primitive != WMMAPrimitive16x16x16 {
		t.Errorf("tile.Primitive = %q, want %q", tile.Primitive, WMMAPrimitive16x16x16)
	}
	if tile.Precision != "fp16/bf16/iu8/iu4" {
		t.Errorf("tile.Precision = %q, want %q", tile.Precision, "fp16/bf16/iu8/iu4")
	}

	// 2. Integer GEMM precisions tune to the single 16x16x16 primitive, TileK=16.
	for _, prec := range []string{"int8", "int4", "q8", "q4"} {
		cfg, err := a.TuneCooperativeMatrixGEMM(128, 128, 128, prec)
		if err != nil {
			t.Fatalf("TuneCooperativeMatrixGEMM(%s): %v", prec, err)
		}
		if cfg.Primitive != WMMAPrimitive16x16x16 {
			t.Errorf("%s: cfg.Primitive = %q, want %q", prec, cfg.Primitive, WMMAPrimitive16x16x16)
		}
		if cfg.TileK != 16 {
			t.Errorf("%s: cfg.TileK = %d, want 16 (one gfx11 hardware WMMA step)", prec, cfg.TileK)
		}
		if cfg.SubgroupSize != 32 {
			t.Errorf("%s: cfg.SubgroupSize = %d, want 32", prec, cfg.SubgroupSize)
		}
	}

	// 3. FP8 has no gfx11 WMMA path: fail closed with the typed sentinel.
	for _, prec := range []string{"fp8", "e4m3", "e5m2"} {
		_, err := a.TuneCooperativeMatrixGEMM(128, 128, 128, prec)
		if !errors.Is(err, ErrFP8WMMAUnsupported) {
			t.Errorf("%s: err = %v, want errors.Is(err, ErrFP8WMMAUnsupported)", prec, err)
		}
	}
}
