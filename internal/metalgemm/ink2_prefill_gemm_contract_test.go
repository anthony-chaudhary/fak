package metalgemm

import (
	"os"
	"strings"
	"testing"
)

// TestInk2PrefillGemmContract verifies cross-platform structural invariants of
// the Inkling-2 batched Metal prefill GEMM shader so Darwin-only MSL edits
// cannot silently drop adaptive short-tile splits, cooperative offset lookup,
// or any supported packed/scalar quantization format.
func TestInk2PrefillGemmContract(t *testing.T) {
	raw, err := os.ReadFile("ink2_prefill_gemm.metal")
	if err != nil {
		t.Fatalf("read ink2_prefill_gemm.metal: %v", err)
	}
	src := string(raw)

	for _, want := range []string{
		"inline void ink2g_rows_tile(",
		"#define INK2G_ROWS_DISPATCH(",
		"ink2g_rows_tile<Ink2GDec_##TYPE, TW, 2, 2, 1, INK2G_R_SN / 2>",
		"inline bool ink2g_grouped_find_tg(",
		"#define INK2G_GROUPED_DISPATCH(",
		"ink2g_grouped_tile<Ink2GDec_##TYPE, TW, 2, 2, 1, INK2G_G_SN / 2>",
		"if (!ink2g_grouped_find_tg<INK2G_G_BM>(offsets, nexp, group.y, tg, tid, e, first, count)) return;",
		"if (!ink2g_grouped_find_tg<32>(offsets, nexp, group.y, tg, tid, e, first, count)) return;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("ink2_prefill_gemm.metal missing %q", want)
		}
	}

	// All 14 Inkling-2 GGUF weight dtypes must remain instantiated for batched prefill.
	for _, dtype := range []string{
		"INK2G_PACKED(q8_0, 32, float)",
		"INK2G_PACKED(q3_K, 256, float)",
		"INK2G_PACKED(q4_K, 256, float)",
		"INK2G_PACKED(q5_K, 256, float)",
		"INK2G_PACKED(q6_K, 256, float)",
		"INK2G_PACKED(iq2_xs, 256, float)",
		"INK2G_PACKED(iq2_s, 256, float)",
		"INK2G_PACKED(iq3_xxs, 256, float)",
		"INK2G_PACKED(iq3_s, 256, float)",
		"INK2G_PACKED(iq4_nl, 32, float)",
		"INK2G_PACKED(nvfp4, 64, half)",
		"INK2G_SCALAR(f32, float, float)",
		"INK2G_SCALAR(f16, half, half)",
		"INK2G_SCALAR(bf16, ushort, float)",
	} {
		if !strings.Contains(src, dtype) {
			t.Errorf("ink2_prefill_gemm.metal missing instantiation %q", dtype)
		}
	}
}
