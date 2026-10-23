package compute

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
// Source witness only: the routed-expert CUDA GEMV accumulates through the two
// explicitly rounded PTX ops (mul.rn.f32 / add.rn.f32, no .ftz), so nvcc cannot contract
// a product into its add or flush a subnormal under any -fmad / -ftz / --use_fast_math.
// No compiler or GPU proof; the -tags cuda suite holds the bits.
func TestRoutedExpertGemvSourceIsFusionFree(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(filepath.FromSlash("routed_expert_gemv.cu"))
	if err != nil {
		t.Fatal(err)
	}
	code := stripCComments(strings.ReplaceAll(string(raw), "\r\n", "\n"))
	if got := strings.Count(code, "asm("); got != 2 {
		t.Fatalf("routed_expert_gemv.cu has %d inline asm statements, want exactly the mul.rn/add.rn pair", got)
	}
	for _, token := range []string{
		`asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));`,
		`asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));`,
		"__global__ void __launch_bounds__(REX_THREADS) k_rex_gemv_fp8_block32_e8m0(",
		"__global__ void __launch_bounds__(REX_THREADS) k_rex_gemv_mxfp4_e8m0(",
	} {
		if !strings.Contains(code, token) {
			t.Errorf("routed_expert_gemv.cu lacks %q", token)
		}
	}
	// Each kernel has a full-word loop and a tail loop; all four accumulate one rounded
	// product into acc with one rounded add.
	if got := strings.Count(code, "acc = rex_add_rn(acc, rex_mul_rn(w, sX["); got != 4 {
		t.Errorf("found %d rounded accumulations, want 4 (word + tail loop per kernel)", got)
	}
	if got := strings.Count(code, "const float w = rex_mul_rn("); got != 4 {
		t.Errorf("found %d rounded weight decodes, want 4", got)
	}
	if got := strings.Count(code, "acc"); got != 4*2+2+2 {
		t.Errorf("acc is touched %d times, want only the 4 rounded accumulations, 2 inits and 2 stores", got)
	}
	lower := strings.ToLower(code)
	for _, forbidden := range []string{
		"fma", // fmaf, __fmaf_rn, __fma_rn, fma.rn
		"mad.",
		".ftz",
		"__fmul", "__fadd", "__fsub", "__fdiv", "__fsqrt", "__frcp",
		"__expf", "__logf", "__powf",
		"#pragma fp_contract",
		"use_fast_math",
	} {
		if strings.Contains(lower, forbidden) {
			t.Errorf("routed_expert_gemv.cu code contains fused or flushing form %q", forbidden)
		}
	}
}
