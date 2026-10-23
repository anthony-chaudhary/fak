package compute

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// Source witness only (Ref #13668): the ordinary strict scalar twin keeps the
// reviewed witness logit-scale body, declares its three F32 modes
// unconditionally, has one exit, and loads only under the selected device's
// controls and the exact scalar proof. No compiler, SPIR-V or GPU proof.
func TestV41ScaleExactSourceContract(t *testing.T) {
	t.Parallel()
	read := func(path string) string {
		t.Helper()
		raw, err := os.ReadFile(filepath.FromSlash(path))
		if err != nil {
			t.Fatal(err)
		}
		return strings.ReplaceAll(string(raw), "\r\n", "\n")
	}
	exact := read("shaders/v41_scale_exact.comp")
	logit := read("shaders/v41_logit_scale.comp")
	for _, token := range []string{
		"layout(local_size_x = 64) in;",
		"layout(std430, binding = 0) readonly buffer Input { float X[]; };",
		"layout(std430, binding = 1) writeonly buffer Output { float Y[]; };",
		"layout(push_constant) uniform Push { uint count; float scale; } pc;",
		"uint i = gl_GlobalInvocationID.x;",
		"precise float value = X[i] * pc.scale;",
		"Y[i] = value;",
	} {
		if !strings.Contains(exact, token) || !strings.Contains(logit, token) {
			t.Errorf("strict twin and v41_logit_scale.comp disagree on %q", token)
		}
	}
	body := exact[strings.Index(exact, "void main()"):]
	// The >= test keeps the proof's single admitted unsigned comparison; an
	// empty then-arm keeps one exit, so glslc -O emits no merged-return OpSwitch.
	if !strings.Contains(body, "if (i >= pc.count) {\n    } else {") {
		t.Error("strict twin lost its single-exit >= bounds guard")
	}
	for _, forbidden := range []string{"return", "fma(", "+", "-", "/", "float16_t", "double", "for (", "while"} {
		if strings.Contains(body, forbidden) {
			t.Errorf("forbidden strict twin behavior %q", forbidden)
		}
	}
	header := exact[:strings.Index(exact, "layout(local_size_x = 64) in;")]
	for _, token := range []string{
		"#extension GL_EXT_spirv_intrinsics : require",
		`spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4464], 4459, 32);`,
		`spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4466], 4461, 32);`,
		`spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 32);`,
	} {
		if !strings.Contains(header, token) {
			t.Errorf("strict twin header lacks %q", token)
		}
	}
	for _, forbidden := range []string{"#ifdef", "#if ", "#ifndef", "WITNESS"} {
		if strings.Contains(header, forbidden) {
			t.Errorf("strict twin modes must be unconditional; found %q", forbidden)
		}
	}
	shim := read("vulkan_shim.cpp")
	for _, token := range []string{
		`buildKernel(g_kern[K_V41_SCALE_EXACT], P("v41_scale_exact.spv"), 2, 8);`,
		"const bool v41ScaleExactBuilt = g_v41IndexerFloatControls &&",
		"(nbuf != 2 || pcsize != 8 || !g_v41IndexerFloatControls || !v41LogitScaleABI(code))",
		"g_have_v41_scale_exact = 0;",
		"g_have_v41_scale_exact = v41ScaleExactBuilt ? 1 : 0;",
		"return g_ready && g_v41IndexerFloatControls && g_have_v41_scale_exact &&",
		"if (!dispatch(g_kern[K_V41_SCALE_EXACT], bufs, &pc, sizeof(pc), groups)) {",
		`"v41_tail_rope_exact", "v41_scale_exact",`,
	} {
		if !strings.Contains(shim, token) {
			t.Errorf("missing strict scalar twin native contract %q", token)
		}
	}
	// Membership in the checked dispatch expression must not depend on adjacent
	// kernels; adding another checked kernel must preserve this source witness.
	checkedExpression := func(source string) string {
		_, rest, found := strings.Cut(source, "const bool checkedV41 =")
		if !found {
			return ""
		}
		expression, _, terminated := strings.Cut(rest, ";")
		if !terminated {
			return ""
		}
		return expression
	}
	const scaleExactTerm = "&k == &g_kern[K_V41_SCALE_EXACT]"
	hasScaleExact := func(source string) bool {
		for _, term := range strings.Split(checkedExpression(source), "||") {
			if strings.TrimSpace(term) == scaleExactTerm {
				return true
			}
		}
		return false
	}
	if !hasScaleExact(shim) {
		t.Fatal("checkedV41 expression must include the strict scalar twin")
	}
	// Keep other SCALE_EXACT references intact: only checked dispatch admission
	// is removed, so unrelated occurrences cannot satisfy the source witness.
	expression := checkedExpression(shim)
	broken := strings.Replace(shim, expression, strings.ReplaceAll(expression, scaleExactTerm, "false"), 1)
	if hasScaleExact(broken) {
		t.Error("source witness admitted SCALE_EXACT without checked dispatch")
	}
	// The ordinary exports must not hide behind the witness-archive guard.
	exports := shim[strings.Index(shim, "int fvk_v41_scale_exact_module_abi("):]
	if guard := strings.Index(exports, "#ifdef FAK_V41_LOGIT_SCALE_WITNESS"); guard < 0 || guard < strings.Index(exports, "int fvk_v41_scale_exact_f32(") {
		t.Error("ordinary scalar exports moved under the witness-archive guard")
	}
	marker := read("vulkan_v41_scale_exact.go")
	if !strings.Contains(marker, "//go:build vulkan && (windows || linux) && cgo") ||
		!strings.Contains(marker, "return C.fvk_have_v41_scale_exact() != 0") ||
		!strings.Contains(marker, "C.fvk_v41_scale_exact_f32(source.ptr, output.ptr, C.int(input.Shape[0]), C.float(scale))") {
		t.Fatal("ordinary marker must bind the admitted twin in every Vulkan archive")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41ScaleExactValidation(t *testing.T) {
	t.Parallel()
	row := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{64}}
	if bytes, err := validateV41ScaleExact(row, float32(math.Copysign(0, -1))); err != nil || bytes != 256 {
		t.Fatalf("finite signed-zero scale refused: bytes=%d err=%v", bytes, err)
	}
	if _, err := validateV41ScaleExact(row, math.Float32frombits(1)); err != nil {
		t.Fatalf("subnormal scale refused: %v", err)
	}
	for name, tc := range map[string]struct {
		input Tensor
		scale float32
	}{
		"nan scale":   {row, float32(math.NaN())},
		"inf scale":   {row, float32(math.Inf(-1))},
		"empty":       {Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{0}}, 1},
		"matrix":      {Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2, 32}}, 1},
		"bf16 dtype":  {Tensor{Dtype: BF16, Layout: RowMajor, Shape: []int{64}}, 1},
		"quantized":   {Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{64}, Quant: &QuantSpec{}}, 1},
		"column-maj":  {Tensor{Dtype: F32, Layout: ColMajor, Shape: []int{64}}, 1},
		"over int32":  {Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{math.MaxInt32 + 1}}, 1},
		"zero length": {Tensor{Dtype: F32, Layout: RowMajor}, 1},
	} {
		if _, err := validateV41ScaleExact(tc.input, tc.scale); err == nil {
			t.Errorf("%s admitted", name)
		}
	}
}
