package compute

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// Source witness only (#13816): the strict twin keeps the reviewed tail body,
// declares its three F32 modes unconditionally, and loads/routes only under the
// selected device's controls. No compiler invocation, SPIR-V or GPU proof.
func TestV41TailRoPEExactSourceContract(t *testing.T) {
	t.Parallel()
	read := func(path string) string {
		t.Helper()
		raw, err := os.ReadFile(filepath.FromSlash(path))
		if err != nil {
			t.Fatal(err)
		}
		return strings.ReplaceAll(string(raw), "\r\n", "\n")
	}
	exact := read("shaders/v41_tail_rope_exact.comp")
	qk := read("shaders/v41_tail_rope_qk.comp")
	// Both modules carry the same bindings, push ABI, prefix copy and separately
	// rounded rotation. The twin keeps one exit so glslc -O emits no merged-return
	// OpSwitch, which the strict ABI proof rejects.
	for _, token := range []string{
		"layout(local_size_x = 256) in;",
		"layout(std430, binding = 0) readonly buffer QBuf { uint Q[]; };",
		"layout(std430, binding = 1) readonly buffer KVBuf { uint KV[]; };",
		"layout(std430, binding = 2) writeonly buffer QOutBuf { uint QOut[]; };",
		"layout(std430, binding = 3) writeonly buffer KVOutBuf { uint KVOut[]; };",
		"layout(std430, binding = 4) readonly buffer SinCosBuf { float SinCos[]; };",
		"int heads;\n    int headDim;\n    int rotaryDim;\n} pc;",
		"uint qCount = uint(pc.heads) * uint(pc.headDim);",
		"uint index = isKV ? i - qCount : i;", "uint dim = index % uint(pc.headDim);",
		"uint tail = uint(pc.headDim - pc.rotaryDim);",
		"KVOut[index] = KV[index];", "QOut[index] = Q[index];",
		"float a = uintBitsToFloat(isKV ? KV[index] : Q[index]);",
		"float b = uintBitsToFloat(isKV ? KV[index + 1u] : Q[index + 1u]);",
		"float s = SinCos[lane];", "float c = SinCos[lane + 1u];",
		"precise float ac = a * c;", "precise float bs = b * s;", "precise float bc = b * c;", "precise float as = a * s;",
		"precise float x = ac - bs;", "precise float y = bc + as;",
		"KVOut[index] = floatBitsToUint(x);", "KVOut[index + 1u] = floatBitsToUint(y);",
		"QOut[index] = floatBitsToUint(x);", "QOut[index + 1u] = floatBitsToUint(y);",
	} {
		if !strings.Contains(exact, token) || !strings.Contains(qk, token) {
			t.Errorf("strict twin and v41_tail_rope_qk.comp disagree on %q", token)
		}
	}
	if !strings.Contains(exact, "if (i < qCount + uint(pc.headDim)) {") || !strings.Contains(exact, "} else if (((dim - tail) & 1u) == 0u) {") {
		t.Error("strict twin lost its single-exit bounds and even-lane guards")
	}
	for _, forbidden := range []string{"return", "sin(", "cos(", "pow(", "fma(", "float16_t", "halfRotary"} {
		if strings.Contains(exact[strings.Index(exact, "void main()"):], forbidden) {
			t.Errorf("forbidden strict twin behavior %q", forbidden)
		}
	}
	header := exact[:strings.Index(exact, "layout(local_size_x = 256) in;")]
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
		`buildKernel(g_kern[K_V41_TAIL_ROPE_EXACT], P("v41_tail_rope_exact.spv"), 5, 3 * sizeof(int));`,
		"v41TailRoPEExactBuilt = g_v41IndexerFloatControls &&",
		"(nbuf != 5 || pcsize != 12 || !g_v41IndexerFloatControls || !v41TailRoPEABI(code, true))",
		"g_have_v41_tail_rope_exact = 0;",
		"g_have_v41_tail_rope_exact = v41TailRoPEExactBuilt ? 1 : 0;",
		"return g_ready && g_v41IndexerFloatControls && g_have_v41_tail_rope_exact &&",
		"return v41TailRoPEDispatch(g_kern[K_V41_TAIL_ROPE_EXACT]",
		"case K_V41_TAIL_ROPE_QK: case K_V41_TAIL_ROPE_EXACT:",
		"&k == &g_kern[K_V41_TAIL_ROPE_QK] || &k == &g_kern[K_V41_TAIL_ROPE_EXACT] || &k == &g_kern[K_V41_SHARED_ATTENTION]",
	} {
		if !strings.Contains(shim, token) {
			t.Errorf("missing strict twin native contract %q", token)
		}
	}
	rows := read("vulkan_v41_rope.go")
	for _, token := range []string{
		"if C.fvk_have_v41_tail_rope_exact() != 0 {",
		"C.fvk_v41_tail_rope_exact_f32(x.ptr, x.ptr, out.ptr, discard.ptr, sinCos.ptr,",
		"v41TailRoPERowsStatusLocked(xb, outBuf, discard, sinCos.buf.(*vulkanBuf), rows, headDim, rotaryDim)",
	} {
		if !strings.Contains(rows, token) {
			t.Errorf("rows panel does not run on the admitted strict twin: missing %q", token)
		}
	}
	marker := read("vulkan_v41_rope_exact.go")
	if !strings.Contains(marker, "//go:build !v41_tail_rope_exact_witness && vulkan") ||
		!strings.Contains(marker, "return C.fvk_have_v41_tail_rope_exact() != 0") {
		t.Fatal("ordinary exact marker must bind the admitted twin and stay out of the witness archive")
	}
}
