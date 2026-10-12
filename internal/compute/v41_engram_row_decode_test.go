package compute

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// Source witness only (Ref #13668): the Engram row decode module declares the
// three F32 modes unconditionally, loads only under the selected device's float
// controls, and keeps the 16-byte four-int32 push ABI its Go wrapper sends. No
// compiler, SPIR-V or GPU proof (TestVulkanV41EngramRowDecode is the device
// witness).
func TestV41EngramRowDecodeSourceContract(t *testing.T) {
	t.Parallel()
	read := func(path string) string {
		t.Helper()
		raw, err := os.ReadFile(filepath.FromSlash(path))
		if err != nil {
			t.Fatal(err)
		}
		return strings.ReplaceAll(string(raw), "\r\n", "\n")
	}
	shader := read("shaders/v41_engram_row_decode.comp")
	header := shader[:strings.Index(shader, "layout(local_size_x = 256) in;")]
	for _, token := range []string{
		`spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4464], 4459, 32);`,
		`spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4466], 4461, 32);`,
		`spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 32);`,
	} {
		if !strings.Contains(header, token) {
			t.Errorf("row decode header lacks %q", token)
		}
	}
	if strings.Contains(header, "#if") {
		t.Error("row decode float modes must be unconditional")
	}
	shim := read("vulkan_shim.cpp")
	for _, token := range []string{
		"const bool v41EngramRowDecodeBuilt = g_v41IndexerFloatControls &&",
		`buildKernel(g_kern[K_V41_ENGRAM_ROW_DECODE], P("v41_engram_row_decode.spv"), 3, 16);`,
		"return g_ready && g_v41IndexerFloatControls && g_have_v41_engram_row_decode &&",
		"&k == &g_kern[K_V41_ENGRAM_ROW_DECODE]",
		"static_assert(sizeof(V41EngramRowDecodePush) == 16",
	} {
		if !strings.Contains(shim, token) {
			t.Errorf("missing Engram row decode native contract %q", token)
		}
	}
	for encoding, stride := range map[V41EngramRowEncoding]int{V41EngramRowPacked: 264, V41EngramRowQ2K: 84, V41EngramRowF32: 1024} {
		if V41EngramRowStride(encoding) != stride || stride%4 != 0 {
			t.Fatalf("encoding %d stride %d, want word-aligned %d", encoding, V41EngramRowStride(encoding), stride)
		}
	}
	for word, want := range map[uint32][3]int{1: {0, 1, 1}, 2: {0, 0, 1}, 1 + 2*255 + 1: {255, 0, 1}, 0: {0, 0, 0}, 1 + 2*256: {0, 0, 0}} {
		j, nanByte, ok := V41EngramRowFault(word)
		if got := [3]int{j, map[bool]int{true: 1}[nanByte], map[bool]int{true: 1}[ok]}; got != want {
			t.Fatalf("V41EngramRowFault(%d) = %v, want %v", word, got, want)
		}
	}
}
