package compute

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// metalAttentionSourceCandidates lists the locations the MSL sources may be read
// from depending on the working directory Go tests run under.
func metalAttentionSourceCandidates(name string) []string {
	return []string{
		filepath.Join("internal", "compute", "shaders", name),
		filepath.Join("shaders", name),
		filepath.Join("..", "..", "internal", "compute", "shaders", name),
	}
}

func metalShimSourceCandidates() []string {
	return []string{
		filepath.Join("internal", "compute", "metal_shim.m"),
		"metal_shim.m",
		filepath.Join("..", "..", "internal", "compute", "metal_shim.m"),
	}
}

// readMetalSource reads the first non-empty candidate path.
func readMetalSource(t *testing.T, candidates []string) string {
	t.Helper()
	for _, p := range candidates {
		b, err := os.ReadFile(p)
		if err == nil && len(b) > 0 {
			return string(b)
		}
	}
	t.Fatalf("could not read metal source from candidate paths: %v", candidates)
	return ""
}

// extractKernelBodies returns a map of kernel name -> body text for every
// `kernel void <name>(...)  { ... }` definition in the MSL source, using
// balanced-brace extraction so a wrapper's body is isolated from its neighbors.
func extractKernelBodies(src string) map[string]string {
	re := regexp.MustCompile(`kernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	bodies := map[string]string{}
	for _, loc := range re.FindAllStringSubmatchIndex(src, -1) {
		name := src[loc[2]:loc[3]]
		brace := strings.IndexByte(src[loc[1]:], '{')
		if brace < 0 {
			continue
		}
		start := loc[1] + brace
		depth := 0
		end := -1
		for i := start; i < len(src); i++ {
			switch src[i] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					end = i
				}
			}
			if end >= 0 {
				break
			}
		}
		if end > start {
			bodies[name] = src[start : end+1]
		}
	}
	return bodies
}

// assertNoKernelToKernelCall fails when any kernel body calls another declared
// kernel name. MSL rejects kernel-to-kernel calls at compile time; shared bodies
// must live in a `static inline` device function helper instead.
func assertNoKernelToKernelCall(t *testing.T, label, src string) {
	t.Helper()
	bodies := extractKernelBodies(src)
	if len(bodies) == 0 {
		t.Fatalf("%s: no `kernel void` definitions found to audit", label)
	}
	for name, body := range bodies {
		for callee := range bodies {
			if callee == name {
				continue
			}
			callRe := regexp.MustCompile(`\b` + regexp.QuoteMeta(callee) + `\s*\(`)
			if callRe.MatchString(body) {
				t.Errorf("%s: kernel %q calls kernel %q (kernel-to-kernel call is illegal MSL)", label, name, callee)
			}
		}
	}
}

// TestMetalAttentionNoKernelToKernelCall is the build-tag-free source-contract
// witness for issue #12817: Metal rejects kernel-to-kernel calls at MSL compile
// time, so the shared attention body must be a `static inline` device helper
// (`attention_f32_impl`) and the `flash_attention_tiled_f32` kernel must call
// that helper, never the `attention_f32` kernel entrypoint.
func TestMetalAttentionNoKernelToKernelCall(t *testing.T) {
	attnSrc := readMetalSource(t, metalAttentionSourceCandidates("attention.metal"))
	shimSrc := readMetalSource(t, metalShimSourceCandidates())

	for label, src := range map[string]string{
		"attention.metal": attnSrc,
		"metal_shim.m":    shimSrc,
	} {
		assertNoKernelToKernelCall(t, label, src)

		if !strings.Contains(src, "attention_f32_impl") {
			t.Errorf("%s: missing shared helper token %q", label, "attention_f32_impl")
		}
		if !strings.Contains(src, "attention_f32_impl(") {
			t.Errorf("%s: missing shared helper call token %q", label, "attention_f32_impl(")
		}
	}

	attnBodies := extractKernelBodies(attnSrc)
	flash, ok := attnBodies["flash_attention_tiled_f32"]
	if !ok {
		t.Fatalf("attention.metal: missing kernel flash_attention_tiled_f32")
	}
	if regexp.MustCompile(`\battention_f32\s*\(`).MatchString(flash) {
		t.Errorf("attention.metal: flash_attention_tiled_f32 calls attention_f32; must call attention_f32_impl")
	}
	if !regexp.MustCompile(`\battention_f32_impl\s*\(`).MatchString(flash) {
		t.Errorf("attention.metal: flash_attention_tiled_f32 must call attention_f32_impl")
	}
}
