package compute

import (
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
// Estimate is unmeasured. This checks source ABI coupling, not compiled SPIR-V,
// pipeline creation, native dispatch, or physical attention numerics.
func TestVulkanDecodeAttentionPushContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return stripGLSLComments(string(b))
	}
	shader, shim := read("shaders/attention.comp"), read("vulkan_shim.cpp")
	capture := func(pattern, source string) string {
		t.Helper()
		m := regexp.MustCompile(pattern).FindStringSubmatch(source)
		if len(m) != 2 {
			t.Fatalf("missing unique source contract %q", pattern)
		}
		if len(regexp.MustCompile(pattern).FindAllStringSubmatch(source, -1)) != 1 {
			t.Fatalf("ambiguous source contract %q", pattern)
		}
		return m[1]
	}
	fields := func(body string) []string {
		t.Helper()
		var out []string
		for _, declaration := range strings.Split(body, ";") {
			parts := strings.Fields(declaration)
			if len(parts) == 0 {
				continue
			}
			if len(parts) != 2 {
				t.Fatalf("unexpected ABI declaration %q", declaration)
			}
			out = append(out, strings.Join(parts, " "))
		}
		return out
	}
	want := []string{"int nPos", "int nH", "int nKV", "int hd", "float scale", "int mode", "int tileCount", "int causal", "int windowSize"}
	shaderFields := fields(capture(`(?s)layout\(push_constant\)\s+uniform\s+Push\s*\{([^}]*)\}\s*pc\s*;`, shader))
	shimFields := fields(capture(`(?s)struct\s+AttentionPush\s*\{([^}]*)\}\s*;`, shim))
	if !reflect.DeepEqual(shaderFields, want) || !reflect.DeepEqual(shimFields, want) {
		t.Fatalf("push layouts shader=%v shim=%v want=%v", shaderFields, shimFields, want)
	}
	compact := strings.Join(strings.Fields(shim), " ")
	for _, token := range []string{
		`sizeof(int) == 4 && sizeof(float) == 4 && sizeof(AttentionPush) == 36`,
		`offsetof(AttentionPush, causal) == 28`,
		`offsetof(AttentionPush, windowSize) == 32`,
		`buildKernel(g_kern[K_ATTENTION], P("attention.spv"), 5, sizeof(AttentionPush))`,
	} {
		if !strings.Contains(compact, token) {
			t.Errorf("missing native ABI coupling %q", token)
		}
	}
	start := strings.Index(shim, "void fvk_attention_f32(")
	if start < 0 {
		t.Fatal("missing actual attention C owner")
	}
	// The next top-level function begins after this owner; extract by balanced
	// braces so neighboring dispatches cannot satisfy the owner's assertions.
	bodyStart := strings.Index(shim[start:], "{") + start
	depth, end := 0, -1
	for i := bodyStart; i < len(shim); i++ {
		switch shim[i] {
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
	if end < 0 {
		t.Fatal("unterminated attention owner")
	}
	body := strings.Join(strings.Fields(shim[bodyStart:end+1]), " ")
	initializer := capture(`AttentionPush\s+pc\s*\{([^}]*)\}\s*;`, body)
	args := strings.Split(initializer, ",")
	for i := range args {
		args[i] = strings.TrimSpace(args[i])
	}
	if !reflect.DeepEqual(args, []string{"nPos", "nH", "nKV", "hd", "scale", "0", "1", "1", "0"}) {
		t.Fatalf("decode defaults changed: %v", args)
	}
	if strings.Count(body, `dispatch(g_kern[K_ATTENTION], bufs, &pc, sizeof(pc),`) != 3 {
		t.Fatal("all three default/split dispatches must send the complete push object")
	}
	for _, token := range []string{`if (!contextSplit)`, `pc.mode = 1;`, `pc.tileCount = (int)tileCount;`, `pc.mode = 2;`} {
		if !strings.Contains(body, token) {
			t.Errorf("context-split contract lost %q", token)
		}
	}
	if regexp.MustCompile(`pc\.(causal|windowSize)\s*=`).MatchString(body) {
		t.Fatal("dispatch overrides supplied-prefix mask defaults")
	}
	causal, err := strconv.Atoi(args[7])
	if err != nil {
		t.Fatal(err)
	}
	window, err := strconv.Atoi(args[8])
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range []string{`int queryPos = pc.nPos - 1;`, `pc.causal != 0`, `pc.windowSize > 0`, `tile + Bc <= minAttendPos`} {
		if !strings.Contains(shader, token) {
			t.Errorf("shader bound semantics changed %q", token)
		}
	}
	// Independent scalar set control for the current single-query mapping.
	// Covers empty prefix and both sides of the 256-row context-split boundary.
	for _, rows := range []int{0, 1, 255, 256, 257, 513} {
		queryPos := rows - 1
		maximum := rows
		if causal != 0 {
			maximum = max(0, queryPos+1)
		}
		minimum := 0
		if window > 0 {
			minimum = max(0, queryPos-window+1)
		}
		if minimum != 0 || maximum != rows {
			t.Fatalf("defaults mask supplied prefix: rows=%d bounds=[%d,%d)", rows, minimum, maximum)
		}
		visited := 0
		for tile := 0; tile < rows; tile += 256 {
			if causal != 0 && tile >= maximum {
				break
			}
			if window > 0 && tile+256 <= minimum {
				continue
			}
			visited += min(256, rows-tile)
		}
		if visited != rows {
			t.Fatalf("default bounds omitted rows: got=%d want=%d", visited, rows)
		}
	}
}
