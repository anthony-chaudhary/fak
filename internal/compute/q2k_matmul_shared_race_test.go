package compute

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestQ2KPlainMatMulAvoidsWorkgroupSharedInput guards #13807: plainMatMul maps the
// invocations of one workgroup onto different tokens when P>1, so any write of its
// input into workgroup-shared memory races with a neighbour's q2kDot read.
func TestQ2KPlainMatMulAvoidsWorkgroupSharedInput(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("shaders", "q2k_matmul.comp"))
	if err != nil {
		t.Fatalf("read q2k_matmul.comp: %v", err)
	}
	src := string(raw)

	shared := regexp.MustCompile(`(?m)^\s*shared\s+\w+\s+(\w+)\s*\[`).FindAllStringSubmatch(src, -1)
	if len(shared) == 0 {
		t.Fatal("q2k_matmul.comp declares no shared arrays; the fused paths need them")
	}

	body := glslFunctionBody(t, src, "void plainMatMul()")
	for _, m := range shared {
		name := m[1]
		if regexp.MustCompile(`\b` + name + `\b`).MatchString(body) {
			t.Errorf("plainMatMul references workgroup-shared %q; its invocations span tokens, so X must be read per invocation", name)
		}
	}
}

func glslFunctionBody(t *testing.T, src, signature string) string {
	t.Helper()
	start := strings.Index(src, signature)
	if start < 0 {
		t.Fatalf("shader has no %q", signature)
	}
	open := strings.Index(src[start:], "{")
	if open < 0 {
		t.Fatalf("%q has no body", signature)
	}
	depth := 0
	for i := start + open; i < len(src); i++ {
		switch src[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return src[start+open : i+1]
			}
		}
	}
	t.Fatalf("%q body is unterminated", signature)
	return ""
}
