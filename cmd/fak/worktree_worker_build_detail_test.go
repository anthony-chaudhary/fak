package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestGoBuildVerifyFailureDetailKeepsFirstError pins that a failed verify build
// keeps the head of the toolchain output, where Go prints the cause (the first
// compile error, or a linker "fatal error:" ahead of its goroutine dump). The
// old tail-only 500-byte cut dropped it, so land refusals such as
// ERR_FLOW_WHOLE_MODULE_BUILD_FAILED carried only trailing errors or stack frames.
func TestGoBuildVerifyFailureDetailKeepsFirstError(t *testing.T) {
	t.Setenv("FAK_SHARED_GOCACHE", "")
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain is not installed")
	}
	wt := t.TempDir()
	if err := os.WriteFile(filepath.Join(wt, "go.mod"), []byte("module example.com/verify\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	pad := strings.Repeat("x", 180)
	var src strings.Builder
	src.WriteString("package verify\n\nfunc broken() {\n")
	src.WriteString("\t_ = firstCauseMarker" + pad + "\n")
	for i := 0; i < 12; i++ {
		fmt.Fprintf(&src, "\t_ = laterError%02d%s\n", i, pad)
	}
	src.WriteString("}\n")
	if err := os.WriteFile(filepath.Join(wt, "verify.go"), []byte(src.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	ok, detail := worktreeWorkerGoBuildVerify(wt)
	if ok {
		t.Fatal("go-build verify passed a package that does not compile")
	}
	if !strings.Contains(detail, "firstCauseMarker") {
		t.Fatalf("failure detail lost the first compile error: %q", detail)
	}
	if !strings.Contains(detail, "too many errors") {
		t.Fatalf("failure detail lost the tail of the build output: %q", detail)
	}
	if len(detail) > 1200 {
		t.Fatalf("failure detail is unbounded (%d bytes)", len(detail))
	}
}
