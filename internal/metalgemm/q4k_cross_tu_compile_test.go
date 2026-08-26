//go:build darwin && arm64 && cgo

package metalgemm

import (
	"os/exec"
	"testing"
)

// TestQ4KQ8BridgeCompilesCallingTranslationUnit keeps the mixed primitive's native boundary
// executable: q4k.m calls helpers defined in q8.m, so declarations must be visible when Clang
// compiles q4k.m independently. A source-name check cannot prove that compile contract.
func TestQ4KQ8BridgeCompilesCallingTranslationUnit(t *testing.T) {
	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Fatalf("darwin/arm64 cgo build requires clang: %v", err)
	}
	cmd := exec.Command(clang, "-fsyntax-only", "-Werror=implicit-function-declaration", "q4k.m")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile q4k.m cross-translation-unit contract: %v\n%s", err, output)
	}
}
