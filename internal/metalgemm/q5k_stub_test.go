//go:build !darwin || !arm64 || !cgo

package metalgemm

import (
	"context"
	"os"
	"os/exec"
	"testing"
	"time"
)

// fak-test:runtime integration est=30s lane=optin
func TestQ5KPortableLoaderBuild(t *testing.T) {
	if os.Getenv("FAK_TEST_Q5K_PORTABLE_BUILD") != "1" {
		t.Skip("set FAK_TEST_Q5K_PORTABLE_BUILD=1 for the portable loader build witness")
	}
	// Keep this test buildable before the stub exists: the real importer must
	// resolve the capability symbol even where its runtime Metal branch is false.
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "build", "-p=1", "../ggufload")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("portable GGUF loader build failed: %v\n%s", err, output)
	}
}
