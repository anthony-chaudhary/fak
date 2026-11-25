package devcmd

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// writeLegacyManifestFixture lays out a module root whose architecture manifest uses the given
// bytes, so the legacy route resolves it exactly as a Windows checkout would.
func writeLegacyManifestFixture(t *testing.T, root, manifest string) {
	t.Helper()
	compute := filepath.Join(root, "internal", "compute")
	if err := os.MkdirAll(compute, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.test/cuda\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(computebuild.CUDAArchManifestPath(root), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyAcceptsArchFromCRLFManifest(t *testing.T) {
	root := t.TempDir()
	writeLegacyManifestFixture(t, root, "sm_80\r\nsm_89\r\nsm_90\r\nsm_100\r\n")
	t.Chdir(root)
	if err := checkCUDALegacyArch("sm_89"); err != nil {
		t.Fatalf("supported arch from CRLF manifest refused: %v", err)
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyRefusesUnknownArchWithCRLFManifestBeforeBuild(t *testing.T) {
	root := t.TempDir()
	writeLegacyManifestFixture(t, root, "sm_80\r\nsm_89\r\n")
	t.Chdir(root)
	t.Setenv("FAK_CUDA_ARCH", "sm_999")
	var errOut bytes.Buffer
	code := runCUDALegacy(io.Discard, &errOut, []string{"check"}, func(context.Context, *computebuild.CUDAConfig) error {
		t.Fatal("unknown arch reached the check verb")
		return nil
	}, noopLegacyCommand(t))
	if code != 2 {
		t.Fatalf("exit = %d, want 2; stderr=%q", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "unsupported CUDA arch 'sm_999'") {
		t.Fatalf("stderr = %q, want unsupported-arch message", errOut.String())
	}
	if strings.Contains(errOut.String(), "\r") {
		t.Fatalf("CRLF leaked into the supported-arch list: %q", errOut.String())
	}
}
