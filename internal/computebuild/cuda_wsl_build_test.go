package computebuild

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
// fak-test:justify why=regression when=changed:internal/computebuild/**
func TestCUDACgoEnvLinksWSLDriverLibcuda(t *testing.T) {
	wsl := filepath.Join(t.TempDir(), "wsl-lib")
	if err := os.MkdirAll(wsl, 0o755); err != nil {
		t.Fatal(err)
	}
	old := wslCUDADriverDir
	wslCUDADriverDir = wsl
	t.Cleanup(func() { wslCUDADriverDir = old })
	t.Setenv("LIBRARY_PATH", "")

	// No libcuda.so linker name: not WSL, no link search path added.
	env := SynthesizeCUDACgoEnv(&Toolchain{}, "", false, nil)
	if _, ok := env["LIBRARY_PATH"]; ok || strings.Contains(env["CGO_LDFLAGS"], QuoteFlag("-L", wsl)) {
		t.Fatalf("non-WSL env gained WSL link path: %v", env)
	}

	if err := os.WriteFile(filepath.Join(wsl, "libcuda.so"), []byte("fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LIBRARY_PATH", "/opt/existing")
	env = SynthesizeCUDACgoEnv(&Toolchain{}, "", false, nil)
	if !strings.Contains(env["CGO_LDFLAGS"], QuoteFlag("-L", wsl)) {
		t.Fatalf("CGO_LDFLAGS=%q lacks WSL driver dir", env["CGO_LDFLAGS"])
	}
	want := wsl + string(os.PathListSeparator) + "/opt/existing"
	if env["LIBRARY_PATH"] != want {
		t.Fatalf("LIBRARY_PATH=%q want %q", env["LIBRARY_PATH"], want)
	}

	win := SynthesizeCUDACgoEnv(&Toolchain{IsWindows: true}, "", false, nil)
	if _, ok := win["LIBRARY_PATH"]; ok {
		t.Fatal("Windows env gained LIBRARY_PATH")
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=regression when=changed:internal/computebuild/**
func TestObserveCUDANVCCResolvesSymlink(t *testing.T) {
	fx := newCUDAProvenanceFixture(t, "sm_80\n")
	link := filepath.Join(t.TempDir(), "nvcc"+execSuffix())
	if err := os.Symlink(fx.nvcc, link); err != nil {
		t.Skipf("symlink unsupported here: %v", err)
	}
	run := stubCUDARunner(nil, "Cuda compilation tools, release 12.6, V12.6.85\n", "")
	direct, err := observeCUDANVCC(context.Background(), &Toolchain{NVCC: fx.nvcc}, run)
	if err != nil {
		t.Fatal(err)
	}
	viaLink, err := observeCUDANVCC(context.Background(), &Toolchain{NVCC: link}, run)
	if err != nil {
		t.Fatalf("symlinked nvcc refused: %v", err)
	}
	if viaLink != direct {
		t.Fatalf("symlink identity %+v != direct %+v", viaLink, direct)
	}
}
