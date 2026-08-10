package compute_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildCUDAScriptRejectsIncompleteGoToolchainBeforeNVCC(t *testing.T) {
	data, err := os.ReadFile("build_cuda.sh")
	if err != nil {
		t.Fatal(err)
	}
	script := string(data)
	preflight := strings.Index(script, "src/encoding/base32/base32.go")
	nvcc := strings.Index(script, `echo "[cuda] nvcc compile kernels`)
	if preflight < 0 || nvcc < 0 {
		t.Fatalf("build_cuda.sh is missing the complete-toolchain preflight or nvcc boundary")
	}
	if preflight > nvcc {
		t.Fatalf("complete-toolchain preflight occurs after the expensive nvcc build")
	}
	for _, required := range []string{"src/encoding/base32/base32.go", "src/go/types/api.go"} {
		if !strings.Contains(script, required) {
			t.Errorf("build_cuda.sh does not verify %s", required)
		}
	}
	if !strings.Contains(script, "go env GOROOT") || !strings.Contains(script, "incomplete Go toolchain") {
		t.Error("build_cuda.sh does not resolve and clearly refuse an incomplete GOROOT")
	}
}

func TestGPUAcceptanceScriptsResolveCleanEnvironmentToolkit(t *testing.T) {
	matches, err := filepath.Glob(filepath.Join("..", "..", "tools", "run_*_acceptance_on_gpu.sh"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no GPU acceptance scripts found")
	}
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		script := string(data)
		if strings.Contains(script, "${CUDA_HOME:-$HOME/") {
			t.Errorf("%s expands unset HOME under `set -u`", filepath.Base(path))
		}
		for _, cache := range []string{"GOCACHE", "GOMODCACHE", "GOPATH"} {
			if !strings.Contains(script, "export "+cache+"=") {
				t.Errorf("%s does not provision %s for a HOME-less clean session", filepath.Base(path), cache)
			}
		}
		if !strings.Contains(script, "[ -x /usr/local/cuda/bin/nvcc ]") {
			t.Errorf("%s does not probe the standard CUDA toolkit outside PATH", filepath.Base(path))
		}
	}
}
