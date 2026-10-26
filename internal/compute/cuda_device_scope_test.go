package compute

import (
	"context"
	_ "embed"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

//go:embed cuda/runtime_services.cuh
var cudaDeviceScopeServices []byte

//go:embed cuda/device_scope.h
var cudaDeviceScopeGuard []byte

//go:embed testdata/cuda_device_scope_test.cpp
var cudaDeviceScopeHarness []byte

// Compile the production on-device operations against a deterministic fake CUDA
// runtime. This exercises host ownership and call ordering, not CUDA execution.
// fak-test:runtime medium est=3s lane=default
func TestCUDADeviceScopeRestoresProductionOperations(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dir := t.TempDir()
	for _, source := range []struct {
		name string
		data []byte
	}{
		{"runtime_services.cuh", cudaDeviceScopeServices},
		{"device_scope.h", cudaDeviceScopeGuard},
		{"device_scope_test.cpp", cudaDeviceScopeHarness},
	} {
		if err := os.WriteFile(filepath.Join(dir, source.name), source.data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	name := "device-scope"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binary := filepath.Join(dir, name)
	args := []string{"-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", dir, filepath.Join(dir, "device_scope_test.cpp"), "-o", binary}
	if out, err := exec.CommandContext(ctx, compiler, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile production CUDA device-scope harness: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production CUDA device-scope contract: %v\n%s", err, out)
	}
}
