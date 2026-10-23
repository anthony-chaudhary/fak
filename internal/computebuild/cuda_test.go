package computebuild

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const cudaArchHelperEnv = "FAK_CUDA_ARCH_TEST_HELPER"

func init() {
	if os.Getenv(cudaArchHelperEnv) != "1" {
		return
	}
	if len(os.Args) == 2 && os.Args[1] == "--version" {
		if os.Getenv("FAK_CUDA_ARCH_TEST_VERSION") == "probe-failure" {
			os.Exit(23)
		}
		fmt.Println(os.Getenv("FAK_CUDA_ARCH_TEST_VERSION"))
		os.Exit(0)
	}
	if err := os.WriteFile(os.Getenv("FAK_CUDA_ARCH_TEST_COMPILE"), []byte(strings.Join(os.Args[1:], " ")), 0o600); err != nil {
		os.Exit(24)
	}
	os.Exit(0)
}

// fak-test:runtime fast est=200ms lane=default
func TestCUDAArchitectureToolchainPreflight(t *testing.T) {
	testBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name    string
		arch    string
		version string
		wantErr string
	}{
		{"sm120-cuda128", "120", "Cuda compilation tools, release 12.8, V12.8.93", ""},
		{"default-cuda128", "", "Cuda compilation tools, release 12.8, V12.8.93", "nvcc reports 12.8"},
		{"default-cuda129", "", "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.9, V12.9.41\nBuild cuda_12.9", ""},
		{"sm103-cuda130", "103", "Cuda compilation tools, release 13.0, V13.0.48\r\n", ""},
		{"default-unparseable", "", "CUDA Version: 12.9", "cannot parse nvcc --version"},
		{"default-probe-failure", "", "probe-failure", "nvcc --version failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			pkgDir := filepath.Join(root, "internal", "compute")
			if err := os.MkdirAll(pkgDir, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(pkgDir, "cuda_arch.txt"), []byte("sm_80\nsm_89\nsm_90\nsm_100\nsm_103\nsm_120\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			compileLog := filepath.Join(root, "compile-args")
			t.Setenv(cudaArchHelperEnv, "1")
			t.Setenv("FAK_CUDA_ARCH_TEST_VERSION", tc.version)
			t.Setenv("FAK_CUDA_ARCH_TEST_COMPILE", compileLog)
			_, err := CompileCUDAKernels(context.Background(), &Toolchain{NVCC: testBinary}, root, tc.arch, false, io.Discard)
			args, compileErr := os.ReadFile(compileLog)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), "sm_103 requires CUDA Toolkit 12.9 or newer") || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("preflight error = %v, want %q", err, tc.wantErr)
				}
				if !os.IsNotExist(compileErr) {
					t.Fatalf("compiler ran despite preflight refusal: args=%q err=%v", args, compileErr)
				}
				return
			}
			if err != nil || compileErr != nil {
				t.Fatalf("compile error=%v, compile log error=%v", err, compileErr)
			}
			if tc.arch == "120" && (strings.Contains(string(args), "compute_103") || !strings.Contains(string(args), "arch=compute_120,code=sm_120")) {
				t.Fatalf("single-arch compile args = %q", args)
			}
			if tc.arch == "" && (!strings.Contains(string(args), "arch=compute_103,code=sm_103") || !strings.Contains(string(args), "arch=compute_120,code=compute_120")) {
				t.Fatalf("default fatbin compile args = %q", args)
			}
		})
	}
}
