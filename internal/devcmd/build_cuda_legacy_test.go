package devcmd

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=50ms
func TestCUDALegacyArgumentsAndEnvironment(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("CUDA_HOME", "")
	t.Setenv("FAK_CUDA_ARCH", "sm_120")
	t.Setenv("FAK_CUDA_NCCL", "1")
	t.Setenv("NCCL_HOME", filepath.Join(home, "NCCL with spaces"))
	t.Setenv("FAK_NVCC_CCBIN", "nvcc-host")
	t.Setenv("CXX", "cgo-host")
	toolkit := filepath.Join(home, "cudaenv")
	if err := os.MkdirAll(filepath.Join(toolkit, "bin"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(toolkit, "bin", "nvcc"), []byte("fixture"), 0755); err != nil {
		t.Fatal(err)
	}
	for _, spec := range []struct {
		args                     []string
		command, pkg, out, model string
		steps                    int
	}{
		{nil, "test", "", "", "", 0},
		{[]string{"build"}, "build", "", "", "", 0},
		{[]string{"binary", "./cmd/fak", "output with spaces/fak"}, "binary", "./cmd/fak", "output with spaces/fak", "", 0},
		{[]string{"bench", "model with spaces", "32"}, "bench", "", "", "model with spaces", 32},
	} {
		var errOut bytes.Buffer
		calls := 0
		code := runBuildCUDAWith(io.Discard, &errOut, append([]string{"legacy"}, spec.args...), func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			calls++
			if cfg.Command != spec.command || cfg.OutPkg != spec.pkg || cfg.OutBin != spec.out || cfg.BenchModelDir != spec.model || cfg.BenchDecodeSteps != spec.steps || !cfg.SkipSmoke || cfg.Smoke {
				t.Fatalf("legacy mapping command=%s cfg=%+v", spec.command, cfg)
			}
			if cfg.Toolchain == nil || cfg.Toolchain.CUDAHome != toolkit || cfg.Toolchain.CXX != "nvcc-host" || cfg.Toolchain.VsDevEnv["CXX"] != "cgo-host" || cfg.Arch != "sm_120" || !cfg.EnableNCCL || cfg.NCCLHome != os.Getenv("NCCL_HOME") {
				t.Fatal("legacy toolchain/environment mapping lost")
			}
			return nil
		}, func(_ context.Context, name string, args []string, _ string, _ []string, out, _ io.Writer) error {
			if name != "go" || len(args) != 2 || args[0] != "env" || args[1] != "GOROOT" {
				t.Fatalf("unexpected compatibility command %s %q", name, args)
			}
			_, err := io.WriteString(out, runtime.GOROOT())
			return err
		})
		if code != 0 || calls != 1 {
			t.Fatalf("legacy exit=%d calls=%d stderr=%s", code, calls, &errOut)
		}
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyRefusesBeforeBuild(t *testing.T) {
	for _, args := range [][]string{{"unknown"}, {"binary", "pkg"}, {"bench", "model", "1", "extra"}, {"build", "extra"}, {"test"}} {
		code := runCUDALegacy(io.Discard, io.Discard, args, func(context.Context, *computebuild.CUDAConfig) error {
			t.Fatal("invalid input or incomplete Go toolchain reached CUDA build")
			return nil
		}, func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error {
			return errors.New("incomplete toolchain")
		})
		if code == 0 {
			t.Fatalf("invalid legacy invocation accepted: %q", args)
		}
	}
	code := runCUDALegacy(io.Discard, io.Discard, []string{"check"}, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
		if cfg.Command != "check" {
			t.Fatal("check mapping lost")
		}
		return nil
	}, func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error {
		t.Fatal("GPU-free check requires no CUDA/Go preflight")
		return nil
	})
	if code != 0 {
		t.Fatal("GPU-free legacy check failed")
	}
}
