package computebuild

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestWindowsCUDAFallbackKeepsToolkitTogether(t *testing.T) {
	root := t.TempDir()
	stale := filepath.Join(root, "stale CUDA")
	selected := filepath.Join(root, "selected CUDA")
	homeOverride := filepath.Join(root, "CUDA_HOME toolkit")
	for _, home := range []string{stale, selected, homeOverride} {
		for _, dir := range []string{"bin", "include", filepath.Join("lib", "x64")} {
			if err := os.MkdirAll(filepath.Join(home, dir), 0755); err != nil {
				t.Fatal(err)
			}
		}
	}
	writeNVCC := func(home string) {
		t.Helper()
		// Discovery inspects this file but must never execute it.
		if err := os.WriteFile(filepath.Join(home, "bin", "nvcc.exe"), nil, 0755); err != nil {
			t.Fatal(err)
		}
	}
	writeNVCC(selected)
	writeNVCC(homeOverride)
	t.Setenv("PATH", filepath.Join(selected, "bin"))
	t.Setenv("CUDA_PATH", stale)
	t.Setenv("CUDA_HOME", homeOverride)

	checkToolkit := func(t *testing.T, wantHome string) {
		t.Helper()
		tc := &Toolchain{IsWindows: true}
		discoverWindowsCUDA(tc)
		if want := filepath.Join(wantHome, "bin", "nvcc.exe"); tc.NVCC != want {
			t.Errorf("NVCC = %q, want %q", tc.NVCC, want)
		}
		if tc.CUDAHome != wantHome {
			t.Errorf("CUDAHome = %q, want %q", tc.CUDAHome, wantHome)
		}
		if want := []string{filepath.Join(wantHome, "include")}; !reflect.DeepEqual(tc.CUDAInc, want) {
			t.Errorf("CUDAInc = %v, want %v", tc.CUDAInc, want)
		}
		if want := []string{filepath.Join(wantHome, "lib", "x64")}; !reflect.DeepEqual(tc.CUDALib, want) {
			t.Errorf("CUDALib = %v, want %v", tc.CUDALib, want)
		}
		env := SynthesizeCUDACgoEnv(tc, "", false, nil)
		if want := QuoteFlag("-I", filepath.Join(wantHome, "include")); env["CGO_CFLAGS"] != want {
			t.Errorf("CGO_CFLAGS = %q, want %q", env["CGO_CFLAGS"], want)
		}
		if want := QuoteFlag("-L", filepath.Join(wantHome, "lib", "x64")); env["CGO_LDFLAGS"] != want {
			t.Errorf("CGO_LDFLAGS = %q, want %q", env["CGO_LDFLAGS"], want)
		}
	}

	t.Run("missing CUDA_PATH compiler falls back coherently", func(t *testing.T) {
		checkToolkit(t, selected)
	})
	t.Run("missing CUDA_HOME compiler falls back coherently", func(t *testing.T) {
		t.Setenv("CUDA_PATH", "")
		t.Setenv("CUDA_HOME", stale)
		checkToolkit(t, selected)
	})
	t.Run("failed PATH fallback preserves missing compiler", func(t *testing.T) {
		t.Setenv("PATH", root)
		tc := &Toolchain{IsWindows: true}
		discoverWindowsCUDA(tc)
		if tc.NVCC != "" || tc.CUDAHome != stale {
			t.Errorf("NVCC = %q, CUDAHome = %q; want no compiler and %q", tc.NVCC, tc.CUDAHome, stale)
		}
	})
	writeNVCC(stale)
	t.Run("usable CUDA_PATH precedes CUDA_HOME and PATH", func(t *testing.T) {
		checkToolkit(t, stale)
	})
	t.Run("usable CUDA_HOME precedes PATH", func(t *testing.T) {
		t.Setenv("CUDA_PATH", "")
		t.Setenv("CUDA_HOME", stale)
		checkToolkit(t, stale)
	})
}
