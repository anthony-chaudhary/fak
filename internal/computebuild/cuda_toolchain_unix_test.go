//go:build !windows

package computebuild

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestCUDAFallbackKeepsToolkitTogether(t *testing.T) {
	root := t.TempDir()
	stale := filepath.Join(root, "stale-cuda")
	selected := filepath.Join(root, "selected-cuda")
	for _, home := range []string{stale, selected} {
		for _, dir := range []string{"bin", "include", "lib64"} {
			if err := os.MkdirAll(filepath.Join(home, dir), 0755); err != nil {
				t.Fatal(err)
			}
		}
	}
	writeNVCC := func(home string) {
		t.Helper()
		// Discovery needs an executable file, but must not execute it.
		if err := os.WriteFile(filepath.Join(home, "bin", "nvcc"), nil, 0755); err != nil {
			t.Fatal(err)
		}
	}
	writeNVCC(selected)
	t.Setenv("CUDA_HOME", stale)
	t.Setenv("PATH", filepath.Join(selected, "bin"))

	checkToolkit := func(t *testing.T, wantHome string) {
		t.Helper()
		tc, err := DiscoverToolchain()
		if err != nil {
			t.Fatal(err)
		}
		if want := filepath.Join(wantHome, "bin", "nvcc"); tc.NVCC != want {
			t.Errorf("NVCC = %q, want %q", tc.NVCC, want)
		}
		if tc.CUDAHome != wantHome {
			t.Errorf("CUDAHome = %q, want %q", tc.CUDAHome, wantHome)
		}
		if want := []string{filepath.Join(wantHome, "include")}; !reflect.DeepEqual(tc.CUDAInc, want) {
			t.Errorf("CUDAInc = %v, want %v", tc.CUDAInc, want)
		}
		if want := []string{filepath.Join(wantHome, "lib64")}; !reflect.DeepEqual(tc.CUDALib, want) {
			t.Errorf("CUDALib = %v, want %v", tc.CUDALib, want)
		}
		env := SynthesizeCUDACgoEnv(tc, "", false, nil)
		if want := QuoteFlag("-I", filepath.Join(wantHome, "include")); env["CGO_CFLAGS"] != want {
			t.Errorf("CGO_CFLAGS = %q, want %q", env["CGO_CFLAGS"], want)
		}
		if want := QuoteFlag("-L", filepath.Join(wantHome, "lib64")); env["CGO_LDFLAGS"] != want {
			t.Errorf("CGO_LDFLAGS = %q, want %q", env["CGO_LDFLAGS"], want)
		}
	}

	t.Run("missing home compiler falls back coherently", func(t *testing.T) {
		checkToolkit(t, selected)
	})
	// A usable explicit home must keep precedence over the PATH toolkit.
	writeNVCC(stale)
	t.Run("usable explicit home keeps precedence", func(t *testing.T) {
		checkToolkit(t, stale)
	})
}
