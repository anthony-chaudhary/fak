package computebuild

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
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
	t.Setenv("LIBRARY_PATH", "/ambient-must-not-leak")
	baseEnv := []string{"LIBRARY_PATH=/opt/existing"}

	// No libcuda.so linker name: not WSL, no link search path added.
	env := SynthesizeCUDACgoEnv(baseEnv, &Toolchain{}, "", false, nil)
	if _, ok := env["LIBRARY_PATH"]; ok || strings.Contains(env["CGO_LDFLAGS"], QuoteFlag("-L", wsl)) {
		t.Fatalf("non-WSL env gained WSL link path: %v", env)
	}

	if got := cgoBaseEnvValue(MergeEnviron(baseEnv, env), "LIBRARY_PATH"); got != "/opt/existing" {
		t.Fatalf("non-WSL inheritance=%q", got)
	}

	if err := os.WriteFile(filepath.Join(wsl, "libcuda.so"), []byte("fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	env = SynthesizeCUDACgoEnv(baseEnv, &Toolchain{}, "", false, nil)
	if !strings.Contains(env["CGO_LDFLAGS"], QuoteFlag("-L", wsl)) {
		t.Fatalf("CGO_LDFLAGS=%q lacks WSL driver dir", env["CGO_LDFLAGS"])
	}
	want := wsl + string(os.PathListSeparator) + "/opt/existing"
	if env["LIBRARY_PATH"] != want {
		t.Fatalf("LIBRARY_PATH=%q want %q", env["LIBRARY_PATH"], want)
	}

	sep := string(os.PathListSeparator)
	for _, tc := range []struct {
		name string
		base []string
		want string
	}{
		{"nil", nil, wsl},
		{"empty", []string{}, wsl},
		{"empty-value", []string{"LIBRARY_PATH="}, wsl},
		{"ordered", []string{"LIBRARY_PATH=" + strings.Join([]string{"", "/first", wsl, "/first", "/last", wsl, ""}, sep)}, wsl + sep + "/first" + sep + "/first" + sep + "/last"},
		{"last-key", []string{"LIBRARY_PATH=/discard", "malformed", "=ignored", "library_path=/wrong-case", "LIBRARY_PATH=/explicit=equals"}, wsl + sep + "/explicit=equals"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := append([]string(nil), tc.base...)
			first := SynthesizeCUDACgoEnv(tc.base, &Toolchain{}, "", false, nil)
			t.Setenv("LIBRARY_PATH", "/different-ambient")
			second := SynthesizeCUDACgoEnv(tc.base, &Toolchain{}, "", false, nil)
			if first["LIBRARY_PATH"] != tc.want || second["LIBRARY_PATH"] != tc.want {
				t.Fatalf("explicit path: first=%q second=%q want=%q", first["LIBRARY_PATH"], second["LIBRARY_PATH"], tc.want)
			}
			if len(tc.base) != len(before) || len(before) > 0 && !reflect.DeepEqual(tc.base, before) {
				t.Fatal("base environment mutated")
			}
		})
	}

	win := SynthesizeCUDACgoEnv(baseEnv, &Toolchain{IsWindows: true}, "", false, nil)
	if _, ok := win["LIBRARY_PATH"]; ok {
		t.Fatal("Windows env gained LIBRARY_PATH")
	}
	if got := cgoBaseEnvValue(MergeEnviron(baseEnv, win), "LIBRARY_PATH"); got != "/opt/existing" {
		t.Fatalf("Windows inheritance=%q", got)
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
