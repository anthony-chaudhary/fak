package devcmd

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fakeGoOnPath makes exec.LookPath("go") succeed without depending on the host toolchain.
func fakeGoOnPath(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	name := "go"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte("fixture"), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
}

// isolateCUDALegacyEnv keeps the host's Go prepend directory and GOTOOLCHAIN out of a test: the
// legacy route mutates both on the process, and t.Setenv restores them at cleanup.
func isolateCUDALegacyEnv(t *testing.T) {
	t.Helper()
	prev := cudaLegacyGoBin
	cudaLegacyGoBin = filepath.Join(t.TempDir(), "absent-go-bin")
	t.Cleanup(func() { cudaLegacyGoBin = prev })
	t.Setenv("GOTOOLCHAIN", "")
}

func noopLegacyCommand(t *testing.T) func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error {
	return func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error {
		t.Fatal("command seam reached")
		return nil
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyArchRule(t *testing.T) {
	for _, tc := range []struct {
		arch string
		ok   bool
	}{
		{"89", true},
		{"sm_120", true},
		{"sm_90a", true},
		{"sm_100f", true},
		{"sm_89a", false},
		{"sm_90f", false},
		{"sm_999", false},
		{"sm_abc", false},
	} {
		if err := checkCUDALegacyArch(tc.arch); (err == nil) != tc.ok {
			t.Fatalf("checkCUDALegacyArch(%q) = %v, want ok=%v", tc.arch, err, tc.ok)
		}
	}
	err := checkCUDALegacyArch("sm_89a")
	if err == nil || !strings.HasPrefix(err.Error(), "ERROR: unsupported CUDA arch 'sm_89a'; choose one from ") || !strings.HasSuffix(err.Error(), "(a/f suffix on sm_90+/sm_100+)") {
		t.Fatalf("unsupported arch message = %v", err)
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyRejectsBadArchBeforeAnyWork(t *testing.T) {
	t.Setenv("FAK_CUDA_ARCH", "sm_89a")
	for _, args := range [][]string{{"check"}, {"build"}, {"test"}, {"binary", "pkg", "out"}} {
		var errOut bytes.Buffer
		code := runCUDALegacy(io.Discard, &errOut, args, func(context.Context, *computebuild.CUDAConfig) error {
			t.Fatalf("bad arch reached build for %q", args)
			return nil
		}, noopLegacyCommand(t))
		if code != 2 || !strings.Contains(errOut.String(), "ERROR: unsupported CUDA arch 'sm_89a'") {
			t.Fatalf("args=%q code=%d stderr=%s", args, code, errOut.String())
		}
	}
}

// fak-test:runtime fast est=10ms
func TestPrependPathDirIsIdempotentAtFront(t *testing.T) {
	sep := string(filepath.ListSeparator)
	dir := cudaLegacyGoBin
	if got := prependPathDir("/a"+sep+"/b", dir); got != dir+sep+"/a"+sep+"/b" {
		t.Fatalf("prepend = %q", got)
	}
	if got := prependPathDir(dir+sep+"/b", dir); got != dir+sep+"/b" {
		t.Fatalf("already first changed: %q", got)
	}
	if got := prependPathDir("", dir); got != dir {
		t.Fatalf("empty path = %q", got)
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyEnvironmentDefaultsGOTOOLCHAIN(t *testing.T) {
	t.Setenv("PATH", os.Getenv("PATH"))
	t.Setenv("GOTOOLCHAIN", "")
	applyCUDALegacyEnvironment()
	if got := os.Getenv("GOTOOLCHAIN"); got != "auto" {
		t.Fatalf("GOTOOLCHAIN = %q, want auto", got)
	}
	t.Setenv("GOTOOLCHAIN", "go1.26.1")
	applyCUDALegacyEnvironment()
	if got := os.Getenv("GOTOOLCHAIN"); got != "go1.26.1" {
		t.Fatalf("explicit GOTOOLCHAIN overwritten: %q", got)
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyRefusesMissingGoBeforeBuild(t *testing.T) {
	isolateCUDALegacyEnv(t)
	t.Setenv("FAK_CUDA_ARCH", "")
	t.Setenv("PATH", t.TempDir())
	var errOut bytes.Buffer
	code := runCUDALegacy(io.Discard, &errOut, []string{"build"}, func(context.Context, *computebuild.CUDAConfig) error {
		t.Fatal("missing go reached CUDA build")
		return nil
	}, noopLegacyCommand(t))
	if code != 2 || strings.TrimSpace(errOut.String()) != "[cuda] Go 1.26+ is required (no go executable on PATH)" {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyRefusesIncompleteGoRoot(t *testing.T) {
	isolateCUDALegacyEnv(t)
	t.Setenv("FAK_CUDA_ARCH", "")
	fakeGoOnPath(t)
	emptyRoot := t.TempDir()
	for _, tc := range []struct {
		name    string
		command func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error
		where   string
	}{
		{"empty goroot", func(_ context.Context, _ string, _ []string, _ string, _ []string, out, _ io.Writer) error {
			_, err := io.WriteString(out, emptyRoot)
			return err
		}, emptyRoot},
		{"go env failure", func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error {
			return errors.New("go env failed")
		}, "<unknown>"},
	} {
		var errOut bytes.Buffer
		code := runCUDALegacy(io.Discard, &errOut, []string{"build"}, func(context.Context, *computebuild.CUDAConfig) error {
			t.Fatalf("%s reached CUDA build", tc.name)
			return nil
		}, tc.command)
		want := "[cuda] incomplete Go toolchain at " + tc.where + ": missing src/encoding/base32/base32.go"
		if code != 2 || !strings.Contains(errOut.String(), want) {
			t.Fatalf("%s: code=%d stderr=%q", tc.name, code, errOut.String())
		}
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyCheckIsStrictAndSkipsCUDAPreflight(t *testing.T) {
	t.Setenv("FAK_CUDA_ARCH", "")
	t.Setenv("CC", "cc-host")
	var seen bool
	code := runCUDALegacy(io.Discard, io.Discard, []string{"check"}, func(ctx context.Context, cfg *computebuild.CUDAConfig) error {
		seen = true
		if cfg.Command != "check" || !computebuild.IsStrictCUDACheck(ctx) {
			t.Fatalf("legacy check not strict: command=%s", cfg.Command)
		}
		if cfg.Toolchain == nil || cfg.Toolchain.CC != "cc-host" {
			t.Fatalf("legacy check toolchain = %+v", cfg.Toolchain)
		}
		return nil
	}, noopLegacyCommand(t))
	if code != 0 || !seen {
		t.Fatalf("legacy check code=%d seen=%v", code, seen)
	}
}

// fak-test:runtime fast est=20ms
func TestCUDALegacyBinaryKeepsOutPathWithSpaces(t *testing.T) {
	isolateCUDALegacyEnv(t)
	t.Setenv("FAK_CUDA_ARCH", "")
	fakeGoOnPath(t)
	goRoot := t.TempDir()
	for _, name := range []string{"src/encoding/base32/base32.go", "src/go/types/api.go"} {
		path := filepath.Join(goRoot, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("package x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	command := func(_ context.Context, _ string, _ []string, _ string, _ []string, out, _ io.Writer) error {
		_, err := io.WriteString(out, goRoot)
		return err
	}
	const outBin = "output with spaces/fak"
	var got *computebuild.CUDAConfig
	code := runCUDALegacy(io.Discard, io.Discard, []string{"binary", "./cmd/fak", outBin}, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
		got = cfg
		return nil
	}, command)
	if code != 0 || got == nil {
		t.Fatalf("binary code=%d seen=%v", code, got != nil)
	}
	if got.OutBin != outBin || got.OutPkg != "./cmd/fak" {
		t.Fatalf("binary out pkg=%q bin=%q, want %q and %q", got.OutPkg, got.OutBin, "./cmd/fak", outBin)
	}
}

// fak-test:runtime fast est=20ms
func TestPlainBuildCUDACheckStaysFailOpen(t *testing.T) {
	var seen bool
	code := runBuildCUDAWith(io.Discard, io.Discard, []string{"check"}, func(ctx context.Context, cfg *computebuild.CUDAConfig) error {
		seen = true
		if computebuild.IsStrictCUDACheck(ctx) {
			t.Fatal("plain build-cuda check became strict")
		}
		return nil
	}, noopLegacyCommand(t))
	if code != 0 || !seen {
		t.Fatalf("plain check code=%d seen=%v", code, seen)
	}
}
