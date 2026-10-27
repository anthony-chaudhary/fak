package computebuild

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func init() {
	if os.Getenv("FAK_CUDA_GRAPH_PHASE_HELPER") == "1" && len(os.Args) > 1 && os.Args[1] == "test" {
		f, err := os.OpenFile("graph-test-passes", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			os.Exit(27)
		}
		_, _ = f.WriteString(os.Getenv("FAK_CUDA_GRAPH") + "\n")
		_ = f.Close()
		if os.Getenv("FAK_CUDA_GRAPH") == "1" && os.Getenv("FAK_CUDA_GRAPH_PHASE_FAIL") == "1" {
			os.Exit(28)
		}
		os.Exit(0)
	}
	if os.Getenv("FAK_CUDA_SHELL_DELEGATION_HELPER") == "1" {
		if !FileExists(filepath.Join("cmd", "fak-dev", "main.go")) || !FileExists("go.mod") {
			os.Exit(29)
		}
		_ = json.NewEncoder(os.Stdout).Encode(os.Args[1:])
		os.Exit(0)
	}
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime medium est=5s
func TestCUDATestGraphPassesShareCompileAndFailureReceipt(t *testing.T) {
	for _, failSecond := range []bool{false, true} {
		t.Run(map[bool]string{false: "passed", true: "graph-failed"}[failSecond], func(t *testing.T) {
			root, tc := cudaIncrementalFixture(t)
			self, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			raw, err := os.ReadFile(self)
			if err != nil {
				t.Fatal(err)
			}
			name := "go"
			if runtime.GOOS == "windows" {
				name += ".exe"
			}
			if err := os.WriteFile(filepath.Join(root, name), raw, 0755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.test/cuda\n\ngo 1.26\n"), 0600); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
			t.Setenv("FAK_CUDA_GRAPH_PHASE_HELPER", "1")
			t.Setenv("FAK_CUDA_GRAPH", "inherited")
			t.Setenv("FAK_CUDA_GRAPH_PHASE_FAIL", map[bool]string{false: "0", true: "1"}[failSecond])
			cfg := &CUDAConfig{Command: "test", RepoRoot: root, Arch: "sm_120", Toolchain: tc, Stdout: io.Discard, Stderr: io.Discard}
			err = RunCUDA(context.Background(), cfg)
			if (err != nil) != failSecond {
				t.Fatalf("graph test result=%v failSecond=%v", err, failSecond)
			}
			passes, err := os.ReadFile(filepath.Join(root, "graph-test-passes"))
			if err != nil || string(passes) != "0\n1\n" {
				t.Fatalf("graph passes=%q err=%v", passes, err)
			}
			if cudaCompileCount(t, root) != 1 {
				t.Fatal("graph passes did not share one native compilation")
			}
			native := cudaReadNativeReceipt(t, root)
			wantOutcome := map[bool]string{false: "success", true: "failed"}[failSecond]
			if native.Outcome != wantOutcome || cfg.Receipt.Outcome != wantOutcome {
				t.Fatalf("main/native outcome=%s/%s", cfg.Receipt.Outcome, native.Outcome)
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/compute/build_cuda.sh,internal/computebuild/**
// fak-test:runtime fast est=1s
func TestBuildCUDAScriptDelegatesToNativeDriver(t *testing.T) {
	bash, err := exec.LookPath("bash")
	if runtime.GOOS == "windows" {
		bash = filepath.Join(os.Getenv("ProgramFiles"), "Git", "bin", "bash.exe")
		_, err = os.Stat(bash)
	}
	if err != nil {
		t.Skip("bash is required to exercise build_cuda.sh")
	}
	script, err := filepath.Abs(filepath.Join("..", "compute", "build_cuda.sh"))
	if err != nil {
		t.Fatal(err)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bash, script, "binary", "./cmd/fak", "output with spaces/fak")
	cmd.Dir = t.TempDir()
	cmd.Env = append(os.Environ(), "GO="+self, "FAK_CUDA_SHELL_DELEGATION_HELPER=1")
	raw, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	var args []string
	if err := json.Unmarshal(raw, &args); err != nil {
		t.Fatal(err)
	}
	want := []string{"run", "./cmd/fak-dev", "build-cuda", "legacy", "binary", "./cmd/fak", "output with spaces/fak"}
	if !slices.Equal(args, want) {
		t.Fatalf("shell native delegation=%q; want %q", args, want)
	}
}

// Two nvcc runs over identical sources differ only in the PID-derived host-stub name.
func fixtureCUDAArchive(code, pid string) []byte {
	return []byte("!<arch>\n\x00\x01" + code + "\x00tmpxft_" + pid + "_00000000-6_cuda_kernels.cudafe1.cpp\x00" +
		"_GLOBAL__sub_I_tmpxft_" + pid + "_00000000_6_cuda_kernels\x00no trailing newline")
}

func TestCUDAArchiveStampKeysOnKernelCodeNotNVCCTempNames(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "libfakcuda.a")
	stampPath := filepath.Join(dir, CUDAArchiveStampFile)
	stamp := func(code, pid string) (string, bool, []byte) {
		t.Helper()
		if err := os.WriteFile(archive, fixtureCUDAArchive(code, pid), 0o644); err != nil {
			t.Fatal(err)
		}
		digest, changed, err := WriteCUDAArchiveStamp(dir, archive)
		if err != nil {
			t.Fatal(err)
		}
		src, err := os.ReadFile(stampPath)
		if err != nil {
			t.Fatal(err)
		}
		return digest, changed, src
	}

	base, changed, baseSrc := stamp("kernel-v1", "000001cb")
	if !changed || !bytes.Contains(baseSrc, []byte(base)) {
		t.Fatal("first archive did not write a stamp carrying its digest")
	}
	if fi, err := os.Stat(stampPath); err != nil || (runtime.GOOS != "windows" && fi.Mode().Perm() != 0o644) {
		t.Fatalf("stamp mode is not 0644 like build_cuda.sh writes: %v", err)
	}
	file, err := parser.ParseFile(token.NewFileSet(), CUDAArchiveStampFile, baseSrc, parser.ParseComments)
	if err != nil || file.Name.Name != "compute" || !ast.IsGenerated(file) {
		t.Fatalf("stamp is not generated compute-package Go source: %v", err)
	}
	for _, tags := range [][]string{nil, {"cuda"}} {
		ctx := build.Default
		ctx.BuildTags = tags
		if match, err := ctx.MatchFile(dir, CUDAArchiveStampFile); err != nil || match != (len(tags) == 1) {
			t.Fatalf("stamp build match under tags %v = %t (%v); want only -tags cuda builds", tags, match, err)
		}
	}

	if again, changed, src := stamp("kernel-v1", "000001e3"); changed || again != base || !bytes.Equal(src, baseSrc) {
		t.Fatal("an nvcc temp-name difference moved the stamp; unchanged rebuilds would miss the cache")
	}
	if next, changed, src := stamp("kernel-v2", "000001e3"); !changed || next == base || bytes.Equal(src, baseSrc) {
		t.Fatal("a kernel code change left the stamp unchanged; the link would stay stale")
	}
	if leftovers, _ := filepath.Glob(filepath.Join(dir, ".cuda-archive-stamp-*")); len(leftovers) != 0 {
		t.Fatalf("stamp write left temp files: %v", leftovers)
	}
}

func TestCUDAArchiveStampRefusesWhatCannotNameAnArchive(t *testing.T) {
	for _, digest := range []string{"", "abc", strings.Repeat("A", 64), strings.Repeat("g", 64), strings.Repeat("a", 66)} {
		if _, err := CUDAArchiveStampSource(digest); !errors.Is(err, ErrCUDAArchiveStamp) {
			t.Fatalf("digest %q accepted: %v", digest, err)
		}
	}
	dir := t.TempDir()
	if _, _, err := WriteCUDAArchiveStamp(dir, filepath.Join(dir, "missing.a")); !errors.Is(err, ErrCUDAArchiveStamp) {
		t.Fatalf("missing archive accepted: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, CUDAArchiveStampFile)); !os.IsNotExist(err) {
		t.Fatal("missing archive still wrote a stamp")
	}
}
