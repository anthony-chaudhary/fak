package computebuild

import (
	"bytes"
	"errors"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

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

// build_cuda.sh and fak-dev build-cuda must write identical stamps, or alternating between
// them recompiles the compute package on every build.
// fak-test:justify why=contract when=changed:internal/compute/build_cuda.sh,internal/computebuild/**
// fak-test:runtime fast est=1s
func TestBuildCUDAScriptWritesTheGoArchiveStamp(t *testing.T) {
	t.Parallel()
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skip("bash is required to exercise build_cuda.sh")
	}
	script, err := os.ReadFile(filepath.Join("..", "compute", "build_cuda.sh"))
	if err != nil {
		t.Fatal(err)
	}
	text := strings.ReplaceAll(string(script), "\r\n", "\n")
	start := strings.Index(text, "write_archive_stamp() {\n")
	end := strings.Index(text[max(start, 0):], "\n}\n")
	if start < 0 || end < 0 {
		t.Fatal("build_cuda.sh has no write_archive_stamp function")
	}
	fn := text[start : start+end+3]
	if !strings.Contains(text[start+end:], "\n  write_archive_stamp\n") {
		t.Fatal("build_cuda.sh never stamps the archive it builds")
	}
	if out, err := exec.Command(bash, "-c", "command -v sed && command -v sha256sum").CombinedOutput(); err != nil {
		t.Skipf("bash lacks sed/sha256sum: %v (%s)", err, strings.TrimSpace(string(out)))
	}
	for _, pid := range []string{"000001cb", "000001e3"} {
		dir := t.TempDir()
		archive := filepath.Join(dir, "libfakcuda.a")
		if err := os.WriteFile(archive, fixtureCUDAArchive("kernel-v1", pid), 0o644); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command(bash, "-c", "set -euo pipefail\n"+fn+"write_archive_stamp\n")
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("write_archive_stamp: %v\n%s", err, out)
		}
		got, err := os.ReadFile(filepath.Join(dir, CUDAArchiveStampFile))
		if err != nil {
			t.Fatal(err)
		}
		digest, err := CUDAArchiveDigest(archive)
		if err != nil {
			t.Fatal(err)
		}
		want, err := CUDAArchiveStampSource(digest)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("build_cuda.sh stamp differs from computebuild:\n got %q\nwant %q", got, want)
		}
	}
}
