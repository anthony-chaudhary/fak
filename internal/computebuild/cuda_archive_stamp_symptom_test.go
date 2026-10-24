package computebuild

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"go/build"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Symptom: Go never hashed libfakcuda.a, so after a kernel-only rebuild `go test -c` reused
// the old test binary and a kernel mutant ran the previous kernel and passed. A kernel build
// must leave the archive's digest in a -tags cuda compute source so the next link moves.
// fak-test:justify why=regression when=changed:internal/computebuild/**,internal/compute/build_cuda.sh
// fak-test:runtime fast est=2s
func TestBuildCUDALibStampsComputePackageWithArchiveDigest(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	tool := filepath.Join(root, "toolkit", "cuobjdump")
	if runtime.GOOS == "windows" {
		tool += ".exe"
	}
	source := filepath.Join(root, "toolkit.go")
	const toolkit = `package main
import ("fmt"; "os")
func main() {
	a := os.Args[1:]
	switch {
	case len(a) == 2 && a[0] == "--list-elf":
		fmt.Print("ELF file 1: cuda_kernels.sm_120.cubin\n")
	case len(a) == 2 && a[0] == "--list-ptx":
	case len(a) > 2 && a[0] == "rcs":
		if os.WriteFile(a[1], []byte("kernel archive v1"), 0o644) != nil { os.Exit(3) }
	default:
		for i := 0; i+1 < len(a); i++ {
			if a[i] == "-o" {
				if os.WriteFile(a[i+1], []byte("kernel object"), 0o644) != nil { os.Exit(4) }
				return
			}
		}
		os.Exit(2)
	}
}
`
	for path, body := range map[string]string{
		source: toolkit,
		filepath.Join(root, "internal", "compute", "cuda_arch.txt"):   "sm_120\n",
		filepath.Join(root, "internal", "compute", "cuda_kernels.cu"): "kernel\n",
	} {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if out, err := exec.Command("go", "build", "-o", tool, source).CombinedOutput(); err != nil {
		t.Fatalf("build toolkit fixture: %v\n%s", err, out)
	}

	artifact, err := BuildCUDALib(context.Background(), &Toolchain{NVCC: tool, AR: tool}, root, "sm_120", false, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	archive, err := os.ReadFile(artifact.Path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(archive)
	stamp, err := os.ReadFile(filepath.Join(root, "internal", "compute", "cuda_archive_stamp_gen.go"))
	if err != nil {
		t.Fatalf("kernel build left no archive stamp in the compute package: %v", err)
	}
	if want := "\nvar cudaArchiveDigest = \"" + hex.EncodeToString(sum[:]) + "\"\n"; !strings.Contains(string(stamp), want) {
		t.Fatalf("archive stamp does not declare the archive digest %q:\n%s", want, stamp)
	}
	for _, tags := range [][]string{nil, {"cuda"}} {
		ctx := build.Default
		ctx.BuildTags = tags
		if match, err := ctx.MatchFile(filepath.Join(root, "internal", "compute"), "cuda_archive_stamp_gen.go"); err != nil || match != (len(tags) == 1) {
			t.Fatalf("archive stamp build match under tags %v = %t (%v); want only -tags cuda builds", tags, match, err)
		}
	}
}
