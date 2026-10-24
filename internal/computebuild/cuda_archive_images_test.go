package computebuild

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func cudaImageAuditFixture(t *testing.T) (string, *Toolchain) {
	t.Helper()
	root := t.TempDir()
	tool := filepath.Join(root, "cuobjdump")
	if runtime.GOOS == "windows" {
		tool += ".exe"
	}
	source := filepath.Join(root, "tool.go")
	writeFixtureFile(t, source, `package main
import("fmt"; "os"; "strings")
func main() {
 a := os.Args[1:]
 if len(a) == 0 { os.Exit(2) }
 switch a[0] {
 case "--list-elf", "--list-ptx":
  if len(a) != 2 { os.Exit(3) }
  if os.Getenv("FAK_CUDA_IMAGE_TEST_MODE") == "crash" { fmt.Fprint(os.Stderr, "tool failed"); os.Exit(7) }
  if os.Getenv("FAK_CUDA_IMAGE_TEST_MODE") == "oversized" { fmt.Print(strings.Repeat("x", 2<<20)); return }
  if a[0] == "--list-elf" { fmt.Print(os.Getenv("FAK_CUDA_IMAGE_TEST_ELF")) } else { fmt.Print(os.Getenv("FAK_CUDA_IMAGE_TEST_PTX")) }
  return
 case "rcs":
  if err := os.WriteFile(a[1], []byte("archive-fixture"), 0644); err != nil { os.Exit(4) }; return
 }
 for i:=0; i+1<len(a); i++ {
  if a[i] == "-o" || a[i] == "--output-file" {
   if err := os.WriteFile(a[i+1], []byte("compiled-fixture"), 0644); err != nil { os.Exit(5) }; return
  }
 }
 if a[0] != "build" { os.Exit(6) }
}
`)
	cmd := exec.Command("go", "build", "-o", tool, source)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build native fixture: %v\n%s", err, out)
	}
	goTool := filepath.Join(root, "go")
	if runtime.GOOS == "windows" {
		goTool += ".exe"
	}
	data, err := os.ReadFile(tool)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(goTool, data, 0755); err != nil {
		t.Fatal(err)
	}
	writeFixtureFile(t, filepath.Join(root, "go.mod"), "module example.test/cuda\n\ngo 1.26\n")
	writeFixtureFile(t, filepath.Join(root, "internal", "compute", "cuda_arch.txt"), "sm_100\nsm_103\nsm_120\n")
	writeFixtureFile(t, filepath.Join(root, "internal", "compute", "cuda_kernels.cu"), "fixture\n")
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("FAK_CUDA_IMAGE_TEST_MODE", "")
	t.Setenv("FAK_CUDA_IMAGE_TEST_ELF", "")
	t.Setenv("FAK_CUDA_IMAGE_TEST_PTX", "")
	return root, &Toolchain{NVCC: tool, AR: tool}
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime medium est=3s
func TestCUDAArchiveImageAudit(t *testing.T) {
	root, tc := cudaImageAuditFixture(t)
	archive := filepath.Join(root, "library.a")
	writeFixtureFile(t, archive, "archive-fixture")
	wantDigest := sha256.Sum256([]byte("archive-fixture"))
	wantSHA256 := hex.EncodeToString(wantDigest[:])
	for _, arch := range []string{"sm_100", "sm_103", "sm_120"} {
		t.Run(arch, func(t *testing.T) {
			t.Setenv("FAK_CUDA_IMAGE_TEST_ELF", "member C:\\Toolkit\\library.lib:cuda_kernels.obj:\nELF file    1: cuda_kernels.1."+arch+".cubin\n")
			audit, err := AuditCUDAArchiveImages(context.Background(), tc, root, arch, archive)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(audit.SASS, []string{arch}) || len(audit.PTX) != 0 || audit.ELFImageCount != 1 || audit.ArchiveSHA256 != wantSHA256 {
				t.Fatal("selected image or archive identity was not recorded")
			}
		})
	}
	allELF := "member library.a:cuda_kernels.o:\nELF file    1: cuda_kernels.1.sm_100.cubin\nELF file    2: cuda_kernels.2.sm_103.cubin\nELF file    3: cuda_kernels.3.sm_120.cubin\n"
	t.Setenv("FAK_CUDA_IMAGE_TEST_ELF", allELF)
	t.Setenv("FAK_CUDA_IMAGE_TEST_PTX", "PTX file    1: cuda_kernels.1.sm_120.ptx\n")
	audit, err := AuditCUDAArchiveImages(context.Background(), tc, root, "", archive)
	if err != nil || audit == nil || !reflect.DeepEqual(audit.PTX, []string{"compute_120"}) || audit.ELFImageCount != 3 {
		t.Fatalf("full matrix with PTX was not accepted: %v", err)
	}
	for _, spec := range []struct {
		name, elf, ptx, mode string
	}{
		{"missing-sass", "ELF file 1: x.sm_100.cubin\n", "PTX file 1: x.sm_120.ptx\n", ""},
		{"missing-ptx", allELF, "", ""},
		{"wrong-ptx", allELF, "PTX file 1: x.sm_100.ptx\n", ""},
		{"malformed", "ELF file 1: x.sm_120.cubin.trailing\n", "", ""},
		{"diagnostic-is-not-image", "cuobjdump info    : file x.sm_100.cubin x.sm_103.cubin x.sm_120.cubin\n", "", ""},
		{"tool-failure", allELF, "", "crash"},
		{"bounded-output", allELF, "", "oversized"},
	} {
		t.Run(spec.name, func(t *testing.T) {
			t.Setenv("FAK_CUDA_IMAGE_TEST_ELF", spec.elf)
			t.Setenv("FAK_CUDA_IMAGE_TEST_PTX", spec.ptx)
			t.Setenv("FAK_CUDA_IMAGE_TEST_MODE", spec.mode)
			got, err := AuditCUDAArchiveImages(context.Background(), tc, root, "", archive)
			if got != nil || !errors.Is(err, ErrCUDAImageAuditInvalid) {
				t.Fatalf("invalid image inventory accepted: %v", err)
			}
		})
	}
	missingToolchain := &Toolchain{NVCC: filepath.Join(root, "missing-toolkit", "nvcc"), CUDAHome: root}
	if _, err := AuditCUDAArchiveImages(context.Background(), missingToolchain, root, "sm_120", archive); !errors.Is(err, ErrCUDAImageAuditToolMissing) {
		t.Fatalf("missing selected toolkit did not refuse: %v", err)
	}
}

// fak-test:justify why=integration when=changed:internal/computebuild/**
// fak-test:runtime medium est=3s
func TestCUDABuildRequiresArchiveImages(t *testing.T) {
	root, tc := cudaImageAuditFixture(t)
	t.Setenv("FAK_CUDA_IMAGE_TEST_ELF", "ELF file 1: cuda_kernels.sm_120.cubin\n")
	for _, windows := range []bool{false, true} {
		tc.IsWindows = windows
		if windows {
			tc.AR = ""
		}
		artifact, err := BuildCUDALib(context.Background(), tc, root, "sm_120", false, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		if artifact.CUDAImages == nil || artifact.SHA256 != artifact.CUDAImages.ArchiveSHA256 {
			t.Fatal("archive was returned without observed images")
		}
		requireCUDAArchiveStamp(t, root)
	}
	tc.IsWindows, tc.AR = false, tc.NVCC
	cfg := &CUDAConfig{Command: "binary", RepoRoot: root, Arch: "sm_120", OutPkg: "./cmd/fak", OutBin: filepath.Join(root, "built-fak"), SkipSmoke: true, SkipSign: true, Toolchain: tc, Stdout: io.Discard, Stderr: io.Discard}
	if err := RunCUDA(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	receiptBytes, err := os.ReadFile(cfg.ReceiptPath)
	if err != nil {
		t.Fatal(err)
	}
	var receipt ComputeBuildReceipt
	if err := json.Unmarshal(receiptBytes, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Outcome != "success" || receipt.Artifact == nil || receipt.Artifact.CUDAImages == nil || !reflect.DeepEqual(receipt.Artifact.CUDAImages.SASS, []string{"sm_120"}) {
		t.Fatal("native binary receipt lost the observed archive images")
	}
	t.Setenv("FAK_CUDA_IMAGE_TEST_ELF", "ELF file 1: stale.sm_100.cubin\n")
	staleCUDAArchiveStamp(t, root)
	if err := RunCUDA(context.Background(), cfg); !errors.Is(err, ErrCUDAImageAuditInvalid) {
		t.Fatalf("native build accepted wrong archive images: %v", err)
	}
	requireCUDAArchiveStamp(t, root)
	if cfg.Receipt.Outcome != "failed" || cfg.Receipt.Artifact != nil || len(cfg.Receipt.Phases) != 2 || cfg.Receipt.Phases[1].Outcome != "failed" {
		t.Fatal("wrong archive did not stop the native build before linking")
	}
	if strings.Contains(cfg.Receipt.Error, "tool.go") {
		t.Fatal("fixture source leaked into the bounded audit diagnostic")
	}
	// A toolkit without cuobjdump (the micromamba WSL env) refuses the audit after the archive
	// was replaced; the stamp must still name the archive on disk.
	tool, err := os.ReadFile(tc.NVCC)
	if err != nil {
		t.Fatal(err)
	}
	bare := filepath.Join(root, "no-cuobjdump", strings.Replace(filepath.Base(tc.NVCC), "cuobjdump", "nvcc", 1))
	writeFixtureFile(t, bare, string(tool))
	if err := os.Chmod(bare, 0o755); err != nil {
		t.Fatal(err)
	}
	staleCUDAArchiveStamp(t, root)
	if _, err := BuildCUDALib(context.Background(), &Toolchain{NVCC: bare, AR: bare}, root, "sm_120", false, io.Discard); !errors.Is(err, ErrCUDAImageAuditToolMissing) {
		t.Fatalf("toolkit without cuobjdump did not refuse the audit: %v", err)
	}
	requireCUDAArchiveStamp(t, root)
}

func staleCUDAArchiveStamp(t *testing.T, root string) {
	t.Helper()
	stale, err := CUDAArchiveStampSource(strings.Repeat("0", 64))
	if err != nil {
		t.Fatal(err)
	}
	writeFixtureFile(t, filepath.Join(root, "internal", "compute", CUDAArchiveStampFile), string(stale))
}

func requireCUDAArchiveStamp(t *testing.T, root string) {
	t.Helper()
	digest, err := CUDAArchiveDigest(filepath.Join(root, "internal", "compute", "libfakcuda.a"))
	if err != nil {
		t.Fatal(err)
	}
	want, err := CUDAArchiveStampSource(digest)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(filepath.Join(root, "internal", "compute", CUDAArchiveStampFile)); err != nil || string(got) != string(want) {
		t.Fatalf("compute stamp does not name the archive on disk: %v", err)
	}
}
