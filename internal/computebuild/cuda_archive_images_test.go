package computebuild

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime fast est=10ms
func TestCUDAArchiveImageArchOrder(t *testing.T) {
	t.Parallel()
	// Synthetic declarations and listings, not captured CUDA toolkit output.
	declared := []string{
		"sm_120", "sm_100f", "sm_100000000000000000000", "sm_103a", "sm_100",
		"sm_80", "sm_100a", "sm_090a", "sm_99999999999999999999", "sm_103", "sm_90a",
	}
	wantSASS := []string{
		"sm_80", "sm_090a", "sm_90a", "sm_100", "sm_100a", "sm_100f", "sm_103",
		"sm_103a", "sm_120", "sm_99999999999999999999", "sm_100000000000000000000",
	}
	wantPTX := []string{
		"compute_80", "compute_090a", "compute_90a", "compute_100", "compute_100a", "compute_100f", "compute_103",
		"compute_103a", "compute_120", "compute_99999999999999999999", "compute_100000000000000000000",
	}
	root := t.TempDir()
	matrix := filepath.Join(root, "internal", "compute", "cuda_arch.txt")
	writeFixtureFile(t, matrix, strings.Join(declared, "\n")+"\n")
	t.Run("requested", func(t *testing.T) {
		sass, ptx, err := cudaArchiveImageRequirements(root, "")
		if err != nil || !reflect.DeepEqual(sass, wantSASS) || !reflect.DeepEqual(ptx, []string{"compute_90a"}) {
			t.Fatalf("requirements = %v, %v, %v; want %v and last-declared compute_90a", sass, ptx, err, wantSASS)
		}
	})
	for _, spec := range []struct {
		kind, extension string
		want            []string
	}{{"ELF", "cubin", wantSASS}, {"PTX", "ptx", wantPTX}} {
		t.Run(spec.kind, func(t *testing.T) {
			var listing strings.Builder
			for i, arch := range append(append([]string(nil), declared...), "sm_100") {
				fmt.Fprintf(&listing, "%s file %d: synthetic.%s.%s\n", spec.kind, i+1, arch, spec.extension)
			}
			arches, count, err := parseCUDAImageList([]byte(listing.String()), spec.kind)
			if err != nil || !reflect.DeepEqual(arches, spec.want) || count != len(declared)+1 {
				t.Fatalf("listing = %v, count %d, %v; want %v, count %d", arches, count, err, spec.want, len(declared)+1)
			}
			malformed := fmt.Sprintf("%s file 1: synthetic.sm_90af.%s\n", spec.kind, spec.extension)
			if _, _, err := parseCUDAImageList([]byte(malformed), spec.kind); !errors.Is(err, ErrCUDAImageAuditInvalid) {
				t.Fatalf("malformed architecture error = %v; want ErrCUDAImageAuditInvalid", err)
			}
		})
	}
	t.Run("requested-validation-unchanged", func(t *testing.T) {
		writeFixtureFile(t, matrix, "sm_100\nsm_future\nsm_80\nsm_80\n")
		sass, ptx, err := cudaArchiveImageRequirements(root, "")
		counts := make(map[string]int)
		for _, arch := range sass {
			counts[arch]++
		}
		if err != nil || !reflect.DeepEqual(counts, map[string]int{"sm_100": 1, "sm_future": 1, "sm_80": 2}) || !reflect.DeepEqual(ptx, []string{"compute_80"}) {
			t.Fatalf("requested token preservation changed: %v, %v, %v", sass, ptx, err)
		}
	})
}

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
 case "--version":
  fmt.Print("Cuda compilation tools, release 12.6, V12.6.85\n")
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
	include := filepath.Join(root, "include")
	writeFixtureFile(t, filepath.Join(include, "cublas_api.h"), "#define CUBLASAPI_VERSION 120600\n")
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("FAK_CUDA_IMAGE_TEST_MODE", "")
	t.Setenv("FAK_CUDA_IMAGE_TEST_ELF", "")
	t.Setenv("FAK_CUDA_IMAGE_TEST_PTX", "")
	return root, &Toolchain{NVCC: tool, AR: tool, CUDAInc: []string{include}}
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
		// ArchiveCUDALib records the exact archived bytes: ar writes the
		// rcs fixture payload, the nvcc --lib fallback writes the -o payload.
		wantPayload := "archive-fixture"
		if windows {
			wantPayload = "compiled-fixture"
		}
		wantDigest := sha256.Sum256([]byte(wantPayload))
		if artifact.SHA256 != hex.EncodeToString(wantDigest[:]) {
			t.Fatalf("windows=%v archive digest=%s; want %s", windows, artifact.SHA256, hex.EncodeToString(wantDigest[:]))
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
	// A successful binary build observes its own provenance: the receipt
	// switches to the CUDA successor schema and carries the block on the wire.
	if receipt.Schema != CUDABuildReceiptSchema {
		t.Fatalf("binary receipt kept base schema %s", receipt.Schema)
	}
	strict, err := DecodeStrictCUDABuildReceipt(receiptBytes)
	if err != nil {
		t.Fatal(err)
	}
	if strict.CUDA == nil || strict.CUDA.ArtifactKind != CUDAProvenanceArtifactBinary || !reflect.DeepEqual(strict.CUDA.SASSArchs, []string{"sm_120"}) || strict.CUDA.NVCC.Version != "12.6" || strict.CUDA.CUBLASVersion != "12.6.0" { //boundarylint:ignore CHANGE_DETECTOR_TEST fixture echo: the receipt must carry the nvcc/cublas versions the fixture toolchain declares verbatim
		t.Fatalf("binary receipt missing provenance: %+v", strict.CUDA)
	}
	if strict.CUDA.ArtifactSHA256 != receipt.Artifact.SHA256 {
		t.Fatalf("provenance digest %s does not match artifact identity %s", strict.CUDA.ArtifactSHA256, receipt.Artifact.SHA256)
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
	// A toolkit without cuobjdump (the micromamba or apt WSL env) refuses before any kernel
	// compiles, so the archive on disk and its stamp are left as they were.
	tool, err := os.ReadFile(tc.NVCC)
	if err != nil {
		t.Fatal(err)
	}
	bare := filepath.Join(root, "no-cuobjdump", strings.Replace(filepath.Base(tc.NVCC), "cuobjdump", "nvcc", 1))
	writeFixtureFile(t, bare, string(tool))
	if err := os.Chmod(bare, 0o755); err != nil {
		t.Fatal(err)
	}
	archive := filepath.Join(root, "internal", "compute", "libfakcuda.a")
	before, err := os.ReadFile(archive)
	if err != nil {
		t.Fatal(err)
	}
	_, err = BuildCUDALib(context.Background(), &Toolchain{NVCC: bare, AR: bare}, root, "sm_120", false, io.Discard)
	var refusal *CUDAPreflightRefusal
	if !errors.As(err, &refusal) || refusal.Token != CUDAPreflightCuobjdumpMissing || !errors.Is(err, ErrCUDAImageAuditToolMissing) {
		t.Fatalf("toolkit without cuobjdump did not refuse at preflight: %v", err)
	}
	if after, err := os.ReadFile(archive); err != nil || string(after) != string(before) {
		t.Fatalf("refused preflight replaced the archive: %v", err)
	}
	requireCUDAArchiveStamp(t, root)
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime fast est=10ms
func TestRequireCUDAImageAuditToolRefusesMissingCuobjdump(t *testing.T) {
	toolkit := t.TempDir()
	exe := ""
	if runtime.GOOS == "windows" {
		exe = ".exe"
	}
	// The fixture nvcc is not runnable: a refusal proves no compile was attempted.
	nvcc := filepath.Join(toolkit, "nvcc"+exe)
	writeFixtureFile(t, nvcc, "not a compiler")
	if err := os.Chmod(nvcc, 0o755); err != nil {
		t.Fatal(err)
	}
	tc := &Toolchain{NVCC: nvcc}
	for name, run := range map[string]func() error{
		"preflight": func() error { return RequireCUDAImageAuditTool(tc) },
		"compile": func() error {
			_, err := CompileCUDAKernels(context.Background(), tc, t.TempDir(), "sm_120", false, io.Discard)
			return err
		},
	} {
		err := run()
		var refusal *CUDAPreflightRefusal
		if !errors.As(err, &refusal) || refusal.Token != CUDAPreflightCuobjdumpMissing || !errors.Is(err, ErrCUDAImageAuditToolMissing) {
			t.Fatalf("%s without cuobjdump = %v, want %s", name, err, CUDAPreflightCuobjdumpMissing)
		}
	}
	for _, missing := range []*Toolchain{nil, {}} {
		if err := RequireCUDAImageAuditTool(missing); !errors.Is(err, ErrCUDAImageAuditToolMissing) {
			t.Fatalf("no selected toolkit = %v", err)
		}
	}
	writeFixtureFile(t, filepath.Join(toolkit, "cuobjdump"+exe), "fixture")
	if err := RequireCUDAImageAuditTool(tc); err != nil {
		t.Fatalf("cuobjdump beside nvcc refused: %v", err)
	}
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
