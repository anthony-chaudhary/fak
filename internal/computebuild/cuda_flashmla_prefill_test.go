package computebuild

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=30ms lane=default
func TestCUDAFlashMLAPrefillBuildContract(t *testing.T) {
	if (CUDAConfig{}).EnableFlashMLAPrefill {
		t.Fatal("candidate must be opt-in")
	}
	for _, version := range []string{"release 12.9", "release 13.0", "unknown"} {
		if err := requireCUDAFlashMLAVersion(version); !errors.Is(err, ErrCUDAFlashMLAPrefill) {
			t.Fatalf("version %q: %v", version, err)
		}
	}
	if err := requireCUDAFlashMLAVersion("Cuda compilation tools, release 13.1, V13.1.80"); err != nil {
		t.Fatal(err)
	}
	args := cudaFlashMLACompileArgs(&Toolchain{CXX: "/cxx", CUDAInc: []string{"/cuda/include"}}, "/repo", "/out/flash.o", true)
	var images []string
	for i, arg := range args {
		if arg == "-gencode" {
			images = append(images, args[i+1])
		}
	}
	if !reflect.DeepEqual(images, []string{"arch=compute_100a,code=sm_100a", "arch=compute_103a,code=sm_103a"}) {
		t.Fatal(images)
	}
	joined := " " + strings.Join(args, " ") + " "
	for _, flag := range []string{"-std=c++20", "--use_fast_math", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DFLASH_MLA_IS_BUILD_ON_CUDA", "-DFAK_FLASHMLA_DISABLE_LD_RED", "--warn-on-spills", "--warning-as-error"} {
		if !strings.Contains(joined, flag) {
			t.Errorf("missing effective policy %s", flag)
		}
	}
	if strings.Contains(joined, "code=compute_") || strings.Contains(joined, "sm_100f") {
		t.Fatal("FlashMLA must have no PTX or family-specific fallback")
	}
	if !strings.Contains(joined, "-I/repo/third_party/flashmla/csrc ") || !strings.Contains(joined, "-I/repo/third_party/cutlass/include ") {
		t.Fatal("vendor include roots missing")
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestCUDAVendorPinnedSourceRejectsMutation(t *testing.T) {
	root := t.TempDir()
	file := filepath.Join(root, "third_party/flashmla/csrc/kernel.cuh")
	if err := os.MkdirAll(filepath.Dir(file), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("original"), 0644); err != nil {
		t.Fatal(err)
	}
	digest, err := cudaFileDigest(file)
	if err != nil {
		t.Fatal(err)
	}
	manifest := cudaVendorSourceManifest{Schema: "fak.flashmla.vendor.reconstruction.v1", Files: []cudaVendorSourceFile{{Path: "third_party/flashmla/csrc/kernel.cuh", Bytes: 8, SHA256: digest}}}
	path := filepath.Join(root, "third_party/flashmla/PINNED-SOURCE.json")
	if err := cudaWriteJSON(path, manifest); err != nil {
		t.Fatal(err)
	}
	pin, err := cudaFileDigest(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := verifyCUDAVendorSources(root, "third_party/flashmla/PINNED-SOURCE.json", pin); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("mutation"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyCUDAVendorSources(root, "third_party/flashmla/PINNED-SOURCE.json", pin); !errors.Is(err, ErrCUDAVendorSource) {
		t.Fatalf("changed vendor source: %v", err)
	}
}
