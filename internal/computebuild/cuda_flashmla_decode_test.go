package computebuild

import (
	"context"
	"strings"
	"testing"
)

func TestFlashMLADecodeMatchedSelection(t *testing.T) {
	cfg := &CUDAConfig{EnableArchSpecific: true, EnableFlashMLADecode: true}
	tags := CUDAGoTags(cfg)
	if !strings.Contains(tags, "cudaarchspecific") || !strings.Contains(tags, "fak_cuda_flashmla_decode") {
		t.Fatal(tags)
	}
	if strings.Contains(CUDAGoTags(&CUDAConfig{}), "flashmla_decode") {
		t.Fatal("default activation")
	}
	ctx := context.WithValue(context.Background(), cudaFlashMLADecodeKey{}, &cudaFlashMLAOptions{})
	flags, err := cudaFlashMLADecodeGenericFlags(ctx, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(flags, " "); got != "-DFAK_CUDA_FLASHMLA_DECODE" {
		t.Fatal(got)
	}
	flags, err = cudaFlashMLADecodeGenericFlags(context.Background(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(flags) != 0 {
		t.Fatal("default native activation")
	}
	args := strings.Join(cudaFlashMLADecodeCompileArgs(&Toolchain{}, "/repo", "decode.o", false), " ")
	for _, want := range []string{"-std=c++20", "sm_100a", "sm_103a", "cuda_flashmla_sparse_decode.cu", "--warn-on-spills"} {
		if !strings.Contains(args, want) {
			t.Fatal(want, args)
		}
	}
	for _, bad := range []string{"sm_120", "cuda_flashmla_sparse_prefill.cu", "compute_120", "-DNDEBUG"} {
		if strings.Contains(args, bad) {
			t.Fatal(args)
		}
	}
}
func TestFlashMLADecodeRejectsOtherArchitectures(t *testing.T) {
	for _, arch := range []string{"", "sm_120", "sm_100a", "sm_103a", "100a", "sm_90"} {
		if _, err := prepareCUDAFlashMLADecode(context.Background(), &CUDAConfig{Arch: arch}); err == nil {
			t.Fatalf("admitted %q", arch)
		}
	}
}

func TestFlashMLADecodeAmbientFlagsRefuse(t *testing.T) {
	ctx := context.WithValue(context.Background(), cudaFlashMLADecodeKey{}, &cudaFlashMLAOptions{})
	for _, v := range []string{"NVCC_PREPEND_FLAGS=-DNDEBUG", "nvcc_append_flags=-arch=sm_120"} {
		if _, err := cudaFlashMLADecodeGenericFlags(ctx, nil, []string{v}); err == nil {
			t.Fatal("ambient override admitted")
		}
	}
	if _, err := cudaFlashMLADecodeGenericFlags(ctx, nil, []string{"NVCC_APPEND_FLAGS=  "}); err != nil {
		t.Fatal(err)
	}
}

func TestFlashMLADecodeGenericMatrixComposition(t *testing.T) {
	matrix := "sm_80\nsm_89\nsm_90\nsm_100\nsm_103\nsm_120\n"
	for _, arch := range []string{"sm_100", "sm_103"} {
		if err := validateFlashMLADecodeGenericArch(arch); err != nil {
			t.Fatal(err)
		}
		flags, _, err := ParseCUDAArchMatrix(matrix, arch)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(strings.Join(flags, " "), "code="+arch) {
			t.Fatal(flags)
		}
	}
	for _, arch := range []string{"sm_120", "sm_100a", "sm_103a"} {
		if validateFlashMLADecodeGenericArch(arch) == nil {
			t.Fatal("invalid generic selection", arch)
		}
	}
	// Ordinary matrix support does not qualify an unrelated FlashMLA architecture.
	if _, _, err := ParseCUDAArchMatrix(matrix, "sm_120"); err != nil {
		t.Fatal(err)
	}
}
