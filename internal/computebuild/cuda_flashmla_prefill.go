package computebuild

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

var ErrCUDAFlashMLAPrefill = errors.New("computebuild: FlashMLA sparse prefill unavailable")

const cudaFlashMLAManifestSHA256 = "23b68f0257c09dc8b48c5ae7bbffda31538f08523034864854220f170ceaeb22"

type cudaFlashMLAKey struct{}
type cudaFlashMLAOptions struct {
	disableLDRed bool
	receipt      CUDAFlashMLAPrefillReceipt
}

// CUDAFlashMLAPrefillReceipt describes the explicit build candidate, not GPU qualification.
type CUDAFlashMLAPrefillReceipt struct {
	ManifestSHA256    string                 `json:"manifest_sha256"`
	FlashMLACommit    string                 `json:"flashmla_commit"`
	CUTLASSCommit     string                 `json:"cutlass_commit"`
	MinimumNVCC       string                 `json:"minimum_nvcc"`
	DisableLDRed      bool                   `json:"disable_ld_red"`
	HardwareQualified bool                   `json:"hardware_qualified"`
	ObjectImages      *CUDAArchiveImageAudit `json:"object_images,omitempty"`
}

func requireCUDAFlashMLAVersion(text string) error {
	match := cudaNVCCRelease.FindStringSubmatch(text)
	if len(match) != 2 || compareVersionStrings(match[1], "13.1") < 0 {
		return fmt.Errorf("%w: nvcc release >= 13.1 required", ErrCUDAFlashMLAPrefill)
	}
	return nil
}

func prepareCUDAFlashMLA(ctx context.Context, cfg *CUDAConfig) (*cudaFlashMLAOptions, error) {
	if !cfg.EnableArchSpecific || cfg.Toolchain == nil || cfg.Toolchain.NVCC == "" || cfg.Toolchain.IsWindows || runtime.GOOS != "linux" {
		return nil, fmt.Errorf("%w: explicit arch-specific Linux nvcc build required", ErrCUDAFlashMLAPrefill)
	}
	out, err := exec.CommandContext(ctx, cfg.Toolchain.NVCC, "--version").Output()
	if err != nil {
		return nil, fmt.Errorf("%w: nvcc version: %v", ErrCUDAFlashMLAPrefill, err)
	}
	if err := requireCUDAFlashMLAVersion(string(out)); err != nil {
		return nil, err
	}
	manifest, err := verifyCUDAVendorSources(cfg.RepoRoot, "third_party/flashmla/PINNED-SOURCE.json", cudaFlashMLAManifestSHA256)
	if err != nil {
		return nil, err
	}
	return &cudaFlashMLAOptions{disableLDRed: cfg.DisableFlashMLALDRed, receipt: CUDAFlashMLAPrefillReceipt{
		ManifestSHA256: cudaFlashMLAManifestSHA256, FlashMLACommit: manifest.FlashMLACommit, CUTLASSCommit: manifest.CUTLASSCommit,
		MinimumNVCC: "13.1", DisableLDRed: cfg.DisableFlashMLALDRed,
	}}, nil
}

func cudaFlashMLACompileArgs(tc *Toolchain, root, object string, disableLDRed bool) []string {
	args := []string{"-O3", "-std=c++20", "-gencode", "arch=compute_100a,code=sm_100a", "-gencode", "arch=compute_103a,code=sm_103a",
		"-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
		"--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
		"--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use,--warning-as-error",
		"-lineinfo", "--source-in-ptx", "--diag-suppress=445", "-DNDEBUG", "-DFLASH_MLA_IS_BUILD_ON_CUDA"}
	if disableLDRed {
		args = append(args, "-DFAK_FLASHMLA_DISABLE_LD_RED")
	}
	if tc.CXX != "" {
		args = append(args, "-ccbin", tc.CXX)
	}
	for _, include := range tc.CUDAInc {
		args = append(args, "-I"+include, "-I"+filepath.Join(include, "cccl"))
	}
	if tc.CUDAHome != "" {
		for _, target := range []string{"x86_64-linux", "sbsa-linux"} {
			args = append(args, "-I"+filepath.Join(tc.CUDAHome, "targets", target, "include", "cccl"))
		}
	}
	for _, include := range []string{"third_party/flashmla/csrc", "third_party/flashmla/csrc/3rdparty/kerutils/include", "third_party/cutlass/include"} {
		args = append(args, "-I"+filepath.Join(root, filepath.FromSlash(include)))
	}
	return append(args, "-Xcompiler", "-fPIC", "-c", "cuda_flashmla_sparse_prefill.cu", "-o", object)
}

func compileCUDAFlashMLA(ctx context.Context, tc *Toolchain, root string, env []string, receipt *CUDANativeBuildReceipt, stdout io.Writer) (string, error) {
	plan, ok := ctx.Value(cudaFlashMLAKey{}).(*cudaFlashMLAOptions)
	if !ok {
		return "", nil
	}
	pkg := filepath.Join(root, "internal", "compute")
	object := filepath.Join(pkg, "cuda_flashmla_sparse_prefill.o")
	args := cudaFlashMLACompileArgs(tc, root, object, plan.disableLDRed)
	if err := compileCUDAObject(ctx, tc, pkg, "cuda_flashmla_sparse_prefill.cu", object, args, env, receipt, stdout); err != nil {
		return "", fmt.Errorf("%w: dedicated translation unit: %v", ErrCUDAFlashMLAPrefill, err)
	}
	images, err := auditCUDAImages(ctx, tc, object, []string{"sm_100a", "sm_103a"}, nil)
	if err != nil {
		return "", err
	}
	plan.receipt.ObjectImages = images
	if receipt.FlashMLAPrefill != nil {
		receipt.FlashMLAPrefill.ObjectImages = images
	}
	return object, nil
}

func cudaFlashMLAGenericFlags(ctx context.Context, args []string) []string {
	if _, ok := ctx.Value(cudaFlashMLAKey{}).(*cudaFlashMLAOptions); ok {
		return append(args, "-DFAK_CUDA_FLASHMLA_PREFILL")
	}
	return args
}

func cudaFlashMLAGoTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableFlashMLAPrefill {
		return strings.TrimSuffix(tags, ",") + ",fak_cuda_flashmla_prefill"
	}
	return tags
}
