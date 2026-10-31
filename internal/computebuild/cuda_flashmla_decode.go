package computebuild

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"
)

const cudaFlashMLADecodeManifestSHA256 = "97dd0fce45408709cd30a9b27ccb91ba9b4f4b11923e218a913db7433cbeb2c8"

type cudaFlashMLADecodeKey struct{}

// Same evidence fields; a decode build receipt never implies hardware acceptance.
type CUDAFlashMLADecodeReceipt = CUDAFlashMLAPrefillReceipt

func prepareCUDAFlashMLADecode(ctx context.Context, cfg *CUDAConfig) (*cudaFlashMLAOptions, error) {
	if err := validateFlashMLADecodeGenericArch(cfg.Arch); err != nil {
		return nil, err
	}
	plan, err := prepareCUDAFlashMLA(ctx, cfg)
	if err != nil {
		return nil, err
	}
	manifest, err := verifyCUDAVendorSources(cfg.RepoRoot, "third_party/flashmla/DECODE-PINNED-SOURCE.json", cudaFlashMLADecodeManifestSHA256)
	if err != nil {
		return nil, err
	}
	if manifest.FlashMLACommit != plan.receipt.FlashMLACommit || manifest.CUTLASSCommit != plan.receipt.CUTLASSCommit {
		return nil, fmt.Errorf("FlashMLA decode donor pin mismatch")
	}
	plan.receipt.ManifestSHA256 = cudaFlashMLADecodeManifestSHA256
	return plan, nil
}
func cudaFlashMLADecodeCompileArgs(tc *Toolchain, root, object string, disableLDRed bool) []string {
	args := cudaFlashMLACompileArgs(tc, root, object, disableLDRed)
	out := args[:0]
	for _, v := range args {
		if v == "-DNDEBUG" {
			continue
		} // Keep donor qualification assertions active.
		if v == "cuda_flashmla_sparse_prefill.cu" {
			v = "cuda_flashmla_sparse_decode.cu"
		}
		out = append(out, v)
	}
	return out
}
func compileCUDAFlashMLADecode(ctx context.Context, tc *Toolchain, root string, env []string, receipt *CUDANativeBuildReceipt, stdout io.Writer) (string, error) {
	plan, ok := ctx.Value(cudaFlashMLADecodeKey{}).(*cudaFlashMLAOptions)
	if !ok {
		return "", nil
	}
	if err := checkFlashMLADecodeEnvironment(env); err != nil {
		return "", err
	}
	pkg := filepath.Join(root, "internal", "compute")
	object := filepath.Join(pkg, "cuda_flashmla_sparse_decode.o")
	args := cudaFlashMLADecodeCompileArgs(tc, root, object, plan.disableLDRed)
	if err := compileCUDAObject(ctx, tc, pkg, "cuda_flashmla_sparse_decode.cu", object, args, env, receipt, stdout); err != nil {
		return "", err
	}
	images, err := auditCUDAImages(ctx, tc, object, []string{"sm_100a", "sm_103a"}, nil)
	if err != nil {
		return "", err
	}
	plan.receipt.ObjectImages = images
	if receipt.FlashMLADecode != nil {
		receipt.FlashMLADecode.ObjectImages = images
	}
	return object, nil
}
func checkFlashMLADecodeEnvironment(env []string) error {
	for _, item := range env {
		key, value, ok := strings.Cut(item, "=")
		if ok && (strings.EqualFold(key, "NVCC_PREPEND_FLAGS") || strings.EqualFold(key, "NVCC_APPEND_FLAGS")) && strings.TrimSpace(value) != "" {
			return fmt.Errorf("FlashMLA decode requires recorded flags; unset %s", key)
		}
	}
	return nil
}
func cudaFlashMLADecodeGenericFlags(ctx context.Context, args, env []string) ([]string, error) {
	if _, ok := ctx.Value(cudaFlashMLADecodeKey{}).(*cudaFlashMLAOptions); ok {
		if err := checkFlashMLADecodeEnvironment(env); err != nil {
			return nil, err
		}
		return append(args, "-DFAK_CUDA_FLASHMLA_DECODE"), nil
	}
	return args, nil
}
func cudaFlashMLADecodeGoTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableFlashMLADecode {
		return strings.TrimSuffix(tags, ",") + ",fak_cuda_flashmla_decode"
	}
	return tags
}

// cfg.Arch selects the existing generic CUDA matrix. The dedicated FlashMLA TU
// has separate architecture-specific images and never uses a generic SM120 path.
func validateFlashMLADecodeGenericArch(arch string) error {
	if arch != "sm_100" && arch != "sm_103" {
		return fmt.Errorf("FlashMLA decode requires generic sm_100 or sm_103 plus explicit arch-specific SM100a/103a TU")
	}
	return nil
}
