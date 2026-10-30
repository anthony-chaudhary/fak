package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaV41SharedAttentionKey struct{}

// prepareCUDAV41SharedAttention keeps this build arm explicit and bounded to
// Raider SM120. The portable kernel is included by cuda_kernels.cu, not a
// separate translation unit. Source presence is required: older trees must not
// silently build a tagged binary whose archive omits the candidate ABI.
func prepareCUDAV41SharedAttention(cfg *CUDAConfig) error {
	if cfg.Arch != "sm_120" && cfg.Arch != "120" {
		return fmt.Errorf("V4.1 shared attention candidate requires explicit --arch sm_120")
	}
	if _, _, err := LoadCUDAArchMatrix(cfg.RepoRoot, cfg.Arch); err != nil {
		return err
	}
	for _, source := range []struct{ path, marker string }{
		{"internal/compute/cuda_kernels.cu", `#include "cuda_v41_shared_attention.cu"`},
		{"internal/compute/cuda_v41_shared_attention.cu", "#ifdef FAK_CUDA_V41_SHARED_ATTENTION"},
		{"internal/compute/cuda_v41_shared_attention.go", "//go:build cuda && cgo && fak_cuda_v41_shared_attention"},
	} {
		data, err := os.ReadFile(filepath.Join(cfg.RepoRoot, source.path))
		if err != nil {
			return fmt.Errorf("V4.1 shared attention candidate source %s: %w", source.path, err)
		}
		if !strings.Contains(string(data), source.marker) {
			return fmt.Errorf("V4.1 shared attention candidate source %s lacks paired opt-in contract", source.path)
		}
	}
	return nil
}

func cudaV41SharedAttentionFlags(ctx context.Context, args []string) []string {
	if enabled, _ := ctx.Value(cudaV41SharedAttentionKey{}).(bool); enabled {
		return append(args, "-DFAK_CUDA_V41_SHARED_ATTENTION")
	}
	return args
}

func cudaV41SharedAttentionTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableV41SharedAttention {
		return tags + ",fak_cuda_v41_shared_attention"
	}
	return tags
}
