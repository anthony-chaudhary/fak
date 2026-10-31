package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaEngramIQ4NLQualificationKey struct{}

func cudaEngramIQ4NLQualificationTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableV41EngramIQ4NLQualification {
		return tags + ",fak_cuda_v41_embed_gather,fak_cuda_v41_engram_iq4nl"
	}
	return tags
}

// Reuse the checked-stream native seam and existing flag/dependency cache
// identity. The pinned native decoder and Go witness must both be selected;
// this does not qualify model routing or advertise public serving support.
func cudaEngramIQ4NLQualificationFlags(ctx context.Context, args, env []string) ([]string, error) {
	enabled, _ := ctx.Value(cudaEngramIQ4NLQualificationKey{}).(bool)
	if !enabled {
		return args, nil
	}
	for _, item := range env {
		key, value, ok := strings.Cut(item, "=")
		if ok && (strings.EqualFold(key, "NVCC_PREPEND_FLAGS") || strings.EqualFold(key, "NVCC_APPEND_FLAGS")) && strings.TrimSpace(value) != "" {
			return nil, fmt.Errorf("Engram IQ4_NL qualification requires recorded flags; unset %s", key)
		}
	}
	out := append([]string(nil), args...)
	return append(out, "-DFAK_CUDA_V41_EMBED_GATHER", "-DFAK_CUDA_V41_EMBED_IQ4NL"), nil
}

// Presence checks are fail-closed prerequisites, not numerical/device proof.
func prepareCUDAEngramIQ4NLQualification(cfg *CUDAConfig) error {
	if !cfg.EnableV41EngramIQ4NLQualification {
		return nil
	}
	pkgDir := cfg.PkgDir
	if pkgDir == "" {
		pkgDir = filepath.Join(cfg.RepoRoot, "internal", "compute")
	}
	checks := []struct{ path, marker string }{
		{"cuda_kernels.cu", "defined(FAK_CUDA_V41_EMBED_GATHER)"},
		{"cuda_kernels.cu", "#define FAK_CUDA_V41_CHECKED_STREAM 1"},
		{"cuda_kernels.cu", "#include \"cuda/v41_checked_stream.cuh\""},
		{"cuda_kernels.cu", "#include \"cuda/v41_embed_gather.cuh\""},
		{"cuda/v41_embed_gather.cuh", "#ifdef FAK_CUDA_V41_EMBED_GATHER"},
		{"cuda/v41_embed_gather.cuh", "#ifdef FAK_CUDA_V41_EMBED_IQ4NL"},
		{"cuda/v41_embed_gather.cuh", "fak_v41_embed_iq4nl_rows_abi"},
		{"cuda/v41_embed_gather.cuh", "fak_v41_embed_iq4nl_upload"},
		{"cuda/v41_embed_gather.cuh", "fak_v41_embed_iq4nl_gather_rows"},
		{"cuda/v41_embed_gather.cuh", "fak_v41_embed_iq4nl_row_poison"},
		{"cuda/v41_embed_iq4nl.h", "v41_eg_iq4nl_rows_geometry"},
		{"cuda_v41_engram_row_decode.go", "cuda && cgo && fak_cuda_v41_embed_gather && fak_cuda_v41_engram_iq4nl"},
		{"cuda_v41_engram_row_decode.go", "func (c *cudaBackend) SupportsV41EngramRowDecode() bool { return false }"},
		{"cuda_v41_engram_row_decode_test.go", "TestCUDAV41EngramIQ4NLRows"},
	}
	for _, check := range checks {
		data, err := os.ReadFile(filepath.Join(pkgDir, check.path))
		if err != nil {
			return fmt.Errorf("Engram IQ4_NL prerequisite %s: %w", check.path, err)
		}
		if !strings.Contains(string(data), check.marker) {
			return fmt.Errorf("Engram IQ4_NL prerequisite %s lacks %q", check.path, check.marker)
		}
	}
	return nil
}
