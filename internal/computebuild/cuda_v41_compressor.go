package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaV41CompressorKey struct{}

// The explicit ring/pool selection does not qualify numerical or hardware
// behavior. Require its paired native/Go sources and failed-drain retention.
func prepareCUDAV41Compressor(cfg *CUDAConfig) error {
	if cfg.EnableV41CompressorPool {
		return fmt.Errorf("--v41-compressor and --v41-compressor-pool are mutually exclusive")
	}
	if cfg.Arch != "sm_120" && cfg.Arch != "120" {
		return fmt.Errorf("V4.1 compressor candidate requires explicit --arch sm_120")
	}
	if _, _, err := LoadCUDAArchMatrix(cfg.RepoRoot, cfg.Arch); err != nil {
		return err
	}
	for _, source := range []struct{ path, marker string }{
		{"internal/compute/cuda_kernels.cu", `#include "cuda/v41_compressor.cuh"`},
		{"internal/compute/cuda_kernels.cu", "if (g_v41_compressor_uncertain) return true;"},
		{"internal/compute/cuda/runtime_memory.cuh", "if (g_v41_compressor_uncertain) return;"},
		{"internal/compute/cuda/runtime_memory.cuh", "if (g_v41_compressor_uncertain) return (int)cudaErrorInvalidValue;"},
		{"internal/compute/cuda/runtime_services.cuh", "if (g_v41_compressor_uncertain) return;"},
		{"internal/compute/cuda/v41_compressor.cuh", "#ifdef FAK_CUDA_V41_COMPRESSOR"},
		{"internal/compute/cuda_v41_compressor.go", "//go:build cuda && cgo && fak_cuda_v41_compressor && !fak_cuda_v41_compressor_pool"},
	} {
		data, err := os.ReadFile(filepath.Join(cfg.RepoRoot, source.path))
		if err != nil {
			return fmt.Errorf("V4.1 compressor candidate source %s: %w", source.path, err)
		}
		if !strings.Contains(string(data), source.marker) {
			return fmt.Errorf("V4.1 compressor candidate source %s lacks paired opt-in contract", source.path)
		}
	}
	return nil
}

// Numeric flags apply to the monolithic object, including peer candidates.
func cudaV41CompressorFlags(ctx context.Context, args, env []string) ([]string, error) {
	enabled, _ := ctx.Value(cudaV41CompressorKey{}).(bool)
	if !enabled {
		return args, nil
	}
	effective := append([]string(nil), args...)
	for _, item := range env {
		key, value, ok := strings.Cut(item, "=")
		if ok && (key == "NVCC_PREPEND_FLAGS" || key == "NVCC_APPEND_FLAGS") {
			effective = append(effective, strings.Fields(value)...)
		}
	}
	for i, arg := range effective {
		arg = strings.Trim(arg, "\"'")
		key, value, assigned := strings.Cut(strings.TrimLeft(arg, "-"), "=")
		if key == "use_fast_math" {
			return nil, fmt.Errorf("V4.1 compressor candidate refuses fast math")
		}
		if key != "ftz" && key != "prec-div" && key != "prec-sqrt" && key != "fmad" {
			continue
		}
		if !assigned && i+1 < len(effective) {
			value = strings.Trim(effective[i+1], "\"'")
		}
		want := "true"
		if key == "ftz" || key == "fmad" {
			want = "false"
		}
		if value != want {
			return nil, fmt.Errorf("V4.1 compressor candidate requires %s=%s (got %q)", key, want, value)
		}
	}
	return append(args, "-DFAK_CUDA_V41_COMPRESSOR", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=false"), nil
}

func cudaV41CompressorTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableV41Compressor {
		return tags + ",fak_cuda_v41_compressor"
	}
	return tags
}
