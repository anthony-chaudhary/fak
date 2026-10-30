package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaV41MHCKey struct{}

// The explicit SM120 witness build does not qualify SupportsV41MHC. Its
// numerical policy is carried by the same monolithic object as the candidate.
func prepareCUDAV41MHC(cfg *CUDAConfig) error {
	if cfg.Arch != "sm_120" && cfg.Arch != "120" {
		return fmt.Errorf("V4.1 mHC candidate requires explicit --arch sm_120")
	}
	if _, _, err := LoadCUDAArchMatrix(cfg.RepoRoot, cfg.Arch); err != nil {
		return err
	}
	for _, source := range []struct{ path, marker string }{
		{"internal/compute/cuda_kernels.cu", `#include "cuda_v41_mhc.cu"`},
		{"internal/compute/cuda_v41_mhc.cu", "#ifdef FAK_CUDA_V41_MHC"},
		{"internal/compute/cuda_v41_mhc.go", "//go:build cuda && cgo && fak_cuda_v41_mhc"},
	} {
		data, err := os.ReadFile(filepath.Join(cfg.RepoRoot, source.path))
		if err != nil {
			return fmt.Errorf("V4.1 mHC candidate source %s: %w", source.path, err)
		}
		if !strings.Contains(string(data), source.marker) {
			return fmt.Errorf("V4.1 mHC candidate source %s lacks paired opt-in contract", source.path)
		}
	}
	return nil
}

func cudaV41MHCFlags(ctx context.Context, args, env []string) ([]string, error) {
	enabled, _ := ctx.Value(cudaV41MHCKey{}).(bool)
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
			return nil, fmt.Errorf("V4.1 mHC candidate refuses fast math")
		}
		if key != "ftz" && key != "prec-div" && key != "prec-sqrt" {
			continue
		}
		if !assigned && i+1 < len(effective) {
			value = strings.Trim(effective[i+1], "\"'")
		}
		want := "true"
		if key == "ftz" {
			want = "false"
		}
		if value != want {
			return nil, fmt.Errorf("V4.1 mHC candidate requires %s=%s (got %q)", key, want, value)
		}
	}
	return append(args, "-DFAK_CUDA_V41_MHC", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"), nil
}

func cudaV41MHCTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableV41MHC {
		return tags + ",fak_cuda_v41_mhc"
	}
	return tags
}
