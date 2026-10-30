package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaV41RouterKey struct{}

// The router's native ABI is always compiled; its Go method set is opt-in.
// Require the matching port and completion-retention/full-score corrections,
// not an invented native define. Source selection is not hardware acceptance.
func prepareCUDAV41Router(cfg *CUDAConfig) error {
	if cfg.Arch != "sm_120" && cfg.Arch != "120" {
		return fmt.Errorf("V4.1 router selection requires explicit --arch sm_120")
	}
	if _, _, err := LoadCUDAArchMatrix(cfg.RepoRoot, cfg.Arch); err != nil {
		return err
	}
	for _, source := range []struct{ path, marker string }{
		{"internal/compute/cuda_kernels.cu", `#include "cuda_v41_router.cu"`},
		{"internal/compute/cuda_kernels.cu", "if (g_v41_router_uncertain) return true;"},
		{"internal/compute/cuda/runtime_memory.cuh", "if (g_v41_router_uncertain) return;"},
		{"internal/compute/cuda/runtime_memory.cuh", "if (g_v41_router_uncertain) return (int)cudaErrorInvalidValue;"},
		{"internal/compute/cuda/runtime_services.cuh", "if (g_v41_router_uncertain) return;"},
		{"internal/compute/cuda_v41_router.cu", `extern "C" int fak_v41_router_topk_f32(`},
		{"internal/compute/cuda_v41_router.cu", "v41_candidates_stream_uncertain()"},
		{"internal/compute/cuda_v41_router.cu", "v41rt_score(float z, bool full)"},
		{"internal/compute/cuda_v41_router.cu", "if (!full && z < -20.0f)"},
		{"internal/compute/cuda_v41_router.go", "//go:build cuda && cgo && fak_cuda_v41_router\n"},
	} {
		data, err := os.ReadFile(filepath.Join(cfg.RepoRoot, source.path))
		if err != nil {
			return fmt.Errorf("V4.1 router source %s: %w", source.path, err)
		}
		if !strings.Contains(strings.ReplaceAll(string(data), "\r\n", "\n"), source.marker) {
			return fmt.Errorf("V4.1 router source %s lacks paired selection contract", source.path)
		}
	}
	return nil
}

// Numeric flags apply to the always-compiled monolithic object and its peers.
func cudaV41RouterFlags(ctx context.Context, args, env []string) ([]string, error) {
	enabled, _ := ctx.Value(cudaV41RouterKey{}).(bool)
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
			return nil, fmt.Errorf("V4.1 router selection refuses fast math")
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
			return nil, fmt.Errorf("V4.1 router selection requires %s=%s (got %q)", key, want, value)
		}
	}
	return append(args, "--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=false"), nil
}

func cudaV41RouterTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableV41Router {
		return tags + ",fak_cuda_v41_router"
	}
	return tags
}
