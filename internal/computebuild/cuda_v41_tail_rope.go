package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaV41TailRoPEKey struct{}

// Select only the existing ordinary QK/Rows candidate. Its idle-stream and
// exact-base retention contract is required; exact arithmetic remains gated.
func prepareCUDAV41TailRoPE(cfg *CUDAConfig) error {
	if cfg.Arch != "sm_120" && cfg.Arch != "120" {
		return fmt.Errorf("V4.1 tail RoPE selection requires explicit --arch sm_120")
	}
	if _, _, err := LoadCUDAArchMatrix(cfg.RepoRoot, cfg.Arch); err != nil {
		return err
	}
	for _, source := range []struct{ path, marker string }{
		{"internal/compute/cuda_kernels.cu", `#include "cuda/model_candidates.cuh"`},
		{"internal/compute/cuda_kernels.cu", "if (g_v41_rope_uncertain) return true;"},
		{"internal/compute/cuda/model_candidates.cuh", "#ifdef FAK_CUDA_V41_TAIL_ROPE"},
		{"internal/compute/cuda/model_candidates.cuh", `extern "C" int fak_v41_tail_rope_qk(`},
		{"internal/compute/cuda/model_candidates.cuh", `extern "C" int fak_v41_tail_rope_rows(`},
		{"internal/compute/cuda/model_candidates.cuh", "if (!g_blas || !g_stream || g_capture_open || v41_candidates_stream_uncertain()) return -3;"},
		{"internal/compute/cuda/model_candidates.cuh", "error = cudaStreamQuery(g_stream);"},
		{"internal/compute/cuda/model_candidates.cuh", "if (error == cudaErrorNotReady) return -3;"},
		{"internal/compute/cuda/model_candidates.cuh", "g_v41_rope_uncertain = true;"},
		{"internal/compute/cuda/runtime_memory.cuh", "for (const void *held : g_v41_rope_quarantine) {\n    if (d == held) return;"},
		{"internal/compute/cuda/runtime_memory.cuh", "for (const void *held : g_v41_rope_quarantine)\n    if (d == held) return (int)cudaErrorInvalidValue;"},
		{"internal/compute/cuda/runtime_services.cuh", "for (const void *held : g_v41_rope_quarantine) {\n    if (d == held) return;"},
		{"internal/compute/cuda_v41_rope.go", "//go:build cuda && cgo && fak_cuda_v41_tail_rope\n"},
	} {
		data, err := os.ReadFile(filepath.Join(cfg.RepoRoot, source.path))
		if err != nil {
			return fmt.Errorf("V4.1 tail RoPE source %s: %w", source.path, err)
		}
		if !strings.Contains(strings.ReplaceAll(string(data), "\r\n", "\n"), source.marker) {
			return fmt.Errorf("V4.1 tail RoPE source %s lacks paired selection contract", source.path)
		}
	}
	return nil
}

// Strict numeric flags apply to the monolithic object, including peer candidates.
func cudaV41TailRoPEFlags(ctx context.Context, args, env []string) ([]string, error) {
	enabled, _ := ctx.Value(cudaV41TailRoPEKey{}).(bool)
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
			return nil, fmt.Errorf("V4.1 tail RoPE selection refuses fast math")
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
			return nil, fmt.Errorf("V4.1 tail RoPE selection requires %s=%s (got %q)", key, want, value)
		}
	}
	return append(args, "-DFAK_CUDA_V41_TAIL_ROPE", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=false"), nil
}

func cudaV41TailRoPETags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableV41TailRoPE {
		return tags + ",fak_cuda_v41_tail_rope"
	}
	return tags
}
