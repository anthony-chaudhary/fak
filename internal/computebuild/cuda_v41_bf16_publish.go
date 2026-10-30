package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaV41BF16PublishKey struct{}

// Select the existing finite publication and final-norm binding, requiring all
// five cleanup guards after its failed-drain latch. This is not a general
// cleanup-drain guarantee or a native/device numerical qualification.
func prepareCUDAV41BF16Publish(cfg *CUDAConfig) error {
	if cfg.Arch != "sm_120" && cfg.Arch != "120" {
		return fmt.Errorf("V4.1 BF16 publication selection requires explicit --arch sm_120")
	}
	if _, _, err := LoadCUDAArchMatrix(cfg.RepoRoot, cfg.Arch); err != nil {
		return err
	}
	for _, source := range []struct{ path, marker string }{
		{"internal/compute/cuda_kernels.cu", `#include "cuda/model_candidates.cuh"`},
		{"internal/compute/cuda_kernels.cu", "if (g_v41_bf16_uncertain) return true;"},
		{"internal/compute/cuda/model_candidates.cuh", "#ifdef FAK_CUDA_V41_BF16_PUBLISH"},
		{"internal/compute/cuda/model_candidates.cuh", `extern "C" int fak_v41_bf16_publish_f32(`},
		{"internal/compute/cuda/model_candidates.cuh", "g_v41_bf16_uncertain = true;"},
		{"internal/compute/cuda/runtime_memory.cuh", "if (g_v41_bf16_uncertain) return;"},
		{"internal/compute/cuda/runtime_memory.cuh", "if (g_v41_bf16_uncertain) return (int)cudaErrorInvalidValue;"},
		{"internal/compute/cuda/runtime_services.cuh", "if (g_v41_bf16_uncertain) return;"},
		{"internal/compute/cuda/runtime_memory.cuh", "extern \"C\" size_t fcuda_panel_release(void) {\n  if (v41_candidates_stream_uncertain()) return g_panel_live.size();"},
		{"internal/compute/cuda/runtime_memory.cuh", "extern \"C\" void fcuda_trim_pool_large(size_t max_keep_bytes) {\n  if (v41_candidates_stream_uncertain()) return;"},
		{"internal/compute/cuda_v41_bf16_publish.go", "//go:build cuda && cgo && fak_cuda_v41_bf16_publish\n"},
		{"internal/compute/cuda_v41_bf16_checked.go", "//go:build cuda && cgo && fak_cuda_v41_bf16_publish\n"},
	} {
		data, err := os.ReadFile(filepath.Join(cfg.RepoRoot, source.path))
		if err != nil {
			return fmt.Errorf("V4.1 BF16 publication source %s: %w", source.path, err)
		}
		if !strings.Contains(strings.ReplaceAll(string(data), "\r\n", "\n"), source.marker) {
			return fmt.Errorf("V4.1 BF16 publication source %s lacks paired selection contract", source.path)
		}
	}
	return nil
}

// Strict numeric flags apply to the monolithic object, including peer candidates.
func cudaV41BF16PublishFlags(ctx context.Context, args, env []string) ([]string, error) {
	enabled, _ := ctx.Value(cudaV41BF16PublishKey{}).(bool)
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
			return nil, fmt.Errorf("V4.1 BF16 publication selection refuses fast math")
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
			return nil, fmt.Errorf("V4.1 BF16 publication selection requires %s=%s (got %q)", key, want, value)
		}
	}
	return append(args, "-DFAK_CUDA_V41_BF16_PUBLISH", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=false"), nil
}

func cudaV41BF16PublishTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableV41BF16Publish {
		return tags + ",fak_cuda_v41_bf16_publish"
	}
	return tags
}
