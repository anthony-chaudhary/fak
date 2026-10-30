package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaV41EngramGateKey struct{}

// The explicit SM120 candidate does not qualify public Engram-gate capability.
// Preserve the same-repository oracle noncontracted binary32 operation order.
// These numeric flags apply to the monolithic object, including peer candidates.
func prepareCUDAV41EngramGate(cfg *CUDAConfig) error {
	if cfg.Arch != "sm_120" && cfg.Arch != "120" {
		return fmt.Errorf("V4.1 Engram-gate candidate requires explicit --arch sm_120")
	}
	if _, _, err := LoadCUDAArchMatrix(cfg.RepoRoot, cfg.Arch); err != nil {
		return err
	}
	for _, source := range []struct{ path, marker string }{
		{"internal/compute/cuda_kernels.cu", `#include "cuda_v41_engram_gate.cu"`},
		{"internal/compute/cuda_kernels.cu", "if (g_v41_engram_uncertain) return true;"},
		{"internal/compute/cuda/runtime_memory.cuh", "g_v41_engram_quarantine"},
		{"internal/compute/cuda/runtime_services.cuh", "g_v41_engram_quarantine"},
		{"internal/compute/cuda_v41_engram_gate.cu", "#ifdef FAK_CUDA_V41_ENGRAM_GATE"},
		{"internal/compute/cuda_v41_engram_gate.go", "//go:build cuda && cgo && fak_cuda_v41_engram_gate"},
	} {
		data, err := os.ReadFile(filepath.Join(cfg.RepoRoot, source.path))
		if err != nil {
			return fmt.Errorf("V4.1 Engram-gate candidate source %s: %w", source.path, err)
		}
		if !strings.Contains(string(data), source.marker) {
			return fmt.Errorf("V4.1 Engram-gate candidate source %s lacks paired opt-in contract", source.path)
		}
	}
	return nil
}

func cudaV41EngramGateFlags(ctx context.Context, args, env []string) ([]string, error) {
	enabled, _ := ctx.Value(cudaV41EngramGateKey{}).(bool)
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
			return nil, fmt.Errorf("V4.1 Engram-gate candidate refuses fast math")
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
			return nil, fmt.Errorf("V4.1 Engram-gate candidate requires %s=%s (got %q)", key, want, value)
		}
	}
	return append(args, "-DFAK_CUDA_V41_ENGRAM_GATE", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=false"), nil
}

func cudaV41EngramGateTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableV41EngramGate {
		return tags + ",fak_cuda_v41_engram_gate"
	}
	return tags
}
