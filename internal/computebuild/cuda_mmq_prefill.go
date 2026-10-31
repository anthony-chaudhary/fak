package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type cudaMMQPrefillKey struct{}

// Order is stable: these explicit args participate in the existing native
// object receipt EffectiveFlags and cache InputSHA256. No new receipt format.
func cudaMMQPrefillSelection(cfg *CUDAConfig) []string {
	var macros []string
	if cfg.EnableQ2KMMQPrefill {
		macros = append(macros, "FAK_CUDA_Q2K_MMQ_PREFILL")
	}
	if cfg.EnableQ3KMMQPrefill {
		macros = append(macros, "FAK_CUDA_Q3K_MMQ_PREFILL")
	}
	if cfg.EnableQ8MMQPrefill {
		macros = append(macros, "FAK_CUDA_Q8_MMQ_PREFILL")
	}
	return macros
}

func cudaMMQPrefillFlags(ctx context.Context, args, env []string) ([]string, error) {
	macros, _ := ctx.Value(cudaMMQPrefillKey{}).([]string)
	if len(macros) == 0 {
		return args, nil
	}
	// Ambient nvcc flags are absent from EffectiveFlags and can override our
	// defines. Refuse them rather than silently disabling cache/provenance.
	for _, item := range env {
		key, value, ok := strings.Cut(item, "=")
		if ok && (strings.EqualFold(key, "NVCC_PREPEND_FLAGS") || strings.EqualFold(key, "NVCC_APPEND_FLAGS")) && strings.TrimSpace(value) != "" {
			return nil, fmt.Errorf("MMQ qualification requires explicit recorded flags; unset %s", key)
		}
	}
	out := append([]string(nil), args...)
	for _, macro := range macros {
		out = append(out, "-D"+macro)
	}
	return out, nil
}

// This fail-closed source preflight does not qualify arithmetic or lifetime.
// The integration reviewer must still verify the complete safety prerequisite
// delta; a substring is not a substitute for source review or device witnesses.
func prepareCUDAMMQPrefill(cfg *CUDAConfig, macros []string) error {
	arch, err := strconv.Atoi(strings.TrimPrefix(cfg.Arch, "sm_"))
	if err != nil || arch < 80 {
		return fmt.Errorf("MMQ qualification requires one explicit supported SM80+ architecture")
	}
	if _, _, err := LoadCUDAArchMatrix(cfg.RepoRoot, cfg.Arch); err != nil {
		return err
	}
	checks := []struct{ path, marker string }{
		{"cuda_kernels.cu", "g_v41_checked_stream_uncertain.load()"},
		{"cuda_kernels.cu", "#define FAK_CUDA_V41_CHECKED_STREAM 1"},
		{"cuda_kernels.cu", `#include "cuda/v41_checked_stream.cuh"`},
		{"cuda/runtime_memory.cuh", "if (v41_candidates_stream_uncertain()) return;"},
		{"cuda/runtime_services.cuh", "if (v41_candidates_stream_uncertain()) return;"},
		{"cuda/v41_checked_stream.cuh", "g_v41_checked_stream_uncertain=true;"},
	}
	for _, macro := range macros {
		name := "q2k"
		if macro == "FAK_CUDA_Q3K_MMQ_PREFILL" {
			name = "q3k"
		} else if macro == "FAK_CUDA_Q8_MMQ_PREFILL" {
			name = "q8"
		}
		checks = append(checks,
			struct{ path, marker string }{"cuda_kernels.cu", "defined(" + macro + ")"},
			struct{ path, marker string }{"cuda_kernels.cu", `#include "cuda/` + name + `_mmq.cuh"`},
			struct{ path, marker string }{"cuda/" + name + "_mmq.cuh", "#if defined(" + macro + ")"},
		)
	}
	for _, check := range checks {
		data, err := os.ReadFile(filepath.Join(cfg.RepoRoot, "internal", "compute", check.path))
		if err != nil {
			return fmt.Errorf("MMQ qualification prerequisite %s: %w", check.path, err)
		}
		if !strings.Contains(string(data), check.marker) {
			return fmt.Errorf("MMQ qualification prerequisite %s lacks %q", check.path, check.marker)
		}
	}
	return nil
}
