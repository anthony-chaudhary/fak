package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaGraphOnceKey struct{}

func cudaGraphOnceTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableCaptureOnceQualification {
		return tags + ",fak_cuda_capture_once_qualification"
	}
	return tags
}

// Explicit arguments flow through the existing EffectiveFlags and InputSHA256.
// No ambient flag is accepted in this qualification mode: it could undo the
// selection without appearing in the receipt/cache identity.
func cudaGraphOnceFlags(ctx context.Context, args, env []string) ([]string, error) {
	enabled, _ := ctx.Value(cudaGraphOnceKey{}).(bool)
	if !enabled {
		return args, nil
	}
	for _, item := range env {
		key, value, ok := strings.Cut(item, "=")
		if ok && (strings.EqualFold(key, "NVCC_PREPEND_FLAGS") || strings.EqualFold(key, "NVCC_APPEND_FLAGS")) && strings.TrimSpace(value) != "" {
			return nil, fmt.Errorf("capture-once qualification requires recorded flags; unset %s", key)
		}
	}
	out := append([]string(nil), args...)
	return append(out, "-DFAK_CUDA_CAPTURE_ONCE_QUALIFICATION"), nil
}

// This is presence checking, not proof of native correctness or device parity.
// The exact native + HAL source packet must separately pass critical review.
func prepareCUDAGraphOnce(cfg *CUDAConfig) error {
	pkgDir := cfg.PkgDir
	if pkgDir == "" {
		pkgDir = filepath.Join(cfg.RepoRoot, "internal", "compute")
	}
	checks := []struct{ path, marker string }{
		{"cuda_kernels.cu", "#ifdef FAK_CUDA_CAPTURE_ONCE_QUALIFICATION"},
		{"cuda_kernels.cu", `#include "cuda/graph_capture_once.cuh"`},
		{"cuda/graph_capture_once.cuh", "fak_graph_once_copy_input"},
		{"cuda/graph_capture_once.cuh", "fak_graph_once_abort"},
		{"cuda_graph_once_qualification.go", "cuda && cgo && fak_cuda_capture_once_qualification"},
		{"cuda_graph_once_qualification.go", "func (c *cudaBackend) GraphOnceReplay"},
	}
	for _, check := range checks {
		data, err := os.ReadFile(filepath.Join(pkgDir, check.path))
		if err != nil {
			return fmt.Errorf("capture-once prerequisite %s: %w", check.path, err)
		}
		if !strings.Contains(string(data), check.marker) {
			return fmt.Errorf("capture-once prerequisite %s lacks %q", check.path, check.marker)
		}
	}
	return nil
}
