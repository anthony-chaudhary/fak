package computebuild

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type cudaAsyncQualificationKey struct{}
type cudaAsyncQualificationProfile struct {
	CompressorPending bool
	Q4KQKV            bool
	Q4KQKVAsync       bool
}

func cudaAsyncQualificationSelection(cfg *CUDAConfig) cudaAsyncQualificationProfile {
	return cudaAsyncQualificationProfile{cfg.EnableV41CompressorPending, cfg.EnableQ4KQKVQualification, cfg.EnableQ4KQKVAsync}
}

// ValidateCUDAAsyncQualifications checks profile dependencies without selecting defaults.
// A successful check is build selection only, never hardware qualification.
func ValidateCUDAAsyncQualifications(cfg *CUDAConfig) error {
	if cfg == nil {
		return fmt.Errorf("nil CUDAConfig")
	}
	if cfg.EnableV41CompressorPending && (!cfg.EnableV41Compressor || cfg.EnableV41CompressorPool) {
		return fmt.Errorf("--v41-compressor-pending requires --v41-compressor and excludes --v41-compressor-pool")
	}
	if cfg.EnableQ4KQKVAsync && !cfg.EnableQ4KQKVQualification {
		return fmt.Errorf("--q4k-qkv-async requires --q4k-qkv-qualification")
	}
	return nil
}

func cudaAsyncQualificationTags(cfg *CUDAConfig, tags string) string {
	if cfg.EnableV41CompressorPending {
		tags += ",fak_cuda_v41_compressor_pending"
	}
	if cfg.EnableQ4KQKVQualification {
		tags += ",fak_cuda_q4k_qkv_qualification,fak_cuda_v41_index_select"
	}
	if cfg.EnableQ4KQKVAsync {
		tags += ",fak_cuda_q4k_qkv_async"
	}
	return tags
}

// Both caller modes share native symbols. The existing compressor flag supplies
// the pending ABI; INDEX_SELECT activates the checked-stream prerequisite for Q4.
// Explicit args participate in EffectiveFlags and the existing object cache hash.
// Distinct Go tags key caller changes even when native archive bytes are shared.
func cudaAsyncQualificationFlags(ctx context.Context, args, env []string) ([]string, error) {
	p, _ := ctx.Value(cudaAsyncQualificationKey{}).(cudaAsyncQualificationProfile)
	if !p.CompressorPending && !p.Q4KQKV && !p.Q4KQKVAsync {
		return args, nil
	}
	for _, item := range env {
		key, value, ok := strings.Cut(item, "=")
		if ok && (strings.EqualFold(key, "NVCC_PREPEND_FLAGS") || strings.EqualFold(key, "NVCC_APPEND_FLAGS")) && strings.TrimSpace(value) != "" {
			return nil, fmt.Errorf("async qualification requires recorded flags; unset %s", key)
		}
	}
	out := append([]string(nil), args...)
	if p.Q4KQKV {
		out = append(out, "-DFAK_CUDA_V41_INDEX_SELECT")
	}
	return out, nil
}

// Presence checks refuse incomplete packets; they do not replace source review,
// native compilation, device parity, or the runtime GraphOnce admission guard.
func prepareCUDAAsyncQualifications(cfg *CUDAConfig) error {
	if err := ValidateCUDAAsyncQualifications(cfg); err != nil {
		return err
	}
	p := cudaAsyncQualificationSelection(cfg)
	if !p.CompressorPending && !p.Q4KQKV && !p.Q4KQKVAsync {
		return nil
	}
	pkgDir := cfg.PkgDir
	if pkgDir == "" {
		pkgDir = filepath.Join(cfg.RepoRoot, "internal", "compute")
	}
	type check struct{ path, marker string }
	var checks []check
	if p.CompressorPending {
		checks = append(checks,
			check{"cuda_kernels.cu", "#include \"cuda/v41_compressor_pending.cuh\""},
			check{"cuda/v41_compressor_pending.cuh", "#ifdef FAK_CUDA_V41_COMPRESSOR"},
			check{"cuda/v41_compressor_pending.cuh", "fak_v41_compressor_pending_retain"},
			check{"cuda/v41_compressor.cuh", "fak_v41_compressor_ring_store_pending_f32"},
			check{"cuda_v41_compressor_pending.go", "fak_cuda_v41_compressor_pending"},
			check{"cuda_v41_compressor_pending_binding.go", "fak_v41_compressor_ring_store_pending_f32"})
	}
	if p.Q4KQKV {
		checks = append(checks,
			check{"cuda_kernels.cu", "defined(FAK_CUDA_V41_INDEX_SELECT)"},
			check{"cuda_kernels.cu", "#define FAK_CUDA_V41_CHECKED_STREAM 1"},
			check{"cuda_kernels.cu", "#include \"cuda/v41_checked_stream.cuh\""},
			check{"cuda_kernels.cu", "#include \"cuda/q4k_shared_input_qualification.cuh\""},
			check{"cuda/q4k_shared_input_qualification.cuh", "v41_checked_stream_owned_bytes"},
			check{"cuda_q4k_qkv_qualification.go", "fak_cuda_q4k_qkv_qualification && fak_cuda_v41_index_select"})
	}
	if p.Q4KQKVAsync {
		checks = append(checks,
			check{"cuda/q4k_shared_input_qualification.cuh", "fcuda_q4k_qkv_async_f32"},
			check{"cuda_q4k_qkv_async.go", "fak_cuda_q4k_qkv_async"},
			check{"cuda_q4k_async_owner.go", "fcuda_q4k_qkv_async_finish"})
	}
	if p.CompressorPending || p.Q4KQKVAsync {
		checks = append(checks, check{"cuda_async_graph_once_blocked.go", "func cudaAsyncGraphOnceAllowed() bool { return false }"})
	}
	for _, c := range checks {
		data, err := os.ReadFile(filepath.Join(pkgDir, c.path))
		if err != nil {
			return fmt.Errorf("async qualification prerequisite %s: %w", c.path, err)
		}
		if !strings.Contains(string(data), c.marker) {
			return fmt.Errorf("async qualification prerequisite %s lacks %q", c.path, c.marker)
		}
	}
	return nil
}
