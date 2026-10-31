package computebuild

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestCUDAAsyncQualificationProfiles(t *testing.T) {
	for mask := 0; mask < 8; mask++ {
		cfg := &CUDAConfig{EnableV41Compressor: true, EnableV41CompressorPending: mask&1 != 0, EnableQ4KQKVQualification: mask&2 != 0, EnableQ4KQKVAsync: mask&4 != 0}
		valid := mask&4 == 0 || mask&2 != 0
		if (ValidateCUDAAsyncQualifications(cfg) == nil) != valid {
			t.Fatalf("dependency mask=%d", mask)
		}
		if !valid {
			continue
		}
		cfg.EnableCaptureOnceQualification = true
		cfg.EnableFlashMLADecode = true
		tags := CUDAGoTags(cfg)
		for _, peer := range []string{"fak_cuda_capture_once_qualification", "fak_cuda_flashmla_decode", "fak_cuda_v41_compressor"} {
			if !strings.Contains(tags, peer) {
				t.Fatalf("lost peer tag %s: %s", peer, tags)
			}
		}
		for bit, tag := range []string{"fak_cuda_v41_compressor_pending", "fak_cuda_q4k_qkv_qualification", "fak_cuda_q4k_qkv_async"} {
			if strings.Contains(tags, tag) != (mask&(1<<bit) != 0) {
				t.Fatalf("mask=%d tags=%s", mask, tags)
			}
		}
		if strings.Contains(tags, "fak_cuda_v41_index_select") != (mask&2 != 0) {
			t.Fatal(tags)
		}
		base := []string{"-O3", "-DFAK_CUDA_Q2K_MMQ_PREFILL", "-DFAK_CUDA_CAPTURE_ONCE_QUALIFICATION", "-DFAK_CUDA_FLASHMLA_DECODE"}
		ctx := context.WithValue(context.Background(), cudaAsyncQualificationKey{}, cudaAsyncQualificationSelection(cfg))
		got, err := cudaAsyncQualificationFlags(ctx, base, nil)
		want := append([]string(nil), base...)
		if mask&2 != 0 {
			want = append(want, "-DFAK_CUDA_V41_INDEX_SELECT")
		}
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("mask=%d flags=%v err=%v", mask, got, err)
		}
		for _, key := range []string{"NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "Nvcc_Append_Flags"} {
			_, err = cudaAsyncQualificationFlags(ctx, base, []string{key + "=-UFAK_CUDA_V41_INDEX_SELECT"})
			if (err != nil) != (mask != 0) {
				t.Fatalf("mask=%d override=%v", mask, err)
			}
		}
	}
	cfg := &CUDAConfig{EnableV41CompressorPending: true}
	if ValidateCUDAAsyncQualifications(cfg) == nil {
		t.Fatal("pending without compressor")
	}
	cfg.EnableV41Compressor = true
	cfg.EnableV41CompressorPool = true
	if ValidateCUDAAsyncQualifications(cfg) == nil {
		t.Fatal("pending with pool")
	}
	cfg = &CUDAConfig{Arch: "sm_120"}
	ApplyCUDAV41Defaults(cfg)
	if cfg.EnableV41CompressorPending || cfg.EnableQ4KQKVQualification || cfg.EnableQ4KQKVAsync {
		t.Fatal("qualification became a default")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestCUDAAsyncQualificationPreflight(t *testing.T) {
	cfg := &CUDAConfig{RepoRoot: t.TempDir(), EnableQ4KQKVQualification: true}
	if prepareCUDAAsyncQualifications(cfg) == nil {
		t.Fatal("missing source accepted")
	}
	files := map[string]string{
		"cuda_kernels.cu":                         "defined(FAK_CUDA_V41_INDEX_SELECT)\n#define FAK_CUDA_V41_CHECKED_STREAM 1\n#include \"cuda/v41_checked_stream.cuh\"\n#include \"cuda/q4k_shared_input_qualification.cuh\"",
		"cuda/q4k_shared_input_qualification.cuh": "v41_checked_stream_owned_bytes fcuda_q4k_qkv_async_f32",
		"cuda_q4k_qkv_qualification.go":           "fak_cuda_q4k_qkv_qualification && fak_cuda_v41_index_select",
	}
	write := func(root string) {
		t.Helper()
		for name, data := range files {
			path := filepath.Join(root, name)
			if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	write(filepath.Join(cfg.RepoRoot, "internal", "compute"))
	if err := prepareCUDAAsyncQualifications(cfg); err != nil {
		t.Fatal(err)
	}
	cfg.PkgDir = t.TempDir()
	if prepareCUDAAsyncQualifications(cfg) == nil {
		t.Fatal("checked wrong PkgDir")
	}
	write(cfg.PkgDir)
	cfg.EnableQ4KQKVAsync = true
	if prepareCUDAAsyncQualifications(cfg) == nil {
		t.Fatal("async without owner accepted")
	}
	files["cuda_q4k_qkv_async.go"] = "fak_cuda_q4k_qkv_async"
	files["cuda_q4k_async_owner.go"] = "fcuda_q4k_qkv_async_finish"
	files["cuda_async_graph_once_blocked.go"] = "func cudaAsyncGraphOnceAllowed() bool { return false }"
	write(cfg.PkgDir)
	if err := prepareCUDAAsyncQualifications(cfg); err != nil {
		t.Fatal(err)
	}
	// Pending mode shares the existing compressor ABI but must require its own
	// caller, retention implementation, and graph-admission companion.
	cfg.EnableV41Compressor = true
	cfg.EnableV41CompressorPending = true
	if prepareCUDAAsyncQualifications(cfg) == nil {
		t.Fatal("pending without native ABI accepted")
	}
	files["cuda_kernels.cu"] += "\n#include \"cuda/v41_compressor_pending.cuh\""
	files["cuda/v41_compressor_pending.cuh"] = "#ifdef FAK_CUDA_V41_COMPRESSOR\nfak_v41_compressor_pending_retain"
	files["cuda/v41_compressor.cuh"] = "fak_v41_compressor_ring_store_pending_f32"
	files["cuda_v41_compressor_pending.go"] = "fak_cuda_v41_compressor_pending"
	files["cuda_v41_compressor_pending_binding.go"] = "fak_v41_compressor_ring_store_pending_f32"
	write(cfg.PkgDir)
	if err := prepareCUDAAsyncQualifications(cfg); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(cfg.PkgDir, "cuda/v41_compressor_pending.cuh")); err != nil {
		t.Fatal(err)
	}
	if prepareCUDAAsyncQualifications(cfg) == nil {
		t.Fatal("missing pending ABI accepted")
	}
	write(cfg.PkgDir)
	if err := os.Remove(filepath.Join(cfg.PkgDir, "cuda_async_graph_once_blocked.go")); err != nil {
		t.Fatal(err)
	}
	if prepareCUDAAsyncQualifications(cfg) == nil {
		t.Fatal("missing GraphOnce guard accepted")
	}
}
