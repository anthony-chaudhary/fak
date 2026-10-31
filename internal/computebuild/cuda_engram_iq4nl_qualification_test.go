package computebuild

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestCUDAEngramIQ4NLQualificationSelection(t *testing.T) {
	base := []string{"-O3", "-DFAK_CUDA_V41_INDEX_SELECT", "-DFAK_CUDA_CAPTURE_ONCE_QUALIFICATION", "-DFAK_CUDA_FLASHMLA_DECODE"}
	for _, enabled := range []bool{false, true} {
		cfg := &CUDAConfig{EnableV41EngramIQ4NLQualification: enabled, EnableV41Compressor: true, EnableV41CompressorPending: true, EnableQ4KQKVQualification: true, EnableQ4KQKVAsync: true, EnableCaptureOnceQualification: true, EnableFlashMLADecode: true}
		tags := CUDAGoTags(cfg)
		for _, tag := range []string{"fak_cuda_v41_embed_gather", "fak_cuda_v41_engram_iq4nl"} {
			if strings.Contains(tags, tag) != enabled {
				t.Fatalf("enabled=%v tags=%s", enabled, tags)
			}
		}
		for _, tag := range []string{"fak_cuda_v41_compressor_pending", "fak_cuda_q4k_qkv_async", "fak_cuda_capture_once_qualification", "fak_cuda_flashmla_decode"} {
			if !strings.Contains(tags, tag) {
				t.Fatalf("lost peer tag %s", tag)
			}
		}
		ctx := context.WithValue(context.Background(), cudaEngramIQ4NLQualificationKey{}, enabled)
		got, err := cudaEngramIQ4NLQualificationFlags(ctx, base, nil)
		want := append([]string(nil), base...)
		if enabled {
			want = append(want, "-DFAK_CUDA_V41_EMBED_GATHER", "-DFAK_CUDA_V41_EMBED_IQ4NL")
		}
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("flags=%v err=%v", got, err)
		}
		for _, key := range []string{"NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "Nvcc_Append_Flags"} {
			_, err = cudaEngramIQ4NLQualificationFlags(ctx, base, []string{key + "=-UFAK_CUDA_V41_EMBED_IQ4NL"})
			if (err != nil) != enabled {
				t.Fatalf("enabled=%v override err=%v", enabled, err)
			}
		}
	}
	cfg := &CUDAConfig{Arch: "sm_120"}
	ApplyCUDAV41Defaults(cfg)
	if cfg.EnableV41EngramIQ4NLQualification {
		t.Fatal("qualification became default")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAEngramIQ4NLQualificationPreflight(t *testing.T) {
	cfg := &CUDAConfig{RepoRoot: t.TempDir(), EnableV41EngramIQ4NLQualification: true}
	if prepareCUDAEngramIQ4NLQualification(cfg) == nil {
		t.Fatal("absent source accepted")
	}
	files := map[string]string{
		"cuda_kernels.cu":                    "defined(FAK_CUDA_V41_EMBED_GATHER)\n#define FAK_CUDA_V41_CHECKED_STREAM 1\n#include \"cuda/v41_checked_stream.cuh\"\n#include \"cuda/v41_embed_gather.cuh\"",
		"cuda/v41_embed_gather.cuh":          "#ifdef FAK_CUDA_V41_EMBED_GATHER\n#ifdef FAK_CUDA_V41_EMBED_IQ4NL\nfak_v41_embed_iq4nl_rows_abi fak_v41_embed_iq4nl_upload fak_v41_embed_iq4nl_gather_rows fak_v41_embed_iq4nl_row_poison",
		"cuda/v41_embed_iq4nl.h":             "v41_eg_iq4nl_rows_geometry",
		"cuda_v41_engram_row_decode.go":      "cuda && cgo && fak_cuda_v41_embed_gather && fak_cuda_v41_engram_iq4nl\nfunc (c *cudaBackend) SupportsV41EngramRowDecode() bool { return false }",
		"cuda_v41_engram_row_decode_test.go": "TestCUDAV41EngramIQ4NLRows",
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
	if err := prepareCUDAEngramIQ4NLQualification(cfg); err != nil {
		t.Fatal(err)
	}
	cfg.PkgDir = t.TempDir()
	if prepareCUDAEngramIQ4NLQualification(cfg) == nil {
		t.Fatal("checked wrong PkgDir")
	}
	write(cfg.PkgDir)
	if err := prepareCUDAEngramIQ4NLQualification(cfg); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(cfg.PkgDir, "cuda/v41_embed_iq4nl.h")); err != nil {
		t.Fatal(err)
	}
	if prepareCUDAEngramIQ4NLQualification(cfg) == nil {
		t.Fatal("missing native math accepted")
	}
}
