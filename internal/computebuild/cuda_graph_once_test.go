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
func TestCUDAGraphOnceSelectionAndFlags(t *testing.T) {
	args := []string{"-O3", "-DFAK_CUDA_Q2K_MMQ_PREFILL"}
	for _, enabled := range []bool{false, true} {
		cfg := &CUDAConfig{EnableCaptureOnceQualification: enabled}
		ctx := context.WithValue(context.Background(), cudaGraphOnceKey{}, enabled)
		got, err := cudaGraphOnceFlags(ctx, args, nil)
		want := append([]string(nil), args...)
		tags := "cuda,fak_cuda_v41_router"
		wantTags := tags
		if enabled {
			want = append(want, "-DFAK_CUDA_CAPTURE_ONCE_QUALIFICATION")
			wantTags += ",fak_cuda_capture_once_qualification"
		}
		if err != nil || !reflect.DeepEqual(got, want) || cudaGraphOnceTags(cfg, tags) != wantTags {
			t.Fatalf("enabled=%v flags=%v tags=%s err=%v", enabled, got, cudaGraphOnceTags(cfg, tags), err)
		}
		if len(args) != 2 || args[1] != "-DFAK_CUDA_Q2K_MMQ_PREFILL" {
			t.Fatal("mutated prior MMQ flags")
		}
		for _, key := range []string{"NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "Nvcc_Append_Flags"} {
			_, err = cudaGraphOnceFlags(ctx, args, []string{key + "=-UFAK_CUDA_CAPTURE_ONCE_QUALIFICATION"})
			if (err != nil) != enabled {
				t.Fatalf("ambient override enabled=%v err=%v", enabled, err)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAGraphOncePreflight(t *testing.T) {
	cfg := &CUDAConfig{RepoRoot: t.TempDir()}
	if prepareCUDAGraphOnce(cfg) == nil {
		t.Fatal("missing prerequisites accepted")
	}
	files := map[string]string{
		"cuda_kernels.cu":                  "#ifdef FAK_CUDA_CAPTURE_ONCE_QUALIFICATION\n#include \"cuda/graph_capture_once.cuh\"",
		"cuda/graph_capture_once.cuh":      "fak_graph_once_copy_input fak_graph_once_abort",
		"cuda_graph_once_qualification.go": "cuda && cgo && fak_cuda_capture_once_qualification\nfunc (c *cudaBackend) GraphOnceReplay",
	}
	// Source markers alone only exercise refusal/selection, never qualification.
	for p, data := range files {
		path := filepath.Join(cfg.RepoRoot, "internal", "compute", p)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := prepareCUDAGraphOnce(cfg); err != nil {
		t.Fatal(err)
	}
	// A marker-complete default tree must not authorize a different actual TU.
	cfg.PkgDir = t.TempDir()
	if prepareCUDAGraphOnce(cfg) == nil {
		t.Fatal("checked RepoRoot rather than actual PkgDir")
	}
	for p, data := range files {
		path := filepath.Join(cfg.PkgDir, p)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := prepareCUDAGraphOnce(cfg); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(cfg.PkgDir, "cuda/graph_capture_once.cuh")
	if err := os.WriteFile(path, []byte("fak_graph_once_copy_input"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := prepareCUDAGraphOnce(cfg); err == nil || !strings.Contains(err.Error(), "fak_graph_once_abort") {
		t.Fatalf("missing abort accepted: %v", err)
	}
}
