package computebuild

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestCUDAMMQPrefillSelectionAndFlags(t *testing.T) {
	base := []string{"-O3", "-DFAK_CUDA_V41_COMPRESSOR", "--fmad=false"}
	for mask := 0; mask < 8; mask++ {
		cfg := &CUDAConfig{EnableQ2KMMQPrefill: mask&1 != 0, EnableQ3KMMQPrefill: mask&2 != 0, EnableQ8MMQPrefill: mask&4 != 0}
		selection := cudaMMQPrefillSelection(cfg)
		ctx := context.WithValue(context.Background(), cudaMMQPrefillKey{}, selection)
		got, err := cudaMMQPrefillFlags(ctx, base, nil)
		want := append([]string(nil), base...)
		for _, macro := range selection {
			want = append(want, "-D"+macro)
		}
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("mask %d: flags=%v err=%v", mask, got, err)
		}
		for _, key := range []string{"NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "nvcc_prepend_flags", "Nvcc_Append_Flags"} {
			_, err := cudaMMQPrefillFlags(ctx, base, []string{key + "=-UFAK_CUDA_Q2K_MMQ_PREFILL"})
			if (err != nil) != (mask != 0) {
				t.Fatalf("mask %d: ambient override err=%v", mask, err)
			}
		}
	}
	cfg := &CUDAConfig{Arch: "sm_120"}
	ApplyCUDAV41Defaults(cfg)
	if len(cudaMMQPrefillSelection(cfg)) != 0 {
		t.Fatal("MMQ must never be defaulted")
	}
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime slow est=10s
// Uses the existing synthetic nvcc helper, not a CUDA compiler or GPU. Exercises
// the actual CompileCUDAKernels -> compileCUDAObject receipt/cache path.
func TestCUDAMMQPrefillNativeReceiptCache(t *testing.T) {
	t.Setenv("NVCC_PREPEND_FLAGS", "")
	t.Setenv("NVCC_APPEND_FLAGS", "")
	root, tc := cudaIncrementalFixture(t)
	compile := func(macros []string, wantAction string) string {
		t.Helper()
		ctx := context.WithValue(context.Background(), cudaMMQPrefillKey{}, macros)
		if _, err := CompileCUDAKernels(ctx, tc, root, "sm_120", false, io.Discard); err != nil {
			t.Fatal(err)
		}
		r := cudaReadNativeReceipt(t, root)
		if len(r.Objects) != 1 || r.Objects[0].Action != wantAction {
			t.Fatalf("native objects: %+v", r.Objects)
		}
		o := r.Objects[0]
		var got []string
		for _, arg := range o.EffectiveFlags {
			if strings.HasPrefix(arg, "-DFAK_CUDA_Q") && strings.HasSuffix(arg, "_MMQ_PREFILL") {
				got = append(got, strings.TrimPrefix(arg, "-D"))
			}
		}
		if !reflect.DeepEqual(got, macros) || o.InputSHA256 == "" {
			t.Fatalf("receipt lost selection/key: %+v", o)
		}
		return o.InputSHA256
	}
	baseline := compile(nil, "executed")
	macros := cudaMMQPrefillSelection(&CUDAConfig{EnableQ2KMMQPrefill: true, EnableQ3KMMQPrefill: true, EnableQ8MMQPrefill: true})
	selected := compile(macros, "executed")
	if selected == baseline || compile(macros, "reused") != selected {
		t.Fatal("MMQ flags did not change/preserve cache identity")
	}
	if compile(nil, "executed") != baseline {
		t.Fatal("default build failed to recover its original key")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAMMQPrefillNativeIntegration(t *testing.T) {
	data, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	if strings.Count(s, "nvccArgs, err = cudaMMQPrefillFlags(ctx, nvccArgs, env)") != 2 || !strings.Contains(s, "prepareCUDAMMQPrefill(cfg, selection)") || !strings.Contains(s, "context.WithValue(ctx, cudaMMQPrefillKey{}, selection)") {
		t.Fatal("both OS compiler branches and RunCUDA preflight must use the selection")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAMMQPrefillPrerequisites(t *testing.T) {
	root := t.TempDir()
	cfg := &CUDAConfig{RepoRoot: root, Arch: "sm_120", EnableQ2KMMQPrefill: true, EnableQ3KMMQPrefill: true, EnableQ8MMQPrefill: true}
	macros := cudaMMQPrefillSelection(cfg)
	files := map[string]string{
		"cuda_arch.txt":               "sm_120\n",
		"cuda_kernels.cu":             "g_v41_checked_stream_uncertain.load()\n#define FAK_CUDA_V41_CHECKED_STREAM 1\n#include \"cuda/v41_checked_stream.cuh\"\n",
		"cuda/runtime_memory.cuh":     "if (v41_candidates_stream_uncertain()) return;",
		"cuda/runtime_services.cuh":   "if (v41_candidates_stream_uncertain()) return;",
		"cuda/v41_checked_stream.cuh": "g_v41_checked_stream_uncertain=true;",
	}
	for i, name := range []string{"q2k", "q3k", "q8"} {
		files["cuda_kernels.cu"] += "defined(" + macros[i] + ")\n#include \"cuda/" + name + "_mmq.cuh\"\n"
		files["cuda/"+name+"_mmq.cuh"] = "#if defined(" + macros[i] + ")"
	}
	write := func(path, body string) {
		t.Helper()
		path = filepath.Join(root, "internal", "compute", path)
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	for path, body := range files {
		write(path, body)
	}
	if err := prepareCUDAMMQPrefill(cfg, macros); err != nil {
		t.Fatal(err)
	}
	for path, body := range files {
		write(path, "")
		if err := prepareCUDAMMQPrefill(cfg, macros); err == nil {
			t.Fatalf("missing prerequisite %s accepted", path)
		}
		write(path, body)
	}
	for _, arch := range []string{"", "sm_75", "sm_120,sm_89", "sm_999"} {
		cfg.Arch = arch
		if err := prepareCUDAMMQPrefill(cfg, macros); err == nil {
			t.Fatalf("invalid/unlisted arch %q accepted", arch)
		}
	}
}
