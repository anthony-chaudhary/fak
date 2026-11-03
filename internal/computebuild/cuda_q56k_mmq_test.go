package computebuild

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestCUDAQ56KMMQTypedSelection(t *testing.T) {
	const macro = "-DFAK_CUDA_Q56K_MMQ_QUALIFICATION"
	args := []string{"-O3", "-DFAK_CUDA_Q2K_MMQ_PREFILL", "-DFAK_CUDA_Q56K_MMVQ_QUALIFICATION", "-DFAK_CUDA_TOKENOMOS_STAGING_QUALIFICATION", "-DFAK_CUDA_PAGED_KV_COPY_QUALIFICATION"}
	originalArgs := slices.Clone(args)
	for _, enabled := range []bool{false, true} {
		ctx := context.WithValue(context.Background(), cudaQ56KMMQKey{}, enabled)
		got, err := cudaQ56KMMQFlags(ctx, args, nil)
		want := append([]string(nil), args...)
		if enabled {
			want = append(want, macro)
		}
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("enabled=%v flags=%v err=%v", enabled, got, err)
		}
		if !slices.Equal(args, originalArgs) {
			t.Fatal("mutated peer flags")
		}
		for _, key := range []string{"NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "Nvcc_Append_Flags"} {
			for _, ambient := range []string{"-O0", macro, "-UFAK_CUDA_Q56K_MMQ_QUALIFICATION"} {
				_, err = cudaQ56KMMQFlags(ctx, args, []string{key + "=" + ambient})
				wantError := enabled || ambient != "-O0"
				if (err != nil) != wantError {
					t.Fatalf("enabled=%v ambient=%s err=%v", enabled, ambient, err)
				}
			}
		}
	}
	for _, arch := range []string{"", "sm_89", "sm_120"} {
		cfg := &CUDAConfig{Arch: arch}
		ApplyCUDAV41Defaults(cfg)
		if cfg.EnableQ56KMMQQualification {
			t.Fatal("ordinary architecture defaults selected qualification")
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAQ56KMMQPrerequisites(t *testing.T) {
	cfg := &CUDAConfig{RepoRoot: t.TempDir()}
	if err := prepareCUDAQ56KMMQ(cfg); err != nil {
		t.Fatal("disabled route inspected optional sources:", err)
	}
	cfg.EnableQ56KMMQQualification = true
	if prepareCUDAQ56KMMQ(cfg) == nil {
		t.Fatal("missing native prerequisites accepted")
	}
	files := map[string]string{
		"cuda_kernels.cu":   "#include \"cuda/q56k_mmq.cuh\"",
		"cuda/q56k_mmq.cuh": "vec_dot_q6_K_q8_1_mma load_tiles_q5_K fcuda_q56k_mmq_matmul_checked_f32",
		"cuda_backend.h":    "fcuda_q56k_mmq_preflight",
		"cuda_q56k_mmq.go":  "C.fcuda_q56k_mmq_preflight C.fcuda_q56k_mmq_matmul_checked_f32",
		"cuda.go":           "c.validateQ56KProjectionOperands",
		"cuda_kernels.go":   "c.qwen35Q56KMMQFenceLocked",
	}
	write := func(name, body string) {
		t.Helper()
		path := filepath.Join(cfg.RepoRoot, "internal", "compute", name)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	for name, body := range files {
		write(name, body)
	}
	if err := prepareCUDAQ56KMMQ(cfg); err != nil {
		t.Fatal(err)
	}
	// Marker presence is only admission plumbing, not compilation or correctness.
	for name, body := range files {
		write(name, "")
		if prepareCUDAQ56KMMQ(cfg) == nil {
			t.Fatalf("missing prerequisite %s accepted", name)
		}
		write(name, body)
	}
	cfg.PkgDir = t.TempDir()
	if err := prepareCUDAQ56KMMQ(cfg); err == nil || !strings.Contains(err.Error(), "canonical") {
		t.Fatalf("different package directory qualified: %v", err)
	}
	cfg.PkgDir = filepath.Join(cfg.RepoRoot, "internal", "compute")
	if err := prepareCUDAQ56KMMQ(cfg); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAQ56KMMQBothCompilerBranches(t *testing.T) {
	data, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	if strings.Count(source, "nvccArgs, err = cudaQ56KMMQFlags(ctx, nvccArgs, env)") != 2 ||
		!strings.Contains(source, "prepareCUDAQ56KMMQ(cfg)") ||
		!strings.Contains(source, "context.WithValue(ctx, cudaQ56KMMQKey{}, cfg.EnableQ56KMMQQualification)") {
		t.Fatal("typed selection must reach RunCUDA and both native compiler branches")
	}
}

// Uses the existing synthetic compiler helper, never nvcc or a CUDA device.
// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime slow est=10s
func TestCUDAQ56KMMQReceiptCache(t *testing.T) {
	t.Setenv("NVCC_PREPEND_FLAGS", "")
	t.Setenv("NVCC_APPEND_FLAGS", "")
	root, tc := cudaIncrementalFixture(t)
	compile := func(enabled bool, action string) string {
		t.Helper()
		ctx := context.WithValue(context.Background(), cudaQ56KMMQKey{}, enabled)
		if _, err := CompileCUDAKernels(ctx, tc, root, "sm_120", false, io.Discard); err != nil {
			t.Fatal(err)
		}
		receipt := cudaReadNativeReceipt(t, root)
		if len(receipt.Objects) != 1 || receipt.Objects[0].Action != action {
			t.Fatalf("wrong native action: %+v", receipt.Objects)
		}
		object := receipt.Objects[0]
		selected := slices.Contains(object.EffectiveFlags, "-DFAK_CUDA_Q56K_MMQ_QUALIFICATION")
		if selected != enabled || object.InputSHA256 == "" {
			t.Fatalf("receipt lost typed selection: %+v", object)
		}
		return object.InputSHA256
	}
	baseline := compile(false, "executed")
	selected := compile(true, "executed")
	if selected == baseline || compile(true, "reused") != selected {
		t.Fatal("qualification did not change and then preserve the cache identity")
	}
	if compile(false, "executed") != baseline {
		t.Fatal("default build did not recover its original key")
	}
}
