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
func TestCUDAQ56KMMVQTypedSelection(t *testing.T) {
	const macro = "-DFAK_CUDA_Q56K_MMVQ_QUALIFICATION"
	args := []string{"-O3", "-DFAK_CUDA_Q2K_MMQ_PREFILL"}
	for _, enabled := range []bool{false, true} {
		ctx := context.WithValue(context.Background(), cudaQ56KMMVQKey{}, enabled)
		got, err := cudaQ56KMMVQFlags(ctx, args, nil)
		want := append([]string(nil), args...)
		if enabled {
			want = append(want, macro)
		}
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("enabled=%v flags=%v err=%v", enabled, got, err)
		}
		if len(args) != 2 || args[1] != "-DFAK_CUDA_Q2K_MMQ_PREFILL" {
			t.Fatal("mutated peer flags")
		}
		for _, key := range []string{"NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "Nvcc_Append_Flags"} {
			for _, ambient := range []string{"-O0", macro, "-UFAK_CUDA_Q56K_MMVQ_QUALIFICATION"} {
				_, err = cudaQ56KMMVQFlags(ctx, args, []string{key + "=" + ambient})
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
		if cfg.EnableQ56KMMVQQualification {
			t.Fatal("ordinary architecture defaults selected qualification")
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAQ56KMMVQPrerequisites(t *testing.T) {
	cfg := &CUDAConfig{RepoRoot: t.TempDir()}
	if err := prepareCUDAQ56KMMVQ(cfg); err != nil {
		t.Fatal("disabled route inspected optional sources:", err)
	}
	cfg.EnableQ56KMMVQQualification = true
	if prepareCUDAQ56KMMVQ(cfg) == nil {
		t.Fatal("missing native prerequisites accepted")
	}
	files := map[string]string{
		"cuda_kernels.cu":       "#include \"cuda/q4k_mmvq.cuh\"",
		"cuda/q4k_mmvq.cuh":     "struct dot_q5\nstruct dot_q6",
		"cuda/quant_matmul.cuh": "fcuda_q56k_mmvq_preflight",
		"cuda_backend.h":        "fcuda_q6k_matmul_checked_f32",
		"cuda.go":               "validateQ56KMMVQOperands",
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
	if err := prepareCUDAQ56KMMVQ(cfg); err != nil {
		t.Fatal(err)
	}
	// Marker presence is only admission plumbing, not compilation or correctness.
	for name, body := range files {
		write(name, "")
		if prepareCUDAQ56KMMVQ(cfg) == nil {
			t.Fatalf("missing prerequisite %s accepted", name)
		}
		write(name, body)
	}
	cfg.PkgDir = t.TempDir()
	if err := prepareCUDAQ56KMMVQ(cfg); err == nil || !strings.Contains(err.Error(), "canonical") {
		t.Fatalf("different package directory qualified: %v", err)
	}
	cfg.PkgDir = filepath.Join(cfg.RepoRoot, "internal", "compute")
	if err := prepareCUDAQ56KMMVQ(cfg); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAQ56KMMVQBothCompilerBranches(t *testing.T) {
	data, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	if strings.Count(source, "nvccArgs, err = cudaQ56KMMVQFlags(ctx, nvccArgs, env)") != 2 ||
		!strings.Contains(source, "prepareCUDAQ56KMMVQ(cfg)") ||
		!strings.Contains(source, "context.WithValue(ctx, cudaQ56KMMVQKey{}, cfg.EnableQ56KMMVQQualification)") {
		t.Fatal("typed selection must reach RunCUDA and both native compiler branches")
	}
}

// Uses the existing synthetic compiler helper, never nvcc or a CUDA device.
// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime slow est=10s
func TestCUDAQ56KMMVQReceiptCache(t *testing.T) {
	t.Setenv("NVCC_PREPEND_FLAGS", "")
	t.Setenv("NVCC_APPEND_FLAGS", "")
	root, tc := cudaIncrementalFixture(t)
	compile := func(enabled bool, action string) string {
		t.Helper()
		ctx := context.WithValue(context.Background(), cudaQ56KMMVQKey{}, enabled)
		if _, err := CompileCUDAKernels(ctx, tc, root, "sm_120", false, io.Discard); err != nil {
			t.Fatal(err)
		}
		receipt := cudaReadNativeReceipt(t, root)
		if len(receipt.Objects) != 1 || receipt.Objects[0].Action != action {
			t.Fatalf("wrong native action: %+v", receipt.Objects)
		}
		object := receipt.Objects[0]
		selected := slices.Contains(object.EffectiveFlags, "-DFAK_CUDA_Q56K_MMVQ_QUALIFICATION")
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
