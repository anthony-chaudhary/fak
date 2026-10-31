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
func TestCUDATokenomosStagingTypedSelection(t *testing.T) {
	const macro = "-DFAK_CUDA_TOKENOMOS_STAGING_QUALIFICATION"
	args := []string{"-O3", "-DFAK_CUDA_Q2K_MMQ_PREFILL"}
	for _, enabled := range []bool{false, true} {
		ctx := context.WithValue(context.Background(), cudaTokenomosStagingKey{}, enabled)
		got, err := cudaTokenomosStagingFlags(ctx, args, nil)
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
			for _, ambient := range []string{"-O0", macro, "-UFAK_CUDA_TOKENOMOS_STAGING_QUALIFICATION"} {
				_, err = cudaTokenomosStagingFlags(ctx, args, []string{key + "=" + ambient})
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
		if cfg.EnableTokenomosStagingQualification {
			t.Fatal("ordinary architecture defaults selected qualification")
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDATokenomosStagingPrerequisites(t *testing.T) {
	cfg := &CUDAConfig{RepoRoot: t.TempDir()}
	if err := prepareCUDATokenomosStaging(cfg); err != nil {
		t.Fatal("disabled route inspected optional sources:", err)
	}
	cfg.EnableTokenomosStagingQualification = true
	if prepareCUDATokenomosStaging(cfg) == nil {
		t.Fatal("missing native prerequisites accepted")
	}
	files := map[string]string{
		"cuda_kernels.cu":           "#include \"cuda/model_candidates.cuh\"",
		"cuda/model_candidates.cuh": "#ifdef FAK_CUDA_TOKENOMOS_STAGING_QUALIFICATION\n#include \"tokenomos_staging.h\"\nfak_tokenomos_staging_copy(dIDs, hIDs, idsBytes)\nfak_tokenomos_staging_error(launch)",
		"cuda/tokenomos_staging.h":  "struct fak_tokenomos_staging_pool\ncudaEventSynchronize(a.done)\ng_cuda_cleanup_status",
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
	if err := prepareCUDATokenomosStaging(cfg); err != nil {
		t.Fatal(err)
	}
	// Marker presence is only admission plumbing, not compilation or correctness.
	for name, body := range files {
		write(name, "")
		if prepareCUDATokenomosStaging(cfg) == nil {
			t.Fatalf("missing prerequisite %s accepted", name)
		}
		write(name, body)
	}
	cfg.PkgDir = t.TempDir()
	if err := prepareCUDATokenomosStaging(cfg); err == nil || !strings.Contains(err.Error(), "canonical") {
		t.Fatalf("different package directory qualified: %v", err)
	}
	cfg.PkgDir = filepath.Join(cfg.RepoRoot, "internal", "compute")
	if err := prepareCUDATokenomosStaging(cfg); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDATokenomosStagingBothCompilerBranches(t *testing.T) {
	data, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	if strings.Count(source, "nvccArgs, err = cudaTokenomosStagingFlags(ctx, nvccArgs, env)") != 2 ||
		!strings.Contains(source, "prepareCUDATokenomosStaging(cfg)") ||
		!strings.Contains(source, "context.WithValue(ctx, cudaTokenomosStagingKey{}, cfg.EnableTokenomosStagingQualification)") {
		t.Fatal("typed selection must reach RunCUDA and both native compiler branches")
	}
}

// Uses the existing synthetic compiler helper, never nvcc or a CUDA device.
// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime slow est=10s
func TestCUDATokenomosStagingReceiptCache(t *testing.T) {
	t.Setenv("NVCC_PREPEND_FLAGS", "")
	t.Setenv("NVCC_APPEND_FLAGS", "")
	root, tc := cudaIncrementalFixture(t)
	compile := func(enabled bool, action string) string {
		t.Helper()
		ctx := context.WithValue(context.Background(), cudaTokenomosStagingKey{}, enabled)
		if _, err := CompileCUDAKernels(ctx, tc, root, "sm_120", false, io.Discard); err != nil {
			t.Fatal(err)
		}
		receipt := cudaReadNativeReceipt(t, root)
		if len(receipt.Objects) != 1 || receipt.Objects[0].Action != action {
			t.Fatalf("wrong native action: %+v", receipt.Objects)
		}
		object := receipt.Objects[0]
		selected := slices.Contains(object.EffectiveFlags, "-DFAK_CUDA_TOKENOMOS_STAGING_QUALIFICATION")
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
