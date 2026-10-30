package computebuild

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestCUDAV41CompressorBuildContract(t *testing.T) {
	cfg := &CUDAConfig{}
	base := []string{"-O3", "-std=c++17"}
	strict := []string{"--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=false"}
	if cfg.EnableV41Compressor || cudaV41CompressorTags(cfg, "cuda,nccl") != "cuda,nccl" {
		t.Fatal("compressor must default off")
	}
	if got, err := cudaV41CompressorFlags(context.Background(), base, []string{"NVCC_APPEND_FLAGS=--use_fast_math"}); err != nil || !reflect.DeepEqual(got, base) {
		t.Fatalf("default flags changed: %v %v", got, err)
	}
	cfg.EnableV41Compressor = true
	ctx := context.WithValue(context.Background(), cudaV41CompressorKey{}, true)
	want := append(append(append([]string(nil), base...), "-DFAK_CUDA_V41_COMPRESSOR"), strict...)
	if got, err := cudaV41CompressorFlags(ctx, base, nil); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("flags=%v err=%v want=%v", got, err, want)
	}
	if got := cudaV41CompressorTags(cfg, "cuda,nccl"); got != "cuda,nccl,fak_cuda_v41_compressor" {
		t.Fatal(got)
	}
	for _, bad := range []string{"--use_fast_math", "-use_fast_math", "--ftz=true", "--ftz true", "--prec-div=false", "--prec-sqrt false", "--fmad=true", "--fmad true"} {
		if _, err := cudaV41CompressorFlags(ctx, append(append([]string(nil), base...), strings.Fields(bad)...), nil); err == nil {
			t.Fatalf("accepted args %q", bad)
		}
		for _, key := range []string{"NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"} {
			if _, err := cudaV41CompressorFlags(ctx, base, []string{key + "=" + bad}); err == nil {
				t.Fatalf("accepted %s=%q", key, bad)
			}
		}
	}
	if _, err := cudaV41CompressorFlags(ctx, base, []string{"NVCC_APPEND_FLAGS=" + strings.Join(strict, " ")}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	cfg.RepoRoot = root
	fixture := map[string]string{
		"cuda_arch.txt": "sm_120\n",
		"cuda_kernels.cu": "#include \"cuda/v41_compressor.cuh\"\n" +
			"if (g_v41_compressor_uncertain) return true;\n",
		"cuda/v41_compressor.cuh": "#ifdef FAK_CUDA_V41_COMPRESSOR\n#endif\n",
		"cuda_v41_compressor.go":  "//go:build cuda && cgo && fak_cuda_v41_compressor && !fak_cuda_v41_compressor_pool\n",
		"cuda/runtime_memory.cuh": "if (g_v41_compressor_uncertain) return;\n" +
			"if (g_v41_compressor_uncertain) return (int)cudaErrorInvalidValue;\n",
		"cuda/runtime_services.cuh": "if (g_v41_compressor_uncertain) return;\n",
	}
	for name, data := range fixture {
		path := filepath.Join(root, "internal", "compute", name)
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
	}
	for _, arch := range []string{"", "sm_100a", "sm_103f"} {
		cfg.Arch = arch
		if err := prepareCUDAV41Compressor(cfg); err == nil {
			t.Fatalf("accepted arch %q", arch)
		}
	}
	for _, arch := range []string{"sm_120", "120"} {
		cfg.Arch = arch
		if err := prepareCUDAV41Compressor(cfg); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"cuda_kernels.cu", "cuda/v41_compressor.cuh", "cuda_v41_compressor.go", "cuda/runtime_memory.cuh", "cuda/runtime_services.cuh"} {
		path := filepath.Join(root, "internal", "compute", name)
		if err := os.WriteFile(path, []byte("unpaired"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41Compressor(cfg); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("unpaired source %s: %v", name, err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41Compressor(cfg); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("missing source %s: %v", name, err)
		}
		if err := os.WriteFile(path, []byte(fixture[name]), 0644); err != nil {
			t.Fatal(err)
		}
	}
	for _, enabled := range []bool{false, true} {
		ctx := context.WithValue(context.Background(), cudaV41CompressorKey{}, enabled)
		r := newCUDANativeReceipt(ctx, root, "sm_120", false)
		if r.V41Compressor != enabled || r.V41CompressorPool || r.Outcome != "failed" || r.Mode != "not_started" || r.Executed != 0 {
			t.Fatalf("selection receipt=%+v", r)
		}
	}
	conflict := &CUDAConfig{EnableV41Compressor: true, EnableV41CompressorPool: true}
	if err := RunCUDA(context.Background(), conflict); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("conflicting config reached repository/toolchain discovery: %v", err)
	}
	if err := prepareCUDAV41Compressor(conflict); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("conflicting source preparation: %v", err)
	}
	data, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(data), "nvccArgs, err = cudaV41CompressorFlags(ctx, nvccArgs, env)") != 2 || !strings.Contains(string(data), "goTags = cudaV41CompressorTags(cfg, goTags)") {
		t.Fatal("compressor selection must reach both native compiler paths and Go build tags")
	}
}
