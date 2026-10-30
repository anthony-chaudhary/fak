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
func TestCUDAV41RouterBuildContract(t *testing.T) {
	cfg := &CUDAConfig{}
	base := []string{"-O3", "-std=c++17"}
	strict := []string{"--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=false"}
	if cfg.EnableV41Router || cudaV41RouterTags(cfg, "cuda,nccl") != "cuda,nccl" {
		t.Fatal("router must default off")
	}
	if got, err := cudaV41RouterFlags(context.Background(), base, []string{"NVCC_APPEND_FLAGS=--use_fast_math"}); err != nil || !reflect.DeepEqual(got, base) {
		t.Fatalf("default flags changed: %v %v", got, err)
	}
	cfg.EnableV41Router = true
	ctx := context.WithValue(context.Background(), cudaV41RouterKey{}, true)
	want := append(append([]string(nil), base...), strict...)
	if got, err := cudaV41RouterFlags(ctx, base, nil); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("flags=%v err=%v want=%v", got, err, want)
	}
	if got := cudaV41RouterTags(cfg, "cuda,nccl"); got != "cuda,nccl,fak_cuda_v41_router" {
		t.Fatal(got)
	}
	for _, bad := range []string{"--use_fast_math", "-use_fast_math", "--ftz=true", "--ftz true", "--prec-div=false", "--prec-sqrt false", "--fmad=true", "--fmad true"} {
		if _, err := cudaV41RouterFlags(ctx, append(append([]string(nil), base...), strings.Fields(bad)...), nil); err == nil {
			t.Fatalf("accepted args %q", bad)
		}
		for _, key := range []string{"NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"} {
			if _, err := cudaV41RouterFlags(ctx, base, []string{key + "=" + bad}); err == nil {
				t.Fatalf("accepted %s=%q", key, bad)
			}
		}
	}
	if _, err := cudaV41RouterFlags(ctx, base, []string{"NVCC_APPEND_FLAGS=" + strings.Join(strict, " ")}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	cfg.RepoRoot = root
	fixture := map[string]string{
		"cuda_arch.txt": "sm_120\n",
		"cuda_kernels.cu": "#include \"cuda_v41_router.cu\"\n" +
			"if (g_v41_router_uncertain) return true;\n",
		"cuda_v41_router.cu": "extern \"C\" int fak_v41_router_topk_f32(\n" +
			"v41_candidates_stream_uncertain()\nv41rt_score(float z, bool full)\nif (!full && z < -20.0f)\n",
		"cuda_v41_router.go": "//go:build cuda && cgo && fak_cuda_v41_router\n",
		"cuda/runtime_memory.cuh": "if (g_v41_router_uncertain) return;\n" +
			"if (g_v41_router_uncertain) return (int)cudaErrorInvalidValue;\n",
		"cuda/runtime_services.cuh": "if (g_v41_router_uncertain) return;\n",
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
		if err := prepareCUDAV41Router(cfg); err == nil {
			t.Fatalf("accepted arch %q", arch)
		}
	}
	for _, arch := range []string{"sm_120", "120"} {
		cfg.Arch = arch
		if err := prepareCUDAV41Router(cfg); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"cuda_kernels.cu", "cuda_v41_router.cu", "cuda_v41_router.go", "cuda/runtime_memory.cuh", "cuda/runtime_services.cuh"} {
		path := filepath.Join(root, "internal", "compute", name)
		if err := os.WriteFile(path, []byte("unpaired"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41Router(cfg); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("unpaired source %s: %v", name, err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41Router(cfg); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("missing source %s: %v", name, err)
		}
		if err := os.WriteFile(path, []byte(fixture[name]), 0644); err != nil {
			t.Fatal(err)
		}
	}
	tagPath := filepath.Join(root, "internal", "compute", "cuda_v41_router.go")
	for _, tag := range []string{"//go:build cuda && cgo && fak_cuda_v41_router_topk\n", "//go:build cuda && cgo && fak_cuda_v41_router\r\n"} {
		if err := os.WriteFile(tagPath, []byte(tag), 0644); err != nil {
			t.Fatal(err)
		}
		err := prepareCUDAV41Router(cfg)
		if (err != nil) != strings.Contains(tag, "_topk") {
			t.Fatalf("tag %q: %v", tag, err)
		}
	}
	for _, enabled := range []bool{false, true} {
		ctx := context.WithValue(context.Background(), cudaV41RouterKey{}, enabled)
		r := newCUDANativeReceipt(ctx, root, "sm_120", false)
		if r.V41Router != enabled || r.V41Compressor || r.Outcome != "failed" || r.Mode != "not_started" || r.Executed != 0 {
			t.Fatalf("selection receipt=%+v", r)
		}
	}
	data, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(data), "nvccArgs, err = cudaV41RouterFlags(ctx, nvccArgs, env)") != 2 || !strings.Contains(string(data), "goTags = cudaV41RouterTags(cfg, goTags)") {
		t.Fatal("router selection must reach both native compiler paths and Go build tags")
	}
}
