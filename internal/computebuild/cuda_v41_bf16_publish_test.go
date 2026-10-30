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
func TestCUDAV41BF16PublishBuildContract(t *testing.T) {
	cfg := &CUDAConfig{}
	base := []string{"-O3", "-std=c++17"}
	strict := []string{"--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=false"}
	if cfg.EnableV41BF16Publish || cudaV41BF16PublishTags(cfg, "cuda,nccl") != "cuda,nccl" {
		t.Fatal("bf16_publish must default off")
	}
	if got, err := cudaV41BF16PublishFlags(context.Background(), base, []string{"NVCC_APPEND_FLAGS=--use_fast_math"}); err != nil || !reflect.DeepEqual(got, base) {
		t.Fatalf("default flags changed: %v %v", got, err)
	}
	cfg.EnableV41BF16Publish = true
	ctx := context.WithValue(context.Background(), cudaV41BF16PublishKey{}, true)
	want := append(append(append([]string(nil), base...), "-DFAK_CUDA_V41_BF16_PUBLISH"), strict...)
	if got, err := cudaV41BF16PublishFlags(ctx, base, nil); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("flags=%v err=%v want=%v", got, err, want)
	}
	if got := cudaV41BF16PublishTags(cfg, "cuda,nccl"); got != "cuda,nccl,fak_cuda_v41_bf16_publish" {
		t.Fatal(got)
	}
	for _, bad := range []string{"--use_fast_math", "-use_fast_math", "--ftz=true", "--ftz true", "--prec-div=false", "--prec-sqrt false", "--fmad=true", "--fmad true"} {
		if _, err := cudaV41BF16PublishFlags(ctx, append(append([]string(nil), base...), strings.Fields(bad)...), nil); err == nil {
			t.Fatalf("accepted args %q", bad)
		}
		for _, key := range []string{"NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"} {
			if _, err := cudaV41BF16PublishFlags(ctx, base, []string{key + "=" + bad}); err == nil {
				t.Fatalf("accepted %s=%q", key, bad)
			}
		}
	}
	if _, err := cudaV41BF16PublishFlags(ctx, base, []string{"NVCC_APPEND_FLAGS=" + strings.Join(strict, " ")}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	cfg.RepoRoot = root
	fixture := map[string]string{
		"cuda_arch.txt": "sm_120\n",
		"cuda_kernels.cu": "#include \"cuda/model_candidates.cuh\"\n" +
			"if (g_v41_bf16_uncertain) return true;\n",
		"cuda/model_candidates.cuh": "#ifdef FAK_CUDA_V41_BF16_PUBLISH\n" +
			"extern \"C\" int fak_v41_bf16_publish_f32(\ng_v41_bf16_uncertain = true;\n",
		"cuda_v41_bf16_publish.go": "//go:build cuda && cgo && fak_cuda_v41_bf16_publish\n",
		"cuda_v41_bf16_checked.go": "//go:build cuda && cgo && fak_cuda_v41_bf16_publish\n",
		"cuda/runtime_memory.cuh": "if (g_v41_bf16_uncertain) return;\n" +
			"if (g_v41_bf16_uncertain) return (int)cudaErrorInvalidValue;\n" +
			"extern \"C\" size_t fcuda_panel_release(void) {\n  if (v41_candidates_stream_uncertain()) return g_panel_live.size();\n" +
			"extern \"C\" void fcuda_trim_pool_large(size_t max_keep_bytes) {\n  if (v41_candidates_stream_uncertain()) return;\n",
		"cuda/runtime_services.cuh": "if (g_v41_bf16_uncertain) return;\n",
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
		if err := prepareCUDAV41BF16Publish(cfg); err == nil {
			t.Fatalf("accepted arch %q", arch)
		}
	}
	for _, arch := range []string{"sm_120", "120"} {
		cfg.Arch = arch
		if err := prepareCUDAV41BF16Publish(cfg); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"cuda_kernels.cu", "cuda/model_candidates.cuh", "cuda_v41_bf16_publish.go", "cuda_v41_bf16_checked.go", "cuda/runtime_memory.cuh", "cuda/runtime_services.cuh"} {
		path := filepath.Join(root, "internal", "compute", name)
		if err := os.WriteFile(path, []byte("unpaired"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41BF16Publish(cfg); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("unpaired source %s: %v", name, err)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41BF16Publish(cfg); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("missing source %s: %v", name, err)
		}
		if err := os.WriteFile(path, []byte(fixture[name]), 0644); err != nil {
			t.Fatal(err)
		}
	}
	// Each of the five post-failure cleanup guards is independently required.
	for _, check := range []struct{ path, marker string }{
		{"cuda/runtime_memory.cuh", "if (g_v41_bf16_uncertain) return;"},
		{"cuda/runtime_memory.cuh", "if (g_v41_bf16_uncertain) return (int)cudaErrorInvalidValue;"},
		{"cuda/runtime_services.cuh", "if (g_v41_bf16_uncertain) return;"},
		{"cuda/runtime_memory.cuh", "if (v41_candidates_stream_uncertain()) return g_panel_live.size();"},
		{"cuda/runtime_memory.cuh", "if (v41_candidates_stream_uncertain()) return;"},
	} {
		path := filepath.Join(root, "internal", "compute", check.path)
		if err := os.WriteFile(path, []byte(strings.Replace(fixture[check.path], check.marker, "", 1)), 0644); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41BF16Publish(cfg); err == nil {
			t.Fatalf("accepted missing cleanup guard %q", check.marker)
		}
		if err := os.WriteFile(path, []byte(fixture[check.path]), 0644); err != nil {
			t.Fatal(err)
		}
	}
	tagPath := filepath.Join(root, "internal", "compute", "cuda_v41_bf16_publish.go")
	for _, tag := range []string{"//go:build cuda && cgo && fak_cuda_v41_bf16_publish_test\n", "//go:build cuda && cgo && fak_cuda_v41_bf16_publish\r\n"} {
		if err := os.WriteFile(tagPath, []byte(tag), 0644); err != nil {
			t.Fatal(err)
		}
		err := prepareCUDAV41BF16Publish(cfg)
		if (err != nil) != strings.Contains(tag, "_test") {
			t.Fatalf("tag %q: %v", tag, err)
		}
	}
	for _, enabled := range []bool{false, true} {
		ctx := context.WithValue(context.Background(), cudaV41BF16PublishKey{}, enabled)
		r := newCUDANativeReceipt(ctx, root, "sm_120", false)
		if r.V41BF16Publish != enabled || r.V41Compressor || r.V41Router || r.V41TailRoPE || r.Outcome != "failed" || r.Mode != "not_started" || r.Executed != 0 {
			t.Fatalf("selection receipt=%+v", r)
		}
	}
	data, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(data), "nvccArgs, err = cudaV41BF16PublishFlags(ctx, nvccArgs, env)") != 2 || !strings.Contains(string(data), "goTags = cudaV41BF16PublishTags(cfg, goTags)") {
		t.Fatal("bf16_publish selection must reach both native compiler paths and Go build tags")
	}
}
