package computebuild

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=30ms lane=default
func TestCUDAV41SharedAttentionBuildContract(t *testing.T) {
	cfg := &CUDAConfig{}
	if cfg.EnableV41SharedAttention {
		t.Fatal("candidate must default off")
	}
	base := []string{"-O3", "-std=c++17"}
	if got := cudaV41SharedAttentionFlags(context.Background(), base); !reflect.DeepEqual(got, base) {
		t.Fatal(got)
	}
	if got := cudaV41SharedAttentionTags(cfg, "cuda,nccl"); got != "cuda,nccl" {
		t.Fatal(got)
	}
	cfg.EnableV41SharedAttention = true
	ctx := context.WithValue(context.Background(), cudaV41SharedAttentionKey{}, true)
	if got := cudaV41SharedAttentionFlags(ctx, base); !reflect.DeepEqual(got, []string{"-O3", "-std=c++17", "-DFAK_CUDA_V41_SHARED_ATTENTION"}) {
		t.Fatal(got)
	}
	if got := cudaV41SharedAttentionTags(cfg, "cuda,nccl"); got != "cuda,nccl,fak_cuda_v41_shared_attention" {
		t.Fatal(got)
	}
	root := t.TempDir()
	cfg.RepoRoot = root
	for _, arch := range []string{"", "sm_80", "sm_100a", "sm_103f"} {
		cfg.Arch = arch
		if err := prepareCUDAV41SharedAttention(cfg); err == nil {
			t.Fatalf("accepted non-Raider arm %q", arch)
		}
	}
	dir := filepath.Join(root, "internal", "compute")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	fixture := map[string]string{
		"cuda_arch.txt":                "sm_120\n",
		"cuda_kernels.cu":              "#include \"cuda_v41_shared_attention.cu\"\n",
		"cuda_v41_shared_attention.cu": "#ifdef FAK_CUDA_V41_SHARED_ATTENTION\n#endif\n",
		"cuda_v41_shared_attention.go": "//go:build cuda && cgo && fak_cuda_v41_shared_attention\n",
	}
	for name, data := range fixture {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
	}
	for _, arch := range []string{"sm_120", "120"} {
		cfg.Arch = arch
		if err := prepareCUDAV41SharedAttention(cfg); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"cuda_kernels.cu", "cuda_v41_shared_attention.cu", "cuda_v41_shared_attention.go"} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("unpaired source"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41SharedAttention(cfg); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("unpaired %s: %v", name, err)
		}
		if err := os.Remove(p); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41SharedAttention(cfg); err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("missing %s: %v", name, err)
		}
		if err := os.WriteFile(p, []byte(fixture[name]), 0644); err != nil {
			t.Fatal(err)
		}
	}
	if r := newCUDANativeReceipt(ctx, root, "sm_120", false); !r.V41SharedAttention {
		t.Fatal("receipt omitted opt-in")
	}
	if r := newCUDANativeReceipt(context.Background(), root, "sm_120", false); r.V41SharedAttention {
		t.Fatal("receipt opted in by default")
	}
}
