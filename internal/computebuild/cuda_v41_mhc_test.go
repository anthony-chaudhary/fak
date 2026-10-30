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
func TestCUDAV41MHCBuildContract(t *testing.T) {
	cfg := &CUDAConfig{}
	base := []string{"-O3", "-std=c++17"}
	if cfg.EnableV41MHC {
		t.Fatal("candidate must default off")
	}
	if got, err := cudaV41MHCFlags(context.Background(), base, []string{"NVCC_APPEND_FLAGS=--use_fast_math"}); err != nil || !reflect.DeepEqual(got, base) {
		t.Fatalf("default changed: %v %v", got, err)
	}
	if got := cudaV41MHCTags(cfg, "cuda"); got != "cuda" {
		t.Fatal(got)
	}
	cfg.EnableV41MHC = true
	ctx := context.WithValue(context.Background(), cudaV41MHCKey{}, true)
	want := []string{"-O3", "-std=c++17", "-DFAK_CUDA_V41_MHC", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"}
	if got, err := cudaV41MHCFlags(ctx, base, nil); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("flags %v %v", got, err)
	}
	if got := cudaV41MHCTags(cfg, "cuda,fak_cuda_v41_shared_attention"); got != "cuda,fak_cuda_v41_shared_attention,fak_cuda_v41_mhc" {
		t.Fatal(got)
	}
	for _, bad := range []string{"--use_fast_math", "-use_fast_math", "--ftz=true", "--ftz true", "-prec-div=false", "--prec-sqrt=false", "--prec-div false"} {
		if _, err := cudaV41MHCFlags(ctx, append(append([]string(nil), base...), strings.Fields(bad)...), nil); err == nil {
			t.Fatalf("accepted args %q", bad)
		}
		for _, key := range []string{"NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"} {
			if _, err := cudaV41MHCFlags(ctx, base, []string{key + "=" + bad}); err == nil {
				t.Fatalf("accepted %s=%q", key, bad)
			}
		}
	}
	if _, err := cudaV41MHCFlags(ctx, base, []string{"NVCC_APPEND_FLAGS=--ftz=false --prec-div=true --prec-sqrt=true"}); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	cfg.RepoRoot = root
	for _, arch := range []string{"", "sm_100a", "sm_103f"} {
		cfg.Arch = arch
		if err := prepareCUDAV41MHC(cfg); err == nil {
			t.Fatalf("accepted arch %q", arch)
		}
	}
	dir := filepath.Join(root, "internal", "compute")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	fixture := map[string]string{
		"cuda_arch.txt":   "sm_120\n",
		"cuda_kernels.cu": "#include \"cuda_v41_mhc.cu\"\n",
		"cuda_v41_mhc.cu": "#ifdef FAK_CUDA_V41_MHC\n#endif\n",
		"cuda_v41_mhc.go": "//go:build cuda && cgo && fak_cuda_v41_mhc\n",
	}
	for name, data := range fixture {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
	}
	for _, arch := range []string{"sm_120", "120"} {
		cfg.Arch = arch
		if err := prepareCUDAV41MHC(cfg); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"cuda_kernels.cu", "cuda_v41_mhc.cu", "cuda_v41_mhc.go"} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("unpaired"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41MHC(cfg); err == nil {
			t.Fatalf("accepted unpaired %s", name)
		}
		if err := os.Remove(p); err != nil {
			t.Fatal(err)
		}
		if err := prepareCUDAV41MHC(cfg); err == nil {
			t.Fatalf("accepted absent %s", name)
		}
		if err := os.WriteFile(p, []byte(fixture[name]), 0644); err != nil {
			t.Fatal(err)
		}
	}
	if r := newCUDANativeReceipt(ctx, root, "sm_120", false); !r.V41MHC {
		t.Fatal("receipt omitted candidate")
	}
	if r := newCUDANativeReceipt(context.Background(), root, "sm_120", false); r.V41MHC {
		t.Fatal("receipt enabled by default")
	}
}
