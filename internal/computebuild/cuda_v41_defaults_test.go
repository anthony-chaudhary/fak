package computebuild

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// The default sm_120 set is the closed device-witnessed list; the ds4-port
// compressor is the default and yields to an explicit pool-only selection.
func TestCUDAV41DefaultTags(t *testing.T) {
	want := []string{"fak_cuda_v41_compressor", "fak_cuda_v41_router", "fak_cuda_v41_shared_attention"}
	if got := CUDAV41DefaultTags(); !reflect.DeepEqual(got, want) {
		t.Fatalf("default tags = %v, want %v", got, want)
	}
	for _, tc := range []struct {
		name    string
		cfg     CUDAConfig
		applied []string
		tags    string
	}{
		{"sm_120", CUDAConfig{Arch: "sm_120"}, want, "cuda,fak_cuda_v41_compressor,fak_cuda_v41_router,fak_cuda_v41_shared_attention"},
		{"bare 120", CUDAConfig{Arch: "120"}, want, "cuda,fak_cuda_v41_compressor,fak_cuda_v41_router,fak_cuda_v41_shared_attention"},
		{"opt out", CUDAConfig{Arch: "sm_120", DisableV41Defaults: true}, nil, "cuda"},
		{"fat build", CUDAConfig{}, nil, "cuda"},
		{"sm_89", CUDAConfig{Arch: "sm_89"}, nil, "cuda"},
		{"pool explicit", CUDAConfig{Arch: "sm_120", EnableV41CompressorPool: true}, want[1:], "cuda,fak_cuda_v41_compressor_pool,fak_cuda_v41_router,fak_cuda_v41_shared_attention"},
		{"router explicit", CUDAConfig{Arch: "sm_120", EnableV41Router: true}, []string{want[0], want[2]}, "cuda,fak_cuda_v41_compressor,fak_cuda_v41_router,fak_cuda_v41_shared_attention"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			got := ApplyCUDAV41Defaults(&cfg)
			if !reflect.DeepEqual(got, tc.applied) || !reflect.DeepEqual(cfg.V41Defaults, tc.applied) {
				t.Fatalf("applied = %v recorded = %v, want %v", got, cfg.V41Defaults, tc.applied)
			}
			if cfg.EnableV41Compressor && cfg.EnableV41CompressorPool {
				t.Fatal("defaults selected both compressor tags")
			}
			if tags := CUDAGoTags(&cfg); tags != tc.tags {
				t.Fatalf("tags = %q, want %q", tags, tc.tags)
			}
			if again := ApplyCUDAV41Defaults(&cfg); again != nil {
				t.Fatalf("second apply = %v, want idempotent nil", again)
			}
		})
	}
	ctx := context.WithValue(context.Background(), cudaV41DefaultsKey{}, want)
	if r := newCUDANativeReceipt(ctx, t.TempDir(), "sm_120", false); !reflect.DeepEqual(r.V41Defaults, want) {
		t.Fatalf("receipt v41_defaults = %v", r.V41Defaults)
	}
	if r := newCUDANativeReceipt(context.Background(), t.TempDir(), "sm_120", false); r.V41Defaults != nil {
		t.Fatalf("receipt without defaults = %v", r.V41Defaults)
	}
	for _, tag := range want {
		if !strings.HasPrefix(tag, "fak_cuda_v41_") {
			t.Fatalf("tag %q outside the V4.1 family", tag)
		}
	}
}
