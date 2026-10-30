package devcmd

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDAV41TailRoPEFlag(t *testing.T) {
	for _, tc := range []struct {
		name                               string
		args                               []string
		tailRoPE, router, compressor, pool bool
	}{
		{"default", nil, false, false, false, false},
		{"tailRoPE", []string{"--v41-tail-rope"}, true, false, false, false},
		{"compressor", []string{"--v41-tail-rope", "--v41-router", "--v41-compressor"}, true, true, true, false},
		{"pool", []string{"--v41-tail-rope", "--v41-compressor-pool"}, true, false, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, errout bytes.Buffer
			called := false
			args := append([]string{"build", "--arch", "sm_120", "--v41-defaults=false"}, tc.args...)
			code := runBuildCUDA(&out, &errout, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
				called = true
				if cfg.EnableV41TailRoPE != tc.tailRoPE || cfg.EnableV41Router != tc.router || cfg.EnableV41Compressor != tc.compressor || cfg.EnableV41CompressorPool != tc.pool {
					t.Fatalf("incorrect selection config: %+v", cfg)
				}
				return nil
			})
			if code != 0 || !called {
				t.Fatalf("code=%d called=%v stderr=%s", code, called, errout.String())
			}
		})
	}
	data, err := os.ReadFile("build_cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "if cfg.EnableV41TailRoPE {\n\t\ttags += \",fak_cuda_v41_tail_rope\"\n\t}") {
		t.Fatal("acceptance family tags lost tailRoPE selection")
	}
}
