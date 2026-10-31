package devcmd

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDAV41RouterFlag(t *testing.T) {
	for _, tc := range []struct {
		name                     string
		args                     []string
		router, compressor, pool bool
	}{
		{"default", nil, false, false, false},
		{"router", []string{"--v41-router"}, true, false, false},
		{"compressor", []string{"--v41-router", "--v41-compressor"}, true, true, false},
		{"pool", []string{"--v41-router", "--v41-compressor-pool"}, true, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, errout bytes.Buffer
			called := false
			args := append([]string{"build", "--arch", "sm_120", "--v41-defaults=false"}, tc.args...)
			code := runBuildCUDA(&out, &errout, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
				called = true
				if cfg.EnableV41Router != tc.router || cfg.EnableV41Compressor != tc.compressor || cfg.EnableV41CompressorPool != tc.pool {
					t.Fatalf("incorrect selection config: %+v", cfg)
				}
				// Check the canonical emitted tag, not a duplicate source-text implementation.
				tags := "," + computebuild.CUDAGoTags(cfg) + ","
				if got := strings.Contains(tags, ",fak_cuda_v41_router,"); got != tc.router {
					t.Fatalf("canonical tags %q lost %s selection=%t", tags, "router", tc.router)
				}
				return nil
			})
			if code != 0 || !called {
				t.Fatalf("code=%d called=%v stderr=%s", code, called, errout.String())
			}
		})
	}
}
