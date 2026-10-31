package devcmd

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDAV41CompressorFlags(t *testing.T) {
	for _, tc := range []struct {
		name                       string
		args                       []string
		compressor, pool, conflict bool
	}{
		{"default", nil, false, false, false},
		{"compressor", []string{"--v41-compressor"}, true, false, false},
		{"pool", []string{"--v41-compressor-pool"}, false, true, false},
		{"conflict", []string{"--v41-compressor", "--v41-compressor-pool"}, false, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, errout bytes.Buffer
			called := false
			args := append([]string{"build", "--arch", "sm_120", "--v41-defaults=false"}, tc.args...)
			code := runBuildCUDA(&out, &errout, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
				called = true
				if cfg.EnableV41Compressor != tc.compressor || cfg.EnableV41CompressorPool != tc.pool {
					t.Fatalf("incorrect candidate config: %+v", cfg)
				}
				// Check the canonical emitted tag, not a duplicate source-text implementation.
				tags := "," + computebuild.CUDAGoTags(cfg) + ","
				if got := strings.Contains(tags, ",fak_cuda_v41_compressor,"); got != tc.compressor {
					t.Fatalf("canonical tags %q lost %s selection=%t", tags, "compressor", tc.compressor)
				}
				if got := strings.Contains(tags, ",fak_cuda_v41_compressor_pool,"); got != tc.pool {
					t.Fatalf("canonical tags %q lost pool selection=%t", tags, tc.pool)
				}
				return nil
			})
			if tc.conflict {
				if code != 2 || called || !strings.Contains(errout.String(), "mutually exclusive") {
					t.Fatalf("code=%d called=%v stderr=%s", code, called, errout.String())
				}
			} else if code != 0 || !called {
				t.Fatalf("code=%d called=%v stderr=%s", code, called, errout.String())
			}
		})
	}
}
