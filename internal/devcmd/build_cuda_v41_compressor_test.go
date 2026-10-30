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
			args := append([]string{"build", "--arch", "sm_120"}, tc.args...)
			code := runBuildCUDA(&out, &errout, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
				called = true
				if cfg.EnableV41Compressor != tc.compressor || cfg.EnableV41CompressorPool != tc.pool {
					t.Fatalf("incorrect candidate config: %+v", cfg)
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
	data, err := os.ReadFile("build_cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "if cfg.EnableV41Compressor {\n\t\ttags += \",fak_cuda_v41_compressor\"\n\t}") {
		t.Fatal("acceptance family tags lost compressor pairing")
	}
}
