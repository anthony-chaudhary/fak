package devcmd

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDAMMQPrefillFlags(t *testing.T) {
	t.Setenv(computebuild.CUDAV41DefaultsEnv, "")
	for mask := 0; mask < 8; mask++ {
		args := []string{"build", "--arch", "sm_120"}
		for bit, flag := range []string{"--q2k-mmq-prefill", "--q3k-mmq-prefill", "--q8-mmq-prefill"} {
			if mask&(1<<bit) != 0 {
				args = append(args, flag)
			}
		}
		var out, errout bytes.Buffer
		called := false
		code := runBuildCUDA(&out, &errout, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			called = true
			if cfg.EnableQ2KMMQPrefill != (mask&1 != 0) || cfg.EnableQ3KMMQPrefill != (mask&2 != 0) || cfg.EnableQ8MMQPrefill != (mask&4 != 0) {
				t.Fatalf("mask %d lost explicit selection: %+v", mask, cfg)
			}
			if !cfg.EnableV41Compressor || !cfg.EnableV41Router || !cfg.EnableV41SharedAttention {
				t.Fatal("MMQ selection changed existing SM120 defaults")
			}
			return nil
		})
		if code != 0 || !called {
			t.Fatalf("mask %d code=%d called=%v stderr=%s", mask, code, called, errout.String())
		}
	}
}
