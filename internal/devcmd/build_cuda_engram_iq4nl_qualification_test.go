package devcmd

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDAEngramIQ4NLQualificationFlag(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		args := []string{"build", "--arch", "sm_120", "--v41-defaults=false", "--v41-compressor", "--v41-compressor-pending", "--q4k-qkv-qualification", "--q4k-qkv-async", "--capture-once-qualification", "--q2k-mmq-prefill"}
		if enabled {
			args = append(args, "--v41-engram-iq4nl-qualification")
		}
		var out, stderr bytes.Buffer
		called := false
		code := runBuildCUDA(&out, &stderr, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			called = true
			if cfg.EnableV41EngramIQ4NLQualification != enabled {
				t.Fatal("lost explicit Engram selection")
			}
			if !cfg.EnableV41CompressorPending || !cfg.EnableQ4KQKVAsync || !cfg.EnableCaptureOnceQualification || !cfg.EnableQ2KMMQPrefill {
				t.Fatal("lost peer selections")
			}
			return nil
		})
		if code != 0 || !called {
			t.Fatalf("enabled=%v code=%d called=%v stderr=%s", enabled, code, called, stderr.String())
		}
	}
}
