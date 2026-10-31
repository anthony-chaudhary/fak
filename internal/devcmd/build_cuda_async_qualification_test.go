package devcmd

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDAAsyncQualificationFlags(t *testing.T) {
	for _, tc := range []struct {
		args  []string
		valid bool
	}{
		{nil, true},
		{[]string{"--v41-compressor", "--v41-compressor-pending", "--q4k-qkv-qualification", "--q4k-qkv-async"}, true},
		{[]string{"--v41-compressor-pending"}, false},
		{[]string{"--q4k-qkv-async"}, false},
		{[]string{"--v41-compressor-pool", "--v41-compressor-pending"}, false},
	} {
		args := append([]string{"build", "--arch", "sm_120", "--v41-defaults=false", "--capture-once-qualification", "--q2k-mmq-prefill"}, tc.args...)
		var out, stderr bytes.Buffer
		called := false
		code := runBuildCUDA(&out, &stderr, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			called = true
			if !cfg.EnableCaptureOnceQualification || !cfg.EnableQ2KMMQPrefill {
				t.Fatal("lost peer fields")
			}
			selected := len(tc.args) > 0
			if cfg.EnableV41CompressorPending != selected || cfg.EnableQ4KQKVQualification != selected || cfg.EnableQ4KQKVAsync != selected {
				t.Fatal("lost explicit profile")
			}
			return nil
		})
		if called != tc.valid || (code == 0) != tc.valid {
			t.Fatalf("args=%v code=%d called=%v stderr=%s", args, code, called, stderr.String())
		}
	}
}
