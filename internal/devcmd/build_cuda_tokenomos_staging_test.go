package devcmd

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDATokenomosStagingFlag(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want bool
	}{
		{nil, false},
		{[]string{"--tokenomos-staging-qualification"}, true},
		{[]string{"--tokenomos-staging-qualification=false"}, false},
	} {
		args := append([]string{"build", "--arch", "sm_120", "--v41-defaults=false", "--capture-once-qualification", "--q2k-mmq-prefill"}, tc.args...)
		var stdout, stderr bytes.Buffer
		called := false
		code := runBuildCUDA(&stdout, &stderr, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			called = true
			if cfg.EnableTokenomosStagingQualification != tc.want {
				t.Fatalf("args=%v selected=%v", args, cfg.EnableTokenomosStagingQualification)
			}
			if !cfg.EnableCaptureOnceQualification || !cfg.EnableQ2KMMQPrefill {
				t.Fatal("lost independent peer selections")
			}
			return nil
		})
		if code != 0 || !called {
			t.Fatalf("args=%v code=%d called=%v stderr=%s", args, code, called, stderr.String())
		}
	}
}
