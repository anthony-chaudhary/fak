package devcmd

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDAQ56KMMVQFlag(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want bool
	}{
		{nil, false},
		{[]string{"--q56k-mmvq-qualification"}, true},
		{[]string{"--q56k-mmvq-qualification=false"}, false},
	} {
		args := append([]string{"build", "--arch", "sm_120", "--v41-defaults=false", "--capture-once-qualification", "--q2k-mmq-prefill", "--tokenomos-staging-qualification"}, tc.args...)
		var stdout, stderr bytes.Buffer
		called := false
		code := runBuildCUDA(&stdout, &stderr, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			called = true
			if cfg.EnableQ56KMMVQQualification != tc.want {
				t.Fatalf("args=%v selected=%v", args, cfg.EnableQ56KMMVQQualification)
			}
			if !cfg.EnableCaptureOnceQualification || !cfg.EnableQ2KMMQPrefill || !cfg.EnableTokenomosStagingQualification {
				t.Fatal("lost independent peer selections")
			}
			return nil
		})
		if code != 0 || !called {
			t.Fatalf("args=%v code=%d called=%v stderr=%s", args, code, called, stderr.String())
		}
	}
}
