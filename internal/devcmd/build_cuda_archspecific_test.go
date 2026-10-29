package devcmd

import (
	"bytes"
	"context"
	"github.com/anthony-chaudhary/fak/internal/computebuild"
	"testing"
)

func TestBuildCUDAArchSpecificFlag(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		var out, errout bytes.Buffer
		args := []string{"build"}
		if enabled {
			args = append(args, "--arch-specific")
		}
		called := false
		code := runBuildCUDA(&out, &errout, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			called = true
			if cfg.EnableArchSpecific != enabled {
				t.Fatalf("flag=%v config=%v", enabled, cfg.EnableArchSpecific)
			}
			return nil
		})
		if code != 0 || !called {
			t.Fatalf("code=%d called=%v stderr=%s", code, called, errout.String())
		}
	}
}
