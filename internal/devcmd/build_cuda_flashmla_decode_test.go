package devcmd

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// Real CLI parsing/configuration via the established injected runner seam.
// Native architecture and donor admission remain in the unchanged builder.
// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDAFlashMLADecodeFlag(t *testing.T) {
	for mask := 0; mask < 4; mask++ {
		args := []string{"build", "--arch", "sm_100", "--arch-specific", "--v41-defaults=false"}
		if mask&1 != 0 {
			args = append(args, "--flashmla-prefill")
		}
		if mask&2 != 0 {
			args = append(args, "--flashmla-decode")
		}
		if mask != 0 {
			args = append(args, "--flashmla-disable-ld-red")
		}
		var out, stderr bytes.Buffer
		called := false
		code := runBuildCUDA(&out, &stderr, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			called = true
			if cfg.EnableFlashMLADecode != (mask&2 != 0) || cfg.EnableFlashMLAPrefill != (mask&1 != 0) {
				t.Fatalf("mask=%d lost or implicitly enabled FlashMLA selection", mask)
			}
			if cfg.Arch != "sm_100" || !cfg.EnableArchSpecific || !cfg.DisableV41Defaults || cfg.DisableFlashMLALDRed != (mask != 0) {
				t.Fatal("changed explicit architecture/defaults/LD.RED configuration")
			}
			tags := computebuild.CUDAGoTags(cfg)
			if strings.Contains(tags, "fak_cuda_flashmla_decode") != (mask&2 != 0) || strings.Contains(tags, "fak_cuda_flashmla_prefill") != (mask&1 != 0) {
				t.Fatalf("mask=%d tags=%s", mask, tags)
			}
			return nil
		})
		if code != 0 || !called {
			t.Fatalf("mask=%d code=%d called=%v stderr=%s", mask, code, called, stderr.String())
		}
	}
}
