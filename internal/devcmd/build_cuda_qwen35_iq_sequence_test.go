package devcmd

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildCUDAQwen35IQ2SequenceFlag(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want bool
	}{
		{nil, false},
		{[]string{"--qwen35-iq2-sequence-qualification"}, true},
		{[]string{"--qwen35-iq2-sequence-qualification=false"}, false},
	} {
		args := append([]string{"build", "--arch", "sm_120", "--v41-defaults=false", "--kv16-prefill-gqa4-qualification", "--tokenomos-staging-qualification", "--paged-kv-copy-qualification", "--q56k-mmvq-qualification"}, tc.args...)
		var stdout, stderr bytes.Buffer
		called := false
		code := runBuildCUDA(&stdout, &stderr, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			called = true
			if cfg.EnableQwen35IQ2SequenceQualification != tc.want || strings.Contains(computebuild.CUDAGoTags(cfg), "fak_cuda_qwen35_iq2_sequence_qualification") != tc.want {
				t.Fatalf("args=%v selection not preserved in recorded Go tags", args)
			}
			if !cfg.EnableKV16PrefillGQA4Qualification || !cfg.EnableTokenomosStagingQualification || !cfg.EnablePagedKVCopyQualification || !cfg.EnableQ56KMMVQQualification {
				t.Fatal("lost independent peer selections")
			}
			return nil
		})
		if code != 0 || !called {
			t.Fatalf("code=%d called=%v stderr=%s", code, called, stderr.String())
		}
	}
}
