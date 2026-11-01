package computebuild

import (
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestCUDAQwen35IQ2SequenceRecordedSelection(t *testing.T) {
	const tag = "fak_cuda_qwen35_iq2_sequence_qualification"
	for _, arch := range []string{"", "sm_89", "sm_120"} {
		cfg := &CUDAConfig{Arch: arch, EnableNCCL: true, EnableCaptureOnceQualification: true, EnableKV16PrefillGQA4Qualification: true}
		ApplyCUDAV41Defaults(cfg)
		before := CUDAGoTags(cfg)
		if cfg.EnableQwen35IQ2SequenceQualification || strings.Contains(before, tag) {
			t.Fatal("ordinary defaults selected IQ2 qualification")
		}
		cfg.EnableQwen35IQ2SequenceQualification = true
		after := CUDAGoTags(cfg)
		if strings.Count(after, tag) != 1 || strings.ReplaceAll(after, ","+tag, "") != before {
			t.Fatalf("tag composition changed peer selection: before=%q after=%q", before, after)
		}
		if !cfg.EnableKV16PrefillGQA4Qualification || !cfg.EnableCaptureOnceQualification {
			t.Fatal("mutated independent policy")
		}
		cfg.EnableQwen35IQ2SequenceQualification = false
		if CUDAGoTags(cfg) != before {
			t.Fatal("qualification selection leaked")
		}
	}
}
