package compute

import (
	"strings"
	"testing"
)

// TestDecodeSeamsRecordEveryCandidate checks that the data file records one seam per shortlist candidate,
// and that no candidate falls back to the unrecorded marker.
func TestDecodeSeamsRecordEveryCandidate(t *testing.T) {
	a, _ := LookupNVArch(SM80)
	report := DecodeGapReport(a, A100SMs, smolLM2_135M, 40)
	recorded := 0
	for _, line := range strings.Split(strings.TrimSpace(decodeSeamsText), "\n") {
		if _, _, ok := strings.Cut(line, "\t"); ok {
			recorded++
		}
	}
	if recorded != len(report) {
		t.Fatalf("data file records %d seams, want %d", recorded, len(report))
	}
	for _, gap := range report {
		if strings.HasPrefix(gap.Seam, "seam not recorded") {
			t.Errorf("candidate %s has no recorded seam", gap.Candidate)
		}
	}
}
