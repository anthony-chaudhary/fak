package disambiguation

import (
	"errors"
	"testing"
)

func TestEvaluateCommittedFreshnessVerdicts(t *testing.T) {
	tests := []struct {
		name               string
		committed, overlay []byte
		probeErr           error
		verdict            CommittedFreshnessVerdict
	}{
		{"clean", []byte("same"), []byte("same"), nil, CommittedFreshnessClean},
		{"overlay drift", []byte("committed"), []byte("overlay"), nil, CommittedFreshnessOverlayDrift},
		{"unavailable", nil, nil, errors.New("git unavailable"), CommittedFreshnessUnavailable},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EvaluateCommittedFreshness(tt.committed, tt.overlay, tt.probeErr)
			if got.Verdict != tt.verdict {
				t.Fatalf("verdict=%q want %q", got.Verdict, tt.verdict)
			}
			if tt.verdict == CommittedFreshnessClean && (!got.CommittedClean || got.OverlayDrift || !got.ProbeAvailable) {
				t.Fatalf("clean report=%+v", got)
			}
			if tt.verdict == CommittedFreshnessOverlayDrift && (!got.CommittedClean || !got.OverlayDrift || !got.ProbeAvailable) {
				t.Fatalf("drift report=%+v", got)
			}
			if tt.verdict == CommittedFreshnessUnavailable && (got.CommittedClean || got.OverlayDrift || got.ProbeAvailable || got.Reason == "") {
				t.Fatalf("unavailable report=%+v", got)
			}
		})
	}
}
