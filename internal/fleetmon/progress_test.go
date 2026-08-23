package fleetmon

import (
	"testing"
	"time"
)

func TestEvaluateProgressSeparatesPIDLivenessFromDurableProgress(t *testing.T) {
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	window := 2 * time.Minute
	cases := []struct {
		name     string
		live     bool
		known    bool
		moved    bool
		evidence ProgressEvidence
		want     ProgressState
	}{
		{"advancing transcript", true, true, true, ProgressEvidence{now.Add(-time.Second), "transcript-jsonl"}, Progressing},
		{"quiet inside window", true, true, false, ProgressEvidence{now.Add(-time.Minute), "terminal-receipt"}, QuietWithinWindow},
		{"parent alive child wedged", true, true, false, ProgressEvidence{now.Add(-3 * time.Minute), "transcript-jsonl"}, Wedged},
		{"alive without telemetry", true, true, false, ProgressEvidence{}, ProgressUnknown},
		{"unknown process scan", true, false, true, ProgressEvidence{now, "write-heartbeat"}, ProgressUnknown},
		{"dead is not a productive-liveness claim", false, true, false, ProgressEvidence{now, "transcript-jsonl"}, ProgressUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := EvaluateProgress(tc.live, tc.known, tc.moved, tc.evidence, now, window); got != tc.want {
				t.Fatalf("got %s, want %s", got, tc.want)
			}
		})
	}
}
