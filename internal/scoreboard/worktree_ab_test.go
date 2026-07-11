package scoreboard

import (
	"strings"
	"testing"
)

func TestWorktreeABRendersBothArms(t *testing.T) {
	r := FoldWorktreeAB(
		WorktreeABArm{WaveID: "fixed-wave", Resolved: 3, DurationSeconds: 1800, PoisonIncidents: 2, PeakConcurrency: 5},
		WorktreeABArm{WaveID: "fixed-wave", Resolved: 3, DurationSeconds: 1200, PoisonIncidents: 0, PeakConcurrency: 8},
	)
	if r.Schema != WorktreeABSchema || r.Verdict != "ISOLATION_POISON_FREE" || !WorktreeABEquivalentWave(r.Baseline, r.Isolated) {
		t.Fatalf("report=%+v", r)
	}
	text := WorktreeABUpdate(r).Text()
	for _, want := range []string{"baseline: 6.00 issues/h, 2 poison, 1800.0s, peak 5", "isolated: 9.00 issues/h, 0 poison, 1200.0s, peak 8"} {
		if !strings.Contains(text, want) {
			t.Fatalf("render missing %q:\n%s", want, text)
		}
	}
}

func TestWorktreeABRequiresMeasuredPoisonFreeIsolatedArm(t *testing.T) {
	r := FoldWorktreeAB(WorktreeABArm{WaveID: "fixed-wave", Resolved: 1, DurationSeconds: 10}, WorktreeABArm{WaveID: "fixed-wave", Resolved: 1, DurationSeconds: 10, PoisonIncidents: 1})
	if r.Verdict != "NOT_PROVEN" {
		t.Fatalf("verdict=%s", r.Verdict)
	}
}
