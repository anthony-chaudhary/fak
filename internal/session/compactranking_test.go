package session

import (
	"strings"
	"testing"
)

func TestWriteCompactTrajectoryRankingPeakResident(t *testing.T) {
	reports := []CompactSessionReport{
		{SessionID: "cumulative", CumulativeInputTokens: 9000, PeakResidentTokens: 90, FinalResidentTokens: 80, ContextWindow: 200, Verdict: VerdictNoFireBounded},
		{SessionID: "resident", CumulativeInputTokens: 1000, PeakResidentTokens: 210, FinalResidentTokens: 40, ContextWindow: 200, Verdict: VerdictFiredAndHeld, Fires: []CompactFire{{}}},
	}
	var out strings.Builder
	if err := writeCompactTrajectoryRanking(&out, reports, 2, CompactRankPeakResident); err != nil {
		t.Fatal(err)
	}
	got := out.String()
	if strings.Index(got, "resident") > strings.Index(got, "cumulative") {
		t.Fatalf("peak-resident ranking did not put the highest live trajectory first:\n%s", got)
	}
	for _, want := range []string{"top 2 sessions by peak-resident tokens", "210/200 (105.0%)", "final RESIDENT 40", "cumulative input 1000", "1 fires"} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q:\n%s", want, got)
		}
	}
}

func TestWriteCompactTrajectoryRankingCumulativeInput(t *testing.T) {
	reports := []CompactSessionReport{
		{SessionID: "resident", CumulativeInputTokens: 1000, PeakResidentTokens: 210},
		{SessionID: "cumulative", CumulativeInputTokens: 9000, PeakResidentTokens: 90},
	}
	var out strings.Builder
	if err := writeCompactTrajectoryRanking(&out, reports, 1, CompactRankCumulativeInput); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); !strings.Contains(got, "cumulative") || strings.Contains(got, "resident") {
		t.Fatalf("cumulative-input ranking =\n%s", got)
	}
}

func TestWriteCompactTrajectoryRankingRejectsUnknownRank(t *testing.T) {
	if err := writeCompactTrajectoryRanking(&strings.Builder{}, nil, 1, "rollout-bytes"); err == nil {
		t.Fatal("unknown ranking accepted")
	}
}
