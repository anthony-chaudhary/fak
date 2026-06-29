package milestonepost

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"
)

// --- resolution -------------------------------------------------------------

func TestResolveTokenAndChannelFromMilestoneEnv(t *testing.T) {
	t.Setenv("FAK_MILESTONE_TOKEN", "xoxb-milestone-token")
	t.Setenv("FAK_MILESTONE_CHANNEL", "C_MILESTONE_ENV")
	if got := ResolveToken(); got != "xoxb-milestone-token" {
		t.Fatalf("ResolveToken env = %q, want xoxb-milestone-token", got)
	}
	if got := ResolveChannel(); got != "C_MILESTONE_ENV" {
		t.Fatalf("ResolveChannel env = %q, want C_MILESTONE_ENV", got)
	}
}

func TestResolveChannelDefaultsToPublicMilestoneChannel(t *testing.T) {
	t.Setenv("FAK_MILESTONE_CHANNEL", "")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel default = %q, want the public milestones channel %q", got, ChannelDefault)
	}
	if ChannelDefault != "C0BDYFRSW6S" {
		t.Fatalf("ChannelDefault = %q, want the #milestones channel C0BDYFRSW6S", ChannelDefault)
	}
}

func TestResolveChannelDoesNotInheritScoreboardChannel(t *testing.T) {
	t.Setenv("FAK_MILESTONE_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C_SCOREBOARD_MUST_NOT_LEAK")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel inherited the scoreboard channel: got %q, want %q", got, ChannelDefault)
	}
}

func TestResolveTokenFallsBackToScoreboardToken(t *testing.T) {
	t.Setenv("FAK_MILESTONE_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-scoreboard-token")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token-must-not-leak")
	chdir(t, t.TempDir())
	if got := ResolveToken(); got != "xoxb-scoreboard-token" {
		t.Fatalf("ResolveToken fallback = %q, want the scoreboard token", got)
	}
}

// --- render -----------------------------------------------------------------

func sampleReport() milestonereport.Report {
	m := milestonereport.InterpretMaturity([]covmatrix.Cell{
		{Family: "a", Backend: "cpu", Support: covmatrix.Supported},
		{Family: "b", Backend: "cpu", Support: covmatrix.Undefined},
	})
	e := milestonereport.InterpretEpics(
		[]milestonereport.EpicSpec{{Number: 42, Title: "the answer"}},
		[]milestonereport.EpicCounts{{Number: 42, Closed: 3, Total: 4, Source: "label"}}, "")
	r := milestonereport.Fold(m, e, milestonereport.FoldOpts{Date: "2026-06-29", Commit: "abc"})
	return r.WithTrend(milestonereport.TrendVsLast(milestonereport.RowFromReport(r), nil))
}

func TestCardTextCarriesBothDimensions(t *testing.T) {
	c := Fold(sampleReport())
	c.Source = "agent"
	out := c.Text()
	for _, want := range []string{cardTitle, "climb:", "ladder:", "M0:", "roadmap:", "#42 the answer", "75% (3/4)", "[label]", "S/N self-score", "posted by agent"} {
		if !strings.Contains(out, want) {
			t.Fatalf("Text missing %q\n%s", want, out)
		}
	}
}

func TestCardBlocksMarshalAndCarryFacts(t *testing.T) {
	c := Fold(sampleReport())
	blocks := c.Blocks()
	if len(blocks) == 0 {
		t.Fatal("Blocks must be non-empty")
	}
	b, err := json.Marshal(blocks)
	if err != nil {
		t.Fatalf("blocks must marshal to JSON: %v", err)
	}
	js := string(b)
	for _, want := range []string{"climb:", "roadmap:", "#42 the answer", "S/N self-score"} {
		if !strings.Contains(js, want) {
			t.Fatalf("Blocks missing %q\n%s", want, js)
		}
	}
}

func TestCardRendersGhFailureHonestly(t *testing.T) {
	m := milestonereport.InterpretMaturity([]covmatrix.Cell{{Support: covmatrix.Supported}})
	e := milestonereport.InterpretEpics([]milestonereport.EpicSpec{{Number: 7, Title: "z"}}, nil, "gh: not found")
	c := Fold(milestonereport.Fold(m, e, milestonereport.FoldOpts{Date: "2026-06-29"}))
	out := c.Text()
	if !strings.Contains(out, "gh read failed") {
		t.Fatalf("an unreadable epic must render 'gh read failed'\n%s", out)
	}
	if strings.Contains(out, "#7 z — 0%") {
		t.Fatalf("must never fabricate a 0%% for an unreadable epic\n%s", out)
	}
}

func TestSignalNoiseIsFinite(t *testing.T) {
	c := Fold(sampleReport())
	sn := c.signalNoise()
	if sn.Signal < 1 || sn.Noise < 1 || sn.Ratio <= 0 {
		t.Fatalf("signal/noise must be positive and finite, got %+v", sn)
	}
}

// --- chdir helper (mirrors cachevaluepost's test helper) --------------------

func chdir(t *testing.T, dir string) {
	t.Helper()
	prev, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(prev) })
}
