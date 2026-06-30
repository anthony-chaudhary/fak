package dojopost

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

// --- resolution -------------------------------------------------------------

func TestResolveTokenAndChannelFromDojoEnv(t *testing.T) {
	t.Setenv("FAK_DOJO_TOKEN", "xoxb-dojo-token")
	t.Setenv("FAK_DOJO_CHANNEL", "C_DOJO_ENV")
	if got := ResolveToken(); got != "xoxb-dojo-token" {
		t.Fatalf("ResolveToken env = %q, want xoxb-dojo-token", got)
	}
	if got := ResolveChannel(); got != "C_DOJO_ENV" {
		t.Fatalf("ResolveChannel env = %q, want C_DOJO_ENV", got)
	}
}

func TestResolveTokenFallsBackToScoreboardToken(t *testing.T) {
	// The dedicated key is unset; the dojo channel shares the scoreboard workspace, so
	// the token must fall back to FAK_SCOREBOARD_TOKEN — never to the lab SLACK_BOT_TOKEN.
	t.Setenv("FAK_DOJO_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-scoreboard-token")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token-must-not-leak")
	chdir(t, t.TempDir()) // no .env.slack.local
	if got := ResolveToken(); got != "xoxb-scoreboard-token" {
		t.Fatalf("ResolveToken fallback = %q, want the scoreboard token", got)
	}
}

func TestResolveTokenNeverLeaksLabToken(t *testing.T) {
	t.Setenv("FAK_DOJO_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token")
	chdir(t, t.TempDir())
	if got := ResolveToken(); got != "" {
		t.Fatalf("ResolveToken leaked a token: got %q, want empty", got)
	}
}

func TestResolveChannelDefaultsToPublicDojoChannel(t *testing.T) {
	// Unlike the bench channel, the dojo channel id is a public, non-secret default so
	// the surface lands with zero config.
	t.Setenv("FAK_DOJO_CHANNEL", "")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel default = %q, want the public dojo channel %q", got, ChannelDefault)
	}
}

func TestResolveChannelDoesNotInheritScoreboardChannel(t *testing.T) {
	// FAK_SCOREBOARD_CHANNEL is the scoreboard CLI's #scoreboard default; the dojo
	// surface must NOT misroute to it — it owns its own default.
	t.Setenv("FAK_DOJO_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C_SCOREBOARD_MUST_NOT_LEAK")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel inherited the scoreboard channel: got %q, want %q", got, ChannelDefault)
	}
}

func TestResolveFromEnvFileWhenEnvUnset(t *testing.T) {
	t.Setenv("FAK_DOJO_TOKEN", "")
	t.Setenv("FAK_DOJO_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")

	dir := t.TempDir()
	envBody := "# comment\n" +
		"export FAK_DOJO_TOKEN=xoxb-file-dojo\n" +
		"FAK_DOJO_CHANNEL=C_FILE_DOJO\n" +
		"SLACK_BOT_TOKEN=xoxb-lab-token-must-not-leak\n"
	if err := os.WriteFile(filepath.Join(dir, ".env.slack.local"), []byte(envBody), 0o600); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(dir, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	chdir(t, sub)

	if got := ResolveToken(); got != "xoxb-file-dojo" {
		t.Fatalf("ResolveToken file = %q, want xoxb-file-dojo", got)
	}
	if got := ResolveChannel(); got != "C_FILE_DOJO" {
		t.Fatalf("ResolveChannel file = %q, want C_FILE_DOJO", got)
	}
}

// --- folds ------------------------------------------------------------------

func TestRollupFromReportMeasuredRun(t *testing.T) {
	r := dojo.Report{
		Commit:       "abcdef1234567890",
		LeverCount:   1,
		EpisodeCount: 2,
		Measured:     2,
		Calibrated:   1,
		MeanCalibErr: 0.341,
		Grade:        "C",
		NextAction:   "inspect the cold-write regression before changing policy",
		Episodes: []dojo.Episode{
			{Lever: "resume-posture", Metric: "cold_write_share", Claimed: 0.85, Realized: 0.40, CalibErr: 0.53, Verdict: dojo.VerdictOverClaim, Grade: "D", Provenance: dojo.Observed, Sample: 40},
			{Lever: "resume-posture", Metric: "posture_accuracy", Claimed: 1.0, Realized: 0.98, CalibErr: 0.02, Verdict: dojo.VerdictCalibrated, Grade: "A", Provenance: dojo.Observed, Sample: 1000},
		},
	}
	got := RollupFromReport(r, 8).Text()

	// The lead carries the aggregate and the (truncated) commit.
	if !strings.Contains(got, "mean calib-err 0.341") || !strings.Contains(got, "grade C") {
		t.Fatalf("rollup lead missing aggregate:\n%s", got)
	}
	if !strings.Contains(got, "@abcdef123456") { // 12-char short commit
		t.Fatalf("rollup lead missing short commit:\n%s", got)
	}
	// Worst-first: the OVER_CLAIM cold_write_share (calib-err 0.53) must precede the
	// CALIBRATED posture_accuracy (0.02).
	cold := strings.Index(got, "cold_write_share")
	acc := strings.Index(got, "posture_accuracy")
	if cold < 0 || acc < 0 || cold > acc {
		t.Fatalf("episodes not worst-first (cold=%d acc=%d):\n%s", cold, acc, got)
	}
	// Provenance is carried through (conflation honesty).
	if !strings.Contains(got, "OVER_CLAIM") || !strings.Contains(got, "OBSERVED") {
		t.Fatalf("rollup dropped verdict/provenance:\n%s", got)
	}
	for _, want := range []string{
		"operator: inspect the cold-write regression",
		"current: 1 lever(s), 2 episode(s), 2 measured, 0 unmeasured, 1 calibrated",
		"worst lever: `resume-posture`",
		"worst metric `cold_write_share`",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rollup missing operator-friendly line %q:\n%s", want, got)
		}
	}
}

func TestRollupFromReportUnmeasuredSurfacesReason(t *testing.T) {
	r := dojo.Report{
		Grade:    "n/a",
		Measured: 0,
		Reason:   "dojo run incomplete — no episode had ground truth to score against",
	}
	got := RollupFromReport(r, 8).Text()
	if !strings.Contains(got, "no episode had ground truth") {
		t.Fatalf("unmeasured rollup must surface the reason:\n%s", got)
	}
}

func TestTrendFromLedgerOrdersRecentFirstAndTrends(t *testing.T) {
	rows := []dojo.LedgerRow{
		{Schema: dojo.LedgerSchema, Date: "2026-06-25", Commit: "c1", GeneratedAt: "2026-06-25T01:00:00Z", LeverCount: 3, EpisodeCount: 7, MeanCalibErr: 0.70, Grade: "F", Calibrated: 2, Measured: 6},
		{Schema: dojo.LedgerSchema, Date: "2026-06-26", Commit: "c2", GeneratedAt: "2026-06-26T01:00:00Z", LeverCount: 3, EpisodeCount: 7, MeanCalibErr: 0.50, Grade: "D", Calibrated: 2, Measured: 6},
		{Schema: dojo.LedgerSchema, Date: "2026-06-27", Commit: "c3", GeneratedAt: "2026-06-27T01:00:00Z", LeverCount: 3, EpisodeCount: 7, MeanCalibErr: 0.34, Grade: "C", Calibrated: 2, Measured: 6},
	}
	got := TrendFromLedger(rows, 3).Text()

	// The latest row's grade leads.
	if !strings.Contains(got, "grade C") {
		t.Fatalf("trend lead must carry the latest grade:\n%s", got)
	}
	// 0.50 -> 0.34 is an improvement; the summary must say so.
	if !strings.Contains(got, "improved") {
		t.Fatalf("trend must report the improvement direction:\n%s", got)
	}
	// Most-recent-first: 2026-06-27 precedes 2026-06-25 in the body.
	newest := strings.Index(got, "2026-06-27")
	oldest := strings.Index(got, "2026-06-25")
	if newest < 0 || oldest < 0 || newest > oldest {
		t.Fatalf("trend rows not most-recent-first (new=%d old=%d):\n%s", newest, oldest, got)
	}
	for _, want := range []string{
		"current: 3 lever(s), 7 episode(s), 6 measured, 1 unmeasured, 2 calibrated",
		"operator: claims moved closer to billed reality",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("trend missing operator-friendly line %q:\n%s", want, got)
		}
	}
}

func TestTrendFromLedgerEmptyIsHonest(t *testing.T) {
	got := TrendFromLedger(nil, 6).Text()
	if !strings.Contains(got, "no dojo history yet") {
		t.Fatalf("empty ledger must yield an honest card:\n%s", got)
	}
}

// --- helpers ----------------------------------------------------------------

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
