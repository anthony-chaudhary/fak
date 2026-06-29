package cachevaluepost

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

// --- resolution -------------------------------------------------------------

func TestResolveTokenAndChannelFromCachevalueEnv(t *testing.T) {
	t.Setenv("FAK_CACHEVALUE_TOKEN", "xoxb-cachevalue-token")
	t.Setenv("FAK_CACHEVALUE_CHANNEL", "C_CACHEVALUE_ENV")
	if got := ResolveToken(); got != "xoxb-cachevalue-token" {
		t.Fatalf("ResolveToken env = %q, want xoxb-cachevalue-token", got)
	}
	if got := ResolveChannel(); got != "C_CACHEVALUE_ENV" {
		t.Fatalf("ResolveChannel env = %q, want C_CACHEVALUE_ENV", got)
	}
}

func TestResolveTokenFallsBackToScoreboardToken(t *testing.T) {
	// The dedicated key is unset; the cache-value channel shares the scoreboard workspace,
	// so the token falls back to FAK_SCOREBOARD_TOKEN — never to the lab SLACK_BOT_TOKEN.
	t.Setenv("FAK_CACHEVALUE_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-scoreboard-token")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token-must-not-leak")
	chdir(t, t.TempDir())
	if got := ResolveToken(); got != "xoxb-scoreboard-token" {
		t.Fatalf("ResolveToken fallback = %q, want the scoreboard token", got)
	}
}

func TestResolveTokenNeverLeaksLabToken(t *testing.T) {
	t.Setenv("FAK_CACHEVALUE_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token")
	chdir(t, t.TempDir())
	if got := ResolveToken(); got != "" {
		t.Fatalf("ResolveToken leaked a token: got %q, want empty", got)
	}
}

func TestResolveChannelDefaultsToPublicCachevalueChannel(t *testing.T) {
	t.Setenv("FAK_CACHEVALUE_CHANNEL", "")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel default = %q, want the public cache-value channel %q", got, ChannelDefault)
	}
	if ChannelDefault != "C0BDSB81XDZ" {
		t.Fatalf("ChannelDefault = %q, want the epic #1301 channel C0BDSB81XDZ", ChannelDefault)
	}
}

func TestResolveChannelDoesNotInheritScoreboardChannel(t *testing.T) {
	// FAK_SCOREBOARD_CHANNEL is the scoreboard CLI's #scoreboard default; a cache-value
	// card must NOT misroute to it — the surface owns its own default.
	t.Setenv("FAK_CACHEVALUE_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C_SCOREBOARD_MUST_NOT_LEAK")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel inherited the scoreboard channel: got %q, want %q", got, ChannelDefault)
	}
}

func TestResolveFromEnvFileWhenEnvUnset(t *testing.T) {
	t.Setenv("FAK_CACHEVALUE_TOKEN", "")
	t.Setenv("FAK_CACHEVALUE_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")

	dir := t.TempDir()
	envBody := "# comment\n" +
		"export FAK_CACHEVALUE_TOKEN=xoxb-file-cachevalue\n" +
		"FAK_CACHEVALUE_CHANNEL=C_FILE_CACHEVALUE\n" +
		"SLACK_BOT_TOKEN=xoxb-lab-token-must-not-leak\n"
	if err := os.WriteFile(filepath.Join(dir, ".env.slack.local"), []byte(envBody), 0o600); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(dir, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	chdir(t, sub)

	if got := ResolveToken(); got != "xoxb-file-cachevalue" {
		t.Fatalf("ResolveToken file = %q, want xoxb-file-cachevalue", got)
	}
	if got := ResolveChannel(); got != "C_FILE_CACHEVALUE" {
		t.Fatalf("ResolveChannel file = %q, want C_FILE_CACHEVALUE", got)
	}
}

// --- render: the fold + card contract ---------------------------------------

// multiTurnRows builds a trending two-week corpus: week 25 at 60% reuse, week 26 at 80%,
// both multi-turn, so the fold yields a MEASURED report with an upward trend.
func multiTurnRows() []cachevalueledger.Row {
	return []cachevalueledger.Row{
		{Date: "2026-06-15", SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 600},
		{Date: "2026-06-22", SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
	}
}

func TestFoldMeasuredCardRendersTrendAndFence(t *testing.T) {
	now := time.Date(2026, 6, 29, 0, 0, 0, 0, time.UTC)
	report := cachevaluereport.Fold(multiTurnRows(), now)
	if report.Verdict != "MEASURED" {
		t.Fatalf("precondition: report should be MEASURED, got %q (%s)", report.Verdict, report.Finding)
	}
	card := Fold(report)
	card.Source = "agent"
	got := card.Text()

	if !strings.Contains(got, ":bar_chart:") {
		t.Fatalf("MEASURED card should lead with the chart glyph:\n%s", got)
	}
	if !strings.Contains(got, "MEASURED") {
		t.Fatalf("card must carry the verdict:\n%s", got)
	}
	// The latest bucket's realized reuse (80.0%) must be visible.
	if !strings.Contains(got, "80.0%") {
		t.Fatalf("card dropped the latest realized reuse 80.0%%:\n%s", got)
	}
	// The trend must read as improved (60% -> 80%).
	if !strings.Contains(got, "improved") {
		t.Fatalf("card should show the upward trend:\n%s", got)
	}
	// A sparkline must render (block glyphs).
	if !strings.Contains(got, "trend ") || !strings.ContainsAny(got, "▁▂▃▄▅▆▇█") {
		t.Fatalf("card missing the reuse sparkline:\n%s", got)
	}
	// The #1066 honesty fence must ride into the channel verbatim.
	if !strings.Contains(got, "marginal-over-tuned-warm-KV") || !strings.Contains(got, "#1066") {
		t.Fatalf("card dropped the #1066 honesty fence:\n%s", got)
	}
	if !strings.Contains(got, "_posted by agent_") {
		t.Fatalf("card dropped the source line:\n%s", got)
	}

	// The Block Kit payload must carry the verdict and the fence too.
	bt := blocksText(card)
	if !strings.Contains(bt, "MEASURED") || !strings.Contains(bt, "marginal-over-tuned-warm-KV") {
		t.Fatalf("Blocks() dropped the verdict/fence:\n%s", bt)
	}
}

func TestFoldInsufficientCardIsHonest(t *testing.T) {
	// A single-turn-only corpus has no multi-turn reuse to trend → INSUFFICIENT, but the
	// card must still render honestly (no fabricated trend) and carry the fence + next step.
	now := time.Date(2026, 6, 29, 0, 0, 0, 0, time.UTC)
	rows := []cachevalueledger.Row{
		{Date: "2026-06-22", SessionType: "run", Turns: 1, PromptTokens: 500, ReusedTokens: 0},
	}
	report := cachevaluereport.Fold(rows, now)
	if report.Verdict != "INSUFFICIENT" {
		t.Fatalf("precondition: single-turn corpus should be INSUFFICIENT, got %q", report.Verdict)
	}
	got := Fold(report).Text()
	if !strings.Contains(got, ":hourglass_flowing_sand:") {
		t.Fatalf("INSUFFICIENT card should use the accumulating glyph:\n%s", got)
	}
	if !strings.Contains(got, "INSUFFICIENT") {
		t.Fatalf("card must carry the INSUFFICIENT verdict:\n%s", got)
	}
	if !strings.Contains(got, "next:") {
		t.Fatalf("INSUFFICIENT card should name the next step:\n%s", got)
	}
	if !strings.Contains(got, "marginal-over-tuned-warm-KV") {
		t.Fatalf("card dropped the #1066 honesty fence:\n%s", got)
	}
}

func TestSparklineNormalizesAndHandlesEdges(t *testing.T) {
	if s := sparkline(nil); s != "" {
		t.Fatalf("empty series should render empty, got %q", s)
	}
	if s := sparkline([]float64{0.5}); []rune(s)[0] != '▅' || len([]rune(s)) != 1 {
		t.Fatalf("single point should render one mid block, got %q", s)
	}
	// A rising series should end higher than it starts.
	s := []rune(sparkline([]float64{0.1, 0.5, 0.9}))
	if len(s) != 3 || s[0] >= s[2] {
		t.Fatalf("rising series should rise: %q", string(s))
	}
}

// --- helpers ----------------------------------------------------------------

// blocksText flattens the Block Kit payload's mrkdwn text so a test can assert on the
// rendered content without re-deriving the block structure.
func blocksText(c Card) string {
	var sb strings.Builder
	for _, blk := range c.Blocks() {
		m, ok := blk.(map[string]any)
		if !ok {
			continue
		}
		if txt, ok := m["text"].(map[string]any); ok {
			if s, ok := txt["text"].(string); ok {
				sb.WriteString(s)
				sb.WriteString("\n")
			}
		}
		if elems, ok := m["elements"].([]any); ok {
			for _, e := range elems {
				if em, ok := e.(map[string]any); ok {
					if s, ok := em["text"].(string); ok {
						sb.WriteString(s)
						sb.WriteString("\n")
					}
				}
			}
		}
	}
	return sb.String()
}

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
