package blockerpost

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- resolution -------------------------------------------------------------

func TestResolveTokenAndChannelFromBlockersEnv(t *testing.T) {
	t.Setenv("FAK_BLOCKERS_TOKEN", "xoxb-blockers-token")
	t.Setenv("FAK_BLOCKERS_CHANNEL", "C_BLOCKERS_ENV")
	if got := ResolveToken(); got != "xoxb-blockers-token" {
		t.Fatalf("ResolveToken env = %q, want xoxb-blockers-token", got)
	}
	if got := ResolveChannel(); got != "C_BLOCKERS_ENV" {
		t.Fatalf("ResolveChannel env = %q, want C_BLOCKERS_ENV", got)
	}
}

func TestResolveTokenFallsBackToScoreboardToken(t *testing.T) {
	// The dedicated key is unset; the blockers channel shares the scoreboard workspace,
	// so the token falls back to FAK_SCOREBOARD_TOKEN — never to the lab SLACK_BOT_TOKEN.
	t.Setenv("FAK_BLOCKERS_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-scoreboard-token")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token-must-not-leak")
	chdir(t, t.TempDir()) // no .env.slack.local
	if got := ResolveToken(); got != "xoxb-scoreboard-token" {
		t.Fatalf("ResolveToken fallback = %q, want the scoreboard token", got)
	}
}

func TestResolveTokenNeverLeaksLabToken(t *testing.T) {
	t.Setenv("FAK_BLOCKERS_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token")
	chdir(t, t.TempDir())
	if got := ResolveToken(); got != "" {
		t.Fatalf("ResolveToken leaked a token: got %q, want empty", got)
	}
}

func TestResolveChannelDefaultsToPublicBlockersChannel(t *testing.T) {
	// The blockers channel id is a public, non-secret default so the surface lands with
	// zero config.
	t.Setenv("FAK_BLOCKERS_CHANNEL", "")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel default = %q, want the public blockers channel %q", got, ChannelDefault)
	}
}

func TestResolveChannelDoesNotInheritScoreboardChannel(t *testing.T) {
	// FAK_SCOREBOARD_CHANNEL is the scoreboard CLI's #scoreboard default; a blocker must
	// NOT misroute to it — the surface owns its own default.
	t.Setenv("FAK_BLOCKERS_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C_SCOREBOARD_MUST_NOT_LEAK")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel inherited the scoreboard channel: got %q, want %q", got, ChannelDefault)
	}
}

func TestResolveFromEnvFileWhenEnvUnset(t *testing.T) {
	t.Setenv("FAK_BLOCKERS_TOKEN", "")
	t.Setenv("FAK_BLOCKERS_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")

	dir := t.TempDir()
	envBody := "# comment\n" +
		"export FAK_BLOCKERS_TOKEN=xoxb-file-blockers\n" +
		"FAK_BLOCKERS_CHANNEL=C_FILE_BLOCKERS\n" +
		"SLACK_BOT_TOKEN=xoxb-lab-token-must-not-leak\n"
	if err := os.WriteFile(filepath.Join(dir, ".env.slack.local"), []byte(envBody), 0o600); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(dir, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	chdir(t, sub)

	if got := ResolveToken(); got != "xoxb-file-blockers" {
		t.Fatalf("ResolveToken file = %q, want xoxb-file-blockers", got)
	}
	if got := ResolveChannel(); got != "C_FILE_BLOCKERS" {
		t.Fatalf("ResolveChannel file = %q, want C_FILE_BLOCKERS", got)
	}
}

// --- severity ---------------------------------------------------------------

func TestParseSeverity(t *testing.T) {
	cases := map[string]Severity{
		"":         SeverityStatus,
		"status":   SeverityStatus,
		"OPERATOR": SeverityOperator,
		" clear ":  SeverityClear,
	}
	for in, want := range cases {
		got, ok := ParseSeverity(in)
		if !ok || got != want {
			t.Fatalf("ParseSeverity(%q) = (%q,%v), want (%q,true)", in, got, ok, want)
		}
	}
	if _, ok := ParseSeverity("urgent"); ok {
		t.Fatal("ParseSeverity(\"urgent\") accepted an unknown severity — a typo must not silently downgrade a page")
	}
}

// --- render: the two-tier surfacing contract --------------------------------

func TestStatusBlockerIsMutedAndDoesNotPage(t *testing.T) {
	b := Blocker{
		Severity: SeverityStatus,
		Title:    "GPU-gated, waiting on DGX hours",
		Detail:   "Rungs 1/2/3/5 need the private DGX-A100.",
		Source:   "agent",
	}
	got := b.Text()
	if !strings.Contains(got, ":hourglass_flowing_sand:") {
		t.Fatalf("status blocker missing the muted glyph:\n%s", got)
	}
	if strings.Contains(got, "<!here>") || strings.Contains(got, "<!channel>") {
		t.Fatalf("status blocker MUST NOT broadcast — it paged:\n%s", got)
	}
	if !strings.Contains(got, "ongoing") {
		t.Fatalf("status blocker should frame as ongoing:\n%s", got)
	}
	// The Block Kit path must also stay silent.
	if blocksText(b) != "" && strings.Contains(blocksText(b), "<!here>") {
		t.Fatalf("status blocker Blocks() broadcast — it paged")
	}
}

func TestOperatorBlockerIsSurfacedAndPages(t *testing.T) {
	b := Blocker{
		Severity:  SeverityOperator,
		Title:     "CPU host unreachable",
		Detail:    "CPU GLM-5.2 node is not responding.",
		Action:    "restart the CPU-host serve",
		ActionURL: "https://example.invalid/runbook",
		Ref:       "host:cpu-server-a",
		Source:    "ci",
	}
	got := b.Text()
	if !strings.Contains(got, ":rotating_light:") {
		t.Fatalf("operator blocker missing the red glyph:\n%s", got)
	}
	if !strings.Contains(got, "<!here>") {
		t.Fatalf("operator blocker MUST page (default <!here>) — it did not:\n%s", got)
	}
	if !strings.Contains(got, "BLOCKER — needs operator") {
		t.Fatalf("operator blocker missing the surfaced banner:\n%s", got)
	}
	if !strings.Contains(got, "restart the CPU-host serve") || !strings.Contains(got, "https://example.invalid/runbook") {
		t.Fatalf("operator blocker dropped the do-this-next affordance:\n%s", got)
	}
	// The broadcast must ALSO be in the Block Kit payload (the mention pages in both paths).
	bt := blocksText(b)
	if !strings.Contains(bt, "<!here>") {
		t.Fatalf("operator Blocks() did not carry the broadcast mention:\n%s", bt)
	}
	if !hasLinkButton(b, "https://example.invalid/runbook") {
		t.Fatal("operator Blocks() missing the link-button affordance")
	}
}

func TestOperatorBlockerPagesNamedOwner(t *testing.T) {
	b := Blocker{Severity: SeverityOperator, Title: "secret missing", Owner: "<@U0OPERATOR>"}
	got := b.Text()
	if !strings.Contains(got, "<@U0OPERATOR>") {
		t.Fatalf("operator blocker did not page the named owner:\n%s", got)
	}
	if strings.Contains(got, "<!here>") {
		t.Fatalf("a named owner should replace the default <!here>:\n%s", got)
	}
}

func TestClearBlockerIsGreenAndQuiet(t *testing.T) {
	b := Blocker{Severity: SeverityClear, Title: "no standing blockers", Detail: "0 open `blocked` issues."}
	got := b.Text()
	if !strings.Contains(got, ":white_check_mark:") || !strings.Contains(got, "all clear") {
		t.Fatalf("clear blocker should be a green all-clear:\n%s", got)
	}
	if strings.Contains(got, "<!here>") {
		t.Fatalf("clear blocker MUST NOT page:\n%s", got)
	}
}

// --- feed fold --------------------------------------------------------------

func TestFoldIssuesEmptyIsClear(t *testing.T) {
	b := FoldIssues(nil, "blocked", "https://github.com/o/r")
	if b.Severity != SeverityClear {
		t.Fatalf("no issues should fold to clear, got %q", b.Severity)
	}
	if !strings.Contains(b.Text(), "no standing blockers") {
		t.Fatalf("clear fold missing the all-clear headline:\n%s", b.Text())
	}
}

func TestFoldIssuesUnownedSurfacesToOperator(t *testing.T) {
	issues := []Issue{
		{Number: 12, Title: "owned thing", URL: "https://x/12", Assignees: []Assignee{{Login: "alice"}}},
		{Number: 5, Title: "adrift blocker", URL: "https://x/5"}, // no assignee
	}
	b := FoldIssues(issues, "blocked", "https://github.com/o/r")
	if b.Severity != SeverityOperator {
		t.Fatalf("an unowned blocker must surface to operator, got %q", b.Severity)
	}
	got := b.Text()
	if !strings.Contains(got, "<!here>") {
		t.Fatalf("operator fold must page:\n%s", got)
	}
	// Worst-first: the UNOWNED #5 must precede the owned #12.
	five := strings.Index(got, "#5")
	twelve := strings.Index(got, "#12")
	if five < 0 || twelve < 0 || five > twelve {
		t.Fatalf("unowned issue not listed first (5=%d 12=%d):\n%s", five, twelve, got)
	}
	if !strings.Contains(got, "UNOWNED") {
		t.Fatalf("the adrift issue should be marked UNOWNED:\n%s", got)
	}
	// The query is url.QueryEscape'd, so the label filter rides as label%3Ablocked.
	if !strings.Contains(b.ActionURL, "/issues?q=") || !strings.Contains(b.ActionURL, "label%3Ablocked") {
		t.Fatalf("operator fold should link the filtered backlog, got %q", b.ActionURL)
	}
}

func TestFoldIssuesAllOwnedIsBackgroundStatus(t *testing.T) {
	issues := []Issue{
		{Number: 1, Title: "a", Assignees: []Assignee{{Login: "bob"}}},
		{Number: 2, Title: "b", Assignees: []Assignee{{Login: "carol"}}},
	}
	b := FoldIssues(issues, "blocked", "")
	if b.Severity != SeverityStatus {
		t.Fatalf("all-owned blockers should be background status, got %q", b.Severity)
	}
	if strings.Contains(b.Text(), "<!here>") {
		t.Fatalf("an all-owned (in-progress) fold MUST NOT page:\n%s", b.Text())
	}
}

func TestFoldIssuesTruncatesLargeBacklog(t *testing.T) {
	var issues []Issue
	for i := 0; i < maxFeedLines+5; i++ {
		issues = append(issues, Issue{Number: i + 1, Title: "x", Assignees: []Assignee{{Login: "x"}}})
	}
	b := FoldIssues(issues, "blocked", "")
	if !strings.Contains(b.Text(), "and 5 more") {
		t.Fatalf("a large backlog should summarize the overflow:\n%s", b.Text())
	}
}

// --- helpers ----------------------------------------------------------------

// blocksText flattens the Block Kit payload's mrkdwn text so a test can assert on the
// rendered content without re-deriving the block structure.
func blocksText(b Blocker) string {
	var sb strings.Builder
	for _, blk := range b.Blocks() {
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
	}
	return sb.String()
}

// hasLinkButton reports whether the Block Kit payload contains an actions block with a
// URL link-button pointing at want.
func hasLinkButton(b Blocker, want string) bool {
	for _, blk := range b.Blocks() {
		m, ok := blk.(map[string]any)
		if !ok || m["type"] != "actions" {
			continue
		}
		elems, ok := m["elements"].([]any)
		if !ok {
			continue
		}
		for _, e := range elems {
			em, ok := e.(map[string]any)
			if ok && em["url"] == want {
				return true
			}
		}
	}
	return false
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
