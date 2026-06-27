package nodeusagepost

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleet"
)

// --- resolution -------------------------------------------------------------

func TestResolveTokenAndChannelFromNodeUsageEnv(t *testing.T) {
	t.Setenv("FAK_NODE_USAGE_TOKEN", "xoxb-node-token")
	t.Setenv("FAK_NODE_USAGE_CHANNEL", "C_NODE")
	if got := ResolveToken(); got != "xoxb-node-token" {
		t.Fatalf("ResolveToken env = %q, want xoxb-node-token", got)
	}
	if got := ResolveChannel(); got != "C_NODE" {
		t.Fatalf("ResolveChannel env = %q, want C_NODE", got)
	}
}

func TestResolveTokenFallsBackToScoreboardToken(t *testing.T) {
	// The dedicated key is unset; the node-usage channel shares the scoreboard
	// workspace, so the token must fall back to FAK_SCOREBOARD_TOKEN — never to the lab
	// SLACK_BOT_TOKEN.
	t.Setenv("FAK_NODE_USAGE_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-scoreboard-token")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token-must-not-leak")
	chdir(t, t.TempDir()) // no .env.slack.local
	if got := ResolveToken(); got != "xoxb-scoreboard-token" {
		t.Fatalf("ResolveToken fallback = %q, want the scoreboard token", got)
	}
}

func TestResolveTokenNeverLeaksLabToken(t *testing.T) {
	t.Setenv("FAK_NODE_USAGE_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token")
	chdir(t, t.TempDir())
	if got := ResolveToken(); got != "" {
		t.Fatalf("ResolveToken leaked a token: got %q, want empty", got)
	}
}

func TestResolveFromEnvFileWhenEnvUnset(t *testing.T) {
	t.Setenv("FAK_NODE_USAGE_TOKEN", "")
	t.Setenv("FAK_NODE_USAGE_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")

	dir := t.TempDir()
	envBody := "# comment\n" +
		"export FAK_NODE_USAGE_TOKEN=xoxb-file-node\n" +
		"FAK_NODE_USAGE_CHANNEL=C_FILE_NODE\n" +
		"SLACK_BOT_TOKEN=xoxb-lab-token-must-not-leak\n"
	if err := os.WriteFile(filepath.Join(dir, ".env.slack.local"), []byte(envBody), 0o600); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(dir, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	chdir(t, sub)

	if got := ResolveToken(); got != "xoxb-file-node" {
		t.Fatalf("ResolveToken file = %q, want xoxb-file-node", got)
	}
	if got := ResolveChannel(); got != "C_FILE_NODE" {
		t.Fatalf("ResolveChannel file = %q, want C_FILE_NODE", got)
	}
}

func TestResolveChannelEmptyWhenUnset(t *testing.T) {
	// The real channel id is never a tracked default — an unset channel is "" so the
	// caller requires an explicit --channel and never silently posts to #scoreboard.
	t.Setenv("FAK_NODE_USAGE_CHANNEL", "")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != "" {
		t.Fatalf("ResolveChannel unset = %q, want empty", got)
	}
}

// --- fold: FromSnapshot -----------------------------------------------------

// foldRoster builds a real snapshot through fleet.Fold so the test exercises the exact
// shape `fak lab status --json` emits, not a hand-rolled struct that could drift.
func foldRoster(boxes []fleet.Box, reports []fleet.Report) fleet.Snapshot {
	ro := fleet.Roster{Schema: fleet.RosterSchema, Boxes: boxes}
	return fleet.Fold(ro, reports, fleet.FoldOpts{})
}

func TestFromSnapshotHealthyFleetIsOKAndGraded(t *testing.T) {
	boxes := []fleet.Box{
		{ID: "a1", Class: "a100x8"},
		{ID: "a2", Class: "a100x8"},
	}
	reports := []fleet.Report{
		{State: fleet.StateLive, Version: "v1"},
		{State: fleet.StateLive, Version: "v1"},
	}
	snap := foldRoster(boxes, reports)
	up := FromSnapshot(snap, "agent")

	if up.Title != "node usage" {
		t.Fatalf("title = %q, want %q", up.Title, "node usage")
	}
	if up.Verdict != "OK" {
		t.Fatalf("verdict = %q, want OK for an all-live fleet", up.Verdict)
	}
	if up.Grade != "A" {
		t.Fatalf("grade = %q, want A for a 100-readiness fleet (score=%d)", up.Grade, snap.Score)
	}
	if up.Score != "2/2 reachable" {
		t.Fatalf("score line = %q, want %q", up.Score, "2/2 reachable")
	}
	if up.Source != "agent" {
		t.Fatalf("source = %q, want agent", up.Source)
	}
	lines := strings.Join(up.Lines, " | ")
	if !strings.Contains(lines, "live: 2") {
		t.Fatalf("expected live count line, got: %s", lines)
	}
	if !strings.Contains(lines, "a100x8=2") {
		t.Fatalf("expected per-class line a100x8=2, got: %s", lines)
	}
	if !strings.Contains(lines, "readiness: 100") {
		t.Fatalf("expected readiness line, got: %s", lines)
	}
}

func TestFromSnapshotDownNodeIsACTIONWithDetail(t *testing.T) {
	boxes := []fleet.Box{
		{ID: "a1", Class: "a100x8"},
		{ID: "a2", Class: "a100x8"},
	}
	reports := []fleet.Report{
		{State: fleet.StateLive, Version: "v1"},
		{State: fleet.StateDown},
	}
	snap := foldRoster(boxes, reports)
	up := FromSnapshot(snap, "ci")

	if up.Verdict != "ACTION" {
		t.Fatalf("verdict = %q, want ACTION when a node is down", up.Verdict)
	}
	if !strings.Contains(up.Detail, "down or unreachable") {
		t.Fatalf("detail = %q, want a down/unreachable headline", up.Detail)
	}
	// A `down` box IS reachable in the fleet model — knowing a box is down is a real,
	// trustworthy observation (only silence/unknown is unreachable). So both boxes
	// count as reachable; the ACTION verdict, not the reachable count, carries "down".
	if up.Score != "2/2 reachable" {
		t.Fatalf("score line = %q, want %q (a down box still returned a report)", up.Score, "2/2 reachable")
	}
	lines := strings.Join(up.Lines, " | ")
	if !strings.Contains(lines, "down: 1") {
		t.Fatalf("expected a down-state count line, got: %s", lines)
	}
}

func TestGradeOfBands(t *testing.T) {
	cases := []struct {
		score int
		want  string
	}{
		{100, "A"}, {90, "A"}, {89, "B"}, {75, "B"}, {74, "C"},
		{50, "C"}, {49, "D"}, {25, "D"}, {24, "F"}, {0, "F"},
	}
	for _, c := range cases {
		if got := gradeOf(c.score); got != c.want {
			t.Errorf("gradeOf(%d) = %q, want %q", c.score, got, c.want)
		}
	}
}

func TestParseSnapshotRoundTrips(t *testing.T) {
	snap := foldRoster(
		[]fleet.Box{{ID: "x", Class: "cpu"}},
		[]fleet.Report{{State: fleet.StateIdle, Version: "v2"}},
	)
	raw, err := json.Marshal(snap)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ParseSnapshot(raw)
	if err != nil {
		t.Fatalf("ParseSnapshot: %v", err)
	}
	if got.Total != 1 || got.Reachable != 1 {
		t.Fatalf("round-trip = total %d reachable %d, want 1/1", got.Total, got.Reachable)
	}
}

func TestParseSnapshotRejectsGarbage(t *testing.T) {
	if _, err := ParseSnapshot([]byte("not json")); err == nil {
		t.Fatal("ParseSnapshot accepted non-JSON")
	}
}

// chdir switches to dir for the test and restores the prior cwd after.
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
