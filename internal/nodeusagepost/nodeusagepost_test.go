package nodeusagepost

import (
	"encoding/json"
	"fmt"
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
	if !strings.Contains(lines, "reporting: 2/2") {
		t.Fatalf("expected reporting line, got: %s", lines)
	}
	if !strings.Contains(lines, "a100x8=2") {
		t.Fatalf("expected per-class line a100x8=2, got: %s", lines)
	}
	if !strings.Contains(lines, "readiness: 100") {
		t.Fatalf("expected readiness line, got: %s", lines)
	}
}

// TestFromSnapshotAllSilentIsVisibilityGapNotOutage is the bug regression: a fleet
// where every box is silent (no report → state unknown) must read as a VISIBILITY GAP
// (no grade, no verdict → the neutral :bar_chart: glyph), NOT a red F/ACTION outage.
func TestFromSnapshotAllSilentIsVisibilityGapNotOutage(t *testing.T) {
	boxes := []fleet.Box{{ID: "a1", Class: "a100x8"}, {ID: "a2", Class: "metal"}}
	// No reports at all → every box folds to unknown.
	snap := foldRoster(boxes, nil)
	up := FromSnapshot(snap, "ci")

	if up.Grade != "" {
		t.Fatalf("grade = %q, want empty (visibility gap is not a graded failure)", up.Grade)
	}
	if up.Verdict != "" {
		t.Fatalf("verdict = %q, want empty (silence is not ACTION)", up.Verdict)
	}
	if strings.Contains(up.Detail, "down or unreachable") {
		t.Fatalf("detail must NOT call silent boxes down: %q", up.Detail)
	}
	if !strings.Contains(up.Detail, "not down") {
		t.Fatalf("detail = %q, want it to clarify the boxes are silent, not down", up.Detail)
	}
	lines := strings.Join(up.Lines, " | ")
	if !strings.Contains(lines, "reporting: 0/2") {
		t.Fatalf("expected a 0/N reporting line, got: %s", lines)
	}
	if !strings.Contains(lines, "populate liveness") {
		t.Fatalf("expected the populate-liveness guidance, got: %s", lines)
	}
}

func TestFromSnapshotRealDownIsACTIONNamedFromCount(t *testing.T) {
	boxes := []fleet.Box{{ID: "a1"}, {ID: "a2"}}
	reports := []fleet.Report{
		{State: fleet.StateLive, Version: "v1"},
		{State: fleet.StateDown},
	}
	snap := foldRoster(boxes, reports)
	up := FromSnapshot(snap, "ci")

	if up.Verdict != "ACTION" {
		t.Fatalf("verdict = %q, want ACTION when a node reported down", up.Verdict)
	}
	if up.Grade != "F" {
		t.Fatalf("grade = %q, want F (a reported-down box forces red)", up.Grade)
	}
	if !strings.Contains(up.Detail, "reported down") {
		t.Fatalf("detail = %q, want a 'reported down' headline built from the count", up.Detail)
	}
	if strings.Contains(up.Detail, "unreachable") {
		t.Fatalf("detail must name down from the count, not the conflated title: %q", up.Detail)
	}
}

// TestFromSnapshotDownWithErrorIsStillACTION is the down-hidden-as-silence regression:
// a box that reported `down` AND carries a read error drives snap.Reachable to 0, so a
// Reachable-based silence check would route a total outage into the no-visibility
// branch and print "not down". Classifying off ByState[StateDown] keeps it ACTION.
func TestFromSnapshotDownWithErrorIsStillACTION(t *testing.T) {
	// Hand-build the snapshot shape directly (a deserialized/bridge-produced snapshot),
	// since fleet.Fold derives Reachable from the report itself.
	snap := fleet.Snapshot{
		Schema:    fleet.SnapshotSchema,
		Total:     2,
		Reachable: 0, // both boxes errored → not reachable, even though they reported down
		ByState:   map[fleet.State]int{fleet.StateDown: 2},
	}
	up := FromSnapshot(snap, "ci")

	if up.Verdict != "ACTION" {
		t.Fatalf("verdict = %q, want ACTION (down-with-error is a real outage)", up.Verdict)
	}
	if up.Grade != "F" {
		t.Fatalf("grade = %q, want F for an all-down fleet", up.Grade)
	}
	if strings.Contains(up.Detail, "not down") {
		t.Fatalf("must not print 'not down' over a fleet that reported down: %q", up.Detail)
	}
	if !strings.Contains(up.Detail, "reported down") {
		t.Fatalf("detail = %q, want it to name the down boxes", up.Detail)
	}
}

// TestFromSnapshotMostlyHealthyOneDownNotMaskedGreen is the down-hidden-as-green
// regression: 9 live + 1 down scores ~92, and the card renderer picks its glyph from
// the grade prefix BEFORE the verdict — so an A/B grade would render green despite
// ACTION. The grade must be clamped so the glyph is red.
func TestFromSnapshotMostlyHealthyOneDownNotMaskedGreen(t *testing.T) {
	var boxes []fleet.Box
	var reports []fleet.Report
	for i := 0; i < 9; i++ {
		boxes = append(boxes, fleet.Box{ID: fmt.Sprintf("h%d", i)})
		reports = append(reports, fleet.Report{State: fleet.StateLive, Version: "v1"})
	}
	boxes = append(boxes, fleet.Box{ID: "d0"})
	reports = append(reports, fleet.Report{State: fleet.StateDown})
	snap := foldRoster(boxes, reports)
	up := FromSnapshot(snap, "ci")

	if up.Verdict != "ACTION" {
		t.Fatalf("verdict = %q, want ACTION with a real down present", up.Verdict)
	}
	if up.Grade == "A" || up.Grade == "B" {
		t.Fatalf("grade = %q, must NOT be A/B (the renderer would paint a real down green); score=%d", up.Grade, snap.Score)
	}
}

func TestFromSnapshotSkewAmongReportersIsACTIONNotGreen(t *testing.T) {
	var boxes []fleet.Box
	var reports []fleet.Report
	for i := 0; i < 9; i++ {
		boxes = append(boxes, fleet.Box{ID: fmt.Sprintf("h%d", i)})
		reports = append(reports, fleet.Report{State: fleet.StateLive, Version: "v1"})
	}
	boxes = append(boxes, fleet.Box{ID: "skew"})
	reports = append(reports, fleet.Report{State: fleet.StateLive, Version: "v2"}) // off the modal version
	snap := foldRoster(boxes, reports)
	up := FromSnapshot(snap, "ci")

	if up.Verdict != "ACTION" {
		t.Fatalf("verdict = %q, want ACTION for version skew among reporters", up.Verdict)
	}
	if up.Grade == "A" || up.Grade == "B" {
		t.Fatalf("grade = %q, must be clamped below B so skew is not painted green/yellow", up.Grade)
	}
}

func TestFromSnapshotPartialVisibilityIsOKWithReportingLine(t *testing.T) {
	boxes := []fleet.Box{{ID: "a1"}, {ID: "a2"}, {ID: "a3"}}
	reports := []fleet.Report{
		{State: fleet.StateLive, Version: "v1"},
		// a2, a3 silent (no report aligned → unknown)
	}
	snap := foldRoster(boxes, reports)
	up := FromSnapshot(snap, "ci")

	if up.Verdict != "OK" {
		t.Fatalf("verdict = %q, want OK (silence alone never escalates)", up.Verdict)
	}
	lines := strings.Join(up.Lines, " | ")
	if !strings.Contains(lines, "reporting: 1/3 (2 silent=unknown, not down)") {
		t.Fatalf("expected the partial-visibility reporting line, got: %s", lines)
	}
	if !strings.Contains(up.Detail, "silent (unknown, not down)") {
		t.Fatalf("detail = %q, want the non-alarming partial-coverage headline", up.Detail)
	}
}

func TestFromSnapshotEmptyRosterIsNeutralNotF(t *testing.T) {
	snap := foldRoster(nil, nil)
	up := FromSnapshot(snap, "ci")

	if up.Grade == "F" {
		t.Fatalf("grade = %q, an empty roster is a config state, not an F outage", up.Grade)
	}
	if strings.HasPrefix(up.Grade, "A") || strings.HasPrefix(up.Grade, "B") {
		t.Fatalf("grade = %q must not prefix-match A/B (would render green for an empty roster)", up.Grade)
	}
	if up.Detail != "no nodes in the roster" {
		t.Fatalf("detail = %q, want 'no nodes in the roster'", up.Detail)
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

func TestClampBelowB(t *testing.T) {
	cases := map[string]string{"A": "C", "B": "C", "C": "C", "D": "D", "F": "F", "": ""}
	for in, want := range cases {
		if got := clampBelowB(in); got != want {
			t.Errorf("clampBelowB(%q) = %q, want %q", in, got, want)
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
