package main

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/steerpr"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// The two stamped SHAs in prPlanFakeLog (defined in release_prplan_test.go): the
// gateway unit is feat(aaa1111) + fix(bbb2222).
const (
	steerFeatSHA = "aaa1111111111111111111111111111111111111"
	steerFixSHA  = "bbb2222222222222222222222222222222222222"
)

func steerFakeVerdicts(feat steerpr.Verdict) func(string, string, string) map[string]steerpr.Verdict {
	return func(_, _, _ string) map[string]steerpr.Verdict {
		return map[string]steerpr.Verdict{
			steerFeatSHA: feat,
			steerFixSHA:  steerpr.VerdictWitnessed,
		}
	}
}

func withSteerFakes(t *testing.T, log string, feat steerpr.Verdict) {
	t.Helper()
	origGit, origVerdicts := releasePRPlanGit, steerPRsVerdicts
	releasePRPlanGit = prPlanFakeGit(log)
	steerPRsVerdicts = steerFakeVerdicts(feat)
	t.Cleanup(func() { releasePRPlanGit, steerPRsVerdicts = origGit, origVerdicts })
}

func TestSteerActorFlagHelpPreservesCommandWording(t *testing.T) {
	tests := []struct {
		name string
		run  func(io.Writer, io.Writer, []string) int
		want string
	}{
		{name: "comment", run: runSteerComment, want: "who is annotating"},
		{name: "pause", run: runSteerPause, want: "who is pausing"},
		{name: "resume", run: runSteerResume, want: "who is releasing"},
		{name: "ack", run: runSteerAck, want: "who looked"},
		{name: "redirect", run: runSteerRedirect, want: "who is steering"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := tt.run(&stdout, &stderr, []string{"--help"}); code != 2 {
				t.Fatalf("help exit = %d, want 2", code)
			}
			want := tt.want + " (default: git config user.name; the row must be attributable)"
			if !strings.Contains(stderr.String(), want) {
				t.Fatalf("help missing %q:\n%s", want, stderr.String())
			}
		})
	}
}

// An unwitnessed member floors its whole unit to RESIDUAL (the worst member
// wins), and the operator view surfaces exactly that unit as owing attention.
func TestSteerPRsBandsWorstMemberResidual(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)

	view, err := buildSteerPRsView(t.TempDir(), "baseref", "headref")
	if err != nil {
		t.Fatalf("buildSteerPRsView: %v", err)
	}
	if view["schema"] != steerpr.Schema {
		t.Fatalf("schema = %v, want %s", view["schema"], steerpr.Schema)
	}
	if view["residual_count"].(int) != 1 {
		t.Fatalf("residual_count = %v, want 1", view["residual_count"])
	}
	units := view["units"].([]steerpr.Unit)
	if len(units) != 1 || units[0].Band != steerpr.BandResidual {
		t.Fatalf("units = %#v, want one RESIDUAL gateway unit", units)
	}
	// The unit folds to its worst member, but the individual verdicts survive:
	// the fix is witnessed, the feat is the one that reds the unit.
	if units[0].Commits[0].Verdict != steerpr.VerdictUnwitnessed {
		t.Fatalf("feat verdict = %q, want unwitnessed (it is what reds the unit)", units[0].Commits[0].Verdict)
	}
}

// When every member is witnessed the unit clears, and --check passes.
func TestSteerPRsAllWitnessedClears(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictWitnessed)

	view, err := buildSteerPRsView(t.TempDir(), "baseref", "headref")
	if err != nil {
		t.Fatalf("buildSteerPRsView: %v", err)
	}
	if view["residual_count"].(int) != 0 {
		t.Fatalf("residual_count = %v, want 0", view["residual_count"])
	}
	if units := view["units"].([]steerpr.Unit); units[0].Band != steerpr.BandCleared {
		t.Fatalf("band = %q, want CLEARED", units[0].Band)
	}
}

// --check is a REPORTING gate: exit 1 iff a RESIDUAL unit exists, 0 otherwise. It
// reports; it never blocks a merge.
func TestSteerPRsCheckReportsResidual(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--check", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("exit = %d, want 1 (a RESIDUAL unit present); stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "RESIDUAL") {
		t.Fatalf("check refusal should name the RESIDUAL band: %s", stderr.String())
	}

	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictWitnessed)
	stdout.Reset()
	stderr.Reset()
	if code := runSteerPRs(&stdout, &stderr, []string{"--check", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0 (no RESIDUAL unit); stderr=%s", code, stderr.String())
	}
}

func TestSteerPRsJSONPayload(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--json", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var payload struct {
		Schema        string `json:"schema"`
		ResidualCount int    `json:"residual_count"`
		Units         []struct {
			Leaf string `json:"leaf"`
			Band string `json:"band"`
		} `json:"units"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("decode json: %v\n%s", err, stdout.String())
	}
	if payload.Schema != steerpr.Schema || payload.ResidualCount != 1 {
		t.Fatalf("payload = %#v, want schema %s + residual 1", payload, steerpr.Schema)
	}
	if len(payload.Units) != 1 || payload.Units[0].Band != string(steerpr.BandResidual) {
		t.Fatalf("units = %#v, want one RESIDUAL gateway unit", payload.Units)
	}
}

func TestSteerPRsHumanRenderWorstFirst(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"# Forming operator PRs — baseref..headref",
		"1 RESIDUAL",
		"## [RESIDUAL] gateway",
		"[UNWITNESSED]",
		"Closes #1146.",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

// An empty range forms nothing and never trips --check.
func TestSteerPRsEmptyRange(t *testing.T) {
	withSteerFakes(t, "", steerpr.VerdictWitnessed)
	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--check", "--base", "sameref", "--head", "sameref2"}); code != 0 {
		t.Fatalf("exit = %d, want 0 on empty range; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Nothing forming") {
		t.Fatalf("empty-range render wrong:\n%s", stdout.String())
	}
}

// The band maps a dos commit-audit row through the SAME keep-bit the dispatch
// sweep uses: witnessed only on OK + diff-witnessed; a subject-only claim is
// RESIDUAL; everything else abstains.
func TestMapAuditVerdictMirrorsKeepBit(t *testing.T) {
	cases := []struct {
		verdict, witness string
		want             steerpr.Verdict
	}{
		{"OK", "diff-witnessed", steerpr.VerdictWitnessed},
		{"ok", "diff-witnessed", steerpr.VerdictWitnessed},
		{"OK", "subject-only", steerpr.VerdictAbstain}, // OK but not diff-witnessed is not a keep
		{"CLAIM_UNWITNESSED", "subject-only", steerpr.VerdictUnwitnessed},
		{"ABSTAIN", "", steerpr.VerdictAbstain},
	}
	for _, c := range cases {
		if got := mapAuditVerdict(c.verdict, c.witness); got != c.want {
			t.Fatalf("mapAuditVerdict(%q,%q) = %q, want %q", c.verdict, c.witness, got, c.want)
		}
	}
}

func TestMatchVerdictByShortSHAPrefix(t *testing.T) {
	verdicts := map[string]steerpr.Verdict{"aaa1111": steerpr.VerdictUnwitnessed}
	if v, ok := matchVerdict(steerFeatSHA, verdicts); !ok || v != steerpr.VerdictUnwitnessed {
		t.Fatalf("prefix match = %q,%v, want unwitnessed,true", v, ok)
	}
	if _, ok := matchVerdict(steerFixSHA, verdicts); ok {
		t.Fatalf("unrelated SHA should not match")
	}
}

// runSteer dispatches the prs subcommand and rejects an unknown one.
func TestRunSteerDispatch(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runSteer(&stdout, &stderr, []string{"bogus"}); code != 2 {
		t.Fatalf("unknown subcommand exit = %d, want 2", code)
	}
	stdout.Reset()
	stderr.Reset()
	if code := runSteer(&stdout, &stderr, []string{"--help"}); code != 0 {
		t.Fatalf("--help exit = %d, want 0", code)
	}
}

// This captured operator render is #5114's spine witness: a live issue-bound
// trajctl objective is joined onto a CLEARED overlay unit, and its declining W3
// curve remains emphatic rather than disappearing behind the cleared band.
func TestSteerPRsRendersDriftCurveOnClearedUnit(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictWitnessed)
	orig := steerPRsTrajState
	steerPRsTrajState = func(string) trajctl.State {
		return trajctl.Fold([]trajctl.Row{
			trajctl.ObjectiveRecord(trajctl.Objective{ID: "issue-1146", Statement: "ship gateway", Status: trajctl.StatusActive}),
			trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "issue-1146", Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3, Value: .8, UnixMillis: 1}),
			trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "issue-1146", Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3, Value: .5, UnixMillis: 2}),
		})
	}
	t.Cleanup(func() { steerPRsTrajState = orig })

	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{
		"## [CLEARED] gateway",
		"**DRIFT HIDDEN BY CLEARED BAND**",
		"curve: DRIFT [W3]",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("captured render missing %q:\n%s", want, got)
		}
	}
}

func TestSteerPRsUnitWithoutObjectiveHasNoCurve(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictWitnessed)
	orig := steerPRsTrajState
	steerPRsTrajState = func(string) trajctl.State { return trajctl.State{} }
	t.Cleanup(func() { steerPRsTrajState = orig })

	view, err := buildSteerPRsView(t.TempDir(), "baseref", "headref")
	if err != nil {
		t.Fatalf("buildSteerPRsView: %v", err)
	}
	units := view["units"].([]steerpr.Unit)
	if len(units) != 1 || units[0].Curve != nil {
		t.Fatalf("units = %#v, want one unit with no curve", units)
	}
	if got := writeSteerPRs(view, 20); strings.Contains(got, "curve:") || strings.Contains(got, "DRIFT HIDDEN") {
		t.Fatalf("curve-free unit rendered a curve warning:\n%s", got)
	}
}
