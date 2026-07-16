package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/modedebt"
)

func writeModeDebtScorecard(t *testing.T, dir string, sc modedebt.Scorecard) string {
	t.Helper()
	b, err := json.Marshal(sc)
	if err != nil {
		t.Fatalf("marshal scorecard: %v", err)
	}
	path := filepath.Join(dir, "mode-debt.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write scorecard: %v", err)
	}
	return path
}

func runModeDebtDispatchJSON(t *testing.T, args []string) dogfoodissues.Result {
	t.Helper()
	var out, errb bytes.Buffer
	code := runModeDebtDispatch(&out, &errb, args)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var result dogfoodissues.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatalf("json output: %v\n%s", err, out.String())
	}
	return result
}

// TestModeDebtDispatchOneHardUnliftedDial proves the positive direction: a
// scorecard with exactly one HARD un-lifted dial -- plus a HARD *lifted* dial and a
// *soft* un-lifted dial that must NOT dispatch -- yields exactly one deduped
// candidate, keyed on the content-stable mode-debt/<slug> marker and routed to the
// permission-regime backlog (#2389 / #2405). A re-run converges on the same key.
func TestModeDebtDispatchOneHardUnliftedDial(t *testing.T) {
	dir := t.TempDir()
	sc := modedebt.Scorecard{
		Schema: modedebt.Schema,
		Dials: []modedebt.Dial{
			{Slug: "auto-approve-edits", Name: "auto-approve edits", Grade: "HARD", Lifted: false, Regime: "yolo"},
			{Slug: "auto-approve-bash", Grade: "HARD", Lifted: true},  // lifted -> excluded
			{Slug: "prompt-on-network", Grade: "soft", Lifted: false}, // soft -> excluded
		},
	}
	scorecard := writeModeDebtScorecard(t, dir, sc)

	result := runModeDebtDispatchJSON(t, []string{"--scorecard", scorecard, "--cap", "10", "--parent-issue", "4397", "--parent-baseline-points", "20", "--completion-standard", "development", "--json"})
	if result.Mode != "dry-run" {
		t.Fatalf("mode=%q, want dry-run", result.Mode)
	}
	if len(result.Planned) != 1 {
		t.Fatalf("planned len=%d, want 1 (soft + lifted dials must not dispatch); skipped=%v", len(result.Planned), result.Skipped)
	}
	const wantKey = "mode-debt/auto-approve-edits"
	if got := result.Planned[0].Key; got != wantKey {
		t.Fatalf("planned key=%q, want %q", got, wantKey)
	}
	if act := result.Planned[0].Action; act != "create" {
		t.Fatalf("action=%q, want create", act)
	}

	// Target assertion: the mapped ActionItem routes to the permission-regime epics.
	dials := modedebt.SelectHardUnlifted(sc)
	if len(dials) != 1 {
		t.Fatalf("SelectHardUnlifted len=%d, want 1", len(dials))
	}
	if got := dials[0].Key(); got != wantKey {
		t.Fatalf("dial key=%q, want %q", got, wantKey)
	}
	item := dials[0].ToActionItem(scorecard)
	for _, want := range []string{modedebt.TargetRegime, modedebt.TargetRegimeBacklog} {
		if !strings.Contains(item.ParentRef, want) {
			t.Fatalf("ActionItem ParentRef %q missing target %q", item.ParentRef, want)
		}
	}

	// Re-run converges: feeding the previously-filed issue back as an existing row
	// makes the dispatcher UPDATE the same marker instead of opening a duplicate.
	existing := []dogfoodissues.Issue{{
		Number: 9001,
		Title:  item.Title,
		Body:   dogfoodissues.IssueBody(item),
		State:  "open",
	}}
	existingPath := filepath.Join(dir, "existing.json")
	b, err := json.Marshal(existing)
	if err != nil {
		t.Fatalf("marshal existing: %v", err)
	}
	if err := os.WriteFile(existingPath, b, 0o644); err != nil {
		t.Fatalf("write existing: %v", err)
	}
	rerun := runModeDebtDispatchJSON(t, []string{"--scorecard", scorecard, "--cap", "10", "--existing-json", existingPath, "--parent-issue", "4397", "--parent-baseline-points", "20", "--completion-standard", "development", "--json"})
	if len(rerun.Planned) != 1 {
		t.Fatalf("rerun planned len=%d, want 1 (dedup, not duplicate)", len(rerun.Planned))
	}
	if rerun.Planned[0].Action != "update" {
		t.Fatalf("rerun action=%q, want update (dedup by stable marker)", rerun.Planned[0].Action)
	}
	if rerun.Planned[0].Key != wantKey {
		t.Fatalf("rerun key=%q, want %q", rerun.Planned[0].Key, wantKey)
	}
}

// TestModeDebtDispatchCleanScorecardYieldsNothing proves the negative direction: a
// CLEAN scorecard -- every HARD dial lifted, soft dials aside -- files nothing.
func TestModeDebtDispatchCleanScorecardYieldsNothing(t *testing.T) {
	dir := t.TempDir()
	sc := modedebt.Scorecard{
		Schema: modedebt.Schema,
		Dials: []modedebt.Dial{
			{Slug: "auto-approve-edits", Grade: "HARD", Lifted: true},
			{Slug: "auto-approve-bash", Grade: "HARD", Lifted: true},
			{Slug: "prompt-on-network", Grade: "soft", Lifted: false},
		},
	}
	scorecard := writeModeDebtScorecard(t, dir, sc)
	result := runModeDebtDispatchJSON(t, []string{"--scorecard", scorecard, "--parent-issue", "4397", "--parent-baseline-points", "20", "--completion-standard", "development", "--json"})
	if len(result.Planned) != 0 {
		t.Fatalf("planned len=%d, want 0 for a CLEAN fully-lifted scorecard", len(result.Planned))
	}
	if got := len(modedebt.SelectHardUnlifted(sc)); got != 0 {
		t.Fatalf("SelectHardUnlifted len=%d, want 0 for a CLEAN scorecard", got)
	}
}

// TestModeDebtDispatchCapBoundsFanout proves --cap bounds a noisy scorecard: three
// HARD un-lifted dials capped at 2 yield exactly two candidates.
func TestModeDebtDispatchCapBoundsFanout(t *testing.T) {
	dir := t.TempDir()
	sc := modedebt.Scorecard{
		Schema: modedebt.Schema,
		Dials: []modedebt.Dial{
			{Slug: "dial-a", Grade: "HARD", Lifted: false},
			{Slug: "dial-b", Grade: "HARD", Lifted: false},
			{Slug: "dial-c", Grade: "HARD", Lifted: false},
		},
	}
	scorecard := writeModeDebtScorecard(t, dir, sc)
	result := runModeDebtDispatchJSON(t, []string{"--scorecard", scorecard, "--cap", "2", "--parent-issue", "4397", "--parent-baseline-points", "20", "--completion-standard", "development", "--json"})
	if len(result.Planned) != 2 {
		t.Fatalf("planned len=%d, want 2 under --cap 2", len(result.Planned))
	}
}

// TestModeDebtDispatchMissingScorecardFailsClosed proves absence fails closed: an
// absent scorecard is an error, not a silently-empty dispatch.
func TestModeDebtDispatchMissingScorecardFailsClosed(t *testing.T) {
	dir := t.TempDir()
	var out, errb bytes.Buffer
	code := runModeDebtDispatch(&out, &errb, []string{"--scorecard", filepath.Join(dir, "nope.json"), "--json"})
	if code == 0 {
		t.Fatalf("missing scorecard exit=0, want non-zero (fail closed); stdout=%s", out.String())
	}
}
