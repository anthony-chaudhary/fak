package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// TestTrajctlUsageAndUnknown pins the no-ledger control paths: bare invocation
// and an unknown subcommand are usage errors (exit 2); help is exit 0.
func TestTrajctlUsageAndUnknown(t *testing.T) {
	cases := []struct {
		argv []string
		want int
	}{
		{nil, 2},
		{[]string{"bogus"}, 2},
		{[]string{"declare"}, 2},
		{[]string{"close"}, 2},
		{[]string{"list", "--status", "bogus"}, 2},
		{[]string{"--help"}, 0},
		{[]string{"help"}, 0},
	}
	for _, c := range cases {
		var out, errb bytes.Buffer
		if got := runTrajctl(&out, &errb, c.argv); got != c.want {
			t.Fatalf("runTrajctl(%v) = %d, want %d (stderr=%q)", c.argv, got, c.want, errb.String())
		}
	}
}

// TestTrajctlDeclareListCloseEndToEnd is the repro-then-fix witness for #2535:
// before this change there was no CLI surface at all over internal/trajctl's
// ledger (no cmd/fak/trajctl.go existed), so `declare` -> `list` -> `close`
// could not run end to end. This drives the real ledger file through all
// three verbs and checks the transcript at each step.
func TestTrajctlDeclareListCloseEndToEnd(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")

	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{
		"declare", "--id", "obj-1", "--statement", "ship the spine",
		"--plan", "declare", "--plan", "curve",
		"--budget-turns", "10", "--budget-tokens", "5000",
		"--ledger", ledger,
	}); code != 0 {
		t.Fatalf("declare exit=%d stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "declared obj-1 (status=active)") {
		t.Fatalf("declare output = %q", out.String())
	}

	// list defaults to the open set and shows the declared objective.
	out.Reset()
	errb.Reset()
	if code := runTrajctl(&out, &errb, []string{"list", "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("list exit=%d stderr=%q", code, errb.String())
	}
	var objs []trajctl.Objective
	if err := json.Unmarshal(out.Bytes(), &objs); err != nil {
		t.Fatalf("list --json: %v (out=%q)", err, out.String())
	}
	if len(objs) != 1 || objs[0].ID != "obj-1" || objs[0].Status != trajctl.StatusActive {
		t.Fatalf("list --json = %+v, want one active obj-1", objs)
	}
	if len(objs[0].Plan) != 2 || objs[0].Plan[0].ID != "phase-1" || objs[0].Plan[1].ID != "phase-2" {
		t.Fatalf("list --json plan = %+v", objs[0].Plan)
	}
	if objs[0].Budget.Turns != 10 || objs[0].Budget.Tokens != 5000 {
		t.Fatalf("list --json budget = %+v", objs[0].Budget)
	}

	// Witness both declared phases before closing. `met` is adjudicated against the
	// declared plan (internal/depthadmit, DEPTH_NOT_CARRIED), so a close on a plan
	// with no witnessed phase is refused now — see trajctl_depth_test.go for that
	// direction. This test is about the lifecycle, so give it the carried plan a
	// clean close costs.
	witnessPhases(t, ledger, "obj-1", "phase-1", "phase-2")

	// close flips the status; the objective drops out of the default open list.
	out.Reset()
	errb.Reset()
	if code := runTrajctl(&out, &errb, []string{"close", "--id", "obj-1", "--ledger", ledger}); code != 0 {
		t.Fatalf("close exit=%d stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "closed obj-1 (status=met)") {
		t.Fatalf("close output = %q", out.String())
	}

	out.Reset()
	errb.Reset()
	if code := runTrajctl(&out, &errb, []string{"list", "--ledger", ledger}); code != 0 {
		t.Fatalf("list exit=%d stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "no objectives") {
		t.Fatalf("list after close = %q, want the open set empty", out.String())
	}

	// --status all still shows the closed objective.
	out.Reset()
	errb.Reset()
	if code := runTrajctl(&out, &errb, []string{"list", "--ledger", ledger, "--status", "all", "--json"}); code != 0 {
		t.Fatalf("list --status all exit=%d stderr=%q", code, errb.String())
	}
	objs = nil
	if err := json.Unmarshal(out.Bytes(), &objs); err != nil {
		t.Fatalf("list --status all --json: %v", err)
	}
	if len(objs) != 1 || objs[0].Status != trajctl.StatusMet {
		t.Fatalf("list --status all --json = %+v, want one met obj-1", objs)
	}
}

// TestTrajctlCloseUnknownObjective fails closed: closing an id that was never
// declared is an error, not a silent no-op ledger write.
func TestTrajctlCloseUnknownObjective(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{"close", "--id", "ghost", "--ledger", ledger}); code == 0 {
		t.Fatalf("close of an undeclared id should fail, got exit 0 (out=%q)", out.String())
	}
}

// TestTrajctlCurveEndToEnd drives the curve fold through the CLI: a ledger with a
// declining objective and a rising one folds into a schema-pinned, worst-first
// report (DRIFT before HEALTHY), a single --objective fold reports that
// objective's signal, and an unknown id fails closed (exit 1).
func TestTrajctlCurveEndToEnd(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	mustAppend := func(row trajctl.Row) {
		t.Helper()
		if err := trajctl.Append(ledger, row); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	w3 := func(id string, v float64, ms int64) trajctl.Row {
		return trajctl.ScoreRecord(trajctl.ScoreRow{
			ObjectiveID: id, Value: v, Method: trajctl.CommitScorerMethod,
			Version: trajctl.CommitScorerVersion, Witness: trajctl.W3, UnixMillis: ms,
		})
	}
	mustAppend(trajctl.ObjectiveRecord(trajctl.Objective{ID: "obj-drift", Statement: "declining", Status: trajctl.StatusActive}))
	mustAppend(w3("obj-drift", 0.75, 1000))
	mustAppend(w3("obj-drift", 0.25, 2000))
	mustAppend(trajctl.ObjectiveRecord(trajctl.Objective{ID: "obj-healthy", Statement: "rising", Status: trajctl.StatusActive}))
	mustAppend(w3("obj-healthy", 0.25, 1000))
	mustAppend(w3("obj-healthy", 0.75, 2000))

	// No id: worst-first listing, DRIFT ahead of HEALTHY, pinned schema.
	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{"curve", "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("curve --json exit=%d stderr=%q", code, errb.String())
	}
	var rep trajctl.CurveReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("curve --json: %v (out=%q)", err, out.String())
	}
	if rep.Schema != trajctl.CurveSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, trajctl.CurveSchema)
	}
	if len(rep.Objectives) != 2 ||
		rep.Objectives[0].ObjectiveID != "obj-drift" || rep.Objectives[0].Signal != trajctl.SignalDrift ||
		rep.Objectives[1].ObjectiveID != "obj-healthy" || rep.Objectives[1].Signal != trajctl.SignalHealthy {
		t.Fatalf("worst-first report = %+v", rep.Objectives)
	}

	// Single objective fold.
	out.Reset()
	errb.Reset()
	if code := runTrajctl(&out, &errb, []string{"curve", "--objective", "obj-drift", "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("curve --objective exit=%d stderr=%q", code, errb.String())
	}
	rep = trajctl.CurveReport{}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("curve --objective --json: %v", err)
	}
	if len(rep.Objectives) != 1 || rep.Objectives[0].Signal != trajctl.SignalDrift {
		t.Fatalf("single fold = %+v, want one DRIFT", rep.Objectives)
	}

	// Text render carries the signal token.
	out.Reset()
	errb.Reset()
	if code := runTrajctl(&out, &errb, []string{"curve", "--ledger", ledger}); code != 0 {
		t.Fatalf("curve text exit=%d stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "DRIFT") || !strings.Contains(out.String(), "HEALTHY") {
		t.Fatalf("curve text = %q, want DRIFT and HEALTHY", out.String())
	}

	// Unknown objective fails closed.
	out.Reset()
	errb.Reset()
	if code := runTrajctl(&out, &errb, []string{"curve", "--objective", "ghost", "--ledger", ledger}); code != 1 {
		t.Fatalf("curve --objective ghost exit=%d, want 1 (stderr=%q)", code, errb.String())
	}
}

// TestTrajctlDeclareFromGoal imports loopdrive.Spec's Objective/Plan/Budget
// from a real GOAL.md, defaulting --id to the frontmatter 'loop:' id.
func TestTrajctlDeclareFromGoal(t *testing.T) {
	dir := t.TempDir()
	goalPath := filepath.Join(dir, "GOAL.md")
	goal := "---\n" +
		"loop: goal-loop-1\n" +
		"witness: commit-audit\n" +
		"budget: { max_iters: 20, max_tokens: 100000 }\n" +
		"---\n" +
		"# Objective\n" +
		"ship the trajctl declare verb\n" +
		"\n" +
		"# Plan\n" +
		"- [ ] wire declare\n" +
		"- [ ] wire close\n"
	if err := os.WriteFile(goalPath, []byte(goal), 0o644); err != nil {
		t.Fatalf("write GOAL.md: %v", err)
	}
	ledger := filepath.Join(dir, "trajctl.jsonl")

	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{"declare", "--from-goal", goalPath, "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("declare --from-goal exit=%d stderr=%q", code, errb.String())
	}
	var obj trajctl.Objective
	if err := json.Unmarshal(out.Bytes(), &obj); err != nil {
		t.Fatalf("declare --from-goal --json: %v (out=%q)", err, out.String())
	}
	if obj.ID != "goal-loop-1" {
		t.Fatalf("obj.ID = %q, want the GOAL.md loop id", obj.ID)
	}
	if obj.Statement != "ship the trajctl declare verb" {
		t.Fatalf("obj.Statement = %q", obj.Statement)
	}
	if len(obj.Plan) != 2 {
		t.Fatalf("obj.Plan = %+v, want 2 phases", obj.Plan)
	}
	if obj.Budget.Turns != 20 || obj.Budget.Tokens != 100000 {
		t.Fatalf("obj.Budget = %+v", obj.Budget)
	}

	// --from-goal together with --statement is a usage conflict.
	out.Reset()
	errb.Reset()
	if code := runTrajctl(&out, &errb, []string{"declare", "--from-goal", goalPath, "--statement", "x", "--ledger", ledger}); code != 2 {
		t.Fatalf("declare --from-goal --statement exit=%d, want 2 (stderr=%q)", code, errb.String())
	}
}
