package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/witness"
	"github.com/anthony-chaudhary/fak/internal/workflow"
)

// The fixture run: a pure step feeding an effectful one whose completion is claimed on a
// commit. Both evidence rungs — the journaled output hash and the dos_verify ladder — are
// therefore exercised by one resume.
const workflowResumeSpec = `{
	"name": "ship",
	"tasks": [
		{"id": "build", "op": "emit", "payload": "artifact"},
		{"id": "land",  "op": "emit", "payload": "commit", "needs": ["build"]}
	]
}`

const workflowResumeSHA = "9f3c1a2"

// newWorkflowRunDir writes a run directory whose journal narrates BOTH steps as done, with
// hashes that are internally consistent — so anything that re-executes does so because the
// evidence refused it, never because the fixture was malformed.
func newWorkflowRunDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, workflowSpecFile), []byte(workflowResumeSpec), 0o644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	graph, err := workflow.CompileJSON([]byte(workflowResumeSpec))
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	epoch := workflow.GraphEpoch(graph, "")
	outputs := map[string]string{"build": "artifact", "land": workflowResumeSHA}
	kinds := map[string]workflow.StepKind{"build": workflow.StepPure, "land": workflow.StepEffectful}
	claims := map[string]string{"land": "ancestor:" + workflowResumeSHA}

	var buf bytes.Buffer
	hashes := map[string]string{}
	for _, n := range graph.Nodes {
		h := workflow.HashOutput(outputs[n.ID])
		row := workflow.Entry{
			Run: "ship", Step: n.ID, Kind: kinds[n.ID],
			InputsHash: workflow.StepInputsHash(n, hashes), EpochHash: epoch,
			OutputHash: h, Output: outputs[n.ID], Claim: claims[n.ID], TSMS: 1000,
		}
		if err := workflow.AppendEntry(&buf, row); err != nil {
			t.Fatalf("append entry: %v", err)
		}
		hashes[n.ID] = h
	}
	if err := os.WriteFile(filepath.Join(dir, workflowJournalFile), buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write journal: %v", err)
	}
	return dir
}

// workflowFakeGit drives the dos_verify ladder from a table instead of a repo: the claimed
// commit is always an ancestor of HEAD (a `git revert` leaves it one), and `reverted`
// decides whether history also carries the revert of it.
func workflowFakeGit(reverted bool) witness.Runner {
	return func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch {
		case len(args) >= 4 && args[0] == "merge-base" && args[1] == "--is-ancestor":
			return "", 0, nil
		case len(args) >= 3 && args[0] == "log" && args[1] == "--grep":
			if reverted && strings.HasPrefix(args[2], "This reverts commit ") {
				return "c0ffee1\n", 0, nil
			}
			return "", 0, nil
		}
		return "", 1, nil // anything else is unknown evidence, never a confirmation
	}
}

func runWorkflowResume(t *testing.T, dir string, reverted bool, extra ...string) (string, string, int) {
	t.Helper()
	t.Setenv(witness.CacheFlagEnv, "off") // a shared verdict cache would leak between cases
	var stdout, stderr bytes.Buffer
	argv := append([]string{"--now-ms", "1700000000000"}, extra...)
	argv = append(argv, dir)
	corr := workflowClaimOracle(witness.NewWithRunner(workflowFakeGit(reverted), dir))
	code := workflowResume(&stdout, &stderr, argv, corr)
	return stdout.String(), stderr.String(), code
}

// Acceptance: a step is skipped only when its completion is witnessed, and the report names
// the witness that answered — a journaled output hash for the pure step, the dos_verify
// ladder for the effectful one.
func TestWorkflowResume_SkipsWitnessedSteps(t *testing.T) {
	dir := newWorkflowRunDir(t)
	out, errOut, code := runWorkflowResume(t, dir, false)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errOut)
	}
	if !strings.Contains(out, "skipped=2 executed=0") {
		t.Fatalf("want skipped=2 executed=0, got:\n%s", out)
	}
	if !strings.Contains(out, "skip build") || !strings.Contains(out, "witness=journal-hash:") {
		t.Fatalf("the pure step must cite its journaled output hash:\n%s", out)
	}
	if !strings.Contains(out, "witness=dos_verify:ancestor:"+workflowResumeSHA) {
		t.Fatalf("the effectful step must cite the ladder that answered:\n%s", out)
	}
	// Nothing ran, so nothing may be appended: a resume that skips everything is a
	// read-only operation over the journal.
	rows, err := workflowReadJournalFile(filepath.Join(dir, workflowJournalFile))
	if err != nil || len(rows) != 2 {
		t.Fatalf("journal rows=%d err=%v, want the original 2", len(rows), err)
	}
}

// Acceptance: the journal still narrates the effectful step as done, but its claimed commit
// was reverted, so the ladder stops corroborating and the step re-executes.
func TestWorkflowResume_ReExecutesUnwitnessed(t *testing.T) {
	dir := newWorkflowRunDir(t)
	out, errOut, code := runWorkflowResume(t, dir, true)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errOut)
	}
	if !strings.Contains(out, "skipped=1 executed=1") {
		t.Fatalf("want skipped=1 executed=1, got:\n%s", out)
	}
	if !strings.Contains(out, "exec land") || !strings.Contains(out, "reason="+workflow.ReasonClaimUnverified) {
		t.Fatalf("the reverted step must re-execute citing its closed reason:\n%s", out)
	}
	// The re-execution is journaled, so the next resume can witness it by hash.
	rows, err := workflowReadJournalFile(filepath.Join(dir, workflowJournalFile))
	if err != nil || len(rows) != 3 || rows[2].Step != "land" {
		t.Fatalf("journal rows=%v err=%v, want a fresh row for land", rows, err)
	}
}

// A step whose op has no local executor yields a typed refusal, never prose, and the run
// reports it as a failure rather than a silent success.
func TestWorkflowResume_UnknownOpRefusesTyped(t *testing.T) {
	dir := t.TempDir()
	spec := `{"name":"ship","tasks":[{"id":"call","op":"agent","payload":"x"}]}`
	if err := os.WriteFile(filepath.Join(dir, workflowSpecFile), []byte(spec), 0o644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	out, _, code := runWorkflowResume(t, dir, false, "--json")
	if code != 1 {
		t.Fatalf("exit=%d, want 1 on a failed step:\n%s", code, out)
	}
	if !strings.Contains(out, `"refusal": "op_unregistered"`) {
		t.Fatalf("want a typed refusal, got:\n%s", out)
	}
	if !strings.Contains(out, `"reason": "`+workflow.ReasonUnjournaled+`"`) {
		t.Fatalf("an unjournaled step must cite its reason:\n%s", out)
	}
}

// The fold is pure: two runs over identical journals with the same injected clock produce
// byte-identical reports, including the executed half.
func TestWorkflowFold_Deterministic(t *testing.T) {
	for _, reverted := range []bool{false, true} {
		first, _, code := runWorkflowResume(t, newWorkflowRunDir(t), reverted, "--json")
		if code != 0 {
			t.Fatalf("exit=%d", code)
		}
		second, _, _ := runWorkflowResume(t, newWorkflowRunDir(t), reverted, "--json")
		if first != second {
			t.Fatalf("resume is not deterministic (reverted=%v):\n%s\n---\n%s", reverted, first, second)
		}
		if !strings.Contains(first, `"epoch"`) || !strings.Contains(first, `"journal_rows": 2`) {
			t.Fatalf("report lost its fold witness:\n%s", first)
		}
	}
}

// An epoch label is part of every cache key, so re-running under a new one re-executes
// instead of replaying decisions made under the old policy revision.
func TestWorkflowResumeEpochDriftReExecutes(t *testing.T) {
	dir := newWorkflowRunDir(t)
	out, _, code := runWorkflowResume(t, dir, false, "--epoch", "e2")
	if code != 0 {
		t.Fatalf("exit=%d:\n%s", code, out)
	}
	if !strings.Contains(out, "skipped=0 executed=2") {
		t.Fatalf("want every step re-executed under a new epoch, got:\n%s", out)
	}
	if !strings.Contains(out, "reason="+workflow.ReasonEpochDrift) {
		t.Fatalf("want the epoch-drift reason:\n%s", out)
	}
}

// A missing journal is not a cache: the first run of a directory executes everything.
func TestWorkflowResumeWithoutJournalExecutesAll(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, workflowSpecFile), []byte(workflowResumeSpec), 0o644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	out, _, code := runWorkflowResume(t, dir, false)
	if code != 0 {
		t.Fatalf("exit=%d:\n%s", code, out)
	}
	if !strings.Contains(out, "skipped=0 executed=2") {
		t.Fatalf("want a cold run to execute everything, got:\n%s", out)
	}
	// Its own journal now witnesses it: a second resume replays both steps by hash.
	again, _, code := runWorkflowResume(t, dir, false)
	if code != 0 || !strings.Contains(again, "skipped=2 executed=0") {
		t.Fatalf("a re-resume must replay the journaled steps, got exit=%d:\n%s", code, again)
	}
}

func TestWorkflowResumeUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := workflowResume(&stdout, &stderr, nil, nil); code != 2 {
		t.Fatalf("exit=%d, want 2 without a run directory", code)
	}
	if !strings.Contains(stderr.String(), "usage: fak workflow resume") {
		t.Fatalf("stderr=%q", stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := runWorkflow(&stdout, &stderr, strings.NewReader(""), []string{"resume"}); code != 2 {
		t.Fatalf("the verb must route resume, exit=%d", code)
	}
}
