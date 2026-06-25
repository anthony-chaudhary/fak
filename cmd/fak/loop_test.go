package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestLoopStatusJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendLoopTestEvent(t, path, loopmgr.Event{LoopID: "issue-dispatch/default", Kind: loopmgr.EventFire})
	appendLoopTestEvent(t, path, loopmgr.Event{LoopID: "issue-dispatch/default", Kind: loopmgr.EventAdmit, RunID: "run-1", Status: loopmgr.StatusAdmitted})
	appendLoopTestEvent(t, path, loopmgr.Event{LoopID: "issue-dispatch/default", Kind: loopmgr.EventWitness, RunID: "run-1", Status: loopmgr.StatusWitnessedDone})

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"status", "--ledger", path, "--json"})
	if code != 0 {
		t.Fatalf("runLoop code=%d stderr=%s", code, stderr.String())
	}
	var st loopmgr.Status
	if err := json.Unmarshal(stdout.Bytes(), &st); err != nil {
		t.Fatalf("unmarshal status: %v\n%s", err, stdout.String())
	}
	if st.Schema != loopmgr.SchemaStatus {
		t.Fatalf("schema = %q, want %q", st.Schema, loopmgr.SchemaStatus)
	}
	if len(st.Loops) != 1 || st.Loops[0].Witnessed != 1 {
		t.Fatalf("loops = %+v", st.Loops)
	}
}

func TestLoopAppendRecordsHashChainedEvent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"append",
		"--ledger", path,
		"--loop", "issue-dispatch/default",
		"--kind", "witness",
		"--run", "run-1",
		"--source", "issue_resolve_dispatch",
		"--principal", "scheduler",
		"--status", "witnessed_done",
		"--reason", "DONE_WITNESSED",
		"--summary", "issue #717 witnessed",
		"--evidence", "issue=717",
		"--metric", "target_issue=717",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runLoop append code=%d stderr=%s", code, stderr.String())
	}
	var ev loopmgr.Event
	if err := json.Unmarshal(stdout.Bytes(), &ev); err != nil {
		t.Fatalf("unmarshal appended event: %v\n%s", err, stdout.String())
	}
	if ev.Schema != loopmgr.SchemaEvent || ev.Seq != 1 || ev.Hash == "" || ev.Status != loopmgr.StatusWitnessedDone {
		t.Fatalf("event = %+v", ev)
	}
	loaded, err := loopmgr.Load(path)
	if err != nil {
		t.Fatalf("Load appended ledger: %v", err)
	}
	if len(loaded) != 1 || loaded[0].EvidenceRefs[0].Kind != "issue" || loaded[0].Metrics["target_issue"] != 717 {
		t.Fatalf("loaded = %+v", loaded)
	}
}

func TestLoopAppendThenStatus(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	steps := [][]string{
		{"append", "--ledger", path, "--loop", "dispatch/issues", "--kind", "fire", "--source", "task-scheduler"},
		{"append", "--ledger", path, "--loop", "dispatch/issues", "--kind", "admit", "--run", "tick-1", "--status", "admitted"},
		{"append", "--ledger", path, "--loop", "dispatch/issues", "--kind", "start", "--run", "tick-1"},
	}
	for _, argv := range steps {
		var stdout, stderr bytes.Buffer
		if code := runLoop(&stdout, &stderr, argv); code != 0 {
			t.Fatalf("runLoop %v code=%d stderr=%s", argv, code, stderr.String())
		}
	}
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"status", "--ledger", path, "--json"})
	if code != 0 {
		t.Fatalf("runLoop status code=%d stderr=%s", code, stderr.String())
	}
	var st loopmgr.Status
	if err := json.Unmarshal(stdout.Bytes(), &st); err != nil {
		t.Fatalf("unmarshal status: %v\n%s", err, stdout.String())
	}
	if len(st.Loops) != 1 || st.Loops[0].Fires != 1 || st.Loops[0].Admitted != 1 || st.Loops[0].Started != 1 {
		t.Fatalf("status = %+v", st.Loops)
	}
}

func TestLoopRunRecordsSuccess(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	t.Setenv("FAK_LOOP_RUN_HELPER", "success")
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"run",
		"--ledger", path,
		"--loop", "scheduler/test",
		"--source", "cron",
		"--run", "run-success",
		"--",
		os.Args[0], "-test.run=TestLoopRunHelper",
	})
	if code != 0 {
		t.Fatalf("runLoop code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	events, err := loopmgr.Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if gotKinds(events) != "fire,admit,start,end" {
		t.Fatalf("kinds = %s events=%+v", gotKinds(events), events)
	}
	end := events[len(events)-1]
	if end.Status != loopmgr.StatusClaimedDone || end.Metrics["exit_code"] != 0 || end.Metrics["duration_ms"] < 0 {
		t.Fatalf("end = %+v", end)
	}
	if !strings.Contains(stdout.String(), "loop helper success") {
		t.Fatalf("stdout missing child output: %q", stdout.String())
	}
}

func TestLoopRunPropagatesFailureAndRecordsEnd(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	t.Setenv("FAK_LOOP_RUN_HELPER", "fail")
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"run",
		"--ledger", path,
		"--loop", "scheduler/test",
		"--source", "task-scheduler",
		"--run", "run-fail",
		"--",
		os.Args[0], "-test.run=TestLoopRunHelper",
	})
	if code != 7 {
		t.Fatalf("runLoop code=%d, want 7 stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	events, err := loopmgr.Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if gotKinds(events) != "fire,admit,start,end" {
		t.Fatalf("kinds = %s events=%+v", gotKinds(events), events)
	}
	end := events[len(events)-1]
	if end.Status != loopmgr.StatusFailed || end.Reason != "EXIT_NONZERO" || end.Metrics["exit_code"] != 7 {
		t.Fatalf("end = %+v", end)
	}
}

func TestLoopRunRejectsMissingCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"run", "--ledger", filepath.Join(t.TempDir(), "loops.jsonl"), "--loop", "scheduler/test",
	})
	if code != 2 {
		t.Fatalf("runLoop code=%d, want 2 stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "command is required") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestLoopStatusHumanOutput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendLoopTestEvent(t, path, loopmgr.Event{LoopID: "dogfood/mac", Kind: loopmgr.EventFire})
	appendLoopTestEvent(t, path, loopmgr.Event{LoopID: "dogfood/mac", Kind: loopmgr.EventWitness, RunID: "run-2", Status: loopmgr.StatusWitnessedDone})

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"status", "--ledger", path})
	if code != 0 {
		t.Fatalf("runLoop code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"loop ledger=", "dogfood/mac", "witnessed=1", "run-2:witnessed_done"} {
		if !strings.Contains(out, want) {
			t.Fatalf("human output missing %q:\n%s", want, out)
		}
	}
}

func TestLoopRunHelper(t *testing.T) {
	switch os.Getenv("FAK_LOOP_RUN_HELPER") {
	case "":
		return
	case "success":
		fmt.Fprintln(os.Stdout, "loop helper success")
		return
	case "fail":
		fmt.Fprintln(os.Stderr, "loop helper fail")
		os.Exit(7)
	default:
		os.Exit(9)
	}
}

func TestLoopStatusMissingLedger(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"status", "--ledger", filepath.Join(t.TempDir(), "missing.jsonl")})
	if code != 0 {
		t.Fatalf("runLoop code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "no loops found") {
		t.Fatalf("stdout = %q, want empty status", stdout.String())
	}
}

func TestLoopRejectsUnknownSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"fire"})
	if code != 2 {
		t.Fatalf("runLoop code=%d, want 2 stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestLoopAppendRejectsBadMetric(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"append", "--ledger", filepath.Join(t.TempDir(), "loops.jsonl"),
		"--loop", "loop-a", "--kind", "fire", "--metric", "target=not-int",
	})
	if code != 2 {
		t.Fatalf("runLoop code=%d, want 2 stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "invalid value") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func appendLoopTestEvent(t *testing.T, path string, ev loopmgr.Event) {
	t.Helper()
	if _, err := loopmgr.Append(path, ev, loopmgr.WithClock(func() time.Time {
		return time.Unix(0, 1000).UTC()
	})); err != nil {
		t.Fatalf("Append(%s): %v", ev.Kind, err)
	}
}

func gotKinds(events []loopmgr.Event) string {
	parts := make([]string, 0, len(events))
	for _, ev := range events {
		parts = append(parts, string(ev.Kind))
	}
	return strings.Join(parts, ",")
}

func TestLoopAdmitNoPolicyAdmits(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "loops.jsonl")
	appendLoopTestEvent(t, ledger, loopmgr.Event{LoopID: "issue-dispatch/default", Kind: loopmgr.EventFire})

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"admit", "--ledger", ledger,
		"--policy", filepath.Join(dir, "absent-policy.json"), "--json"})
	if code != 0 {
		t.Fatalf("admit code=%d stderr=%s", code, stderr.String())
	}
	var out struct {
		Schema    string             `json:"schema"`
		Decisions []loopmgr.Decision `json:"decisions"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if out.Schema != "fak.loop-admit.v1" {
		t.Fatalf("schema = %q", out.Schema)
	}
	if len(out.Decisions) != 1 || !out.Decisions[0].Admit {
		t.Fatalf("decisions = %+v", out.Decisions)
	}
}

func TestLoopAdmitPausedRefusesWithExit3(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "loops.jsonl")
	appendLoopTestEvent(t, ledger, loopmgr.Event{LoopID: "issue-dispatch/default", Kind: loopmgr.EventFire})

	policy := filepath.Join(dir, "loop-policy.json")
	doc := `{"schema":"fak.loop-policy.v1","loops":{"issue-dispatch/default":{"paused":true}}}`
	if err := os.WriteFile(policy, []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"admit", "--ledger", ledger,
		"--policy", policy, "--loop", "issue-dispatch/default"})
	if code != 3 {
		t.Fatalf("paused loop must exit 3, got %d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "REFUSE") || !strings.Contains(stdout.String(), loopmgr.ReasonLoopPaused) {
		t.Fatalf("expected REFUSE/%s in output: %s", loopmgr.ReasonLoopPaused, stdout.String())
	}
}

func TestLoopAdmitUnknownLoopGetsVerdict(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "loops.jsonl")
	policy := filepath.Join(dir, "loop-policy.json")
	doc := `{"schema":"fak.loop-policy.v1","loops":{"future/loop":{"disabled":true}}}`
	if err := os.WriteFile(policy, []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"admit", "--ledger", ledger,
		"--policy", policy, "--loop", "future/loop"})
	if code != 3 {
		t.Fatalf("disabled unseen loop must exit 3, got %d", code)
	}
	if !strings.Contains(stdout.String(), loopmgr.ReasonLoopDisabled) {
		t.Fatalf("expected %s: %s", loopmgr.ReasonLoopDisabled, stdout.String())
	}
}

func TestLoopAdmitBadPolicyExits2(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "loops.jsonl")
	policy := filepath.Join(dir, "loop-policy.json")
	if err := os.WriteFile(policy, []byte(`{"schema":"wrong"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"admit", "--ledger", ledger, "--policy", policy})
	if code != 2 {
		t.Fatalf("bad policy must exit 2, got %d", code)
	}
}
