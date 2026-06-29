package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/repoguard"
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
		"--no-guard",
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
	if !strings.Contains(stderr.String(), "--no-guard disables fak guard containment") {
		t.Fatalf("stderr missing no-guard warning: %q", stderr.String())
	}
	if events[1].Reason != "GUARD_DISABLED" || events[1].Metrics["guard_enabled"] != 0 {
		t.Fatalf("admit event did not record no-guard opt-out: %+v", events[1])
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
		"--no-guard",
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

func TestLoopRunUsesFakGuardByDefault(t *testing.T) {
	oldExecutable := loopExecutable
	oldNewCommand := loopNewCommand
	defer func() {
		loopExecutable = oldExecutable
		loopNewCommand = oldNewCommand
	}()

	loopExecutable = func() (string, error) { return "C:/tools/fak.exe", nil }
	var captured []string
	loopNewCommand = func(argv []string, stdout, stderr io.Writer) loopCommand {
		captured = append([]string(nil), argv...)
		return &fakeLoopCommand{pid: 4242}
	}

	path := filepath.Join(t.TempDir(), "loops.jsonl")
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"run",
		"--ledger", path,
		"--loop", "scheduler/test",
		"--source", "cron",
		"--run", "run-guarded",
		"--",
		"echo", "ok",
	})
	if code != 0 {
		t.Fatalf("runLoop code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	want := []string{"C:/tools/fak.exe", "guard", "--", "echo", "ok"}
	if !reflect.DeepEqual(captured, want) {
		t.Fatalf("child argv = %v, want %v", captured, want)
	}
	events, err := loopmgr.Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if gotKinds(events) != "fire,admit,start,end" {
		t.Fatalf("kinds = %s events=%+v", gotKinds(events), events)
	}
	if events[1].Reason != "GUARD_ADMITTED" || events[1].Metrics["guard_enabled"] != 1 {
		t.Fatalf("admit event did not record default guard: %+v", events[1])
	}
	if events[2].Metrics["pid"] != 4242 {
		t.Fatalf("start pid = %d, want fake pid", events[2].Metrics["pid"])
	}
}

func TestLoopRunGuardRefusesOutOfTreeEffect(t *testing.T) {
	oldNewCommand := loopNewCommand
	defer func() { loopNewCommand = oldNewCommand }()
	spawned := false
	loopNewCommand = func(argv []string, stdout, stderr io.Writer) loopCommand {
		spawned = true
		return &fakeLoopCommand{pid: 1}
	}

	parent, err := os.MkdirTemp(".", ".loop-containment-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(parent) })
	repo := filepath.Join(parent, "repo")
	if err := os.MkdirAll(filepath.Join(repo, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(repo)
	target := filepath.Join(parent, "victim.txt")

	path := filepath.Join(t.TempDir(), "loops.jsonl")
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{
		"run",
		"--ledger", path,
		"--loop", "scheduler/test",
		"--source", "cron",
		"--run", "run-refused",
		"--",
		"bash", "--norc", "-c", "echo bad > ../victim.txt",
	})
	if code != 3 {
		t.Fatalf("runLoop code=%d, want 3 stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if spawned {
		t.Fatal("refused containment command must not spawn the child")
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Fatalf("out-of-tree target was touched: stat err=%v", err)
	}
	events, err := loopmgr.Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if gotKinds(events) != "fire,admit" {
		t.Fatalf("kinds = %s events=%+v", gotKinds(events), events)
	}
	refusal := events[1]
	if refusal.Status != loopmgr.StatusRefused || refusal.Reason != repoguard.Reason {
		t.Fatalf("refusal event = %+v", refusal)
	}
	if refusal.Metrics["violations"] != 1 || refusal.Metrics["guard_enabled"] != 1 {
		t.Fatalf("refusal metrics = %+v", refusal.Metrics)
	}
	if !strings.Contains(stderr.String(), repoguard.Reason) {
		t.Fatalf("stderr missing structured reason %s: %q", repoguard.Reason, stderr.String())
	}
}

type fakeLoopCommand struct {
	pid      int
	startErr error
	waitErr  error
	killed   bool
}

func (c *fakeLoopCommand) Start() error { return c.startErr }
func (c *fakeLoopCommand) Wait() error  { return c.waitErr }
func (c *fakeLoopCommand) PID() int     { return c.pid }
func (c *fakeLoopCommand) Kill() error {
	c.killed = true
	return nil
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

func TestLoopStatusForkedLedgerSurvives(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendLoopTestEvent(t, path, loopmgr.Event{LoopID: "issue-dispatch/default", Kind: loopmgr.EventFire})
	appendLoopTestEvent(t, path, loopmgr.Event{LoopID: "issue-dispatch/default", Kind: loopmgr.EventWitness, RunID: "run-1", Status: loopmgr.StatusWitnessedDone})

	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	lines := strings.Split(strings.TrimRight(string(body), "\n"), "\n")
	if err := os.WriteFile(path, []byte(string(body)+lines[len(lines)-1]+"\n"), 0o644); err != nil {
		t.Fatalf("write forked ledger: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"status", "--ledger", path})
	if code != 0 {
		t.Fatalf("forked status code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	out := stdout.String()
	for _, want := range []string{"loop ledger=", "issue-dispatch/default", "witnessed=1", "run-1:witnessed_done"} {
		if !strings.Contains(out, want) {
			t.Fatalf("forked status output missing %q:\n%s", want, out)
		}
	}
	for _, want := range []string{"ledger integrity break", "recovered 2 event(s)"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("forked status stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestLoopHealthListsLearningDocsDebtAndDarkLoop(t *testing.T) {
	oldLearningDebt := loopLearningDebt
	loopLearningDebt = func(root string) (int64, bool) { return 34, true }
	t.Cleanup(func() { loopLearningDebt = oldLearningDebt })

	dir := t.TempDir()
	ledger := filepath.Join(dir, "loops.jsonl")
	registry := filepath.Join(dir, "loop-registry.json")

	appendLoopTestEventAt(t, ledger, loopmgr.Event{
		LoopID: "learning-docs-freshness",
		Kind:   loopmgr.EventFire,
	}, int64(time.Second))

	reg := loopmgr.Registry{Jobs: map[string]loopmgr.Job{}}
	if err := reg.Put(loopmgr.Job{
		Schedule: loopmgr.Schedule{
			JobID:           "learning-docs-freshness",
			IntervalSeconds: 24 * 60 * 60,
			MissedRun:       loopmgr.MissedCatchUp,
			JitterSeconds:   900,
		},
		State: loopmgr.JobArmed,
	}, time.Unix(0, 0).UTC()); err != nil {
		t.Fatalf("Put registry job: %v", err)
	}
	if err := loopmgr.SaveRegistry(registry, reg); err != nil {
		t.Fatalf("SaveRegistry: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"health", "--ledger", ledger, "--registry", registry})
	if code != 0 {
		t.Fatalf("health code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"learning-docs-freshness",
		"dark-loop",
		"1970-01-01T00:00:01Z",
		"34",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("health output missing %q:\n%s", want, out)
		}
	}

	stdout.Reset()
	stderr.Reset()
	code = runLoop(&stdout, &stderr, []string{"health", "--ledger", ledger, "--registry", registry, "--json"})
	if code != 0 {
		t.Fatalf("health --json code=%d stderr=%s", code, stderr.String())
	}
	var rep loopmgr.HealthReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal health: %v\n%s", err, stdout.String())
	}
	if len(rep.Rows) != 1 {
		t.Fatalf("health rows = %+v, want one docs-freshness row", rep.Rows)
	}
	row := rep.Rows[0]
	if row.State != loopmgr.HealthDark || !row.Dark {
		t.Fatalf("row state = %q dark=%v, want dark/true", row.State, row.Dark)
	}
	if row.LearningDebt == nil || *row.LearningDebt != 34 {
		t.Fatalf("learning debt = %v, want 34", row.LearningDebt)
	}
}

func TestLoopRegistryRegistersLearningDocsFreshness(t *testing.T) {
	reg, err := loopmgr.LoadRegistry(filepath.Join("..", "..", "tools", "loop-registry.json"))
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	job, ok := reg.Get("learning-docs-freshness")
	if !ok {
		t.Fatalf("learning-docs-freshness missing from tools/loop-registry.json")
	}
	if job.State != loopmgr.JobArmed {
		t.Fatalf("learning-docs-freshness state = %q, want armed", job.State)
	}
	if job.Schedule.IntervalSeconds != 24*60*60 {
		t.Fatalf("learning-docs-freshness interval = %d, want daily", job.Schedule.IntervalSeconds)
	}
	if job.Schedule.MissedRun != loopmgr.MissedCatchUp {
		t.Fatalf("learning-docs-freshness missed_run = %q, want catch-up", job.Schedule.MissedRun)
	}
	if job.Schedule.JitterSeconds <= 0 {
		t.Fatalf("learning-docs-freshness jitter = %d, want >0", job.Schedule.JitterSeconds)
	}
}

func TestLearningDocsFreshnessInstallerWiresDurableLoop(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "tools", "register_learning_docs_freshness.ps1"))
	if err != nil {
		t.Fatalf("read installer: %v", err)
	}
	s := string(b)
	for _, want := range []string{
		"learning-docs-freshness",
		"learning_scorecard.py",
		"New-FakLoopScheduledTaskAction",
		"-LogonType S4U",
		"fak loop health",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("installer missing %q:\n%s", want, s)
		}
	}
}

func TestLearningDocsDebtFromJSONParsesNonzeroDebt(t *testing.T) {
	debt, ok := learningDocsDebtFromJSON([]byte(`{"corpus":{"learning_debt":2}}`))
	if !ok || debt != 2 {
		t.Fatalf("learningDocsDebtFromJSON = %d,%v; want 2,true", debt, ok)
	}
}

// loopHelperPlan is the pure decision the re-exec child enacts: given a mode it
// returns what to print and the exit code. Extracting it lets the contract be
// asserted in-process (TestLoopHelperPlan) instead of only observed through a
// subprocess, so the helper test is no longer a bare "doesn't panic" early return.
func loopHelperPlan(mode string) (stdout, stderr string, exit int) {
	switch mode {
	case "success":
		return "loop helper success", "", 0
	case "fail":
		return "", "loop helper fail", 7
	default:
		return "", "", 9
	}
}

// TestLoopHelperPlan asserts the helper's mode->action contract that the three
// re-exec tests above depend on: success prints to stdout and exits 0, fail
// prints to stderr and exits 7, any other mode exits 9. This runs in the normal
// (no-env) test pass -- the case the scorecard flagged as assertionless.
func TestLoopHelperPlan(t *testing.T) {
	cases := []struct {
		mode       string
		wantStdout string
		wantStderr string
		wantExit   int
	}{
		{"success", "loop helper success", "", 0},
		{"fail", "", "loop helper fail", 7},
		{"unknown", "", "", 9},
		{"", "", "", 9},
	}
	for _, c := range cases {
		gotStdout, gotStderr, gotExit := loopHelperPlan(c.mode)
		if gotStdout != c.wantStdout || gotStderr != c.wantStderr || gotExit != c.wantExit {
			t.Errorf("loopHelperPlan(%q) = (%q,%q,%d), want (%q,%q,%d)",
				c.mode, gotStdout, gotStderr, gotExit, c.wantStdout, c.wantStderr, c.wantExit)
		}
	}
}

func TestLoopRunHelper(t *testing.T) {
	mode := os.Getenv("FAK_LOOP_RUN_HELPER")
	if mode == "" {
		// Not the re-exec child. Assert the success contract in-process so the
		// normal test pass exercises a real post-state (not a bare no-op return):
		// "success" must print to stdout and exit 0. The other modes are covered
		// by TestLoopHelperPlan and the re-exec parent tests.
		out, errOut, exit := loopHelperPlan("success")
		if out != "loop helper success" || errOut != "" || exit != 0 {
			t.Fatalf("success plan = (%q,%q,%d), want (%q,%q,0)",
				out, errOut, exit, "loop helper success", "")
		}
		return
	}
	stdout, stderr, exit := loopHelperPlan(mode)
	if stdout != "" {
		fmt.Fprintln(os.Stdout, stdout)
	}
	if stderr != "" {
		fmt.Fprintln(os.Stderr, stderr)
	}
	if exit != 0 {
		os.Exit(exit)
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

// appendLoopTestEventAt appends with a fixed timestamp so cadence/last-run can be
// asserted deterministically (the chain validates seq+prev_hash, not clock order).
func appendLoopTestEventAt(t *testing.T, path string, ev loopmgr.Event, tsNano int64) {
	t.Helper()
	if _, err := loopmgr.Append(path, ev, loopmgr.WithClock(func() time.Time {
		return time.Unix(0, tsNano).UTC()
	})); err != nil {
		t.Fatalf("Append(%s)@%d: %v", ev.Kind, tsNano, err)
	}
}

// loopRollupTestReport mirrors the unexported loopRollupReport for JSON decoding
// in tests (the rollup --json contract: schema, nodes, per-loop fold).
type loopRollupTestReport struct {
	Schema string   `json:"schema"`
	Nodes  []string `json:"nodes"`
	Loops  []struct {
		LoopID            string  `json:"loop_id"`
		Nodes             int     `json:"nodes"`
		Runs              uint64  `json:"runs"`
		Witnessed         uint64  `json:"witnessed"`
		Refused           uint64  `json:"refused"`
		CadenceSeconds    float64 `json:"cadence_seconds"`
		LastEventUnixNano int64   `json:"last_event_unix_nano"`
	} `json:"loops"`
	Skipped []struct {
		Node string `json:"node"`
	} `json:"skipped"`
}

const second = int64(time.Second)

func TestLoopRollupJSONFoldsAcrossNodes(t *testing.T) {
	dir := t.TempDir()
	nodeA := filepath.Join(dir, "node-a.jsonl")
	nodeB := filepath.Join(dir, "node-b.jsonl")
	// Loop "dispatch/issues": 2 fires on node-a (0s, 60s) + 1 fire on node-b (120s),
	// so fleet runs=3 over a merged 120s span (2 gaps) -> cadence 60s, last=120s.
	appendLoopTestEventAt(t, nodeA, loopmgr.Event{LoopID: "dispatch/issues", Kind: loopmgr.EventFire}, 0)
	appendLoopTestEventAt(t, nodeA, loopmgr.Event{LoopID: "dispatch/issues", Kind: loopmgr.EventWitness, RunID: "a1", Status: loopmgr.StatusWitnessedDone}, 30*second)
	appendLoopTestEventAt(t, nodeA, loopmgr.Event{LoopID: "dispatch/issues", Kind: loopmgr.EventFire}, 60*second)
	appendLoopTestEventAt(t, nodeB, loopmgr.Event{LoopID: "dispatch/issues", Kind: loopmgr.EventFire}, 120*second)
	// A second loop only node-b ran, to prove the fold reports every loop fleet-wide.
	appendLoopTestEventAt(t, nodeB, loopmgr.Event{LoopID: "docs/refresh", Kind: loopmgr.EventAdmit, Status: loopmgr.StatusRefused, Reason: "PAUSED"}, 90*second)

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"rollup", "--ledger", nodeA, "--ledger", "edge=" + nodeB, "--json"})
	if code != 0 {
		t.Fatalf("rollup code=%d stderr=%s", code, stderr.String())
	}
	var rep loopRollupTestReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if rep.Schema != "fak.loop-rollup.v1" {
		t.Fatalf("schema = %q", rep.Schema)
	}
	if len(rep.Nodes) != 2 || rep.Nodes[1] != "edge" {
		t.Fatalf("nodes = %v (want [node-a edge]; NODE=PATH must label)", rep.Nodes)
	}
	byLoop := map[string]int{}
	for i, l := range rep.Loops {
		byLoop[l.LoopID] = i
	}
	di, ok := byLoop["dispatch/issues"]
	if !ok {
		t.Fatalf("dispatch/issues missing from %+v", rep.Loops)
	}
	d := rep.Loops[di]
	if d.Runs != 3 {
		t.Fatalf("dispatch/issues runs = %d, want 3 (2 node-a + 1 edge)", d.Runs)
	}
	if d.Nodes != 2 {
		t.Fatalf("dispatch/issues nodes = %d, want 2", d.Nodes)
	}
	if d.Witnessed != 1 {
		t.Fatalf("dispatch/issues witnessed = %d, want 1", d.Witnessed)
	}
	if d.CadenceSeconds != 60 {
		t.Fatalf("dispatch/issues cadence = %v, want 60 (120s merged span / 2 gaps)", d.CadenceSeconds)
	}
	if d.LastEventUnixNano != 120*second {
		t.Fatalf("dispatch/issues last = %d, want %d", d.LastEventUnixNano, 120*second)
	}
	ri, ok := byLoop["docs/refresh"]
	if !ok {
		t.Fatalf("docs/refresh missing (fleet view must include refuse-only loops): %+v", rep.Loops)
	}
	if rep.Loops[ri].Refused != 1 || rep.Loops[ri].Runs != 0 {
		t.Fatalf("docs/refresh = %+v, want refused=1 runs=0", rep.Loops[ri])
	}
}

// TestLoopRollupAddNodeChangesOnlyRollup is the #769 acceptance: adding a node's
// journal changes only the rollup, never any node's behavior. We assert the
// aggregate count grows by the new node's runs AND the pre-existing ledgers are
// byte-for-byte unchanged (the fold is read-only — it ingests, it never writes).
func TestLoopRollupAddNodeChangesOnlyRollup(t *testing.T) {
	dir := t.TempDir()
	nodeA := filepath.Join(dir, "node-a.jsonl")
	nodeB := filepath.Join(dir, "node-b.jsonl")
	appendLoopTestEventAt(t, nodeA, loopmgr.Event{LoopID: "dispatch/issues", Kind: loopmgr.EventFire}, 0)
	appendLoopTestEventAt(t, nodeB, loopmgr.Event{LoopID: "dispatch/issues", Kind: loopmgr.EventFire}, 60*second)

	runsFor := func(args ...string) (uint64, []byte, []byte) {
		var stdout, stderr bytes.Buffer
		if code := runLoop(&stdout, &stderr, append([]string{"rollup", "--json"}, args...)); code != 0 {
			t.Fatalf("rollup code=%d stderr=%s", code, stderr.String())
		}
		var rep loopRollupTestReport
		if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
			t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
		}
		var runs uint64
		for _, l := range rep.Loops {
			if l.LoopID == "dispatch/issues" {
				runs = l.Runs
			}
		}
		a, err := os.ReadFile(nodeA)
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(nodeB)
		if err != nil {
			t.Fatal(err)
		}
		return runs, a, b
	}

	before, aBefore, bBefore := runsFor("--ledger", nodeA, "--ledger", nodeB)
	if before != 2 {
		t.Fatalf("two-node runs = %d, want 2", before)
	}

	// Add a third node's journal and re-fold over all three.
	nodeC := filepath.Join(dir, "node-c.jsonl")
	appendLoopTestEventAt(t, nodeC, loopmgr.Event{LoopID: "dispatch/issues", Kind: loopmgr.EventFire}, 120*second)
	appendLoopTestEventAt(t, nodeC, loopmgr.Event{LoopID: "dispatch/issues", Kind: loopmgr.EventFire}, 180*second)

	after, aAfter, bAfter := runsFor("--ledger", nodeA, "--ledger", nodeB, "--ledger", nodeC)
	if after != 4 {
		t.Fatalf("three-node runs = %d, want 4 (added node-c's 2 fires)", after)
	}
	if !bytes.Equal(aBefore, aAfter) || !bytes.Equal(bBefore, bAfter) {
		t.Fatalf("adding node-c mutated a pre-existing node ledger — rollup must be read-only")
	}
}

func TestLoopRollupDirAndSkipsCorrupt(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.jsonl")
	appendLoopTestEventAt(t, good, loopmgr.Event{LoopID: "dispatch/issues", Kind: loopmgr.EventFire}, 0)
	// A corrupt journal must be skipped (surfaced), not abort the whole fleet view.
	bad := filepath.Join(dir, "bad.jsonl")
	if err := os.WriteFile(bad, []byte("{not valid json\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"rollup", "--dir", dir, "--json"})
	if code != 0 {
		t.Fatalf("rollup --dir code=%d stderr=%s", code, stderr.String())
	}
	var rep loopRollupTestReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if len(rep.Loops) != 1 || rep.Loops[0].Runs != 1 {
		t.Fatalf("loops = %+v, want one loop with runs=1 from the good node", rep.Loops)
	}
	if len(rep.Skipped) != 1 || rep.Skipped[0].Node != "bad" {
		t.Fatalf("skipped = %+v, want the corrupt 'bad' node surfaced", rep.Skipped)
	}
}

func TestLoopRollupHumanOutput(t *testing.T) {
	dir := t.TempDir()
	nodeA := filepath.Join(dir, "mac.jsonl")
	appendLoopTestEventAt(t, nodeA, loopmgr.Event{LoopID: "dogfood/mac", Kind: loopmgr.EventFire}, 0)
	appendLoopTestEventAt(t, nodeA, loopmgr.Event{LoopID: "dogfood/mac", Kind: loopmgr.EventFire}, 120*second)

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"rollup", "--ledger", nodeA})
	if code != 0 {
		t.Fatalf("rollup code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak loop rollup", "LOOP", "RUNS", "CADENCE", "dogfood/mac", "2.0m"} {
		if !strings.Contains(out, want) {
			t.Fatalf("human output missing %q:\n%s", want, out)
		}
	}
}

func TestLoopRollupRequiresNodes(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"rollup", "--json"})
	if code != 2 {
		t.Fatalf("rollup with no nodes code=%d, want 2 stdout=%s", code, stdout.String())
	}
	if !strings.Contains(stderr.String(), "no node ledgers") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestCadenceAndHumanCadence(t *testing.T) {
	if got := cadenceSeconds([]int64{0, 60 * second, 120 * second}); got != 60 {
		t.Fatalf("cadenceSeconds = %v, want 60", got)
	}
	if got := cadenceSeconds([]int64{42 * second}); got != 0 {
		t.Fatalf("single-fire cadence = %v, want 0 (no measurable interval)", got)
	}
	if got := cadenceSeconds([]int64{5, 5, 5}); got != 0 {
		t.Fatalf("zero-span cadence = %v, want 0", got)
	}
	cases := []struct {
		sec  float64
		want string
	}{{0, "-"}, {0.3, "0.3s"}, {45, "45s"}, {120, "2.0m"}, {3960, "1.1h"}, {172800, "2.0d"}}
	for _, c := range cases {
		if got := humanCadence(c.sec); got != c.want {
			t.Fatalf("humanCadence(%v) = %q, want %q", c.sec, got, c.want)
		}
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
