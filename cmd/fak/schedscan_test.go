package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestNormalizeSchedResult proves the signed/unsigned fold: Get-ScheduledTaskInfo
// hands 0x800710E0 back as either the unsigned 2147946720 or the signed Int32
// -2147020576, and both must key the same decode entry.
func TestNormalizeSchedResult(t *testing.T) {
	cases := []struct {
		raw  int64
		want uint32
	}{
		{0, 0},
		{0x800710E0, 0x800710E0},  // unsigned spelling (2147946720)
		{-2147020576, 0x800710E0}, // signed Int32 spelling of the same code
		{-1, 0xFFFFFFFF},          // signed -1 == 0xFFFFFFFF
		{4294967295, 0xFFFFFFFF},  // unsigned spelling of the same code
		{267009, 0x41301},         // SCHED_S_TASK_RUNNING
	}
	for _, c := range cases {
		if got := normalizeSchedResult(c.raw); got != c.want {
			t.Errorf("normalizeSchedResult(%d) = 0x%X, want 0x%X", c.raw, got, c.want)
		}
	}
}

// TestDecodeSchedTaskResult pins the decode of the codes an operator actually sees,
// most importantly the Interactive-logon refusal that the S4U migration targets.
func TestDecodeSchedTaskResult(t *testing.T) {
	// The headline case: 0x800710E0 is a failure and carries the S4U remediation.
	r := decodeSchedTaskResult(0x800710E0)
	if r.Hex != "0x800710E0" {
		t.Errorf("hex = %q, want 0x800710E0", r.Hex)
	}
	if r.Severity != "fail" {
		t.Errorf("severity = %q, want fail", r.Severity)
	}
	if !strings.Contains(r.Message, "refused the request") {
		t.Errorf("message = %q, want the operator-refused text", r.Message)
	}
	if !strings.Contains(strings.ToUpper(r.Hint), "S4U") {
		t.Errorf("hint = %q, want an S4U remediation", r.Hint)
	}
	// The signed spelling decodes identically.
	if got := decodeSchedTaskResult(-2147020576); got != r {
		t.Errorf("signed spelling decoded differently:\n got %+v\nwant %+v", got, r)
	}

	if got := decodeSchedTaskResult(0); got.Severity != "ok" {
		t.Errorf("0x0 severity = %q, want ok", got.Severity)
	}
	if got := decodeSchedTaskResult(0x41301); got.Severity != "running" {
		t.Errorf("0x41301 severity = %q, want running", got.Severity)
	}
	// An unmapped HRESULT failure still classifies as fail via the structural fallback.
	if got := decodeSchedTaskResult(0x80070057); got.Severity != "fail" {
		t.Errorf("unmapped 0x80070057 severity = %q, want fail", got.Severity)
	}
	// An unmapped SCHED_S_* status is informational, not a failure.
	if got := decodeSchedTaskResult(0x4130A); got.Severity != "info" {
		t.Errorf("unmapped 0x4130A severity = %q, want info", got.Severity)
	}
}

func TestClassifySchedTask(t *testing.T) {
	// A refused task reports State=Ready but must classify as failing anyway — the
	// whole reason schedscan exists.
	if status, failing := classifySchedTask("Ready", decodeSchedTaskResult(0x800710E0)); status != "failing" || !failing {
		t.Errorf("refused-but-Ready => (%q,%v), want (failing,true)", status, failing)
	}
	if status, failing := classifySchedTask("Ready", decodeSchedTaskResult(0)); status != "idle" || failing {
		t.Errorf("clean Ready => (%q,%v), want (idle,false)", status, failing)
	}
	if status, _ := classifySchedTask("Running", decodeSchedTaskResult(0x41301)); status != "running" {
		t.Errorf("running => %q, want running", status)
	}
	if status, _ := classifySchedTask("Disabled", decodeSchedTaskResult(0x41302)); status != "disabled" {
		t.Errorf("disabled => %q, want disabled", status)
	}
}

func TestParseSchedTaskJSON(t *testing.T) {
	// Empty pipeline.
	if rows, err := parseSchedTaskJSON("  \n"); err != nil || rows != nil {
		t.Errorf("empty => (%v,%v), want (nil,nil)", rows, err)
	}
	// PowerShell unwraps a single row to a bare object.
	one := `{"TaskName":"FleetResumeWatchdog","State":"Ready","LastTaskResult":-2147020576}`
	rows, err := parseSchedTaskJSON(one)
	if err != nil || len(rows) != 1 || rows[0].TaskName != "FleetResumeWatchdog" {
		t.Fatalf("single-object => (%+v,%v)", rows, err)
	}
	if rows[0].LastTaskResult != -2147020576 {
		t.Errorf("LastTaskResult = %d, want -2147020576", rows[0].LastTaskResult)
	}
	// An array of rows.
	arr := `[{"TaskName":"A","LastTaskResult":0},{"TaskName":"B","LastTaskResult":1}]`
	rows, err = parseSchedTaskJSON(arr)
	if err != nil || len(rows) != 2 {
		t.Fatalf("array => (%+v,%v)", rows, err)
	}
}

// schedScanFixture is a small snapshot mimicking the fleet: a refused watchdog, a
// healthy janitor, a running task, and a non-fleet task that the default filter
// must drop.
const schedScanFixture = `[
  {"TaskName":"FleetResumeWatchdog","TaskPath":"\\","State":"Ready","LogonType":"Interactive","LastRunTime":"2026-07-09T04:00:00.0000000-07:00","LastTaskResult":-2147020576,"NextRunTime":"2026-07-09T05:00:00.0000000-07:00","NumberOfMissedRuns":3},
  {"TaskName":"FakFleetJanitor","TaskPath":"\\","State":"Ready","LogonType":"S4U","LastRunTime":"2026-07-09T04:30:00.0000000-07:00","LastTaskResult":0,"NextRunTime":"2026-07-09T05:30:00.0000000-07:00","NumberOfMissedRuns":0},
  {"TaskName":"FleetSupervisorWatchdog","TaskPath":"\\","State":"Running","LogonType":"S4U","LastTaskResult":267009,"NumberOfMissedRuns":0},
  {"TaskName":"GoogleUpdaterTaskSystem","TaskPath":"\\","State":"Ready","LogonType":"S4U","LastTaskResult":0,"NumberOfMissedRuns":0}
]`

func TestBuildSchedScanDoc_FilterAndOrder(t *testing.T) {
	rows, err := parseSchedTaskJSON(schedScanFixture)
	if err != nil {
		t.Fatal(err)
	}
	doc := buildSchedScanDoc(rows, regexp.MustCompile(schedScanDefaultFilter), "test", "2026-07-09T12:00:00Z")
	// GoogleUpdaterTaskSystem does not match ^(fak|fleet|user) and is dropped.
	if doc.Count != 3 {
		t.Fatalf("count = %d, want 3 (non-fleet task filtered)", doc.Count)
	}
	if doc.FailingCount != 1 {
		t.Errorf("failing_count = %d, want 1", doc.FailingCount)
	}
	// Failing task sorts first.
	if doc.Tasks[0].Name != "FleetResumeWatchdog" || !doc.Tasks[0].Failing {
		t.Errorf("first task = %q failing=%v, want FleetResumeWatchdog failing", doc.Tasks[0].Name, doc.Tasks[0].Failing)
	}
	if doc.Counts["failing"] != 1 || doc.Counts["idle"] != 1 || doc.Counts["running"] != 1 {
		t.Errorf("counts = %v, want failing/idle/running each 1", doc.Counts)
	}
}

func TestRunSchedScan_FromFileJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")
	if err := os.WriteFile(path, []byte(schedScanFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runSchedScan(&out, &errb, []string{"--from", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	var doc schedScanDoc
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if doc.Schema != schedScanSchema {
		t.Errorf("schema = %q", doc.Schema)
	}
	if doc.FailingCount != 1 || doc.Count != 3 {
		t.Errorf("count=%d failing=%d, want 3/1", doc.Count, doc.FailingCount)
	}
}

func TestRunSchedScan_StrictExit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")
	if err := os.WriteFile(path, []byte(schedScanFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	// --strict exits 1 when any task is failing.
	var out, errb bytes.Buffer
	if code := runSchedScan(&out, &errb, []string{"--from", path, "--json", "--strict"}); code != 1 {
		t.Errorf("strict exit = %d, want 1", code)
	}
	// Filtered to only the healthy janitor, --strict is clean.
	out.Reset()
	errb.Reset()
	if code := runSchedScan(&out, &errb, []string{"--from", path, "--json", "--strict", "--filter", "^FakFleetJanitor$"}); code != 0 {
		t.Errorf("strict exit (healthy only) = %d, want 0; stderr=%s", code, errb.String())
	}
}

func TestRunSchedScan_TableFailingOnly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")
	if err := os.WriteFile(path, []byte(schedScanFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runSchedScan(&out, &errb, []string{"--from", path, "--failing-only"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "FleetResumeWatchdog") {
		t.Errorf("table missing the failing task:\n%s", s)
	}
	if strings.Contains(s, "FakFleetJanitor") {
		t.Errorf("--failing-only leaked a healthy task:\n%s", s)
	}
	// The S4U remediation hint travels with the finding.
	if !strings.Contains(strings.ToUpper(s), "S4U") {
		t.Errorf("table missing the S4U remediation hint:\n%s", s)
	}
}
