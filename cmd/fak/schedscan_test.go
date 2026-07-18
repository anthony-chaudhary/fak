package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"
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
	// An unmapped HRESULT failure classifies as fail via the structural fallback,
	// and carries a hex-bearing message + a lookup hint so the table's remediation
	// line is never blank for a failing task.
	if got := decodeSchedTaskResult(0x80070057); got.Severity != "fail" {
		t.Errorf("unmapped 0x80070057 severity = %q, want fail", got.Severity)
	} else {
		if !strings.Contains(got.Message, "0x80070057") {
			t.Errorf("unmapped-HRESULT message = %q, want the hex in it", got.Message)
		}
		if strings.TrimSpace(got.Hint) == "" {
			t.Errorf("unmapped-HRESULT hint is empty; the table would print no remediation")
		}
	}
	// An unmapped SCHED_S_* status is informational, not a failure.
	if got := decodeSchedTaskResult(0x4130A); got.Severity != "info" {
		t.Errorf("unmapped 0x4130A severity = %q, want info", got.Severity)
	}
	// The batch-logon status (SCHED_S_BATCH_LOGON_PROBLEM) is a warn on the leaf's
	// own S4U theme, not a silent info — it must surface as "degraded", not "idle".
	if got := decodeSchedTaskResult(0x4131C); got.Severity != "warn" || !strings.Contains(strings.ToUpper(got.Hint), "S4U") {
		t.Errorf("0x4131C => severity %q hint %q, want warn + S4U remediation", got.Severity, got.Hint)
	}
	// The core correctness guard: a task action that exited with a small non-zero
	// code (e.g. 3, 255) is a FAILED run, never an "unknown"/healthy status — this
	// is the whole reason schedscan does not trust State=Ready.
	for _, code := range []int64{3, 255, 0x420} {
		if got := decodeSchedTaskResult(code); got.Severity != "fail" {
			t.Errorf("non-zero exit 0x%X severity = %q, want fail", code, got.Severity)
		}
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
	// A task action that exited non-zero (here 3) while State=Ready must roll up as
	// failing so --strict catches it — not "idle".
	if status, failing := classifySchedTask("Ready", decodeSchedTaskResult(3)); status != "failing" || !failing {
		t.Errorf("Ready+exit-3 => (%q,%v), want (failing,true)", status, failing)
	}
	// A warn-severity status (batch-logon) surfaces as degraded, not idle.
	if status, failing := classifySchedTask("Ready", decodeSchedTaskResult(0x4131C)); status != "degraded" || failing {
		t.Errorf("Ready+0x4131C => (%q,%v), want (degraded,false)", status, failing)
	}
	// An operator-disabled task with a STALE hard-failure last result must not latch
	// --strict: intentional-off dominates stale history.
	if status, failing := classifySchedTask("Disabled", decodeSchedTaskResult(0x800710E0)); status != "disabled" || failing {
		t.Errorf("Disabled+stale-fail => (%q,%v), want (disabled,false)", status, failing)
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
	// --strict exits 3 (the stallscan-aligned "detection" code) when any task is
	// failing — distinct from 1 (runtime error) / 2 (usage) so a cron gate can tell
	// "a task failed" from "the scan broke".
	var out, errb bytes.Buffer
	if code := runSchedScan(&out, &errb, []string{"--from", path, "--json", "--strict"}); code != 3 {
		t.Errorf("strict exit = %d, want 3", code)
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

// TestRunSchedScan_AllDropsFilter pins the --all escape hatch: it must include the
// non-fleet task and clear the filter. Without this, a regressed no-op --all (filter
// still applied) would be invisible — every other test runs with a filter.
func TestRunSchedScan_AllDropsFilter(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")
	if err := os.WriteFile(path, []byte(schedScanFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runSchedScan(&out, &errb, []string{"--from", path, "--json", "--all"}); code != 0 {
		t.Fatalf("exit = %d; stderr=%s", code, errb.String())
	}
	var doc schedScanDoc
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	if doc.Count != 4 {
		t.Errorf("--all count = %d, want 4 (non-fleet task INCLUDED)", doc.Count)
	}
	if doc.Filter != "" {
		t.Errorf("--all filter = %q, want empty", doc.Filter)
	}
	var names []string
	for _, tk := range doc.Tasks {
		names = append(names, tk.Name)
	}
	if !strings.Contains(strings.Join(names, ","), "GoogleUpdaterTaskSystem") {
		t.Errorf("--all dropped the non-fleet task: %v", names)
	}
}

// TestRunSchedScan_JSONFailingOnly pins that --failing-only prunes the JSON task
// list (previously a silent no-op in JSON mode) while the FailingCount fleet rollup
// is preserved.
func TestRunSchedScan_JSONFailingOnly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")
	if err := os.WriteFile(path, []byte(schedScanFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runSchedScan(&out, &errb, []string{"--from", path, "--json", "--failing-only"}); code != 0 {
		t.Fatalf("exit = %d; stderr=%s", code, errb.String())
	}
	var doc schedScanDoc
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	if doc.Count != 1 || len(doc.Tasks) != 1 {
		t.Fatalf("failing-only JSON count = %d, tasks = %d, want 1/1", doc.Count, len(doc.Tasks))
	}
	if doc.Tasks[0].Name != "FleetResumeWatchdog" || !doc.Tasks[0].Failing {
		t.Errorf("failing-only JSON task = %q failing=%v, want FleetResumeWatchdog failing", doc.Tasks[0].Name, doc.Tasks[0].Failing)
	}
	if doc.FailingCount != 1 {
		t.Errorf("failing_count = %d, want 1 (fleet rollup preserved)", doc.FailingCount)
	}
}

// schedScanMultiFailFixture has TWO tasks failing for the SAME reason (0x800710E0)
// plus one failing for a DIFFERENT reason (0x2), so the table's hint-dedup and
// multi-failing-row render — the fleet-wide-refusal scenario — get a witness.
const schedScanMultiFailFixture = `[
  {"TaskName":"FleetResumeWatchdog","State":"Ready","LogonType":"Interactive","LastTaskResult":-2147020576},
  {"TaskName":"FleetUserResumeWatchdog","State":"Ready","LogonType":"Interactive","LastTaskResult":-2147020576},
  {"TaskName":"FakBadPath","State":"Ready","LogonType":"S4U","LastTaskResult":2}
]`

func TestRunSchedScan_TableDedupsHints(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")
	if err := os.WriteFile(path, []byte(schedScanMultiFailFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runSchedScan(&out, &errb, []string{"--from", path}); code != 0 {
		t.Fatalf("exit = %d; stderr=%s", code, errb.String())
	}
	s := out.String()
	// The shared 0x800710E0 remediation prints exactly once despite two failing tasks.
	if n := strings.Count(s, "migrate it to S4U"); n != 1 {
		t.Errorf("shared S4U hint printed %d times, want exactly 1 (dedup):\n%s", n, s)
	}
	// The distinct 0x2 hint is not collapsed away.
	if !strings.Contains(s, "program path is wrong") {
		t.Errorf("distinct 0x2 hint missing (over-dedup?):\n%s", s)
	}
	// Both failing rows render.
	if !strings.Contains(s, "FleetResumeWatchdog") || !strings.Contains(s, "FleetUserResumeWatchdog") || !strings.Contains(s, "FakBadPath") {
		t.Errorf("a failing row is missing from the table:\n%s", s)
	}
}

// TestSchedActionMasksExit pins the shim detector (#5095): conhost.exe in any
// spelling (bare, full path, quoted) masks the wrapped program's exit code, and
// nothing else does — a direct powershell/python action's exit code is real.
func TestSchedActionMasksExit(t *testing.T) {
	masking := []string{
		"conhost.exe",
		"conhost",
		`C:\WINDOWS\System32\conhost.exe`,
		`"C:\Windows\System32\conhost.exe"`,
		"  CONHOST.EXE  ",
	}
	for _, e := range masking {
		if !schedActionMasksExit(e) {
			t.Errorf("schedActionMasksExit(%q) = false, want true", e)
		}
	}
	direct := []string{
		"",
		"powershell.exe",
		`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`,
		"pythonw.exe",
		`C:\work\fak\fak.exe`,
		"notconhost.exe",
	}
	for _, e := range direct {
		if schedActionMasksExit(e) {
			t.Errorf("schedActionMasksExit(%q) = true, want false", e)
		}
	}
}

// TestApplySchedExitMask pins the demotion rule: ONLY a trusted zero ("idle",
// code 0) is weakened by the conhost shim, to "unverified" with the
// heartbeat-freshness remediation — never to failing (the fleet launches most
// tasks through conhost; latching --strict on the shim would red the fleet
// forever). Every verdict that does not rest on a trusted zero passes through.
func TestApplySchedExitMask(t *testing.T) {
	// The headline case: masked exit 0 stops reading as healthy.
	status, r := applySchedExitMask("idle", decodeSchedTaskResult(0), true)
	if status != "unverified" {
		t.Errorf("masked idle+0 => %q, want unverified", status)
	}
	if r.Severity != "warn" || !strings.Contains(r.Message, "#5095") {
		t.Errorf("masked meaning = severity %q message %q, want warn + #5095", r.Severity, r.Message)
	}
	if !strings.Contains(r.Hint, "heartbeat") {
		t.Errorf("masked hint = %q, want the heartbeat-freshness remediation", r.Hint)
	}

	// Unmasked exit 0 stays idle/ok — a direct action's zero is real.
	if status, r := applySchedExitMask("idle", decodeSchedTaskResult(0), false); status != "idle" || r.Severity != "ok" {
		t.Errorf("unmasked idle+0 => (%q,%q), want (idle,ok)", status, r.Severity)
	}
	// A NON-zero result through the shim is Task Scheduler's own verdict (the
	// launch itself failed) and must stay failing, untouched.
	if status, r := applySchedExitMask("failing", decodeSchedTaskResult(0x800710E0), true); status != "failing" || r.Severity != "fail" {
		t.Errorf("masked refused => (%q,%q), want (failing,fail) untouched", status, r.Severity)
	}
	// Running / disabled don't rest on a trusted zero; no demotion.
	if status, _ := applySchedExitMask("running", decodeSchedTaskResult(0x41301), true); status != "running" {
		t.Errorf("masked running => %q, want running", status)
	}
	if status, _ := applySchedExitMask("disabled", decodeSchedTaskResult(0x41302), true); status != "disabled" {
		t.Errorf("masked disabled => %q, want disabled", status)
	}
}

// schedScanMaskFixture mimics the live fleet's conhost posture: a conhost-shimmed
// task reporting the untrustworthy exit 0, a conhost-shimmed task whose launch
// Task Scheduler itself refused (real failure, must survive), a direct-action
// task whose exit 0 is trustworthy, and a running conhost task.
const schedScanMaskFixture = `[
  {"TaskName":"FleetSupervisorWatchdog","State":"Ready","LogonType":"Interactive","LastRunTime":"2026-07-16T04:00:00.0000000-07:00","LastTaskResult":0,"ActionExecute":"conhost.exe","ActionArguments":"--headless powershell.exe -NoProfile -File watchdog.ps1"},
  {"TaskName":"FleetResumeWatchdog","State":"Ready","LogonType":"Interactive","LastTaskResult":-2147020576,"ActionExecute":"C:\\WINDOWS\\System32\\conhost.exe","ActionArguments":"--headless powershell.exe -File resume.ps1"},
  {"TaskName":"FakFleetJanitor","State":"Ready","LogonType":"S4U","LastTaskResult":0,"ActionExecute":"powershell.exe","ActionArguments":"-NoProfile -File janitor.ps1"},
  {"TaskName":"FleetControlPaneTick","State":"Running","LastTaskResult":267009,"ActionExecute":"conhost.exe","ActionArguments":"--headless cmd.exe /c tick.cmd"}
]`

// TestBuildSchedScanDoc_ConhostMaskedExitZero is the #5095 acceptance witness in
// snapshot form: the conhost-shimmed result=0 task (the FleetSupervisorWatchdog
// that reported result=0 for ~25 days while its target was dead) no longer rolls
// up as healthy-idle — it is "unverified" with result_masked=true — while a real
// failure and a trustworthy direct-action zero keep their verdicts.
func TestBuildSchedScanDoc_ConhostMaskedExitZero(t *testing.T) {
	rows, err := parseSchedTaskJSON(schedScanMaskFixture)
	if err != nil {
		t.Fatal(err)
	}
	doc := buildSchedScanDoc(rows, regexp.MustCompile(schedScanDefaultFilter), "test", "2026-07-16T12:00:00Z")
	if doc.Count != 4 {
		t.Fatalf("count = %d, want 4", doc.Count)
	}
	byName := map[string]schedScanTaskReport{}
	for _, tk := range doc.Tasks {
		byName[tk.Name] = tk
	}
	sup := byName["FleetSupervisorWatchdog"]
	if sup.Status != "unverified" || sup.Failing || !sup.ResultMasked {
		t.Errorf("conhost exit-0 => status=%q failing=%v masked=%v, want unverified/false/true", sup.Status, sup.Failing, sup.ResultMasked)
	}
	if !strings.Contains(sup.Result.Hint, "heartbeat") {
		t.Errorf("conhost exit-0 hint = %q, want heartbeat-freshness remediation", sup.Result.Hint)
	}
	if sup.Action == "" || !strings.Contains(sup.Action, "conhost.exe") {
		t.Errorf("action = %q, want the conhost command line surfaced", sup.Action)
	}
	// The refused conhost task is still a REAL failure — masking must never
	// weaken a non-zero result.
	if res := byName["FleetResumeWatchdog"]; res.Status != "failing" || !res.Failing {
		t.Errorf("masked+refused => (%q,%v), want (failing,true)", res.Status, res.Failing)
	}
	// The direct-action zero is trustworthy and stays idle.
	if jan := byName["FakFleetJanitor"]; jan.Status != "idle" || jan.ResultMasked {
		t.Errorf("direct exit-0 => status=%q masked=%v, want idle/false", jan.Status, jan.ResultMasked)
	}
	// A running conhost task is not demoted.
	if run := byName["FleetControlPaneTick"]; run.Status != "running" {
		t.Errorf("masked running => %q, want running", run.Status)
	}
	if doc.FailingCount != 1 {
		t.Errorf("failing_count = %d, want 1 (unverified is NOT failing)", doc.FailingCount)
	}
	if doc.Counts["unverified"] != 1 {
		t.Errorf("counts = %v, want unverified=1", doc.Counts)
	}
}

// TestRunSchedScan_TableUnverifiedHint pins that the heartbeat remediation
// travels with an unverified (mask-demoted) row in the human table, not only
// with failing rows — otherwise the operator sees UNVERIFIED with no cure.
func TestRunSchedScan_TableUnverifiedHint(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.json")
	if err := os.WriteFile(path, []byte(schedScanMaskFixture), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runSchedScan(&out, &errb, []string{"--from", path}); code != 0 {
		t.Fatalf("exit = %d; stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "UNVERIFIED") {
		t.Errorf("table missing the UNVERIFIED status row:\n%s", s)
	}
	if !strings.Contains(s, "heartbeat freshness") {
		t.Errorf("table missing the heartbeat-freshness remediation for the unverified row:\n%s", s)
	}
	// --strict still keys on real failures only: with the refused task filtered
	// out, an unverified row alone must NOT latch exit 3.
	out.Reset()
	errb.Reset()
	if code := runSchedScan(&out, &errb, []string{"--from", path, "--strict", "--filter", "^FleetSupervisorWatchdog$"}); code != 0 {
		t.Errorf("strict exit (unverified only) = %d, want 0; stderr=%s", code, errb.String())
	}
}

// TestSchedScanShortTime covers the three fallback branches plus rune-safety of the
// non-RFC3339 truncation.
func TestSchedScanShortTime(t *testing.T) {
	// Branch 1: a real RFC3339 stamp renders to minute precision.
	if got := schedScanShortTime("2026-07-09T04:00:00.0000000-07:00"); got != "2026-07-09 04:00" {
		t.Errorf("RFC3339 => %q, want 2026-07-09 04:00", got)
	}
	// Branch 3: empty renders as a dash.
	if got := schedScanShortTime(""); got != "-" {
		t.Errorf("empty => %q, want -", got)
	}
	// Branch 2: a non-RFC3339 string of length >= 16 keeps its leading date+time and
	// turns the ISO 'T' into a space.
	if got := schedScanShortTime("20260709T040000ZZ"); got != "20260709 040000Z" {
		t.Errorf("non-RFC3339 => %q, want 20260709 040000Z", got)
	}
	// Rune-safety: a multibyte rune straddling the 16-byte cut must not be split into
	// invalid UTF-8 (the old byte-slice s[:16] produced mojibake here).
	got := schedScanShortTime("aaaaaaaaaaaaaaaébbb")
	if !utf8.ValidString(got) {
		t.Errorf("multibyte fallback produced invalid UTF-8: %q", got)
	}
}
