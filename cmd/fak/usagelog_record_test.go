package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// usagelog_record_test.go exercises the CLI wiring added on top of the
// already-tested internal/usagelog package: the recorder main() calls
// (recordUsage), its path resolution (usageLogPath), the hook exclusion, and
// the `fak usage` verb's flag handling.

func withUsagePath(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	t.Setenv("FAK_USAGE_LOG_PATH", path)
	return path
}

func TestUsageLogPathOverride(t *testing.T) {
	t.Setenv("FAK_USAGE_LOG_PATH", "")
	if got := usageLogPath(); got != usagelog.DefaultPath() {
		t.Fatalf("usageLogPath() with no override = %q, want usagelog.DefaultPath() %q", got, usagelog.DefaultPath())
	}
	want := filepath.Join(t.TempDir(), "custom-usage.jsonl")
	t.Setenv("FAK_USAGE_LOG_PATH", want)
	if got := usageLogPath(); got != want {
		t.Fatalf("usageLogPath() with FAK_USAGE_LOG_PATH=%q = %q, want %q", want, got, want)
	}
}

func TestUsageGuardDisableSurfaceRendersOutcomeCounts(t *testing.T) {
	withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")
	path, err := guardDisableUsageDefaultPath()
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []guardDisableUsageRow{
		{At: "2026-08-17T10:00:00Z", Outcome: guardDisableUsageSuccess},
		{At: "2026-08-18T10:00:00Z", Outcome: guardDisableUsageChildNonzero},
	} {
		if err := appendGuardDisableUsage(path, row); err != nil {
			t.Fatal(err)
		}
	}

	out := captureStdout(t, func() { cmdUsage([]string{"--guard-disable", "--json"}) })
	var got guardDisableUsageSummary
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != guardDisableUsageSummarySchema || len(got.Weeks) != 1 || got.Weeks[0].Invocations != 2 || got.Weeks[0].Success != 1 || got.Weeks[0].ChildNonzero != 1 {
		t.Fatalf("summary = %+v", got)
	}
}

func TestRecordUsageWritesOneRow(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")

	start := time.Now().Add(-5 * time.Millisecond)
	recordUsage("frontierswe", []string{"--suite", "swebench"}, 0, start)

	rows, err := usagelog.ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1: %+v", len(rows), rows)
	}
	r := rows[0]
	if r.Verb != "frontierswe" {
		t.Errorf("Verb = %q, want frontierswe", r.Verb)
	}
	if r.Argc != 2 {
		t.Errorf("Argc = %d, want 2", r.Argc)
	}
	if r.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", r.ExitCode)
	}
	if r.ArgsDigest == "" {
		t.Error("ArgsDigest is empty, want a salted digest")
	}
	if r.PID != os.Getpid() {
		t.Errorf("PID = %d, want %d", r.PID, os.Getpid())
	}
	if n, err := usagelog.Verify(path); err != nil || n != 1 {
		t.Fatalf("Verify() = (%d, %v), want (1, nil)", n, err)
	}
}

func TestRunObservedGitOperationRecordsCompositeOutcomeDuration(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")
	start := time.Unix(1_700_000_000, 0)
	oldNow := recordUsageNow
	recordUsageNow = func() time.Time { return start.Add(37 * time.Millisecond) }
	t.Cleanup(func() { recordUsageNow = oldNow })

	code := runObservedGitOperation(start, gitOperationName("sync", []string{"push", "--json"}), []string{"push", "--json"}, func() int {
		return syncExitRefused
	})
	if code != syncExitRefused {
		t.Fatalf("code = %d, want %d", code, syncExitRefused)
	}
	rows, err := usagelog.ReadRows(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].Verb != "sync push" || rows[0].ExitCode != syncExitRefused || rows[0].DurationMS != 37 {
		t.Fatalf("rows = %+v, want one 37ms refused sync push", rows)
	}
	stats := usagelog.FoldRows(rows, usagelog.FoldOptions{}).ByOperationOutcome
	if len(stats) != 1 || stats[0].Operation != "sync push" || stats[0].Outcome != usagelog.OutcomeRefused || stats[0].P50MS != 37 {
		t.Fatalf("stats = %+v, want refused sync-push p50=37ms", stats)
	}
}

func TestFoldGitOperationsSeparatesTerminalOutcomes(t *testing.T) {
	rows := []usagelog.Row{
		{Schema: usagelog.SchemaV1, Verb: "sync push", ExitCode: 0, DurationMS: 900},
		{Schema: usagelog.SchemaV1, Verb: "sync push", ExitCode: 0, DurationMS: 1100},
		{Schema: usagelog.SchemaV1, Verb: "sync push", ExitCode: 3, DurationMS: 5},
		{Schema: usagelog.SchemaV1, Verb: "sync push", ExitCode: 2, DurationMS: 2},
		{Schema: usagelog.SchemaV1, Verb: "sync push", ExitCode: 4, DurationMS: 7},
		{Schema: usagelog.SchemaV1, Verb: "dev sync push", ExitCode: 0, DurationMS: 1300},
		{Schema: usagelog.SchemaV1, Verb: "route", ExitCode: 0, DurationMS: 1},
	}
	stats := usagelog.FoldRows(rows, usagelog.FoldOptions{}).ByOperationOutcome
	if len(stats) != 4 {
		t.Fatalf("stats = %+v, want four outcome buckets", stats)
	}
	want := map[usagelog.TerminalOutcome]int64{usagelog.OutcomeSuccess: 1100, usagelog.OutcomeRefused: 5, usagelog.OutcomeUsage: 2, usagelog.OutcomeError: 7}
	for _, stat := range stats {
		if got, ok := want[stat.Outcome]; !ok || stat.P50MS != got {
			t.Fatalf("stat = %+v, want p50 %d (known=%v)", stat, got, ok)
		}
	}
}

func TestGitOperationNameIsClosedAndArgumentSafe(t *testing.T) {
	tests := []struct {
		verb string
		argv []string
		want string
	}{
		{"commit", []string{"--path", "secret/customer.txt", "-m", "token=do-not-log"}, "commit local"},
		{"commit", []string{"--push", "--path", "x"}, "commit push"},
		{"commit", []string{"-m", "--push", "--path", "x"}, "commit local"},
		{"commit", []string{"--path=--push", "-m=x"}, "commit local"},
		{"commit", []string{"--", "--push"}, "commit local"},
		{"dev commit", []string{"--preview", "-m", "x"}, "dev commit"},
		{"sweep", []string{"--apply", "--push", "--no-origin"}, "sweep apply push"},
		{"sync", []string{"apply", "--fetch", "--remote", "private-origin"}, "sync apply fetch"},
		{"sync", []string{"--repo", "secret/path"}, "sync check"},
	}
	for _, tc := range tests {
		if got := gitOperationName(tc.verb, tc.argv); got != tc.want {
			t.Errorf("gitOperationName(%q, %q) = %q, want %q", tc.verb, tc.argv, got, tc.want)
		}
	}
}

func TestRecordUsageMultipleInvocationsChain(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")

	recordUsage("audit", nil, 0, time.Now())
	recordUsage("audit", []string{"verify", "x.jsonl"}, 1, time.Now())
	recordUsage("", nil, 2, time.Now()) // the no-verb help path

	rows, err := usagelog.ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("got %d rows, want 3", len(rows))
	}
	if rows[2].Verb != "" || rows[2].ExitCode != 2 {
		t.Errorf("row 3 = %+v, want verb=\"\" exit_code=2", rows[2])
	}
	if n, err := usagelog.Verify(path); err != nil || n != 3 {
		t.Fatalf("Verify() = (%d, %v), want (3, nil)", n, err)
	}
}

func TestRecordUsageRespectsOptOut(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "off")

	recordUsage("guard", nil, 0, time.Now())

	if _, err := os.Stat(path); err == nil {
		t.Fatalf("usage log %s was created despite FAK_USAGE_LOG=off", path)
	} else if !os.IsNotExist(err) {
		t.Fatalf("stat %s: %v", path, err)
	}
}

func TestRecordUsageExcludesHookVerb(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")

	recordUsage("hook", nil, 0, time.Now())

	if _, err := os.Stat(path); err == nil {
		t.Fatalf("usage log %s was created for the excluded 'hook' verb", path)
	} else if !os.IsNotExist(err) {
		t.Fatalf("stat %s: %v", path, err)
	}
}

// captureStdout runs fn with os.Stdout redirected to a pipe and returns what
// it wrote.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	defer func() { os.Stdout = orig }()

	fn()

	_ = w.Close()
	os.Stdout = orig
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read pipe: %v", err)
	}
	return string(out)
}

func TestCmdUsageEmptyJournal(t *testing.T) {
	withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")

	out := captureStdout(t, func() { cmdUsage(nil) })
	if !bytes.Contains([]byte(out), []byte("no rows recorded yet")) {
		t.Fatalf("cmdUsage on an empty journal = %q, want a no-rows message", out)
	}
}

func TestCmdUsageTextAndJSON(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")
	recordUsage("route", []string{"x"}, 0, time.Now())
	recordUsage("route", []string{"y"}, 1, time.Now())
	recordUsage("guard", nil, 0, time.Now())

	text := captureStdout(t, func() { cmdUsage([]string{"--by-verb"}) })
	for _, want := range []string{path, "total: 3", "errors: 1", "route", "guard"} {
		if !bytes.Contains([]byte(text), []byte(want)) {
			t.Errorf("text output missing %q; got:\n%s", want, text)
		}
	}

	jsonOut := captureStdout(t, func() { cmdUsage([]string{"--json"}) })
	var fold usagelog.Fold
	if err := json.Unmarshal([]byte(jsonOut), &fold); err != nil {
		t.Fatalf("unmarshal --json output: %v\noutput: %s", err, jsonOut)
	}
	if fold.Total != 3 || fold.Errors != 1 {
		t.Errorf("fold = %+v, want Total=3 Errors=1", fold)
	}
}

func TestCmdUsageGitOpsKeepsOutcomeLatencySeparate(t *testing.T) {
	withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")
	recordUsage("sync push", []string{"push"}, 0, time.Now().Add(-900*time.Millisecond))
	recordUsage("sync push", []string{"push"}, syncExitRefused, time.Now().Add(-5*time.Millisecond))

	text := captureStdout(t, func() { cmdUsage([]string{"--git-ops"}) })
	for _, want := range []string{"process observation, not a downstream-effect witness", "sync push", "success", "refused"} {
		if !strings.Contains(text, want) {
			t.Fatalf("git-op usage text missing %q:\n%s", want, text)
		}
	}

	jsonOut := captureStdout(t, func() { cmdUsage([]string{"--git-ops", "--json"}) })
	var stats []usagelog.OperationOutcomeStat
	if err := json.Unmarshal([]byte(jsonOut), &stats); err != nil {
		t.Fatalf("unmarshal git-op JSON: %v\n%s", err, jsonOut)
	}
	if len(stats) != 2 || stats[0].Outcome != usagelog.OutcomeSuccess || stats[1].Outcome != usagelog.OutcomeRefused {
		t.Fatalf("git-op stats = %+v, want separate success/refused rows", stats)
	}
}

func TestRecordGuardUsageAcceptsHelperShapedArgv(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_USAGE_LOG", "")
	t.Setenv("FAK_USAGE_LOG_PATH", filepath.Join(dir, "usage.jsonl"))
	oldNow, oldStart, oldOnce, oldArgs := recordUsageNow, guardUsageStart, guardUsageOnce, os.Args
	t.Cleanup(func() { recordUsageNow, guardUsageStart, guardUsageOnce, os.Args = oldNow, oldStart, oldOnce, oldArgs })
	start := time.Unix(1_700_000_000, 0)
	recordUsageNow = func() time.Time { return start.Add(time.Second) }
	guardUsageStart = start
	guardUsageOnce = new(sync.Once)
	os.Args = []string{"fak.test"}

	recordGuardUsage(0)
	data, err := os.ReadFile(filepath.Join(dir, "usage.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var row usagelog.Row
	if err := json.Unmarshal(bytes.TrimSpace(data), &row); err != nil {
		t.Fatal(err)
	}
	if row.Verb != "guard" || row.ExitCode != 0 {
		t.Fatalf("guard usage row = %+v, want successful guard row", row)
	}
}

func TestRecordGuardUsageClosesDirectExitGap(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_USAGE_LOG", "")
	t.Setenv("FAK_USAGE_LOG_PATH", filepath.Join(dir, "usage.jsonl"))
	oldNow, oldStart, oldOnce, oldArgs := recordUsageNow, guardUsageStart, guardUsageOnce, os.Args
	t.Cleanup(func() { recordUsageNow, guardUsageStart, guardUsageOnce, os.Args = oldNow, oldStart, oldOnce, oldArgs })
	start := time.Unix(1_700_000_000, 0)
	recordUsageNow = func() time.Time { return start.Add(1500 * time.Millisecond) }
	guardUsageStart = start
	guardUsageOnce = new(sync.Once)
	os.Args = []string{"fak", "guard", "--", "claude"}

	recordGuardUsage(7)
	recordGuardUsage(0)
	data, err := os.ReadFile(filepath.Join(dir, "usage.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("got %d rows, want one: %s", len(lines), data)
	}
	var row usagelog.Row
	if err := json.Unmarshal([]byte(lines[0]), &row); err != nil {
		t.Fatal(err)
	}
	if row.Verb != "guard" || row.ExitCode != 7 || row.DurationMS != 1500 {
		t.Fatalf("guard usage row = %+v", row)
	}
}
