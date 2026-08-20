package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestGuardDisableUsageAppendsPrivacySafeOutcomesAndFoldsWeeks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "guard-disable.jsonl")
	rows := []guardDisableUsageRow{
		{At: "2026-08-03T10:00:00Z", Outcome: guardDisableUsageSuccess},
		{At: "2026-08-04T10:00:00Z", Outcome: guardDisableUsageChildNonzero},
		{At: "2026-08-10T10:00:00Z", Outcome: guardDisableUsageLaunchError},
	}
	for _, row := range rows {
		if err := appendGuardDisableUsage(path, row); err != nil {
			t.Fatal(err)
		}
	}
	weeks, err := foldGuardDisableUsage(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(weeks) != 2 || weeks[0].Week != "2026-W32" || weeks[0].Invocations != 2 || weeks[0].Success != 1 || weeks[0].ChildNonzero != 1 || weeks[1].LaunchError != 1 {
		t.Fatalf("weeks = %+v", weeks)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"reason", "command", "workspace", "hostname", "username", `C:\\`, "/home/"} {
		if strings.Contains(string(raw), forbidden) {
			t.Fatalf("ledger leaks %q: %s", forbidden, raw)
		}
	}
}

func TestGuardDisableUsageRecordsRealChildOutcome(t *testing.T) {
	path := filepath.Join(t.TempDir(), "guard-disable.jsonl")
	var stdout, stderr bytes.Buffer
	command := guardDisableTestExitCommand(19)
	code := runGuardDisableWithUsage("guard", strings.NewReader(""), &stdout, &stderr,
		append([]string{"--reason", "private repair detail", "--"}, command...), path, nil)
	if code != 19 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	weeks, err := foldGuardDisableUsage(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(weeks) != 1 || weeks[0].Invocations != 1 || weeks[0].ChildNonzero != 1 {
		t.Fatalf("weeks = %+v", weeks)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "private repair detail") || strings.Contains(string(raw), command[0]) {
		t.Fatalf("ledger captured reason or command: %s", raw)
	}
}

func TestGuardDisableUsageFlagRendersJSONFoldWithoutStartingChild(t *testing.T) {
	path := filepath.Join(t.TempDir(), "guard-disable.jsonl")
	if err := appendGuardDisableUsage(path, guardDisableUsageRow{At: time.Now().UTC().Format(time.RFC3339), Outcome: guardDisableUsageSuccess}); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runGuardDisableWithUsage("guard", strings.NewReader(""), &out, &errOut, []string{"--usage", "--json"}, path, nil); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got guardDisableUsageSummary
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != guardDisableUsageSummarySchema || len(got.Weeks) != 1 || got.Weeks[0].Success != 1 {
		t.Fatalf("summary = %+v", got)
	}
	if !strings.Contains(out.String(), `"success": 1`) {
		t.Fatalf("captured readout missing outcome count: %s", out.String())
	}
}

func guardDisableTestExitCommand(code int) []string {
	if runtime.GOOS == "windows" {
		return []string{"cmd", "/c", "exit", strconv.Itoa(code)}
	}
	return []string{"sh", "-c", "exit " + strconv.Itoa(code)}
}
