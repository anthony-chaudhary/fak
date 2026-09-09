package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestOpsScheduleUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOpsSchedule(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("expected exit code 0 on --help, got %d (stderr: %s)", code, stderr.String())
	}
	out := stderr.String()
	if !strings.Contains(out, "Usage: fak ops schedule") {
		t.Errorf("expected usage output, got:\n%s", out)
	}
}

func TestOpsScheduleFlagsValidation(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"non-positive interval issue", []string{"--interval-issue", "0s"}},
		{"negative interval debt", []string{"--interval-debt", "-10m"}},
		{"negative interval sync", []string{"--interval-sync", "-5m"}},
		{"zero timeout issue", []string{"--timeout-issue", "0s"}},
		{"negative timeout debt", []string{"--timeout-debt", "-1m"}},
		{"negative timeout sync", []string{"--timeout-sync", "-1m"}},
		{"zero run hours", []string{"--run-hours", "0"}},
		{"excessive run hours", []string{"--run-hours", "200"}},
		{"unknown target", []string{"--target", "invalid-target"}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runOpsSchedule(&stdout, &stderr, tc.args)
			if code != 2 {
				t.Errorf("expected exit code 2 on invalid flags %v, got %d", tc.args, code)
			}
		})
	}
}

func TestOpsScheduleDryRunTargets(t *testing.T) {
	targets := []string{"taskscheduler", "systemd", "launchd"}

	for _, target := range targets {
		t.Run(target, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runOpsSchedule(&stdout, &stderr, []string{
				"--target", target,
				"--interval-issue", "10m",
				"--interval-debt", "15m",
				"--interval-sync", "1h",
				"--timeout-issue", "25m",
				"--timeout-debt", "25m",
				"--timeout-sync", "8m",
				"--dry-run",
			})
			if code != 0 {
				t.Fatalf("expected code 0, got %d (stderr: %s)", code, stderr.String())
			}

			out := stdout.String()
			if !strings.Contains(out, "IssueOrchestrator") {
				t.Errorf("missing IssueOrchestrator in %s dry-run output:\n%s", target, out)
			}
			if !strings.Contains(out, "DebtOrchestrator") {
				t.Errorf("missing DebtOrchestrator in %s dry-run output:\n%s", target, out)
			}
			if !strings.Contains(out, "GitSync") {
				t.Errorf("missing GitSync in %s dry-run output:\n%s", target, out)
			}

			switch target {
			case "taskscheduler":
				if !strings.Contains(out, "Register-ScheduledTask") {
					t.Errorf("expected Register-ScheduledTask in taskscheduler output")
				}
			case "systemd":
				if !strings.Contains(out, "[Timer]") || !strings.Contains(out, "[Service]") {
					t.Errorf("expected systemd Timer/Service units in output")
				}
			case "launchd":
				if !strings.Contains(out, "<plist version=\"1.0\">") {
					t.Errorf("expected launchd plist in output")
				}
			}
		})
	}
}

func TestOpsScheduleDryRunJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOpsSchedule(&stdout, &stderr, []string{
		"--target", "taskscheduler",
		"--interval-issue", "10m",
		"--interval-debt", "15m",
		"--interval-sync", "1h",
		"--dry-run",
		"--json",
	})
	if code != 0 {
		t.Fatalf("expected code 0, got %d (stderr: %s)", code, stderr.String())
	}

	var report OpsScheduleReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to decode report JSON: %v\nOutput was:\n%s", err, stdout.String())
	}

	if report.Schema != "fak-ops-schedule/1" {
		t.Errorf("expected schema fak-ops-schedule/1, got %q", report.Schema)
	}
	if report.Action != "dry_run" {
		t.Errorf("expected action dry_run, got %q", report.Action)
	}
	if len(report.Tasks) != 3 {
		t.Fatalf("expected 3 tasks, got %d", len(report.Tasks))
	}

	t1 := report.Tasks[0]
	if t1.Workload != "issue-orchestrator" || t1.Interval != "10m" || t1.Timeout != "25m" {
		t.Errorf("unexpected task 1: %+v", t1)
	}
	if !strings.Contains(t1.Definition, "Register-ScheduledTask") {
		t.Errorf("task 1 definition missing Register-ScheduledTask: %s", t1.Definition)
	}

	t2 := report.Tasks[1]
	if t2.Workload != "debt-orchestrator" || t2.Interval != "15m" || t2.Timeout != "25m" {
		t.Errorf("unexpected task 2: %+v", t2)
	}
	if !strings.Contains(t2.Definition, "Register-ScheduledTask") {
		t.Errorf("task 2 definition missing Register-ScheduledTask: %s", t2.Definition)
	}

	t3 := report.Tasks[2]
	if t3.Workload != "git-sync" || t3.Interval != "1h" || t3.Timeout != "8m" {
		t.Errorf("unexpected task 3: %+v", t3)
	}
	if !strings.Contains(t3.Definition, "Register-ScheduledTask") {
		t.Errorf("task 3 definition missing Register-ScheduledTask: %s", t3.Definition)
	}
}

func TestFormatOpsDurationAndTaskName(t *testing.T) {
	if got := formatOpsDuration(10 * time.Minute); got != "10m" {
		t.Errorf("expected 10m, got %s", got)
	}
	if got := formatOpsDuration(15 * time.Minute); got != "15m" {
		t.Errorf("expected 15m, got %s", got)
	}
	if got := formatOpsDuration(1 * time.Hour); got != "1h" {
		t.Errorf("expected 1h, got %s", got)
	}
	if got := formatOpsDuration(90 * time.Second); got != "1m30s" {
		t.Errorf("expected 1m30s, got %s", got)
	}

	taskName := formatOpsTaskName("FakOps", "DebtOrchestrator", 15*time.Minute)
	if taskName != "FakOps-15m-DebtOrchestrator" {
		t.Errorf("expected FakOps-15m-DebtOrchestrator, got %s", taskName)
	}
}

func TestOpsScheduleStatusLive(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOpsSchedule(&stdout, &stderr, []string{"--status"})
	if code != 0 {
		t.Fatalf("expected code 0 on --status, got %d (stderr: %s)", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Developer Workstation Ops Status") {
		t.Errorf("expected header in --status output, got:\n%s", out)
	}
	if !strings.Contains(out, "issue-orchestrator") || !strings.Contains(out, "debt-orchestrator") || !strings.Contains(out, "git-sync") {
		t.Errorf("expected all 3 workloads in --status output, got:\n%s", out)
	}
}
