package main

import (
	"bytes"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestCronEmitRunnerLaunchd(t *testing.T) {
	out := emitCron(t,
		"--runner",
		"--target", "launchd",
		"--job", "db-vacuum",
		"--ledger", "/var/log/cron.jsonl",
		"--interval", "2h",
		"--timeout", "15m",
		"--", "vacuumdb", "--all",
	)
	assertWellFormedXML(t, out)
	mustContain(t, out,
		"<string>fak</string>",
		"<string>cron</string>",
		"<string>run</string>",
		"<string>--job</string>",
		"<string>db-vacuum</string>",
		"<string>--ledger</string>",
		"<string>/var/log/cron.jsonl</string>",
		"<string>--interval</string>",
		"<string>2h0m0s</string>",
		"<string>--timeout</string>",
		"<string>15m0s</string>",
		"<string>--</string>",
		"<string>vacuumdb</string>",
		"<string>--all</string>",
		"<key>StartInterval</key>",
		"<integer>7200</integer>",
		"<string>fak-cron-db-vacuum</string>",
	)
}

func TestCronEmitRunnerSystemd(t *testing.T) {
	out := emitCron(t,
		"--runner",
		"--target", "systemd",
		"--job", "cache-prune",
		"--ledger", "/var/log/cron.jsonl",
		"--interval", "30m",
		"--timeout", "5m",
		"--", "find", "/tmp/cache", "-type", "f", "-delete",
	)
	mustContain(t, out,
		"# === fak-cron-cache-prune.service ===",
		"# === fak-cron-cache-prune.timer ===",
		"Type=oneshot",
		"ExecStart=fak cron run --job cache-prune --ledger /var/log/cron.jsonl --interval 30m0s --timeout 5m0s -- find /tmp/cache -type f -delete",
		"OnUnitActiveSec=1800s",
		"OnBootSec=1800s",
		"WantedBy=timers.target",
	)
}

func TestCronEmitRunnerTaskScheduler(t *testing.T) {
	out := emitCron(t,
		"--runner",
		"--target", "taskscheduler",
		"--job", "log-rotate",
		"--ledger", "C:\\fak\\cron.jsonl",
		"--interval", "1h",
		"--timeout", "10m",
		"--", "powershell", "-Command", "Clear-History",
	)
	mustContain(t, out,
		"Register-ScheduledTask",
		"New-ScheduledTaskAction -Execute 'fak' -Argument 'cron run --job log-rotate --ledger C:\\fak\\cron.jsonl --interval 1h0m0s --timeout 10m0s -- powershell -Command Clear-History'",
		"New-TimeSpan -Seconds 3600",
		"-TaskName 'fak-cron-log-rotate'",
	)
}

func TestCronEmitRunnerArgvExecutionAgainstRunCron(t *testing.T) {
	tmpDir := t.TempDir()
	ledger := filepath.ToSlash(filepath.Join(tmpDir, "cron_runner_exec.jsonl"))

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd", "/c", "echo executed-from-runner"}
	} else {
		cmdArgs = []string{"echo", "executed-from-runner"}
	}

	emitArgs := append([]string{
		"--runner",
		"--target", "systemd",
		"--job", "runner-exec-test",
		"--ledger", ledger,
		"--interval", "1h",
		"--timeout", "10s",
		"--",
	}, cmdArgs...)

	var emitOut, emitErr bytes.Buffer
	code := runCron(&emitOut, &emitErr, append([]string{"emit"}, emitArgs...))
	if code != 0 {
		t.Fatalf("cron emit failed: code=%d, stderr=%s", code, emitErr.String())
	}

	// In the emitted systemd text, ExecStart gives the exact command line
	lines := strings.Split(emitOut.String(), "\n")
	var execLine string
	for _, l := range lines {
		if strings.HasPrefix(l, "ExecStart=") {
			execLine = strings.TrimPrefix(l, "ExecStart=")
			break
		}
	}
	if execLine == "" {
		t.Fatalf("ExecStart not found in emit output:\n%s", emitOut.String())
	}

	parsedArgs, ok := cronShlexSplit(execLine)
	if !ok {
		t.Fatalf("failed to split ExecStart line: %s", execLine)
	}

	// parsedArgs[0] is "fak", parsedArgs[1:] is ["cron", "run", ...]
	if len(parsedArgs) < 2 || parsedArgs[0] != "fak" || parsedArgs[1] != "cron" {
		t.Fatalf("unexpected parsed argv: %v", parsedArgs)
	}

	// Run through runCron directly using the emitted args after "fak cron"
	cronArgv := parsedArgs[2:]
	var runStdout, runStderr bytes.Buffer
	runCode := runCron(&runStdout, &runStderr, cronArgv)
	if runCode != 0 {
		t.Fatalf("runCron failed with code %d: stderr=%s", runCode, runStderr.String())
	}

	if !strings.Contains(runStdout.String(), "executed-from-runner") {
		t.Fatalf("runStdout did not contain output: %s", runStdout.String())
	}

	runs, err := cronReadRuns(ledger)
	if err != nil {
		t.Fatalf("cronReadRuns failed: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("expected 1 run record, got %d", len(runs))
	}
	if runs[0].Job != "runner-exec-test" || runs[0].Outcome != cronRunOutcomeSucceeded {
		t.Fatalf("unexpected run record: %+v", runs[0])
	}
}

func TestCronEmitRunnerMutualExclusivityAndValidation(t *testing.T) {
	cases := []struct {
		name string
		argv []string
	}{
		{
			name: "runner with command",
			argv: []string{"emit", "--runner", "--target", "systemd", "--command", "fak foo", "--job", "j1", "--ledger", "l1", "--interval", "1h", "--timeout", "5m", "--", "echo", "1"},
		},
		{
			name: "runner with loop",
			argv: []string{"emit", "--runner", "--target", "systemd", "--loop", "loop1", "--job", "j1", "--ledger", "l1", "--interval", "1h", "--timeout", "5m", "--", "echo", "1"},
		},
		{
			name: "runner missing job",
			argv: []string{"emit", "--runner", "--target", "systemd", "--ledger", "l1", "--interval", "1h", "--timeout", "5m", "--", "echo", "1"},
		},
		{
			name: "runner missing ledger",
			argv: []string{"emit", "--runner", "--target", "systemd", "--job", "j1", "--interval", "1h", "--timeout", "5m", "--", "echo", "1"},
		},
		{
			name: "runner zero timeout",
			argv: []string{"emit", "--runner", "--target", "systemd", "--job", "j1", "--ledger", "l1", "--interval", "1h", "--timeout", "0s", "--", "echo", "1"},
		},
		{
			name: "runner missing trailing args",
			argv: []string{"emit", "--runner", "--target", "systemd", "--job", "j1", "--ledger", "l1", "--interval", "1h", "--timeout", "5m"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runCron(&stdout, &stderr, tc.argv)
			if code != 2 {
				t.Fatalf("runCron(%v) = %d, want 2 (stdout=%s stderr=%s)", tc.argv, code, stdout.String(), stderr.String())
			}
		})
	}
}
