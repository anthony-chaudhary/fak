package main

import (
	"bytes"
	"encoding/xml"
	"io"
	"strings"
	"testing"
)

// emitCron runs `fak cron emit ...` and returns stdout/exit, failing on a non-zero
// exit so the per-target tests only assert on the rendered unit.
func emitCron(t *testing.T, argv ...string) string {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runCron(&stdout, &stderr, append([]string{"emit"}, argv...))
	if code != 0 {
		t.Fatalf("cron emit %v: exit=%d stderr=%s", argv, code, stderr.String())
	}
	return stdout.String()
}

// assertWellFormedXML decodes the whole document so a malformed plist fails the
// "round-trips on the host platform" acceptance rung rather than printing garbage.
func assertWellFormedXML(t *testing.T, doc string) {
	t.Helper()
	dec := xml.NewDecoder(strings.NewReader(doc))
	for {
		_, err := dec.Token()
		if err == io.EOF {
			return
		}
		if err != nil {
			t.Fatalf("emitted plist is not well-formed XML: %v\n%s", err, doc)
		}
	}
}

func mustContain(t *testing.T, out string, needles ...string) {
	t.Helper()
	for _, n := range needles {
		if !strings.Contains(out, n) {
			t.Fatalf("emitted unit missing %q\n---\n%s", n, out)
		}
	}
}

func TestCronEmitLaunchd(t *testing.T) {
	out := emitCron(t, "--target", "launchd", "--interval", "5m", "issue-dispatch/default")
	assertWellFormedXML(t, out)
	// The action is the fak loop run vector, fired by launchd's StartInterval.
	mustContain(t, out,
		"<string>fak</string>",
		"<string>loop</string>",
		"<string>run</string>",
		"<string>--loop</string>",
		"<string>issue-dispatch/default</string>",
		"<string>--source</string>",
		"<string>launchd</string>",
		"<string>agent</string>", // the default wrapped tick
		"<key>StartInterval</key>",
		"<integer>300</integer>",
	)
	// Label is derived from the loop id with the path separator sanitized.
	mustContain(t, out, "<string>fak-loop-issue-dispatch-default</string>")
}

func TestCronEmitSystemd(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--interval", "5m", "--loop", "nightly")
	mustContain(t, out,
		"# === fak-loop-nightly.service ===",
		"# === fak-loop-nightly.timer ===",
		"Type=oneshot",
		"ExecStart=fak loop run --loop nightly --source systemd -- fak agent",
		"OnUnitActiveSec=300s",
		"OnBootSec=300s",
		"WantedBy=timers.target",
	)
}

func TestCronEmitTaskScheduler(t *testing.T) {
	out := emitCron(t, "--target", "taskscheduler", "--interval", "5m", "--loop", "nightly")
	mustContain(t, out,
		"Register-ScheduledTask",
		"New-ScheduledTaskAction -Execute 'fak' -Argument 'loop run --loop nightly --source task-scheduler -- fak agent'",
		"New-TimeSpan -Seconds 300",
		"-MultipleInstances IgnoreNew",
		"-TaskName 'fak-loop-nightly'",
	)
}

func TestCronEmitCustomTickAndFakBin(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--loop", "nightly",
		"--fak-bin", "/usr/local/bin/fak", "--", "/usr/local/bin/fak", "agent", "--offline")
	// The wrapped tick lands verbatim after the `--`, and --fak-bin drives the
	// invoked binary on the loop-run side.
	mustContain(t, out,
		"ExecStart=/usr/local/bin/fak loop run --loop nightly --source systemd -- /usr/local/bin/fak agent --offline",
	)
}

func TestCronEmitLedgerPassthrough(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--loop", "nightly", "--ledger", "/var/lib/fak/loops.jsonl")
	mustContain(t, out, "--ledger /var/lib/fak/loops.jsonl")
}

func TestCronEmitRejectsBadInput(t *testing.T) {
	cases := [][]string{
		{"emit"}, // no target, no loop
		{"emit", "--target", "cronjob", "nightly"},                     // unknown target
		{"emit", "--target", "systemd"},                                // no loop id
		{"emit", "--target", "systemd", "--interval", "0s", "nightly"}, // non-positive interval
		{"emit", "--target", "systemd", "--interval", "-5m", "nightly"},
	}
	for _, argv := range cases {
		var stdout, stderr bytes.Buffer
		if code := runCron(&stdout, &stderr, argv); code != 2 {
			t.Fatalf("runCron(%v) = %d, want 2 (stderr=%s stdout=%s)", argv, code, stderr.String(), stdout.String())
		}
	}
}

func TestCronUnknownSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runCron(&stdout, &stderr, []string{"frobnicate"}); code != 2 {
		t.Fatalf("unknown subcommand exit = %d, want 2", code)
	}
}
