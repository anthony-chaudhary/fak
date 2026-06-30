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

// TestCronEmitCommandSystemd proves --command emits a unit whose ExecStart is the
// arbitrary command verbatim — no `fak loop run` wrapper — and matches the garden
// watchdog's spec name so `fak start` auto-heals what `cron emit` produced (#1385).
func TestCronEmitCommandSystemd(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--label", "fleet-stale-work-garden",
		"--command", "fak garden --check", "--interval", "1h")
	mustContain(t, out,
		"# === fleet-stale-work-garden.service ===",
		"# === fleet-stale-work-garden.timer ===",
		"Type=oneshot",
		"ExecStart=fak garden --check",
		"OnUnitActiveSec=3600s",
		"WantedBy=timers.target",
	)
	if strings.Contains(out, "loop run") {
		t.Fatalf("--command unit must not carry a `fak loop run` wrapper:\n%s", out)
	}
}

// TestCronEmitCommandLaunchd proves the arbitrary command lands as ProgramArguments
// (one <string> per token) with the watchdog's com.fleet.* label.
func TestCronEmitCommandLaunchd(t *testing.T) {
	out := emitCron(t, "--target", "launchd", "--label", "com.fleet.stale-work-garden",
		"--command", "fak garden --check", "--interval", "1h")
	assertWellFormedXML(t, out)
	mustContain(t, out,
		"<string>com.fleet.stale-work-garden</string>",
		"<string>fak</string>",
		"<string>garden</string>",
		"<string>--check</string>",
		"<integer>3600</integer>",
	)
	if strings.Contains(out, "<string>loop</string>") || strings.Contains(out, "<string>run</string>") {
		t.Fatalf("--command unit must not carry a `fak loop run` wrapper:\n%s", out)
	}
}

// TestCronEmitCommandTaskScheduler proves the arbitrary command becomes the task's
// -Execute (first token) + -Argument (the rest) with the watchdog's PascalCase name.
func TestCronEmitCommandTaskScheduler(t *testing.T) {
	out := emitCron(t, "--target", "taskscheduler", "--label", "FleetStaleWorkGarden",
		"--command", "fak garden --check", "--interval", "1h")
	mustContain(t, out,
		"Register-ScheduledTask",
		"New-ScheduledTaskAction -Execute 'fak' -Argument 'garden --check'",
		"New-TimeSpan -Seconds 3600",
		"-TaskName 'FleetStaleWorkGarden'",
	)
	if strings.Contains(out, "loop run") {
		t.Fatalf("--command unit must not carry a `fak loop run` wrapper:\n%s", out)
	}
}

// TestCronEmitCommandQuoting proves a multi-word argument (one shlex token with an
// embedded space) survives into each unit re-quoted faithfully, not split apart.
func TestCronEmitCommandQuoting(t *testing.T) {
	const cmd = `fak garden --note "two words" --check`
	// systemd: the spaced token is double-quoted in the ExecStart line.
	sd := emitCron(t, "--target", "systemd", "--label", "garden", "--command", cmd)
	mustContain(t, sd, `ExecStart=fak garden --note "two words" --check`)
	// launchd: each token (including the spaced one) is its own <string>.
	ld := emitCron(t, "--target", "launchd", "--label", "garden", "--command", cmd)
	assertWellFormedXML(t, ld)
	mustContain(t, ld, "<string>two words</string>", "<string>--note</string>")
	// taskscheduler: the spaced token is double-quoted inside the -Argument literal.
	ts := emitCron(t, "--target", "taskscheduler", "--label", "garden", "--command", cmd)
	mustContain(t, ts, `-Argument 'garden --note "two words" --check'`)
}

// TestCronEmitCommandDefaultLabel proves --command with no --label derives a safe
// fak-cron-<verb> name from the command's first token.
func TestCronEmitCommandDefaultLabel(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--command", "fak garden --check")
	mustContain(t, out, "# === fak-cron-fak.service ===")
}

// TestCronEmitDefaultUnchanged is a regression guard: with no --command the systemd
// unit is byte-for-byte the historical `fak loop run` form (the back-compat rung).
func TestCronEmitDefaultUnchanged(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--interval", "5m", "--loop", "nightly")
	mustContain(t, out,
		"Description=fak loop nightly (cron-emitted; OS fires, fak owns overlap-lock + missed-run policy)",
		"Description=Timer for fak loop nightly",
		"ExecStart=fak loop run --loop nightly --source systemd -- fak agent",
	)
}

func TestCronEmitCommandRejectsBadInput(t *testing.T) {
	cases := [][]string{
		{"emit", "--target", "systemd", "--command", `fak "unterminated`}, // unbalanced quote
		{"emit", "--target", "systemd", "--command", "   "},               // empty after split
	}
	for _, argv := range cases {
		var stdout, stderr bytes.Buffer
		if code := runCron(&stdout, &stderr, argv); code != 2 {
			t.Fatalf("runCron(%v) = %d, want 2 (stderr=%s stdout=%s)", argv, code, stderr.String(), stdout.String())
		}
	}
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
