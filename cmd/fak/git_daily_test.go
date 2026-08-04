package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gitdaily"
)

// TestGitDailyEmitUnitTaskScheduler is the "cron thing" acceptance witness on Windows:
// the emitted unit fires THIS verb, on a daily cadence, under the reboot-surviving S4U
// principal, and catches up a firing the box slept through. A daily job registered
// without those last two properties silently stops being daily.
func TestGitDailyEmitUnitTaskScheduler(t *testing.T) {
	root := t.TempDir()
	var out, errOut bytes.Buffer
	if code := runGitDaily(&out, &errOut, []string{"--emit-unit", "taskscheduler", "--root", root}); code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, errOut.String())
	}
	got := out.String()
	for _, want := range []string{
		"Register-ScheduledTask",
		"-TaskName 'fak-git-daily'",
		"-Execute 'fak' -Argument 'git-daily --root " + root,
		"New-TimeSpan -Seconds 86400", // the --interval default: once a day
		"-LogonType S4U",
		"-RunLevel Limited",
		"-Principal $principal",
		"-StartWhenAvailable",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("emitted unit missing %q\n---\n%s", want, got)
		}
	}
	// #3322 forbids the legacy path outright: it cannot express a principal.
	if strings.Contains(got, "schtasks /Create") {
		t.Errorf("emitted the legacy schtasks form:\n%s", got)
	}
}

// TestGitDailyEmitUnitPosixTargets keeps the other two schedulers honest: each fires the
// same verb with the same cadence, so an operator gets the identical contract whichever
// box they install it on.
func TestGitDailyEmitUnitPosixTargets(t *testing.T) {
	root := t.TempDir()
	for _, tc := range []struct{ target, want string }{
		{"systemd", "ExecStart=/usr/local/bin/fak git-daily --root "},
		{"launchd", "<string>--root</string>"},
	} {
		var out, errOut bytes.Buffer
		code := runGitDaily(&out, &errOut, []string{
			"--emit-unit", tc.target, "--fak-bin", "/usr/local/bin/fak", "--root", root,
		})
		if code != 0 {
			t.Fatalf("%s: exit = %d, stderr = %s", tc.target, code, errOut.String())
		}
		if !strings.Contains(out.String(), tc.want) {
			t.Errorf("%s unit missing %q\n---\n%s", tc.target, tc.want, out.String())
		}
		if !strings.Contains(out.String(), "git-daily") {
			t.Errorf("%s unit does not fire the verb\n---\n%s", tc.target, out.String())
		}
	}
}

// TestGitDailyEmitUnitRefusesWithoutARoot: a unit whose command relies on cwd discovery
// exits 2 every day from the scheduler's own working directory while looking installed.
// The refusal moves that failure to install time, where someone is watching.
func TestGitDailyEmitUnitRefusesWithoutARoot(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := emitGitDailyUnit(&out, &errOut, "taskscheduler", "", "fak", "", 24*time.Hour); code != 2 {
		t.Fatalf("exit = %d, want 2 (usage)", code)
	}
	if out.Len() != 0 {
		t.Fatalf("emitted a rootless unit: %s", out.String())
	}
	if !strings.Contains(errOut.String(), "--root") {
		t.Errorf("stderr = %q, want it to name the missing flag", errOut.String())
	}
}

// TestGitDailyEmitUnitRejectsUnknownTarget: a typo must be a usage error, never a
// silently-empty unit an operator installs and believes is scheduled.
func TestGitDailyEmitUnitRejectsUnknownTarget(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runGitDaily(&out, &errOut, []string{"--emit-unit", "cron"}); code != 2 {
		t.Fatalf("exit = %d, want 2 (usage)", code)
	}
	if out.Len() != 0 {
		t.Fatalf("emitted a unit for an unknown target: %s", out.String())
	}
	if !strings.Contains(errOut.String(), "unknown --emit-unit") {
		t.Errorf("stderr = %q, want it to name the bad target", errOut.String())
	}
}

// TestRenderGitDailyStatusSurfacesTheStuckStreak: the readback exists to answer "has
// this been folding, or deferring for a week?" — so a streak has to be visible without
// the operator diffing the rows themselves.
func TestRenderGitDailyStatusSurfacesTheStuckStreak(t *testing.T) {
	var out bytes.Buffer
	writeGitDailyStatus(&out, "C:/repo/.git/"+gitdaily.LedgerName, []gitdaily.Row{
		{Schema: gitdaily.Schema, Day: "2026-08-02", LooseBefore: 4200, LooseAfter: 12, PacksBefore: 3, PacksAfter: 3},
		{Schema: gitdaily.Schema, Day: "2026-08-03", LooseBefore: 5000, LooseAfter: 5000, GraceRefused: "LOCKED"},
		{Schema: gitdaily.Schema, Day: "2026-08-04", LooseBefore: 9000, LooseAfter: 9000, GraceRefused: "LOCKED"},
	})
	got := out.String()
	for _, want := range []string{"folded 4188 loose", "fold tier LOCKED", "held back LOCKED for 2 consecutive runs"} {
		if !strings.Contains(got, want) {
			t.Errorf("status render missing %q\n---\n%s", want, got)
		}
	}
}

// TestRenderGitDailyStatusEmptyLedger: a clone that has never ticked says so plainly,
// rather than rendering a blank block an operator would read as "ran, found nothing".
func TestRenderGitDailyStatusEmptyLedger(t *testing.T) {
	var out bytes.Buffer
	writeGitDailyStatus(&out, "/tmp/x.jsonl", nil)
	if !strings.Contains(out.String(), "no runs recorded yet") {
		t.Errorf("empty-ledger render = %q", out.String())
	}
}

// TestRenderGitDailyTextSkips keeps the two skips legible: both are normal outcomes of a
// coarse OS trigger, so neither may read as a failure, and each must say what to do next.
func TestRenderGitDailyTextSkips(t *testing.T) {
	var out bytes.Buffer
	writeGitDailyText(&out, gitdaily.Result{
		Schema: gitdaily.Schema, Day: "2026-08-04", Apply: true,
		Skipped: gitdaily.SkipAlreadyRanToday, LastRunDay: "2026-08-04",
	})
	if got := out.String(); !strings.Contains(got, "ALREADY_RAN_TODAY") || !strings.Contains(got, "--force") {
		t.Errorf("already-ran render = %q, want the reason and the way back in", got)
	}

	out.Reset()
	writeGitDailyText(&out, gitdaily.Result{
		Schema: gitdaily.Schema, Day: "2026-08-04", Apply: true, Skipped: gitdaily.SkipTickBusy,
	})
	if got := out.String(); !strings.Contains(got, "TICK_BUSY") {
		t.Errorf("busy render = %q", got)
	}
}

// TestGitDailyStatusReportIsValidJSON pins the machine-readable envelope: the ledger path
// travels WITH the rows, so a consumer never has to re-derive which file it read.
func TestGitDailyStatusReportIsValidJSON(t *testing.T) {
	b, err := json.Marshal(gitDailyStatusReport{
		Schema: "fak-git-daily-status/1",
		Ledger: "/repo/.git/" + gitdaily.LedgerName,
		Rows:   []gitdaily.Row{{Schema: gitdaily.Schema, Day: "2026-08-04"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var back map[string]any
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back["ledger"] == "" || back["schema"] != "fak-git-daily-status/1" {
		t.Fatalf("envelope = %s", b)
	}
}
