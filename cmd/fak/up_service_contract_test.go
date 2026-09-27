package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
	"github.com/anthony-chaudhary/fak/internal/systemservice"
)

// Independent contract tests for `fak up off|on|status` (#13535). They reuse
// the fakeUpLaunchd harness from up_service_test.go and pin the obligations
// that file leaves open: the off ordering and its wait, the failure paths that
// must keep the dev-off intent, the lapse waker's wall-clock and token binding,
// every status verdict with its exit code under a read-only launchd, zero side
// effects for usage errors and NOT_SUPPORTED, the cmdUp peel, and the parser
// shapes the stale-definition compare depends on.

// upContractExec is one tool invocation a verb made.
type upContractExec struct {
	name string
	args []string
}

func (e upContractExec) String() string { return e.name + " " + strings.Join(e.args, " ") }

// recordUpContractExecs wraps deps.run so a test sees every tool exec,
// including the read-only launchctl print/print-disabled and plutil calls the
// fake leaves out of f.calls.
func recordUpContractExecs(deps *upServiceDeps) *[]upContractExec {
	var execs []upContractExec
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		execs = append(execs, upContractExec{name: name, args: slices.Clone(args)})
		return inner(ctx, name, args...)
	}
	return &execs
}

// isUpContractPlutilRead reports whether plutil args are one of the two plist
// reads the verbs may exec: the ProgramArguments key as JSON to stdout, or the
// Program key raw to stdout as the fallback. Both only print; neither rewrites
// the file (no -convert/-replace/-insert/-remove, and -o is always stdout).
func isUpContractPlutilRead(args []string, plist string) bool {
	return slices.Equal(args, []string{"-extract", "ProgramArguments", "json", "-o", "-", "--", plist}) ||
		slices.Equal(args, []string{"-extract", "Program", "raw", "-o", "-", "--", plist})
}

// isUpContractReadOnlyExec reports whether an exec only observes: launchctl
// print of the service target, launchctl print-disabled of the domain, or a
// plutil key extraction from the plist to stdout.
func isUpContractReadOnlyExec(e upContractExec) bool {
	switch e.name {
	case "/bin/launchctl":
		return slices.Equal(e.args, []string{"print", fakeUpServiceTgt}) ||
			slices.Equal(e.args, []string{"print-disabled", "gui/501"})
	case "/usr/bin/plutil":
		return isUpContractPlutilRead(e.args, fakeUpServicePlist)
	}
	return false
}

// snapshotUpContractDir returns every file under dir with its bytes, so a test
// can prove a verb left the dev-off state untouched.
func snapshotUpContractDir(t *testing.T, dir string) map[string]string {
	t.Helper()
	snap := map[string]string{}
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		raw, err := os.ReadFile(path)
		snap[path] = string(raw)
		return err
	})
	if err != nil {
		t.Fatalf("snapshot %s: %v", dir, err)
	}
	return snap
}

func upContractExit(code int) error { return errors.New("exit status " + strconv.Itoa(code)) }

func upContractWriteMarker(t *testing.T, deps upServiceDeps, since, until time.Time) {
	t.Helper()
	m := &upDevOffMarker{Schema: upDevOffMarkerSchema, Label: "com.fak.up", Since: since, Until: until, Token: "contract-token", CreatedBy: "fak up off"}
	if err := writeUpDevOffMarker(m, deps); err != nil {
		t.Fatal(err)
	}
}

func runUpContractWaker(t *testing.T, deps upServiceDeps, m *upDevOffMarker) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	rc := runUpService(&stdout, &stderr, []string{"on", "--lapse-token", m.Token, "--lapse-at", strconv.FormatInt(m.Until.Unix(), 10)}, deps)
	return rc, stdout.String(), stderr.String()
}

// TestUpServiceContractOffRecordsIntentThenDisablesThenBootsOut pins the
// indefinite off ordering (no --for): the dev-off marker is on disk before
// launchctl runs (so a failed launchctl still records intent), disable runs
// while the job is still loaded (so a KeepAlive respawn or a login cannot bring
// it back after the bootout), and the marker written first is the one that
// persists. The timed off skips disable; see
// TestUpServiceContractOffForSkipsDisableSoARebootRestores.
func TestUpServiceContractOffRecordsIntentThenDisablesThenBootsOut(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	dir := t.TempDir()
	deps := f.deps(t, dir)
	var tokenAtDisable string
	loadedAtDisable := false
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "disable" {
			m, err := readUpDevOffMarker("com.fak.up", deps)
			if err != nil {
				t.Errorf("dev-off marker not on disk when launchctl disable ran: %v", err)
			} else {
				tokenAtDisable = m.Token
			}
			loadedAtDisable = f.loaded
		}
		return f.run(ctx, name, args...)
	}
	start := f.now
	st, rc, stderr := runUpServiceJSON(t, deps, "off")
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
	}
	want := []string{"launchctl disable " + fakeUpServiceTgt, "launchctl bootout " + fakeUpServiceTgt}
	if !slices.Equal(f.calls, want) {
		t.Fatalf("calls = %v, want %v", f.calls, want)
	}
	if !loadedAtDisable {
		t.Fatal("disable must run while the job is still loaded, before the bootout")
	}
	if _, err := os.Stat(filepath.Join(dir, "com.fak.up.dev-off.json")); err != nil {
		t.Fatalf("marker not at <state>/<label>.dev-off.json: %v", err)
	}
	m, err := readUpDevOffMarker("com.fak.up", deps)
	if err != nil {
		t.Fatal(err)
	}
	if m.Schema != upDevOffMarkerSchema || m.Label != "com.fak.up" || !m.Since.Equal(start) || !m.Until.IsZero() || m.CreatedBy != "fak up off" {
		t.Fatalf("marker = %+v", m)
	}
	if m.Token == "" || m.Token != tokenAtDisable {
		t.Fatalf("persisted token %q, token at disable %q: the recorded intent must not be rewritten", m.Token, tokenAtDisable)
	}
	if st.DevOff == nil || st.DevOff.Token != m.Token || !st.Disabled || st.Loaded {
		t.Fatalf("off record: dev_off=%+v disabled=%v loaded=%v", st.DevOff, st.Disabled, st.Loaded)
	}
}

// TestUpServiceContractOffWaitsForUnloadAndPidNotPort pins what off waits for:
// launchd must have unloaded the job (its unload can lag the bootout) and the
// stopped pid must be dead. The port is reported but never waited on, so an
// unrelated dev server holding the address does not hold off open.
// (Supersedes the earlier "wait for pid AND port" contract: an off that waited
// on the port turned a finished off into STILL_RUNNING whenever anything else
// listened on :8080.)
func TestUpServiceContractOffWaitsForUnloadAndPidNotPort(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	loadLinger, aliveLinger := 2, 3
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "print" && !f.loaded && loadLinger > 0 {
			loadLinger-- // launchd finishes the unload asynchronously
			return []byte(f.printOutput()), nil
		}
		return inner(ctx, name, args...)
	}
	var probed []int
	deps.alive = func(pid int) bool {
		probed = append(probed, pid)
		if aliveLinger > 0 {
			aliveLinger--
			return true
		}
		return f.alive[pid]
	}
	deps.listening = func(string) bool { return true } // another process keeps :8080
	start := f.now
	st, rc, stderr := runUpServiceJSON(t, deps, "off")
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("rc=%d verdict=%s stderr=%s %+v", rc, st.Verdict, stderr, st)
	}
	if loadLinger != 0 || aliveLinger != 0 {
		t.Fatalf("off returned before launchd unloaded the job (%d prints left) and the pid died (%d probes left)", loadLinger, aliveLinger)
	}
	if st.Loaded || st.Stale || !st.Listening || !strings.Contains(st.Detail, "another process still listens on 127.0.0.1:8080") {
		t.Fatalf("off record: loaded=%v stale=%v listening=%v detail=%q", st.Loaded, st.Stale, st.Listening, st.Detail)
	}
	if waited := f.now.Sub(start); waited < 5*500*time.Millisecond || waited >= 3*time.Second {
		t.Fatalf("off waited %s; want the 2 lingering unload polls + 3 lingering pid polls (2.5s) and nothing more for the port", waited)
	}
	if len(probed) == 0 || slices.ContainsFunc(probed, func(p int) bool { return p != 74723 }) {
		t.Fatalf("liveness must be checked for the stopped pid 74723, probed %v", probed)
	}
}

// TestUpServiceContractOffPortHeldByAnotherProcessIsReportedNotWaited pins the
// inverse of the superseded "STILL_RUNNING when the port never closes": once
// the job is unloaded and its pid is dead, off is OFF (exit 0) at once even if
// something else keeps answering a ready /healthz on the service address. The
// listener is named in Detail, the dev-off marker outranks PORT_CONFLICT, and a
// fresh status agrees.
func TestUpServiceContractOffPortHeldByAnotherProcessIsReportedNotWaited(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	deps.listening = func(string) bool { return true }
	start := f.now
	st, rc, stderr := runUpServiceJSON(t, deps, "off")
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
	}
	if !f.now.Equal(start) {
		t.Fatalf("off polled for %s after the job was unloaded and its pid was dead; the port must not be waited on", f.now.Sub(start))
	}
	if !st.Listening || !st.HealthReady || !strings.Contains(st.Detail, "another process still listens on "+st.Addr) {
		t.Fatalf("the foreign listener must be reported: listening=%v ready=%v detail=%q", st.Listening, st.HealthReady, st.Detail)
	}
	if f.loaded || !f.disabled {
		t.Fatalf("launchd side not applied: loaded=%v disabled=%v", f.loaded, f.disabled)
	}
	if st2, rc2, _ := runUpServiceJSON(t, deps, "status"); st2.Verdict != upServiceVerdictOff || rc2 != 0 {
		t.Fatalf("status with a foreign listener during dev-off: verdict=%s rc=%d, want OFF rc=0", st2.Verdict, rc2)
	}
}

// TestUpServiceContractOffStillRunningPastWait pins what STILL_RUNNING (exit 1)
// now means: after --wait, launchd still has the job loaded, or the stopped pid
// is still alive. The wait is bounded by --wait, the dev-off intent stays, and
// the disable still applied.
func TestUpServiceContractOffStillRunningPastWait(t *testing.T) {
	for _, tc := range []struct {
		name       string
		setup      func(f *fakeUpLaunchd, deps *upServiceDeps)
		wantLoaded bool
	}{
		{"the stopped pid outlives --wait", func(f *fakeUpLaunchd, deps *upServiceDeps) {
			deps.alive = func(pid int) bool { return pid == 74723 }
		}, false},
		{"launchd keeps the job loaded past --wait", func(f *fakeUpLaunchd, deps *upServiceDeps) {
			inner := deps.run
			deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
				if name == "/bin/launchctl" && args[0] == "bootout" {
					f.calls = append(f.calls, "launchctl "+strings.Join(args, " "))
					return nil, nil // accepted, but the job never unloads
				}
				return inner(ctx, name, args...)
			}
		}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
			deps := f.deps(t, t.TempDir())
			tc.setup(f, &deps)
			start := f.now
			st, rc, _ := runUpServiceJSON(t, deps, "off", "--wait", "3s")
			if rc != 1 || st.Verdict != upServiceVerdictStillRunning || st.Loaded != tc.wantLoaded {
				t.Fatalf("rc=%d verdict=%s loaded=%v", rc, st.Verdict, st.Loaded)
			}
			if !strings.Contains(st.Detail, "pid 74723") {
				t.Fatalf("detail should name the stopped pid: %q", st.Detail)
			}
			if waited := f.now.Sub(start); waited < 3*time.Second || waited > 10*time.Second {
				t.Fatalf("off waited %s, want bounded by --wait 3s", waited)
			}
			if _, err := readUpDevOffMarker("com.fak.up", deps); err != nil {
				t.Fatalf("dev-off intent lost on STILL_RUNNING: %v", err)
			}
			want := []string{"launchctl disable " + fakeUpServiceTgt, "launchctl bootout " + fakeUpServiceTgt}
			if !slices.Equal(f.calls, want) || !f.disabled {
				t.Fatalf("calls=%v disabled=%v, want %v", f.calls, f.disabled, want)
			}
		})
	}
}

// TestUpServiceContractOffBootoutFailureKeepsIntent pins the launchctl failure
// path: a bootout that fails for a reason other than "not loaded" and leaves the
// job loaded is LAUNCHCTL_FAILED, exit 1, and the marker stays.
func TestUpServiceContractOffBootoutFailureKeepsIntent(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "bootout" {
			f.calls = append(f.calls, "launchctl "+strings.Join(args, " "))
			return []byte("Boot-out failed: 1: Operation not permitted\n"), upContractExit(1)
		}
		return f.run(ctx, name, args...)
	}
	st, rc, _ := runUpServiceJSON(t, deps, "off")
	if rc != 1 || st.Verdict != upServiceVerdictLaunchctlError || !st.Loaded || !strings.Contains(st.Detail, "bootout") {
		t.Fatalf("rc=%d verdict=%s loaded=%v detail=%q", rc, st.Verdict, st.Loaded, st.Detail)
	}
	if _, err := readUpDevOffMarker("com.fak.up", deps); err != nil {
		t.Fatalf("dev-off intent lost on a failed bootout: %v", err)
	}
	if !f.disabled {
		t.Fatal("disable should still have run before the failed bootout")
	}
}

// TestUpServiceContractOffTreatsRacedBootoutAsGone pins that a bootout that
// reports "No such process" (the job unloaded between observe and bootout) is
// success, not LAUNCHCTL_FAILED.
func TestUpServiceContractOffTreatsRacedBootoutAsGone(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "bootout" {
			_, _ = f.run(ctx, name, args...) // the job is gone either way
			return []byte("Boot-out failed: 3: No such process\n"), upContractExit(3)
		}
		return f.run(ctx, name, args...)
	}
	st, rc, stderr := runUpServiceJSON(t, deps, "off")
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
	}
}

// TestUpServiceContractOffTwiceRotatesTokenWithoutBootout pins idempotence when
// the service is already off: a second off does not bootout an unloaded job,
// still succeeds, and its fresh token turns the first off's --for waker into a
// no-op, so the service stays off.
func TestUpServiceContractOffTwiceRotatesTokenWithoutBootout(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	if _, rc, _ := runUpServiceJSON(t, deps, "off", "--for", "90s"); rc != 0 {
		t.Fatalf("first off rc=%d", rc)
	}
	first, err := readUpDevOffMarker("com.fak.up", deps)
	if err != nil {
		t.Fatal(err)
	}
	f.calls = nil
	st, rc, _ := runUpServiceJSON(t, deps, "off")
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("second off rc=%d verdict=%s", rc, st.Verdict)
	}
	if !slices.Equal(f.calls, []string{"launchctl disable " + fakeUpServiceTgt}) {
		t.Fatalf("second off calls = %v, want only the (idempotent) disable", f.calls)
	}
	second, err := readUpDevOffMarker("com.fak.up", deps)
	if err != nil {
		t.Fatal(err)
	}
	if second.Token == first.Token || !second.Until.IsZero() {
		t.Fatalf("second off must replace the window: first=%+v second=%+v", first, second)
	}
	f.calls = nil
	if rc, _, _ := runUpContractWaker(t, deps, first); rc != 0 || len(f.calls) != 0 || f.loaded || !f.disabled {
		t.Fatalf("superseded waker acted: rc=%d calls=%v loaded=%v disabled=%v", rc, f.calls, f.loaded, f.disabled)
	}
}

// TestUpServiceContractLapseWakerNoOpsAfterOnClearedMarker pins that a waker
// whose marker an earlier `on` removed does nothing at all.
func TestUpServiceContractLapseWakerNoOpsAfterOnClearedMarker(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	if _, rc, _ := runUpServiceJSON(t, deps, "off", "--for", "90s"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	m, err := readUpDevOffMarker("com.fak.up", deps)
	if err != nil {
		t.Fatal(err)
	}
	if _, rc, _ := runUpServiceJSON(t, deps, "on"); rc != 0 {
		t.Fatalf("on rc=%d", rc)
	}
	f.calls = nil
	runs := f.runs
	rc, stdout, _ := runUpContractWaker(t, deps, m)
	if rc != 0 || len(f.calls) != 0 || f.runs != runs {
		t.Fatalf("waker acted after on: rc=%d calls=%v runs %d->%d", rc, f.calls, runs, f.runs)
	}
	if !strings.Contains(stdout, "nothing to restore") {
		t.Fatalf("a no-op waker should say so: %q", stdout)
	}
}

// TestUpServiceContractLapseWakerPollsWallClockAcrossSystemSleep pins the
// wall-clock design: the waker polls in bounded steps (never one long sleep
// that a sleeping Mac would stretch), so when the clock jumps past the window
// during a system sleep, it restores on the very next poll.
func TestUpServiceContractLapseWakerPollsWallClockAcrossSystemSleep(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	if _, rc, _ := runUpServiceJSON(t, deps, "off", "--for", "2h"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	m, err := readUpDevOffMarker("com.fak.up", deps)
	if err != nil {
		t.Fatal(err)
	}
	restoring := false
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] != "print" && args[0] != "print-disabled" {
			restoring = true
		}
		return inner(ctx, name, args...)
	}
	var lapseSleeps []time.Duration
	deps.sleep = func(d time.Duration) {
		if !restoring {
			lapseSleeps = append(lapseSleeps, d)
			if len(lapseSleeps) == 1 {
				f.now = f.now.Add(3 * time.Hour) // the lid was closed past the window
				return
			}
		}
		f.now = f.now.Add(d)
	}
	if rc, _, stderr := runUpContractWaker(t, deps, m); rc != 0 {
		t.Fatalf("waker rc=%d stderr=%s", rc, stderr)
	}
	if len(lapseSleeps) != 1 || lapseSleeps[0] <= 0 || lapseSleeps[0] > 30*time.Second {
		t.Fatalf("lapse polls = %v, want one bounded (<=30s) poll before the post-sleep restore", lapseSleeps)
	}
	if !f.loaded || f.disabled {
		t.Fatalf("waker did not restore after the wall clock passed the window: loaded=%v disabled=%v", f.loaded, f.disabled)
	}
}

// TestUpServiceContractLapsedWindowReadsOffLapsedUntilWakerRestores pins the
// lapse handoff: once the window passed, status says OFF_LAPSED (exit 1), and
// a waker that starts late restores at once without sleeping.
func TestUpServiceContractLapsedWindowReadsOffLapsedUntilWakerRestores(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	if _, rc, _ := runUpServiceJSON(t, deps, "off", "--for", "1m"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	m, err := readUpDevOffMarker("com.fak.up", deps)
	if err != nil {
		t.Fatal(err)
	}
	f.now = f.now.Add(2 * time.Minute)
	if st, rc, _ := runUpServiceJSON(t, deps, "status"); st.Verdict != upServiceVerdictOffLapsed || rc != 1 {
		t.Fatalf("status after the window: verdict=%s rc=%d", st.Verdict, rc)
	}
	restoring := false
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "enable" {
			restoring = true
		}
		return inner(ctx, name, args...)
	}
	innerSleep := deps.sleep
	deps.sleep = func(d time.Duration) {
		if !restoring {
			t.Errorf("a late waker slept %s before restoring an already-lapsed window", d)
		}
		innerSleep(d)
	}
	if rc, _, stderr := runUpContractWaker(t, deps, m); rc != 0 {
		t.Fatalf("waker rc=%d stderr=%s", rc, stderr)
	}
	if st, rc, _ := runUpServiceJSON(t, deps, "status"); st.Verdict != upServiceVerdictOn || rc != 0 || st.DevOff != nil {
		t.Fatalf("status after the waker: verdict=%s rc=%d dev_off=%+v", st.Verdict, rc, st.DevOff)
	}
}

// TestUpServiceContractOnBootstrapFailureKeepsIntent pins that the dev-off
// marker is cleared only once the job is loaded: a failed bootstrap is
// LAUNCHCTL_FAILED, exit 1, and the marker stays.
func TestUpServiceContractOnBootstrapFailureKeepsIntent(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	if _, rc, _ := runUpServiceJSON(t, deps, "off"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	f.calls = nil
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "bootstrap" {
			f.calls = append(f.calls, "launchctl "+strings.Join(args, " "))
			return []byte("Bootstrap failed: 5: Input/output error\n"), upContractExit(5)
		}
		return inner(ctx, name, args...)
	}
	st, rc, _ := runUpServiceJSON(t, deps, "on")
	if rc != 1 || st.Verdict != upServiceVerdictLaunchctlError || st.Loaded || !strings.Contains(st.Detail, "bootstrap") {
		t.Fatalf("rc=%d verdict=%s loaded=%v detail=%q", rc, st.Verdict, st.Loaded, st.Detail)
	}
	want := []string{"launchctl enable " + fakeUpServiceTgt, "launchctl bootstrap gui/501 " + fakeUpServicePlist}
	if !slices.Equal(f.calls, want) {
		t.Fatalf("calls = %v, want %v", f.calls, want)
	}
	if _, err := readUpDevOffMarker("com.fak.up", deps); err != nil {
		t.Fatalf("dev-off marker cleared although the bootstrap failed: %v", err)
	}
}

// TestUpServiceContractOnBootstrapErrorButLoadedProceeds pins that a bootstrap
// error with the job loaded anyway (a racing load) is not a failure.
func TestUpServiceContractOnBootstrapErrorButLoadedProceeds(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	if _, rc, _ := runUpServiceJSON(t, deps, "off"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "bootstrap" {
			_, _ = inner(ctx, name, args...)
			return []byte("Bootstrap failed: 17: File exists\n"), upContractExit(17)
		}
		return inner(ctx, name, args...)
	}
	st, rc, stderr := runUpServiceJSON(t, deps, "on")
	if rc != 0 || st.Verdict != upServiceVerdictOn {
		t.Fatalf("rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
	}
	if _, err := readUpDevOffMarker("com.fak.up", deps); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("dev-off marker should be cleared once the job is loaded: %v", err)
	}
}

// TestUpServiceContractOnNotReadyLeavesServiceStarting pins that an on that
// timed out waiting for /healthz leaves a loaded, starting service: status then
// reads STARTING (exit 0), never OFF.
func TestUpServiceContractOnNotReadyLeavesServiceStarting(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	if _, rc, _ := runUpServiceJSON(t, deps, "off"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	f.readyAfter = -1
	start := f.now
	st, rc, _ := runUpServiceJSON(t, deps, "on", "--wait", "10s")
	if rc != 1 || st.Verdict != upServiceVerdictNotReady || !st.Loaded || st.HealthReady || st.HealthStatus != "warming_up" {
		t.Fatalf("rc=%d verdict=%s %+v", rc, st.Verdict, st)
	}
	if waited := f.now.Sub(start); waited < 10*time.Second || waited > 20*time.Second {
		t.Fatalf("on waited %s, want bounded by --wait 10s", waited)
	}
	st, rc, _ = runUpServiceJSON(t, deps, "status")
	if st.Verdict != upServiceVerdictStarting || rc != 0 {
		t.Fatalf("status after a NOT_READY on: verdict=%s rc=%d", st.Verdict, rc)
	}
}

// TestUpServiceContractOnHonorsPlistOverride pins that --plist is the file both
// read (plutil) and bootstrapped.
func TestUpServiceContractOnHonorsPlistOverride(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	f.loaded, f.listening = false, false
	deps := f.deps(t, t.TempDir())
	execs := recordUpContractExecs(&deps)
	const custom = "/Users/example/custom/com.fak.up.plist"
	st, rc, stderr := runUpServiceJSON(t, deps, "on", "--plist", custom)
	if rc != 0 || st.Verdict != upServiceVerdictOn || st.PlistPath != custom {
		t.Fatalf("rc=%d verdict=%s plist=%q stderr=%s", rc, st.Verdict, st.PlistPath, stderr)
	}
	if !slices.Contains(f.calls, "launchctl bootstrap gui/501 "+custom) {
		t.Fatalf("bootstrap did not use --plist: %v", f.calls)
	}
	for _, e := range *execs {
		if e.name == "/usr/bin/plutil" && e.args[len(e.args)-1] != custom {
			t.Fatalf("plutil read %v, want the --plist path", e.args)
		}
	}
}

// upContractPrintXform rewrites the fake's launchctl print output for the
// verdicts the stock (running) shape cannot express.
func upContractPrintXform(deps *upServiceDeps, xform func(string) string) {
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		out, err := inner(ctx, name, args...)
		if err == nil && name == "/bin/launchctl" && args[0] == "print" {
			out = []byte(xform(string(out)))
		}
		return out, err
	}
}

func dropUpContractPID(out string) string {
	lines := slices.DeleteFunc(strings.Split(out, "\n"), func(l string) bool {
		return strings.HasPrefix(strings.TrimSpace(l), "pid = ")
	})
	return strings.Join(lines, "\n")
}

// notRunningUpContractPrint is the print shape of a loaded job with no process.
func notRunningUpContractPrint(s string) string {
	return dropUpContractPID(strings.Replace(s, "\tstate = running\n", "\tstate = not running\n", 1))
}

// upContractBrokenPrints are launchctl print failures that are NOT "service not
// found": each must leave launchd state unknown, never "not loaded".
func upContractBrokenPrints(f *fakeUpLaunchd) map[string]func() ([]byte, error) {
	return map[string]func() ([]byte, error){
		"truncated mid-dump": func() ([]byte, error) {
			out := f.printOutput()
			return []byte(out[:strings.Index(out, "\tenvironment = {")]), nil
		},
		"denied": func() ([]byte, error) {
			return []byte("Could not print service: 1: Operation not permitted\n"), upContractExit(1)
		},
		"empty": func() ([]byte, error) { return nil, nil },
	}
}

// upContractPrintOverride answers launchctl print with broken() while when()
// holds, and defers to the fake otherwise.
func upContractPrintOverride(deps *upServiceDeps, when func() bool, broken func() ([]byte, error)) {
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "print" && when() {
			return broken()
		}
		return inner(ctx, name, args...)
	}
}

// upContractProgramOnlyPlist makes the fake plist carry only a Program key:
// the ProgramArguments extract fails and the raw Program extract answers.
func upContractProgramOnlyPlist(deps *upServiceDeps, program string) {
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/usr/bin/plutil" && len(args) > 1 {
			switch args[1] {
			case "ProgramArguments":
				return []byte("Could not extract value, error: No value at that key path or invalid key path: ProgramArguments\n"), upContractExit(1)
			case "Program":
				return []byte(program + "\n"), nil
			}
		}
		return inner(ctx, name, args...)
	}
}

// TestUpServiceContractStatusVerdictsExitCodesAndReadOnly drives every status
// verdict and pins its documented exit code (0 ON/OFF/STARTING, 1 otherwise),
// the human first line, and that status only observes: every exec is a
// read-only launchctl print/print-disabled or plutil key extraction, and the
// dev-off state dir is byte-for-byte unchanged. ON needs a ready /healthz AND
// a job pid; a ready /healthz the service cannot own is PORT_CONFLICT; a print
// failure other than "not found" is LAUNCHCTL_FAILED, never STOPPED.
func TestUpServiceContractStatusVerdictsExitCodesAndReadOnly(t *testing.T) {
	stopped := func(f *fakeUpLaunchd) { f.loaded, f.listening = false, false }
	cases := []struct {
		name    string
		setup   func(t *testing.T, f *fakeUpLaunchd, deps *upServiceDeps)
		verdict string
		rc      int
		check   func(t *testing.T, st upServiceStatus, execs []upContractExec)
	}{
		{"ready", nil, upServiceVerdictOn, 0, nil},
		{"warming_up with a pid", func(_ *testing.T, f *fakeUpLaunchd, _ *upServiceDeps) { f.readyAfter = -1 }, upServiceVerdictStarting, 0, nil},
		{"spawn scheduled without a pid", func(_ *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			f.listening = false
			upContractPrintXform(d, func(s string) string {
				return dropUpContractPID(strings.Replace(s, "\tstate = running\n", "\tstate = spawn scheduled\n", 1))
			})
		}, upServiceVerdictStarting, 0, nil},
		{"loaded but down after exit 1", func(_ *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			f.listening = false
			upContractPrintXform(d, func(s string) string {
				s = strings.Replace(s, "\tstate = running\n", "\tstate = not running\n", 1)
				s = strings.Replace(s, "last exit code = 0", "last exit code = 1: Operation not permitted", 1)
				return dropUpContractPID(s)
			})
		}, upServiceVerdictDown, 1, nil},
		{"stale definition", func(_ *testing.T, f *fakeUpLaunchd, _ *upServiceDeps) { f.loadedArgs = fakeUpServiceStaleArgs }, upServiceVerdictStale, 1, nil},
		{"off indefinitely", func(t *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			stopped(f)
			upContractWriteMarker(t, *d, f.now.Add(-time.Hour), time.Time{})
		}, upServiceVerdictOff, 0, nil},
		{"off inside a --for window", func(t *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			stopped(f)
			upContractWriteMarker(t, *d, f.now.Add(-time.Minute), f.now.Add(time.Hour))
		}, upServiceVerdictOff, 0, nil},
		{"off window lapsed", func(t *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			stopped(f)
			upContractWriteMarker(t, *d, f.now.Add(-time.Hour), f.now.Add(-time.Second))
		}, upServiceVerdictOffLapsed, 1, nil},
		{"stopped without a marker", func(_ *testing.T, f *fakeUpLaunchd, _ *upServiceDeps) { stopped(f) }, upServiceVerdictStopped, 1, nil},
		{"not installed", func(_ *testing.T, f *fakeUpLaunchd, _ *upServiceDeps) { stopped(f); f.plistArgs = nil }, upServiceVerdictNotInstalled, 1, nil},
		{"ready health on a loaded job without a pid", func(_ *testing.T, _ *fakeUpLaunchd, d *upServiceDeps) {
			upContractPrintXform(d, notRunningUpContractPrint)
		}, upServiceVerdictPortConflict, 1, func(t *testing.T, st upServiceStatus, _ []upContractExec) {
			if !st.Loaded || st.PID != 0 || !st.HealthReady {
				t.Fatalf("loaded=%v pid=%d ready=%v", st.Loaded, st.PID, st.HealthReady)
			}
		}},
		{"ready health, not loaded, no marker", func(_ *testing.T, f *fakeUpLaunchd, _ *upServiceDeps) {
			f.loaded = false // a foreign server still answers on :8080
		}, upServiceVerdictPortConflict, 1, nil},
		{"ready foreign health during a dev-off", func(t *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			f.loaded = false
			upContractWriteMarker(t, *d, f.now.Add(-time.Hour), time.Time{})
		}, upServiceVerdictOff, 0, nil},
		{"launchctl print truncated", func(_ *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			upContractPrintOverride(d, func() bool { return true }, upContractBrokenPrints(f)["truncated mid-dump"])
		}, upServiceVerdictLaunchctlError, 1, nil},
		{"launchctl print denied", func(_ *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			upContractPrintOverride(d, func() bool { return true }, upContractBrokenPrints(f)["denied"])
		}, upServiceVerdictLaunchctlError, 1, nil},
		{"launchctl print empty", func(_ *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			stopped(f) // unknown must not read as STOPPED either
			upContractPrintOverride(d, func() bool { return true }, upContractBrokenPrints(f)["empty"])
		}, upServiceVerdictLaunchctlError, 1, nil},
		{"plist carries only Program", func(_ *testing.T, f *fakeUpLaunchd, d *upServiceDeps) {
			f.loadedArgs = []string{"/Users/example/.local/bin/fak-native"}
			upContractProgramOnlyPlist(d, "/Users/example/.local/bin/fak-native")
		}, upServiceVerdictOn, 0, func(t *testing.T, st upServiceStatus, execs []upContractExec) {
			if !st.Installed || !slices.Equal(st.PlistArgs, []string{"/Users/example/.local/bin/fak-native"}) || st.Stale {
				t.Fatalf("Program fallback: installed=%v plist_args=%q stale=%v", st.Installed, st.PlistArgs, st.Stale)
			}
			var plutil []string
			for _, e := range execs {
				if e.name == "/usr/bin/plutil" {
					plutil = append(plutil, e.args[1])
				}
			}
			if !slices.Equal(plutil, []string{"ProgramArguments", "Program", "ProgramArguments", "Program"}) {
				t.Fatalf("plutil keys read = %v, want ProgramArguments then the Program fallback (json + human run)", plutil)
			}
		}},
		{"plist exists but is unreadable", func(_ *testing.T, _ *fakeUpLaunchd, d *upServiceDeps) {
			d.stat = func(path string) error { return &os.PathError{Op: "stat", Path: path, Err: os.ErrPermission} }
		}, upServiceVerdictOn, 0, func(t *testing.T, st upServiceStatus, _ []upContractExec) {
			if !st.Installed || st.PlistArgs != nil || !strings.Contains(st.Detail, "plist unreadable") {
				t.Fatalf("an unreadable plist is installed, not missing: installed=%v plist_args=%q detail=%q", st.Installed, st.PlistArgs, st.Detail)
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
			dir := t.TempDir()
			deps := f.deps(t, dir)
			if tc.setup != nil {
				tc.setup(t, f, &deps)
			}
			execs := recordUpContractExecs(&deps)
			before := snapshotUpContractDir(t, dir)

			st, rc, _ := runUpServiceJSON(t, deps, "status")
			if st.Verdict != tc.verdict || rc != tc.rc || upServiceStatusExitCode(st.Verdict) != rc {
				t.Fatalf("verdict=%s rc=%d, want %s rc=%d (%+v)", st.Verdict, rc, tc.verdict, tc.rc, st)
			}
			if st.Schema != upServiceStatusSchema || st.Label != "com.fak.up" || st.Target != fakeUpServiceTgt {
				t.Fatalf("status record identity: %+v", st)
			}
			if tc.verdict != upServiceVerdictOn && st.Next == "" {
				t.Fatalf("%s must name the next step", st.Verdict)
			}
			if st.Stale && !st.Loaded {
				t.Fatalf("stale=true with loaded=false: %+v", st)
			}
			if st.Verdict == upServiceVerdictLaunchctlError && (st.LaunchdError == "" || st.Loaded || st.PID != 0) {
				t.Fatalf("unknown launchd state: launchd_error=%q loaded=%v pid=%d", st.LaunchdError, st.Loaded, st.PID)
			}
			var stdout, stderr bytes.Buffer
			hrc := runUpService(&stdout, &stderr, []string{"status"}, deps)
			if hrc != tc.rc || !strings.HasPrefix(stdout.String(), "fak up status: "+tc.verdict+"\n") {
				t.Fatalf("human status rc=%d first line=%q", hrc, strings.SplitN(stdout.String(), "\n", 2)[0])
			}
			if st.LaunchdError != "" && !strings.Contains(stdout.String(), "launchd     unknown") {
				t.Fatalf("human status must say launchd state is unknown, not loaded/not loaded:\n%s", stdout.String())
			}

			if len(f.calls) != 0 {
				t.Fatalf("status changed launchd state: %v", f.calls)
			}
			for _, e := range *execs {
				if !isUpContractReadOnlyExec(e) {
					t.Fatalf("status ran a non-observing exec: %s", e)
				}
				if e.name == "/usr/bin/plutil" && !st.Installed {
					t.Fatalf("plist existence is a stat fact; status ran %s for a missing plist", e)
				}
			}
			if after := snapshotUpContractDir(t, dir); !maps.Equal(before, after) {
				t.Fatalf("status changed the dev-off state dir:\nbefore=%v\nafter=%v", before, after)
			}
			if tc.check != nil {
				tc.check(t, st, *execs)
			}
		})
	}
}

// TestUpServiceContractExitCodeTable pins the documented exit code of every
// verdict the verbs can emit.
func TestUpServiceContractExitCodeTable(t *testing.T) {
	for verdict, want := range map[string]int{
		upServiceVerdictOn:             0,
		upServiceVerdictOff:            0,
		upServiceVerdictStarting:       0,
		upServiceVerdictOffLapsed:      1,
		upServiceVerdictStopped:        1,
		upServiceVerdictStale:          1,
		upServiceVerdictDown:           1,
		upServiceVerdictNotInstalled:   1,
		upServiceVerdictNotReady:       1,
		upServiceVerdictStillRunning:   1,
		upServiceVerdictLaunchctlError: 1,
		upServiceVerdictPortConflict:   1,
		upServiceVerdictNotSupported:   2,
	} {
		if got := upServiceStatusExitCode(verdict); got != want {
			t.Errorf("upServiceStatusExitCode(%s) = %d, want %d", verdict, got, want)
		}
	}
}

// upContractTripwireDeps fails the test on any side effect at all.
func upContractTripwireDeps(t *testing.T, goos string) upServiceDeps {
	t.Helper()
	trip := func(what string) { t.Errorf("%s: %s touched on a path that must have no side effects", goos, what) }
	return upServiceDeps{
		goos:     goos,
		uid:      func() int { trip("uid"); return 0 },
		home:     func() (string, error) { trip("home"); return "", errors.New("tripwire") },
		stateDir: func() (string, error) { trip("stateDir"); return "", errors.New("tripwire") },
		run: func(_ context.Context, name string, args ...string) ([]byte, error) {
			trip("exec " + name + " " + strings.Join(args, " "))
			return nil, errors.New("tripwire")
		},
		httpGet: func(context.Context, string) (int, []byte, error) {
			trip("httpGet")
			return 0, nil, errors.New("tripwire")
		},
		listening:  func(string) bool { trip("listening"); return false },
		alive:      func(int) bool { trip("alive"); return false },
		lease:      func() upServiceLease { trip("lease"); return upServiceLease{} },
		now:        time.Now,
		sleep:      func(time.Duration) { trip("sleep") },
		executable: func() (string, error) { trip("executable"); return "", errors.New("tripwire") },
		spawnLapse: func(string, []string, string) (int, error) {
			trip("spawnLapse")
			return 0, errors.New("tripwire")
		},
	}
}

// TestUpServiceContractNotSupportedTouchesNothing pins NOT_SUPPORTED (exit 2)
// on Linux and Windows for every verb and flag shape, with no exec, no health
// or lease probe, no marker and no waker.
func TestUpServiceContractNotSupportedTouchesNothing(t *testing.T) {
	for _, goos := range []string{"linux", "windows"} {
		for _, argv := range [][]string{
			{"status"},
			{"status", "--json"},
			{"off", "--for", "1h"},
			{"off", "--json"},
			{"on"},
			{"on", "--lapse-token", "tok", "--lapse-at", "1"},
		} {
			var stdout, stderr bytes.Buffer
			rc := runUpService(&stdout, &stderr, argv, upContractTripwireDeps(t, goos))
			if rc != 2 || !strings.Contains(stdout.String()+stderr.String(), upServiceVerdictNotSupported) {
				t.Fatalf("%s %v: rc=%d stdout=%q stderr=%q", goos, argv, rc, stdout.String(), stderr.String())
			}
		}
	}
}

// TestUpServiceContractUsageErrorsExitTwoWithoutSideEffects pins exit 2 for a
// usage error, with nothing executed and no marker written.
func TestUpServiceContractUsageErrorsExitTwoWithoutSideEffects(t *testing.T) {
	for _, argv := range [][]string{
		nil,
		{"restart"},
		{"off", "--for", "-1m"},
		{"off", "--for", "soon"},
		{"off", "now"},
		{"status", "extra"},
		{"on", "--bogus"},
	} {
		var stdout, stderr bytes.Buffer
		if rc := runUpService(&stdout, &stderr, argv, upContractTripwireDeps(t, "darwin")); rc != 2 {
			t.Fatalf("%v: rc=%d, want 2 (stderr=%q)", argv, rc, stderr.String())
		}
		if stderr.Len() == 0 {
			t.Fatalf("%v: a usage error must explain itself on stderr", argv)
		}
	}
}

// captureUpContractStdio runs fn with os.Stdout and os.Stderr redirected.
func captureUpContractStdio(t *testing.T, fn func()) (string, string) {
	t.Helper()
	oldOut, oldErr := os.Stdout, os.Stderr
	rOut, wOut, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	rErr, wErr, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	outC, errC := make(chan string, 1), make(chan string, 1)
	go func() { b, _ := io.ReadAll(rOut); outC <- string(b) }()
	go func() { b, _ := io.ReadAll(rErr); errC <- string(b) }()
	os.Stdout, os.Stderr = wOut, wErr
	defer func() { os.Stdout, os.Stderr = oldOut, oldErr }()
	fn()
	os.Stdout, os.Stderr = oldOut, oldErr
	_ = wOut.Close()
	_ = wErr.Close()
	return <-outC, <-errC
}

// TestUpServiceContractCmdUpPeelsServiceVerbsBeforeTurnkey drives the real
// cmdUp entry point. A service verb in argv[0] must reach the service parser
// (its own usage on stderr), not the turnkey path that boots a model; the same
// words anywhere else stay turnkey. --help keeps both routes side-effect free,
// so a regression cannot boot a model from this test.
func TestUpServiceContractCmdUpPeelsServiceVerbsBeforeTurnkey(t *testing.T) {
	const turnkeyBanner = "Turnkey zero-configuration Apple Silicon model provisioner"
	for _, tc := range []struct {
		argv    []string
		service string // "" = must stay on the turnkey path
	}{
		{[]string{"status", "--help"}, "up status"},
		{[]string{"off", "--for", "1m", "--help"}, "up off"},
		{[]string{"on", "--help"}, "up on"},
		{[]string{"--model", "on", "--help"}, ""},
		{[]string{"--headless", "status", "--help"}, ""},
		{[]string{"help"}, ""},
	} {
		stdout, stderr := captureUpContractStdio(t, func() { cmdUp(tc.argv) })
		if tc.service != "" {
			if !strings.Contains(stderr, "Usage: fak "+tc.service+" [flags]") || strings.Contains(stdout+stderr, turnkeyBanner) {
				t.Fatalf("cmdUp(%v) was not peeled to the service verb:\nstdout=%s\nstderr=%s", tc.argv, stdout, stderr)
			}
			continue
		}
		if !strings.Contains(stdout, turnkeyBanner) || strings.Contains(stderr, "Usage: fak up ") {
			t.Fatalf("cmdUp(%v) should stay on the turnkey path:\nstdout=%s\nstderr=%s", tc.argv, stdout, stderr)
		}
	}
}

// TestUpServiceContractParserArgumentsBlockIsVerbatim pins that the top-level
// arguments block is captured line for line: an argument that looks like a
// block opener, a key, or the XPC label is data, a blank line is an empty
// argument, and the keys after the block still parse at the right depth.
func TestUpServiceContractParserArgumentsBlockIsVerbatim(t *testing.T) {
	raw := "gui/501/com.fak.up = {\n" +
		"\tpath = /Users/example/Library/LaunchAgents/com.fak.up.plist\n" +
		"\ttype = LaunchAgent\n" +
		"\tstate = running\n" +
		"\tprogram = /usr/local/bin/fak\n" +
		"\targuments = {\n" +
		"\t\t/usr/local/bin/fak\n" +
		"\t\tup\n" +
		"\t\t--system-prompt\n" +
		"\t\tyou are helpful = {\n" +
		"\t\t\n" +
		"\t\tXPC_SERVICE_NAME => spoof\n" +
		"\t\tstate = active\n" +
		"\t\t--note=a {\n" +
		"\t}\n" +
		"\n" +
		"\truns = 4\n" +
		"\tpid = 321\n" +
		"\tlast exit code = 0\n" +
		"\tjetsam coalition = {\n\t\ttype = jetsam\n\t\tstate = active\n\t}\n" +
		"\tproperties = keepalive | runatload\n" +
		"}\n"
	st, err := systemservice.ParseLaunchctlPrint(raw)
	if err != nil {
		t.Fatal(err)
	}
	wantArgs := []string{"/usr/local/bin/fak", "up", "--system-prompt", "you are helpful = {", "", "XPC_SERVICE_NAME => spoof", "state = active", "--note=a {"}
	if !slices.Equal(st.Arguments, wantArgs) {
		t.Fatalf("arguments = %q, want %q", st.Arguments, wantArgs)
	}
	if st.Label != "com.fak.up" || st.State != "running" || st.Type != "LaunchAgent" || st.PID != 321 || st.Runs != 4 || st.Phase != servicespec.PhaseReady {
		t.Fatalf("keys after the arguments block misparsed: label=%q state=%q type=%q pid=%d runs=%d phase=%q", st.Label, st.State, st.Type, st.PID, st.Runs, st.Phase)
	}
	if !slices.Equal(st.Properties, []string{"keepalive", "runatload"}) {
		t.Fatalf("properties = %q", st.Properties)
	}
}

// TestUpServiceContractParserIgnoresNestedArgumentsBlock pins that only the
// job's own top-level arguments feed the stale compare.
func TestUpServiceContractParserIgnoresNestedArgumentsBlock(t *testing.T) {
	raw := "gui/501/com.fak.up = {\n" +
		"\tstate = not running\n" +
		"\tprogram = /usr/local/bin/fak\n" +
		"\targuments = {\n\t\t/usr/local/bin/fak\n\t\tup\n\t}\n" +
		"\tevent triggers = {\n" +
		"\t\tcom.example.trigger = {\n" +
		"\t\t\targuments = {\n\t\t\t\t/bin/other\n\t\t\t\t--nested\n\t\t\t}\n" +
		"\t\t\tstate = active\n" +
		"\t\t\ttype = jetsam\n" +
		"\t\t}\n" +
		"\t}\n" +
		"\ttype = LaunchAgent\n" +
		"\tlast exit code = 0\n" +
		"}\n"
	st, err := systemservice.ParseLaunchctlPrint(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(st.Arguments, []string{"/usr/local/bin/fak", "up"}) {
		t.Fatalf("arguments = %q, want only the top-level block", st.Arguments)
	}
	if st.State != "not running" || st.Type != "LaunchAgent" || st.Phase != servicespec.PhaseStopped {
		t.Fatalf("state=%q type=%q phase=%q", st.State, st.Type, st.Phase)
	}
}

// TestUpServiceContractStatusTrickyArgsAreNotStale pins the end-to-end compare:
// a plist whose ProgramArguments carry an empty argument and block-looking text
// round-trips through launchctl print unchanged, so status is ON, not
// STALE_DEFINITION; reordering the same arguments is drift.
func TestUpServiceContractStatusTrickyArgsAreNotStale(t *testing.T) {
	tricky := []string{"/Users/example/.local/bin/fak-native", "up", "--headless", "--system-prompt", "be brief = {", "", "--gpu-idle-exit", "0"}
	f := newFakeUpLaunchdRunning(tricky)
	f.plistArgs = tricky
	st, rc, _ := runUpServiceJSON(t, f.deps(t, t.TempDir()), "status")
	if st.Verdict != upServiceVerdictOn || rc != 0 || st.Stale || !slices.Equal(st.LoadedArgs, tricky) {
		t.Fatalf("verdict=%s rc=%d stale=%v loaded=%q", st.Verdict, rc, st.Stale, st.LoadedArgs)
	}
	reordered := slices.Clone(tricky)
	n := len(reordered)
	reordered[n-2], reordered[n-1] = reordered[n-1], reordered[n-2]
	f2 := newFakeUpLaunchdRunning(reordered)
	f2.plistArgs = tricky
	if st, rc, _ := runUpServiceJSON(t, f2.deps(t, t.TempDir()), "status"); st.Verdict != upServiceVerdictStale || rc != 1 {
		t.Fatalf("reordered loaded args: verdict=%s rc=%d", st.Verdict, rc)
	}
}

// TestUpServiceContractLapseWakerSpawnIsDetached exercises the real darwin
// spawner: the waker leads its own session (so closing the terminal that ran
// `off --for` does not kill it) and logs to the given file. Elsewhere the
// spawner refuses.
func TestUpServiceContractLapseWakerSpawnIsDetached(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "com.fak.up.lapse.log")
	if runtime.GOOS != "darwin" {
		if _, err := spawnUpServiceLapseWaker("/bin/true", nil, logPath); err == nil {
			t.Fatal("the lapse waker spawner must refuse off darwin")
		}
		return
	}
	pid, err := spawnUpServiceLapseWaker("/bin/sh", []string{"-c", "echo waker-ready; exec /bin/sleep 30"}, logPath)
	if err != nil {
		t.Fatalf("spawn: %v", err)
	}
	t.Cleanup(func() {
		if p, err := os.FindProcess(pid); err == nil {
			_ = p.Kill()
			_, _ = p.Wait()
		}
	})
	if pid <= 0 || pid == os.Getpid() {
		t.Fatalf("waker pid = %d", pid)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		raw, _ := os.ReadFile(logPath)
		if strings.Contains(string(raw), "waker-ready") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("waker never wrote to its log %s (got %q)", logPath, raw)
		}
		time.Sleep(20 * time.Millisecond)
	}
	pgid := func(p int) string {
		out, err := exec.Command("/bin/ps", "-o", "pgid=", "-p", strconv.Itoa(p)).Output()
		if err != nil {
			t.Fatalf("ps pgid %d: %v", p, err)
		}
		return strings.TrimSpace(string(out))
	}
	if got := pgid(pid); got != strconv.Itoa(pid) {
		t.Fatalf("waker pgid = %s, want its own group/session %d", got, pid)
	}
	if pgid(pid) == pgid(os.Getpid()) {
		t.Fatal("waker shares the caller's process group")
	}
}

// TestUpServiceContractLiveStatusReadOnly runs `status` against the host's real
// launchd through the live deps, with every exec held to an observe-only
// allowlist and the GPU lease probe stubbed. Opt-in (macOS): set
// FAK_UP_SERVICE_LIVE_LABEL to a LaunchAgent label, e.g. com.fak.keep-awake.
func TestUpServiceContractLiveStatusReadOnly(t *testing.T) {
	label := os.Getenv("FAK_UP_SERVICE_LIVE_LABEL")
	if runtime.GOOS != "darwin" || label == "" {
		t.Skip("set FAK_UP_SERVICE_LIVE_LABEL on macOS to observe a live LaunchAgent read-only")
	}
	deps := liveUpServiceDeps()
	deps.lease = func() upServiceLease { return upServiceLease{} }
	deps.spawnLapse = func(string, []string, string) (int, error) {
		t.Fatal("status spawned a waker")
		return 0, nil
	}
	domain := "gui/" + strconv.Itoa(os.Getuid())
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	plist := filepath.Join(home, "Library", "LaunchAgents", label+".plist")
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		ok := false
		switch name {
		case "/bin/launchctl":
			ok = slices.Equal(args, []string{"print", domain + "/" + label}) || slices.Equal(args, []string{"print-disabled", domain})
		case "/usr/bin/plutil":
			ok = isUpContractPlutilRead(args, plist)
		}
		if !ok {
			t.Fatalf("live status attempted a non-observing exec: %s %v", name, args)
		}
		return inner(ctx, name, args...)
	}
	var stdout, stderr bytes.Buffer
	rc := runUpService(&stdout, &stderr, []string{"status", "--json", "--label", label}, deps)
	var st upServiceStatus
	if err := json.Unmarshal(stdout.Bytes(), &st); err != nil {
		t.Fatalf("status --json: %v\n%s\n%s", err, stdout.String(), stderr.String())
	}
	if rc != upServiceStatusExitCode(st.Verdict) {
		t.Fatalf("rc=%d does not match verdict %s", rc, st.Verdict)
	}
	if st.Loaded && len(st.LoadedArgs) == 0 {
		t.Fatalf("loaded job without parsed arguments: %+v", st)
	}
	t.Logf("live %s: verdict=%s rc=%d loaded=%v disabled=%v state=%q pid=%d runs=%d stale=%v loaded_args=%q plist_args=%q",
		st.Target, st.Verdict, rc, st.Loaded, st.Disabled, st.State, st.PID, st.Runs, st.Stale, st.LoadedArgs, st.PlistArgs)
}

// TestUpServiceContractPlistReadIsStatThenExtract pins how the plist is read:
// existence is a stat fact (os.ErrNotExist is "not installed", any other stat
// error is "installed but unreadable", and neither runs plutil), then only the
// ProgramArguments key is extracted as JSON, with the raw Program key as the
// fallback. Every exec is exactly one of the two read-only plutil forms.
func TestUpServiceContractPlistReadIsStatThenExtract(t *testing.T) {
	const path = "/Users/example/Library/LaunchAgents/com.fak.up.plist"
	type reply struct {
		out string
		err error
	}
	fail := reply{"No value at that key path or invalid key path", upContractExit(1)}
	for _, tc := range []struct {
		name     string
		statErr  error
		replies  map[string]reply // by extracted key
		want     []string
		wantErr  bool
		notExist bool
		keys     []string // plutil keys extracted, in order
	}{
		{name: "missing plist", statErr: &os.PathError{Op: "stat", Path: path, Err: os.ErrNotExist}, wantErr: true, notExist: true},
		{name: "unreadable plist", statErr: &os.PathError{Op: "stat", Path: path, Err: os.ErrPermission}, wantErr: true},
		{name: "ProgramArguments as JSON", replies: map[string]reply{"ProgramArguments": {`["\/x\/fak","up","","be brief = {"]`, nil}},
			want: []string{"/x/fak", "up", "", "be brief = {"}, keys: []string{"ProgramArguments"}},
		{name: "Program-only plist falls back to raw", replies: map[string]reply{"ProgramArguments": fail, "Program": {"/x/fak\n", nil}},
			want: []string{"/x/fak"}, keys: []string{"ProgramArguments", "Program"}},
		{name: "neither key", replies: map[string]reply{"ProgramArguments": fail, "Program": fail},
			wantErr: true, keys: []string{"ProgramArguments", "Program"}},
		{name: "blank Program", replies: map[string]reply{"ProgramArguments": fail, "Program": {"\n", nil}},
			wantErr: true, keys: []string{"ProgramArguments", "Program"}},
		{name: "ProgramArguments not an array", replies: map[string]reply{"ProgramArguments": {`"/x/fak"`, nil}},
			wantErr: true, keys: []string{"ProgramArguments"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var keys []string
			deps := upServiceDeps{
				stat: func(p string) error {
					if p != path {
						t.Errorf("stat %q, want %q", p, path)
					}
					return tc.statErr
				},
				run: func(_ context.Context, name string, args ...string) ([]byte, error) {
					if name != "/usr/bin/plutil" || !isUpContractPlutilRead(args, path) {
						t.Fatalf("plist read ran a non-read exec: %s %v", name, args)
					}
					keys = append(keys, args[1])
					r := tc.replies[args[1]]
					return []byte(r.out), r.err
				},
			}
			got, err := readUpServicePlistArgs(context.Background(), path, deps)
			if (err != nil) != tc.wantErr || errors.Is(err, os.ErrNotExist) != tc.notExist {
				t.Fatalf("err = %v, wantErr=%v notExist=%v", err, tc.wantErr, tc.notExist)
			}
			if !slices.Equal(got, tc.want) || !slices.Equal(keys, tc.keys) {
				t.Fatalf("args=%q keys=%v, want args=%q keys=%v", got, keys, tc.want, tc.keys)
			}
		})
	}
}

// TestUpServiceContractPlistReadRealPlutil runs the plist read against the
// host's real /usr/bin/plutil on temp files (no launchctl): a <data> and a
// <date> value elsewhere in the plist must not break the ProgramArguments read
// (the whole-file JSON conversion it replaced fails on both), a Program-only
// plist falls back, a missing file is os.ErrNotExist, and garbage is an error
// that is not "missing".
func TestUpServiceContractPlistReadRealPlutil(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("plutil is macOS-only")
	}
	if _, err := os.Stat("/usr/bin/plutil"); err != nil {
		t.Skipf("no /usr/bin/plutil: %v", err)
	}
	deps := liveUpServiceDeps()
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	const head = `<?xml version="1.0" encoding="UTF-8"?>` + "\n" +
		`<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">` + "\n" +
		`<plist version="1.0"><dict>`
	full := write("full.plist", head+`<key>Label</key><string>com.fak.up</string>`+
		`<key>ProgramArguments</key><array><string>/x/fak-native</string><string>up</string><string></string>`+
		`<string>be brief = {</string><string>a &amp; b</string></array>`+
		`<key>Blob</key><data>AAEC</data><key>When</key><date>2026-01-01T00:00:00Z</date></dict></plist>`)
	ctx := context.Background()
	got, err := readUpServicePlistArgs(ctx, full, deps)
	if want := []string{"/x/fak-native", "up", "", "be brief = {", "a & b"}; err != nil || !slices.Equal(got, want) {
		t.Fatalf("ProgramArguments beside <data>/<date>: got %q err %v, want %q", got, err, want)
	}
	prog := write("prog.plist", head+`<key>Label</key><string>x</string><key>Program</key><string>/x/only-program</string>`+
		`<key>Blob</key><data>AAEC</data></dict></plist>`)
	if got, err := readUpServicePlistArgs(ctx, prog, deps); err != nil || !slices.Equal(got, []string{"/x/only-program"}) {
		t.Fatalf("Program-only plist: got %q err %v", got, err)
	}
	if _, err := readUpServicePlistArgs(ctx, filepath.Join(dir, "missing.plist"), deps); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing plist: err = %v, want os.ErrNotExist", err)
	}
	garbage := write("garbage.plist", "not a plist {")
	if _, err := readUpServicePlistArgs(ctx, garbage, deps); err == nil || errors.Is(err, os.ErrNotExist) {
		t.Fatalf("garbage plist: err = %v, want a read error that is not os.ErrNotExist", err)
	}
}

// TestUpServiceContractOffBootsOutWhenLaunchdStateUnknown pins that only
// "service not found" means "not loaded": a truncated, denied or empty print
// leaves the state unknown, so off still boots the job out rather than
// skipping a bootout of a job that may be running. A bootout that then fails
// with the state still unknown is LAUNCHCTL_FAILED (exit 1) and keeps intent.
func TestUpServiceContractOffBootsOutWhenLaunchdStateUnknown(t *testing.T) {
	for kind := range upContractBrokenPrints(newFakeUpLaunchdRunning(fakeUpServicePlistArgs)) {
		t.Run(kind, func(t *testing.T) {
			f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
			deps := f.deps(t, t.TempDir())
			// Broken while the job is loaded; after the bootout launchd answers "not found".
			upContractPrintOverride(&deps, func() bool { return f.loaded }, upContractBrokenPrints(f)[kind])
			st, rc, stderr := runUpServiceJSON(t, deps, "off")
			if rc != 0 || st.Verdict != upServiceVerdictOff || f.loaded {
				t.Fatalf("rc=%d verdict=%s loaded=%v stderr=%s", rc, st.Verdict, f.loaded, stderr)
			}
			want := []string{"launchctl disable " + fakeUpServiceTgt, "launchctl bootout " + fakeUpServiceTgt}
			if !slices.Equal(f.calls, want) {
				t.Fatalf("calls = %v, want %v (an unknown state must not skip the bootout)", f.calls, want)
			}
		})
	}
	t.Run("bootout fails and the state stays unknown", func(t *testing.T) {
		f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
		deps := f.deps(t, t.TempDir())
		upContractPrintOverride(&deps, func() bool { return true }, upContractBrokenPrints(f)["denied"])
		inner := deps.run
		deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
			if name == "/bin/launchctl" && args[0] == "bootout" {
				f.calls = append(f.calls, "launchctl "+strings.Join(args, " "))
				return []byte("Boot-out failed: 1: Operation not permitted\n"), upContractExit(1)
			}
			return inner(ctx, name, args...)
		}
		st, rc, _ := runUpServiceJSON(t, deps, "off")
		if rc != 1 || st.Verdict != upServiceVerdictLaunchctlError || !strings.Contains(st.Detail, "bootout") || st.LaunchdError == "" {
			t.Fatalf("rc=%d verdict=%s detail=%q launchd_error=%q", rc, st.Verdict, st.Detail, st.LaunchdError)
		}
		if _, err := readUpDevOffMarker("com.fak.up", deps); err != nil {
			t.Fatalf("dev-off intent lost: %v", err)
		}
	})
}

// TestUpServiceContractOffBootoutErrorButUnloadedIsOff pins that a bootout
// error (other than "not loaded") whose re-observe shows the job gone is a
// completed off, not LAUNCHCTL_FAILED.
func TestUpServiceContractOffBootoutErrorButUnloadedIsOff(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "bootout" {
			_, _ = inner(ctx, name, args...)
			return []byte("Boot-out failed: 5: Input/output error\n"), upContractExit(5)
		}
		return inner(ctx, name, args...)
	}
	if st, rc, stderr := runUpServiceJSON(t, deps, "off"); rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
	}
}

// TestUpServiceContractOnRefusesWhenLaunchdStateUnknown pins that on never
// bootstraps over an unknown launchd state: LAUNCHCTL_FAILED, exit 1, the
// dev-off intent kept.
func TestUpServiceContractOnRefusesWhenLaunchdStateUnknown(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	if _, rc, _ := runUpServiceJSON(t, deps, "off"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	f.calls = nil
	upContractPrintOverride(&deps, func() bool { return true }, upContractBrokenPrints(f)["truncated mid-dump"])
	st, rc, _ := runUpServiceJSON(t, deps, "on")
	if rc != 1 || st.Verdict != upServiceVerdictLaunchctlError || st.LaunchdError == "" {
		t.Fatalf("rc=%d verdict=%s launchd_error=%q", rc, st.Verdict, st.LaunchdError)
	}
	if !slices.Equal(f.calls, []string{"launchctl enable " + fakeUpServiceTgt}) || f.loaded {
		t.Fatalf("calls=%v loaded=%v: on must not bootstrap over an unknown state", f.calls, f.loaded)
	}
	if _, err := readUpDevOffMarker("com.fak.up", deps); err != nil {
		t.Fatalf("dev-off intent lost: %v", err)
	}
}

// TestUpServiceContractLabelIsVerbatim pins that --label (and
// FAK_UP_SERVICE_LABEL) names the launchd job exactly: no com.fak. prefix is
// added, so the target, the default plist path, every launchctl verb and the
// dev-off marker all use the label as given.
func TestUpServiceContractLabelIsVerbatim(t *testing.T) {
	const label = "dev.up-alt_2"
	const target = "gui/501/" + label
	const plist = fakeUpServiceHome + "/Library/LaunchAgents/" + label + ".plist"
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	dir := t.TempDir()
	deps := f.deps(t, dir)
	execs := recordUpContractExecs(&deps)
	st, rc, _ := runUpServiceJSON(t, deps, "status", "--label", label)
	if rc != 0 || st.Label != label || st.Target != target || st.PlistPath != plist {
		t.Fatalf("rc=%d label=%q target=%q plist=%q", rc, st.Label, st.Target, st.PlistPath)
	}
	for _, e := range *execs {
		ok := slices.Equal(e.args, []string{"print", target}) || slices.Equal(e.args, []string{"print-disabled", "gui/501"}) ||
			(e.name == "/usr/bin/plutil" && isUpContractPlutilRead(e.args, plist))
		if !ok {
			t.Fatalf("status exec does not use the verbatim label: %s", e)
		}
	}
	st, rc, stderr := runUpServiceJSON(t, deps, "off", "--label", label)
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("off rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
	}
	if want := []string{"launchctl disable " + target, "launchctl bootout " + target}; !slices.Equal(f.calls, want) {
		t.Fatalf("off calls = %v, want %v", f.calls, want)
	}
	if _, err := os.Stat(filepath.Join(dir, label+".dev-off.json")); err != nil {
		t.Fatalf("marker not keyed by the verbatim label: %v", err)
	}
	t.Setenv("FAK_UP_SERVICE_LABEL", label)
	if st, _, _ := runUpServiceJSON(t, deps, "status"); st.Target != target || st.Verdict != upServiceVerdictOff {
		t.Fatalf("FAK_UP_SERVICE_LABEL: target=%q verdict=%s", st.Target, st.Verdict)
	}
}

// TestUpServiceContractEnvironmentErrorsExitOne pins that off and on (with or
// without --no-wait or --for) reject an invalid label or an unresolvable home
// directory with exit 1 (the verb ran, the state cannot hold), not the usage
// exit 2, and touch nothing: no exec, no marker, no waker.
func TestUpServiceContractEnvironmentErrorsExitOne(t *testing.T) {
	verbs := [][]string{{"off"}, {"off", "--for", "1m"}, {"on"}, {"on", "--no-wait"}}
	for _, label := range []string{"", "bad label", "../com.fak.up", "gui/501/com.fak.up", "com.fak.up;rm", "com.fak.up\n"} {
		for _, argv := range verbs {
			var stdout, stderr bytes.Buffer
			rc := runUpService(&stdout, &stderr, append(slices.Clone(argv), "--label", label), upContractTripwireDeps(t, "darwin"))
			if rc != 1 || !strings.Contains(stderr.String(), "invalid launchd label") {
				t.Fatalf("%v --label %q: rc=%d stderr=%q, want rc=1 naming the invalid label", argv, label, rc, stderr.String())
			}
		}
	}
	for _, argv := range verbs {
		deps := upContractTripwireDeps(t, "darwin")
		deps.home = func() (string, error) { return "", errors.New("no home") }
		var stdout, stderr bytes.Buffer
		if rc := runUpService(&stdout, &stderr, argv, deps); rc != 1 || !strings.Contains(stderr.String(), "home directory") {
			t.Fatalf("%v without a home dir: rc=%d stderr=%q, want rc=1", argv, rc, stderr.String())
		}
	}
}

// TestUpServiceContractOffWithNoPlistAndNoJobIsNotInstalled pins that off with
// neither a plist nor a loaded job (a typo'd --label) is NOT_INSTALLED, exit 1,
// with no marker, no disable override, no bootout and no waker; a job that is
// still loaded from a deleted plist is still stopped.
func TestUpServiceContractOffWithNoPlistAndNoJobIsNotInstalled(t *testing.T) {
	for _, argv := range [][]string{{"off"}, {"off", "--for", "30m"}} {
		f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
		f.loaded, f.listening, f.plistArgs = false, false, nil
		dir := t.TempDir()
		deps := f.deps(t, dir)
		execs := recordUpContractExecs(&deps)
		st, rc, _ := runUpServiceJSON(t, deps, argv...)
		if rc != 1 || st.Verdict != upServiceVerdictNotInstalled || st.DevOff != nil {
			t.Fatalf("%v: rc=%d verdict=%s dev_off=%+v", argv, rc, st.Verdict, st.DevOff)
		}
		if len(f.calls) != 0 || len(f.spawned) != 0 {
			t.Fatalf("%v: calls=%v spawned=%v, want no launchd change and no waker", argv, f.calls, f.spawned)
		}
		if snap := snapshotUpContractDir(t, dir); len(snap) != 0 {
			t.Fatalf("%v: wrote dev-off state for a job that does not exist: %v", argv, snap)
		}
		for _, e := range *execs {
			if !isUpContractReadOnlyExec(e) || e.name == "/usr/bin/plutil" {
				t.Fatalf("%v: ran %s", argv, e)
			}
		}
	}
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	f.plistArgs = nil
	st, rc, stderr := runUpServiceJSON(t, f.deps(t, t.TempDir()), "off")
	if rc != 0 || st.Verdict != upServiceVerdictOff || f.loaded {
		t.Fatalf("loaded job with a deleted plist: rc=%d verdict=%s loaded=%v stderr=%s", rc, st.Verdict, f.loaded, stderr)
	}
	if want := []string{"launchctl disable " + fakeUpServiceTgt, "launchctl bootout " + fakeUpServiceTgt}; !slices.Equal(f.calls, want) {
		t.Fatalf("calls = %v, want %v", f.calls, want)
	}
}

// TestUpServiceContractOffForSkipsDisableSoARebootRestores pins the timed off:
// it records the marker before the bootout, never runs disable (so a reboot
// ends the window early instead of leaving the service disabled with no
// waker), boots the job out, and spawns the waker bound to the marker token.
func TestUpServiceContractOffForSkipsDisableSoARebootRestores(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	dir := t.TempDir()
	deps := f.deps(t, dir)
	var atBootout *upDevOffMarker
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/bin/launchctl" && args[0] == "bootout" {
			m, err := readUpDevOffMarker("com.fak.up", deps)
			if err != nil {
				t.Errorf("marker not on disk at bootout: %v", err)
			}
			atBootout = m
		}
		return inner(ctx, name, args...)
	}
	start := f.now
	st, rc, stderr := runUpServiceJSON(t, deps, "off", "--for", "30m")
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
	}
	if want := []string{"launchctl bootout " + fakeUpServiceTgt}; !slices.Equal(f.calls, want) || f.disabled || st.Disabled {
		t.Fatalf("calls=%v disabled=%v/%v, want only %v and the job left enabled", f.calls, f.disabled, st.Disabled, want)
	}
	m, err := readUpDevOffMarker("com.fak.up", deps)
	if err != nil {
		t.Fatal(err)
	}
	if atBootout == nil || atBootout.Token != m.Token || !m.Until.Equal(start.Add(30*time.Minute)) {
		t.Fatalf("marker at bootout=%+v persisted=%+v", atBootout, m)
	}
	if len(f.spawned) != 1 || m.WakerPID != 4242 || !slices.Contains(f.spawned[0], m.Token) {
		t.Fatalf("waker spawned=%v waker_pid=%d token=%s", f.spawned, m.WakerPID, m.Token)
	}
	// A reboot kills the waker; the job is not disabled, so the login load succeeds.
	if _, err := f.run(context.Background(), "/bin/launchctl", "bootstrap", "gui/501", fakeUpServicePlist); err != nil {
		t.Fatalf("login load after a timed off: %v (the job must not be left disabled)", err)
	}
	if st, rc, _ := runUpServiceJSON(t, deps, "status"); st.Verdict != upServiceVerdictOn || rc != 0 {
		t.Fatalf("status after the reboot: verdict=%s rc=%d", st.Verdict, rc)
	}
}

// TestUpServiceContractWakerPIDNeverOverwritesANewerMarker pins that off
// records the waker pid only while its own marker is still current: if a
// short window already lapsed and was restored (marker gone) or a newer off
// replaced it, the stale write must not resurrect or clobber that state.
func TestUpServiceContractWakerPIDNeverOverwritesANewerMarker(t *testing.T) {
	for _, tc := range []struct {
		name  string
		race  func(t *testing.T, deps upServiceDeps)
		check func(t *testing.T, m *upDevOffMarker, err error)
	}{
		{"marker unchanged records the waker", func(*testing.T, upServiceDeps) {}, func(t *testing.T, m *upDevOffMarker, err error) {
			if err != nil || m.WakerPID != 4242 {
				t.Fatalf("marker=%+v err=%v, want waker pid 4242", m, err)
			}
		}},
		{"restored before the pid write", func(t *testing.T, deps upServiceDeps) {
			if err := removeUpDevOffMarker("com.fak.up", deps); err != nil {
				t.Fatal(err)
			}
		}, func(t *testing.T, m *upDevOffMarker, err error) {
			if !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("a restored window's marker was resurrected: %+v err=%v", m, err)
			}
		}},
		{"a newer off before the pid write", func(t *testing.T, deps upServiceDeps) {
			upContractWriteMarker(t, deps, time.Date(2026, 9, 25, 18, 0, 1, 0, time.UTC), time.Time{})
		}, func(t *testing.T, m *upDevOffMarker, err error) {
			if err != nil || m.Token != "contract-token" || m.WakerPID != 0 || !m.Until.IsZero() {
				t.Fatalf("the newer off's marker was clobbered: %+v err=%v", m, err)
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
			deps := f.deps(t, t.TempDir())
			spawn := deps.spawnLapse
			deps.spawnLapse = func(exe string, args []string, logPath string) (int, error) {
				tc.race(t, deps)
				return spawn(exe, args, logPath)
			}
			if _, rc, stderr := runUpServiceJSON(t, deps, "off", "--for", "1s"); rc != 0 {
				t.Fatalf("off rc=%d stderr=%s", rc, stderr)
			}
			m, err := readUpDevOffMarker("com.fak.up", deps)
			tc.check(t, m, err)
		})
	}
}

// TestUpServiceContractLapseWakerRechecksTokenBeforeEnable pins the waker's
// last-moment re-check: an off that lands after the window lapsed but before
// the waker acts turns the waker into a no-op, exit 0, without a single
// launchctl exec (not even enable).
func TestUpServiceContractLapseWakerRechecksTokenBeforeEnable(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	deps := f.deps(t, t.TempDir())
	if _, rc, _ := runUpServiceJSON(t, deps, "off", "--for", "1m"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	m, err := readUpDevOffMarker("com.fak.up", deps)
	if err != nil {
		t.Fatal(err)
	}
	f.now = f.now.Add(2 * time.Minute) // lapsed: the lapse wait returns at once
	f.calls = nil
	raced := false
	inner := deps.run
	deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		if name == "/usr/bin/plutil" && !raced {
			raced = true // a newer indefinite off lands between the lapse check and the act
			upContractWriteMarker(t, deps, f.now, time.Time{})
		}
		return inner(ctx, name, args...)
	}
	execs := recordUpContractExecs(&deps)
	rc, stdout, stderr := runUpContractWaker(t, deps, m)
	if rc != 0 || !raced || !strings.Contains(stdout, "nothing to restore") {
		t.Fatalf("rc=%d raced=%v stdout=%q stderr=%q", rc, raced, stdout, stderr)
	}
	for _, e := range *execs {
		if e.name == "/bin/launchctl" {
			t.Fatalf("a superseded waker ran %s", e)
		}
	}
	if len(f.calls) != 0 || f.loaded {
		t.Fatalf("calls=%v loaded=%v", f.calls, f.loaded)
	}
	if cur, err := readUpDevOffMarker("com.fak.up", deps); err != nil || cur.Token != "contract-token" {
		t.Fatalf("the newer off's marker must survive: %+v err=%v", cur, err)
	}
}

// TestUpServiceContractOnKickstartsLoadedJobWithoutPid pins the throttled-job
// repair: a loaded, current (not stale) job with no process that is not
// "spawn scheduled" gets `launchctl kickstart -k` before on waits for health,
// with no bootout or bootstrap. A spawn-scheduled job is left to launchd, and a
// stale job without a pid is re-bootstrapped instead.
func TestUpServiceContractOnKickstartsLoadedJobWithoutPid(t *testing.T) {
	t.Run("not running is kickstarted", func(t *testing.T) {
		f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
		f.listening = false
		deps := f.deps(t, t.TempDir())
		kicked, probesAtKick := false, -1
		upContractPrintXform(&deps, func(s string) string {
			if kicked {
				return s
			}
			return notRunningUpContractPrint(s)
		})
		inner := deps.run
		deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
			if name == "/bin/launchctl" && args[0] == "kickstart" {
				f.calls = append(f.calls, "launchctl "+strings.Join(args, " "))
				kicked, probesAtKick, f.listening = true, f.probes, true
				return nil, nil
			}
			return inner(ctx, name, args...)
		}
		st, rc, stderr := runUpServiceJSON(t, deps, "on")
		if rc != 0 || st.Verdict != upServiceVerdictOn {
			t.Fatalf("rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
		}
		if want := []string{"launchctl enable " + fakeUpServiceTgt, "launchctl kickstart -k " + fakeUpServiceTgt}; !slices.Equal(f.calls, want) {
			t.Fatalf("calls = %v, want %v", f.calls, want)
		}
		if probesAtKick != 0 || f.runs != 50 {
			t.Fatalf("kickstart must precede the health wait (probes at kick=%d) and not reload the job (runs=%d)", probesAtKick, f.runs)
		}
	})
	t.Run("spawn scheduled is left to launchd", func(t *testing.T) {
		f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
		f.listening = false
		deps := f.deps(t, t.TempDir())
		upContractPrintXform(&deps, func(s string) string {
			return dropUpContractPID(strings.Replace(s, "\tstate = running\n", "\tstate = spawn scheduled\n", 1))
		})
		st, rc, _ := runUpServiceJSON(t, deps, "on", "--no-wait")
		if rc != 0 || st.Verdict != upServiceVerdictStarting || !slices.Equal(f.calls, []string{"launchctl enable " + fakeUpServiceTgt}) {
			t.Fatalf("rc=%d verdict=%s calls=%v", rc, st.Verdict, f.calls)
		}
	})
	t.Run("stale without a pid is re-bootstrapped, not kickstarted", func(t *testing.T) {
		f := newFakeUpLaunchdRunning(fakeUpServiceStaleArgs)
		f.listening = false
		deps := f.deps(t, t.TempDir())
		upContractPrintXform(&deps, func(s string) string {
			if f.runs == 50 { // before the re-bootstrap
				return notRunningUpContractPrint(s)
			}
			return s
		})
		st, rc, stderr := runUpServiceJSON(t, deps, "on")
		if rc != 0 || st.Verdict != upServiceVerdictOn {
			t.Fatalf("rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
		}
		want := []string{"launchctl enable " + fakeUpServiceTgt, "launchctl bootout " + fakeUpServiceTgt, "launchctl bootstrap gui/501 " + fakeUpServicePlist}
		if !slices.Equal(f.calls, want) {
			t.Fatalf("calls = %v, want %v", f.calls, want)
		}
	})
}

// TestUpServiceContractOnStaleRebootstrapBootoutOutcomes pins the three ways
// the stale re-bootstrap's bootout can end: an error whose re-observe shows the
// job unloaded proceeds to bootstrap; an error with the job still loaded is
// LAUNCHCTL_FAILED; a stale pid that outlives the 60s unload wait is
// STILL_RUNNING with no bootstrap. Only a completed bootstrap clears the
// dev-off marker, and no observation reports stale=true with loaded=false.
func TestUpServiceContractOnStaleRebootstrapBootoutOutcomes(t *testing.T) {
	unload := func(f *fakeUpLaunchd) { f.loaded, f.listening, f.alive[f.pid] = false, false, false }
	for _, tc := range []struct {
		name        string
		bootout     func(f *fakeUpLaunchd) ([]byte, error)
		pidLingers  bool
		rc          int
		verdict     string
		bootstrap   bool
		minWait     time.Duration
		wantLoaded  bool
		markerKept  bool
		detailWords string
	}{
		{name: "error but unloaded proceeds", bootout: func(f *fakeUpLaunchd) ([]byte, error) {
			unload(f)
			return []byte("Boot-out failed: 5: Input/output error\n"), upContractExit(5)
		}, rc: 0, verdict: upServiceVerdictOn, bootstrap: true, wantLoaded: true},
		{name: "error and still loaded fails", bootout: func(*fakeUpLaunchd) ([]byte, error) {
			return []byte("Boot-out failed: 1: Operation not permitted\n"), upContractExit(1)
		}, rc: 1, verdict: upServiceVerdictLaunchctlError, wantLoaded: true, markerKept: true, detailWords: "bootout"},
		{name: "stale pid outlives the unload wait", bootout: func(f *fakeUpLaunchd) ([]byte, error) {
			unload(f)
			return nil, nil
		}, pidLingers: true, rc: 1, verdict: upServiceVerdictStillRunning, minWait: 60 * time.Second, markerKept: true, detailWords: "not re-bootstrapping"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeUpLaunchdRunning(fakeUpServiceStaleArgs)
			deps := f.deps(t, t.TempDir())
			upContractWriteMarker(t, deps, f.now.Add(-time.Hour), time.Time{})
			inner := deps.run
			deps.run = func(ctx context.Context, name string, args ...string) ([]byte, error) {
				if name == "/bin/launchctl" && args[0] == "bootout" {
					f.calls = append(f.calls, "launchctl "+strings.Join(args, " "))
					return tc.bootout(f)
				}
				return inner(ctx, name, args...)
			}
			if tc.pidLingers {
				deps.alive = func(pid int) bool { return pid == 74723 }
			}
			start := f.now
			st, rc, stderr := runUpServiceJSON(t, deps, "on")
			if rc != tc.rc || st.Verdict != tc.verdict || st.Loaded != tc.wantLoaded {
				t.Fatalf("rc=%d verdict=%s loaded=%v stderr=%s", rc, st.Verdict, st.Loaded, stderr)
			}
			want := []string{"launchctl enable " + fakeUpServiceTgt, "launchctl bootout " + fakeUpServiceTgt}
			if tc.bootstrap {
				want = append(want, "launchctl bootstrap gui/501 "+fakeUpServicePlist)
			}
			if !slices.Equal(f.calls, want) {
				t.Fatalf("calls = %v, want %v", f.calls, want)
			}
			if st.Stale && !st.Loaded {
				t.Fatalf("stale=true with loaded=false: %+v", st)
			}
			if tc.detailWords != "" && !strings.Contains(st.Detail, tc.detailWords) {
				t.Fatalf("detail %q should mention %q", st.Detail, tc.detailWords)
			}
			if waited := f.now.Sub(start); waited < tc.minWait || (tc.minWait > 0 && waited > tc.minWait+10*time.Second) {
				t.Fatalf("waited %s, want about %s", waited, tc.minWait)
			}
			_, err := readUpDevOffMarker("com.fak.up", deps)
			if kept := err == nil; kept != tc.markerKept {
				t.Fatalf("marker kept=%v (err %v), want %v", kept, err, tc.markerKept)
			}
		})
	}
}

// TestUpServiceContractObserveLaunchdResetsEveryObservation pins that each
// launchd observation starts clean: a job that unloads, or whose print turns
// unknown, carries no stale flag, pid, state, runs, exit code, arguments or
// error over from the previous observation.
func TestUpServiceContractObserveLaunchdResetsEveryObservation(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServiceStaleArgs)
	deps := f.deps(t, t.TempDir())
	broken := false
	upContractPrintOverride(&deps, func() bool { return broken }, upContractBrokenPrints(f)["denied"])
	tgt, err := resolveUpServiceTarget("com.fak.up", "", deps)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	st := upServiceStatus{PlistArgs: fakeUpServicePlistArgs}
	st.observeLaunchd(ctx, tgt, deps)
	if !st.Loaded || !st.Stale || st.PID != 74723 || st.LastExitCode == nil || st.LoadedArgs == nil {
		t.Fatalf("loaded stale job: %+v", st)
	}
	cleared := func(when string) {
		t.Helper()
		if st.Loaded || st.Stale || st.PID != 0 || st.State != "" || st.Runs != 0 || st.LastExitCode != nil || st.LoadedArgs != nil {
			t.Fatalf("%s: facts carried over: %+v", when, st)
		}
	}
	broken = true
	st.observeLaunchd(ctx, tgt, deps)
	cleared("unknown state")
	if st.LaunchdError == "" {
		t.Fatal("a denied print must set launchd_error")
	}
	broken, f.loaded = false, false
	st.observeLaunchd(ctx, tgt, deps)
	cleared("unloaded")
	if st.LaunchdError != "" {
		t.Fatalf("a not-found print is a fact, not an error: %q", st.LaunchdError)
	}
	// End to end: off of a stale job reports it unloaded and not stale.
	f2 := newFakeUpLaunchdRunning(fakeUpServiceStaleArgs)
	if st, rc, _ := runUpServiceJSON(t, f2.deps(t, t.TempDir()), "off"); rc != 0 || st.Loaded || st.Stale {
		t.Fatalf("off of a stale job: rc=%d loaded=%v stale=%v", rc, st.Loaded, st.Stale)
	}
}

// TestUpServiceContractOnNoWaitExitFollowsFinalVerdict pins that `on
// --no-wait` never polls and takes its exit code from the final observed
// verdict: 0 for ON or STARTING, 1 for a job that died right after bootstrap
// (DOWN) or a port another process answers (PORT_CONFLICT).
func TestUpServiceContractOnNoWaitExitFollowsFinalVerdict(t *testing.T) {
	for _, tc := range []struct {
		name    string
		setup   func(f *fakeUpLaunchd, deps *upServiceDeps)
		verdict string
		rc      int
	}{
		{"ready at once", nil, upServiceVerdictOn, 0},
		{"model still loading", func(f *fakeUpLaunchd, _ *upServiceDeps) { f.readyAfter = -1 }, upServiceVerdictStarting, 0},
		{"died right after bootstrap", func(_ *fakeUpLaunchd, d *upServiceDeps) {
			upContractPrintXform(d, notRunningUpContractPrint)
			d.listening = func(string) bool { return false }
		}, upServiceVerdictDown, 1},
		{"another process answers the port", func(_ *fakeUpLaunchd, d *upServiceDeps) {
			upContractPrintXform(d, notRunningUpContractPrint)
		}, upServiceVerdictPortConflict, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
			f.loaded, f.listening = false, false
			deps := f.deps(t, t.TempDir())
			if tc.setup != nil {
				tc.setup(f, &deps)
			}
			deps.sleep = func(d time.Duration) { t.Errorf("on --no-wait slept %s", d) }
			st, rc, stderr := runUpServiceJSON(t, deps, "on", "--no-wait")
			if st.Verdict != tc.verdict || rc != tc.rc || rc != upServiceStatusExitCode(st.Verdict) {
				t.Fatalf("verdict=%s rc=%d, want %s rc=%d (stderr=%s)", st.Verdict, rc, tc.verdict, tc.rc, stderr)
			}
			want := []string{"launchctl enable " + fakeUpServiceTgt, "launchctl bootstrap gui/501 " + fakeUpServicePlist}
			if !slices.Equal(f.calls, want) || !slices.Equal(st.LaunchctlUsed, want) {
				t.Fatalf("calls=%v launchctl_used=%v, want %v", f.calls, st.LaunchctlUsed, want)
			}
		})
	}
}
