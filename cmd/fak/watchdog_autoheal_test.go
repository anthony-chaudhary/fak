package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testWatchdogAutohealOptions(dir string, now *time.Time, spec watchdogAutohealSpec) watchdogAutohealOptions {
	return watchdogAutohealOptions{
		Verb:     "guard",
		Mode:     watchdogAutohealOn,
		Specs:    []watchdogAutohealSpec{spec},
		StateDir: dir,
		Clock: func() time.Time {
			return *now
		},
		Sleep: func(d time.Duration) {
			*now = now.Add(d)
		},
		LeaseTTL: 30 * time.Second,
		Debounce: 10 * time.Minute,
		RestartPolicy: watchdogRestartPolicy{
			MaxAttempts: 2,
			BaseDelay:   time.Second,
			MaxDelay:    2 * time.Second,
		},
	}
}

func deadInstalledProbe(context.Context) (watchdogProbe, error) {
	return watchdogProbe{Installed: true, Alive: false, Detail: "dead"}, nil
}

func TestWatchdogHealSingleFlightDedupesConcurrentStarts(t *testing.T) {
	now := time.Unix(1000, 0).UTC()
	dir := t.TempDir()
	restarts := 0
	var nested []watchdogAutohealResult
	var opts watchdogAutohealOptions
	spec := watchdogAutohealSpec{
		watchdogService: watchdogService{ID: "fleet-dos-dispatch-watchdog", Manager: "systemd", Unit: "fleet-dos-dispatch-watchdog.timer"},
		Probe:           deadInstalledProbe,
		Restart: func(ctx context.Context) error {
			restarts++
			if restarts == 1 {
				nested = runWatchdogAutoheal(ctx, opts)
			}
			return nil
		},
	}
	opts = testWatchdogAutohealOptions(dir, &now, spec)

	got := runWatchdogAutoheal(context.Background(), opts)
	if restarts != 1 {
		t.Fatalf("restart calls = %d, want 1", restarts)
	}
	if len(got) != 1 || got[0].Action != "restarted" || got[0].Reason != watchdogReasonRestarted {
		t.Fatalf("outer heal = %+v, want restarted", got)
	}
	if len(nested) != 1 || nested[0].Action != "in_flight" || nested[0].Reason != watchdogReasonLeaseHeld {
		t.Fatalf("nested heal = %+v, want in_flight/%s", nested, watchdogReasonLeaseHeld)
	}
}

func TestWatchdogHealDebouncesRecentRestart(t *testing.T) {
	now := time.Unix(2000, 0).UTC()
	dir := t.TempDir()
	if err := writeWatchdogHealState(dir, watchdogHealState{
		Schema:              watchdogAutohealSchema,
		ID:                  "resume",
		LastRestartUnixNano: now.Add(-time.Minute).UnixNano(),
	}); err != nil {
		t.Fatalf("seed state: %v", err)
	}
	restarts := 0
	spec := watchdogAutohealSpec{
		watchdogService: watchdogService{ID: "resume", Manager: "taskscheduler", Unit: "FleetResumeWatchdog"},
		Probe:           deadInstalledProbe,
		Restart: func(context.Context) error {
			restarts++
			return nil
		},
	}
	opts := testWatchdogAutohealOptions(dir, &now, spec)

	got := runWatchdogAutoheal(context.Background(), opts)
	if restarts != 0 {
		t.Fatalf("restart calls = %d, want 0", restarts)
	}
	if len(got) != 1 || got[0].Action != "debounced" || got[0].Reason != watchdogReasonDebounced {
		t.Fatalf("heal = %+v, want debounced/%s", got, watchdogReasonDebounced)
	}
}

func TestWatchdogHealBoundedRetryGivesUp(t *testing.T) {
	now := time.Unix(3000, 0).UTC()
	dir := t.TempDir()
	attempts := 0
	spec := watchdogAutohealSpec{
		watchdogService: watchdogService{ID: "supervisor", Manager: "taskscheduler", Unit: "FleetSupervisorWatchdog"},
		Probe:           deadInstalledProbe,
		Restart: func(context.Context) error {
			attempts++
			return errors.New("boom")
		},
	}
	opts := testWatchdogAutohealOptions(dir, &now, spec)

	got := runWatchdogAutoheal(context.Background(), opts)
	if attempts != 2 {
		t.Fatalf("restart attempts = %d, want 2", attempts)
	}
	if len(got) != 1 || got[0].Action != "give_up" || got[0].Reason != watchdogReasonExhausted {
		t.Fatalf("heal = %+v, want give_up/%s", got, watchdogReasonExhausted)
	}
	st, err := readWatchdogHealState(dir, "supervisor")
	if err != nil {
		t.Fatalf("read state: %v", err)
	}
	if st.Attempts != 2 || st.LastReason != watchdogReasonExhausted {
		t.Fatalf("state = %+v, want attempts=2 reason=%s", st, watchdogReasonExhausted)
	}
}

func TestWatchdogAutohealWarnAndOffModes(t *testing.T) {
	now := time.Unix(4000, 0).UTC()
	dir := t.TempDir()
	restarts := 0
	spec := watchdogAutohealSpec{
		watchdogService: watchdogService{ID: "resume", Manager: "taskscheduler", Unit: "FleetResumeWatchdog"},
		Probe:           deadInstalledProbe,
		Restart: func(context.Context) error {
			restarts++
			return nil
		},
	}
	opts := testWatchdogAutohealOptions(dir, &now, spec)
	opts.Mode = watchdogAutohealWarn

	got := runWatchdogAutoheal(context.Background(), opts)
	if restarts != 0 {
		t.Fatalf("warn mode restart calls = %d, want 0", restarts)
	}
	if len(got) != 1 || got[0].Action != "warn" || got[0].Reason != watchdogReasonWarnOnly {
		t.Fatalf("warn heal = %+v, want warn/%s", got, watchdogReasonWarnOnly)
	}
	opts.Mode = watchdogAutohealOff
	if got := runWatchdogAutoheal(context.Background(), opts); len(got) != 0 {
		t.Fatalf("off mode returned %+v, want no results", got)
	}
}

func TestWatchdogAutohealPlatformProjection(t *testing.T) {
	win := watchdogAutohealServicesForGOOS("windows")
	if !serviceProjectionHas(win, "taskscheduler", "FleetResumeWatchdog") ||
		!serviceProjectionHas(win, "taskscheduler", "FleetSupervisorWatchdog") ||
		!serviceProjectionHas(win, "taskscheduler", "FleetDOSDispatchWatchdog") {
		t.Fatalf("windows projection missing expected Scheduled Tasks: %+v", win)
	}
	if !serviceProjectionHas(win, "taskscheduler", "FleetStaleWorkGarden") {
		t.Fatalf("windows projection missing stale-work garden task: %+v", win)
	}

	darwin := watchdogAutohealServicesForGOOS("darwin")
	if !serviceProjectionHas(darwin, "launchd", "com.fleet.dispatch-supervisor") {
		t.Fatalf("darwin projection missing launchd dispatch supervisor: %+v", darwin)
	}
	if !serviceProjectionHas(darwin, "launchd", "com.fleet.stale-work-garden") {
		t.Fatalf("darwin projection missing launchd stale-work garden: %+v", darwin)
	}
	for _, svc := range darwin {
		if svc.Manager == "launchd" && !strings.Contains(filepath.ToSlash(svc.UnitPath), "LaunchAgents/") {
			t.Fatalf("darwin unit path should target LaunchAgents, got %+v", svc)
		}
	}

	linux := watchdogAutohealServicesForGOOS("linux")
	if !serviceProjectionHas(linux, "systemd", "fleet-dos-dispatch-watchdog.timer") {
		t.Fatalf("linux projection missing systemd dispatch watchdog timer: %+v", linux)
	}
	if !serviceProjectionHas(linux, "systemd", "fleet-stale-work-garden.timer") {
		t.Fatalf("linux projection missing systemd stale-work garden timer: %+v", linux)
	}
}

// TestWatchdogAutohealToSharedStderr pins the sink decision: an attended interactive `fak
// guard` launch must NOT stream the heal JSON to the shared terminal (the agent's alt-screen
// TUI owns it — the bug in the report), while serve and every headless / piped / redirected
// case keep stderr so a captured log stays whole.
func TestWatchdogAutohealToSharedStderr(t *testing.T) {
	cases := []struct {
		name             string
		verb             string
		stderrIsTerminal bool
		childInteractive bool
		wantShared       bool
	}{
		{"guard interactive terminal suppresses (the bug)", "guard", true, true, false},
		{"guard headless child keeps stderr", "guard", true, false, true},
		{"guard redirected stderr keeps stderr", "guard", false, true, true},
		{"serve always keeps stderr", "serve", true, true, true},
		{"serve redirected keeps stderr", "serve", false, false, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := watchdogAutohealToSharedStderr(tc.verb, tc.stderrIsTerminal, tc.childInteractive); got != tc.wantShared {
				t.Fatalf("watchdogAutohealToSharedStderr(%q, %v, %v) = %v, want %v", tc.verb, tc.stderrIsTerminal, tc.childInteractive, got, tc.wantShared)
			}
		})
	}
}

// TestWatchdogAutohealKeepsAgentPaneClean is the render/capture witness for the fix: when guard
// has handed the terminal to an interactive agent, the routing core sends the heal lines to
// autoheal.log under the state dir and ZERO bytes reach the shared terminal. It captures both
// surfaces — the would-be agent pane (a writer standing in for the terminal, which must stay
// empty) and the file (which must hold the full JSON record) — so the proof is the ABSENCE of
// any agent-pane corruption, exactly the `… for agents` fragment in the bug report.
func TestWatchdogAutohealKeepsAgentPaneClean(t *testing.T) {
	dir := t.TempDir()
	results := []watchdogAutohealResult{{
		Schema:  watchdogAutohealSchema,
		Verb:    "guard",
		ID:      "fleet-supervisor-watchdog",
		Manager: "taskscheduler",
		Unit:    "FleetSupervisorWatchdog",
		Action:  "give_up",
		Reason:  watchdogReasonExhausted,
		Summary: "restart attempts exhausted (3/3)",
		Attempt: 3,
	}}

	// agentPane stands in for the terminal the interactive agent owns: a single byte written
	// here is the corruption. Drive the routing core as an interactive guard launch (stderr is
	// a terminal, child is interactive) and prove nothing lands on it.
	var agentPane strings.Builder
	w, closeSink := watchdogAutohealSinkFor("guard", dir, &agentPane, true /*stderrIsTerminal*/, true /*childInteractive*/)
	logWatchdogAutohealResults(w, results)
	closeSink()

	if agentPane.Len() != 0 {
		t.Fatalf("agent pane received %d bytes, want 0 (TUI corruption): %q", agentPane.Len(), agentPane.String())
	}

	logged, err := os.ReadFile(filepath.Join(dir, "autoheal.log"))
	if err != nil {
		t.Fatalf("read autoheal.log: %v", err)
	}
	got := string(logged)
	for _, want := range []string{"fak watchdog-autoheal:", watchdogReasonExhausted, "fleet-supervisor-watchdog"} {
		if !strings.Contains(got, want) {
			t.Fatalf("autoheal.log missing %q; got:\n%s", want, got)
		}
	}

	// The mirror image: a headless guard (non-terminal stderr) keeps the captured-log contract —
	// the JSON streams to the supplied stderr and NO file is created.
	headlessDir := t.TempDir()
	var headlessStderr strings.Builder
	hw, closeHeadless := watchdogAutohealSinkFor("guard", headlessDir, &headlessStderr, false /*stderrIsTerminal*/, true)
	logWatchdogAutohealResults(hw, results)
	closeHeadless()
	if !strings.Contains(headlessStderr.String(), "fak watchdog-autoheal:") {
		t.Fatalf("headless guard: stderr missing heal line; got:\n%s", headlessStderr.String())
	}
	if _, err := os.Stat(filepath.Join(headlessDir, "autoheal.log")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("headless guard wrote an autoheal.log (stat err=%v); want stderr only", err)
	}

	// Appends (a second heal in the same interactive session) accumulate rather than truncate.
	w2, close2 := watchdogAutohealSinkFor("guard", dir, &agentPane, true, true)
	logWatchdogAutohealResults(w2, results)
	close2()
	again, err := os.ReadFile(filepath.Join(dir, "autoheal.log"))
	if err != nil {
		t.Fatalf("re-read autoheal.log: %v", err)
	}
	if n := strings.Count(string(again), "fak watchdog-autoheal:"); n != 2 {
		t.Fatalf("autoheal.log heal lines = %d, want 2 (append, not truncate)", n)
	}
}

func serviceProjectionHas(services []watchdogService, manager, unit string) bool {
	for _, svc := range services {
		if svc.Manager == manager && svc.Unit == unit {
			return true
		}
	}
	return false
}
