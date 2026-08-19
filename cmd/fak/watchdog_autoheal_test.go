package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
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

func TestWatchdogDiscoveryReadmitsExhaustedDiscoveredService(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 7, 11, 12, 0, 0, 0, time.UTC)
	spec := watchdogAutohealSpec{watchdogService: watchdogService{ID: "still-discovered", Manager: "systemd", Unit: "still-discovered.timer"}}
	st := watchdogHealState{
		Schema: watchdogAutohealSchema, ID: spec.ID, Attempts: 3,
		LastFailureUnixNano: now.Add(-time.Minute).UnixNano(), LastReason: watchdogReasonExhausted,
	}
	if err := writeWatchdogHealState(dir, st); err != nil {
		t.Fatal(err)
	}
	opts := watchdogAutohealOptions{
		StateDir: dir, Specs: []watchdogAutohealSpec{spec}, Clock: func() time.Time { return now },
		RestartPolicy: watchdogRestartPolicy{MaxAttempts: 3}, DiscoveryReconcile: 5 * time.Minute,
	}
	reconcileWatchdogDiscovery(opts)
	got, err := readWatchdogHealState(dir, spec.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Attempts != 3 {
		t.Fatalf("first reconcile attempts=%d, want exhausted mark retained", got.Attempts)
	}

	now = now.Add(5 * time.Minute)
	reconcileWatchdogDiscovery(opts)
	got, err = readWatchdogHealState(dir, spec.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Attempts != 0 || got.LastReason != "" || got.LastFailureUnixNano != 0 {
		t.Fatalf("stable authoritative discovery did not readmit service: %+v", got)
	}
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

func TestWatchdogHealReArmsStaleExhaustedStreak(t *testing.T) {
	now := time.Unix(9000, 0).UTC()
	dir := t.TempDir()
	// Seed an EXHAUSTED streak whose last real failed restart is well past the
	// re-arm window. Without re-arm this latches give_up forever (the observed
	// deadlock: exhausted -> never restarts -> never alive -> never resets).
	if err := writeWatchdogHealState(dir, watchdogHealState{
		Schema:              watchdogAutohealSchema,
		ID:                  "supervisor",
		Attempts:            2, // == MaxAttempts
		LastFailureUnixNano: now.Add(-time.Hour).UnixNano(),
		LastReason:          watchdogReasonExhausted,
	}); err != nil {
		t.Fatalf("seed state: %v", err)
	}
	restarts := 0
	spec := watchdogAutohealSpec{
		watchdogService: watchdogService{ID: "supervisor", Manager: "taskscheduler", Unit: "FleetSupervisorWatchdog"},
		Probe:           deadInstalledProbe,
		Restart: func(context.Context) error {
			restarts++
			return nil
		},
	}
	opts := testWatchdogAutohealOptions(dir, &now, spec)
	opts.RestartReArm = 30 * time.Minute

	got := runWatchdogAutoheal(context.Background(), opts)
	if restarts != 1 {
		t.Fatalf("restart calls = %d, want 1 (stale streak re-armed)", restarts)
	}
	if len(got) != 1 || got[0].Action != "restarted" || got[0].Reason != watchdogReasonRestarted {
		t.Fatalf("heal = %+v, want restarted (re-armed)", got)
	}
}

func TestWatchdogHealFreshExhaustedStreakStillGivesUp(t *testing.T) {
	now := time.Unix(9500, 0).UTC()
	dir := t.TempDir()
	// A FRESH exhausted streak (last failure just now) must still give up even with
	// re-arm configured — a genuinely broken unit is not hammered.
	if err := writeWatchdogHealState(dir, watchdogHealState{
		Schema:              watchdogAutohealSchema,
		ID:                  "supervisor",
		Attempts:            2,
		LastFailureUnixNano: now.Add(-time.Minute).UnixNano(),
		LastReason:          watchdogReasonExhausted,
	}); err != nil {
		t.Fatalf("seed state: %v", err)
	}
	restarts := 0
	spec := watchdogAutohealSpec{
		watchdogService: watchdogService{ID: "supervisor", Manager: "taskscheduler", Unit: "FleetSupervisorWatchdog"},
		Probe:           deadInstalledProbe,
		Restart: func(context.Context) error {
			restarts++
			return nil
		},
	}
	opts := testWatchdogAutohealOptions(dir, &now, spec)
	opts.RestartReArm = 30 * time.Minute

	got := runWatchdogAutoheal(context.Background(), opts)
	if restarts != 0 {
		t.Fatalf("restart calls = %d, want 0 (fresh streak still gives up)", restarts)
	}
	if len(got) != 1 || got[0].Action != "give_up" || got[0].Reason != watchdogReasonExhausted {
		t.Fatalf("heal = %+v, want give_up/%s", got, watchdogReasonExhausted)
	}
}

// TestWatchdogHealProbeFreshnessGateSkipsColdProbe is the witness for #3155 fix (1):
// a watchdog confirmed alive within ProbeTTL must NOT be cold-probed again (the
// probe spawns a powershell.exe; on a resume wave that is the host-wedging burst).
// It counts probe invocations directly: a fresh LastProbeAliveUnixNano yields a
// noop/WATCHDOG_PROBE_FRESH with the probe never called, while a stale timestamp
// (or ProbeTTL=0) falls through and probes as before.
func TestWatchdogHealRecentOutputResetsSilenceCanary(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 7, 11, 13, 0, 0, 0, time.UTC)
	probes := 0
	spec := watchdogAutohealSpec{
		watchdogService: watchdogService{ID: "active-output", Manager: "systemd", Unit: "active.timer"},
		LastActivity:    func() (time.Time, error) { return now.Add(-10 * time.Second), nil },
		Probe: func(context.Context) (watchdogProbe, error) {
			probes++
			return watchdogProbe{Alive: false, Installed: true}, nil
		},
		Restart: func(context.Context) error { return nil },
	}
	opts := watchdogAutohealOptions{Mode: watchdogAutohealOn, StateDir: dir, Specs: []watchdogAutohealSpec{spec}, Clock: func() time.Time { return now }, ProbeTTL: time.Minute}
	got := runWatchdogAutoheal(context.Background(), opts)
	if probes != 0 {
		t.Fatalf("probe calls=%d, want 0 while output is recent", probes)
	}
	if len(got) != 1 || got[0].Action != "noop" || got[0].Reason != watchdogReasonProbeFresh {
		t.Fatalf("result=%+v, want activity-reset noop", got)
	}

	spec.LastActivity = func() (time.Time, error) { return now.Add(-2 * time.Minute), nil }
	opts.Specs = []watchdogAutohealSpec{spec}
	got = runWatchdogAutoheal(context.Background(), opts)
	if probes != 1 {
		t.Fatalf("probe calls=%d, want 1 after silence window", probes)
	}
	if len(got) != 1 || got[0].Action == "noop" {
		t.Fatalf("silent result=%+v, want verified probe path", got)
	}
}

func TestWatchdogHealProbeFreshnessGateSkipsColdProbe(t *testing.T) {
	now := time.Unix(6000, 0).UTC()
	const probeTTL = 60 * time.Second

	newSpec := func(probes *int) watchdogAutohealSpec {
		return watchdogAutohealSpec{
			watchdogService: watchdogService{ID: "resume", Manager: "taskscheduler", Unit: "FleetResumeWatchdog"},
			Probe: func(context.Context) (watchdogProbe, error) {
				*probes++
				return watchdogProbe{Installed: true, Alive: true, Detail: "state=Ready"}, nil
			},
			Restart: func(context.Context) error { return nil },
		}
	}

	// Fresh: confirmed alive 30s ago, inside the 60s TTL → probe suppressed.
	t.Run("fresh alive suppresses probe", func(t *testing.T) {
		dir := t.TempDir()
		if err := writeWatchdogHealState(dir, watchdogHealState{
			Schema:                 watchdogAutohealSchema,
			ID:                     "resume",
			LastProbeAliveUnixNano: now.Add(-30 * time.Second).UnixNano(),
		}); err != nil {
			t.Fatalf("seed state: %v", err)
		}
		probes := 0
		local := now
		opts := testWatchdogAutohealOptions(dir, &local, newSpec(&probes))
		opts.ProbeTTL = probeTTL

		got := runWatchdogAutoheal(context.Background(), opts)
		if probes != 0 {
			t.Fatalf("probe calls = %d, want 0 (fresh alive should skip the cold probe)", probes)
		}
		if len(got) != 1 || got[0].Action != "noop" || got[0].Reason != watchdogReasonProbeFresh {
			t.Fatalf("heal = %+v, want noop/%s", got, watchdogReasonProbeFresh)
		}
	})

	// Stale: confirmed alive 90s ago, past the 60s TTL → probe fires.
	t.Run("stale alive falls through to probe", func(t *testing.T) {
		dir := t.TempDir()
		if err := writeWatchdogHealState(dir, watchdogHealState{
			Schema:                 watchdogAutohealSchema,
			ID:                     "resume",
			LastProbeAliveUnixNano: now.Add(-90 * time.Second).UnixNano(),
		}); err != nil {
			t.Fatalf("seed state: %v", err)
		}
		probes := 0
		local := now
		opts := testWatchdogAutohealOptions(dir, &local, newSpec(&probes))
		opts.ProbeTTL = probeTTL

		got := runWatchdogAutoheal(context.Background(), opts)
		if probes != 1 {
			t.Fatalf("probe calls = %d, want 1 (stale alive must re-probe)", probes)
		}
		if len(got) != 1 || got[0].Action != "noop" || got[0].Reason != watchdogReasonAlive {
			t.Fatalf("heal = %+v, want noop/%s", got, watchdogReasonAlive)
		}
	})

	// ProbeTTL=0 (the direct-unit-test default) disables the gate: probe always fires
	// even with a fresh timestamp on record.
	t.Run("zero TTL disables the gate", func(t *testing.T) {
		dir := t.TempDir()
		if err := writeWatchdogHealState(dir, watchdogHealState{
			Schema:                 watchdogAutohealSchema,
			ID:                     "resume",
			LastProbeAliveUnixNano: now.UnixNano(),
		}); err != nil {
			t.Fatalf("seed state: %v", err)
		}
		probes := 0
		local := now
		opts := testWatchdogAutohealOptions(dir, &local, newSpec(&probes))
		// opts.ProbeTTL left at 0.

		got := runWatchdogAutoheal(context.Background(), opts)
		if probes != 1 {
			t.Fatalf("probe calls = %d, want 1 (zero ProbeTTL disables the freshness gate)", probes)
		}
		if len(got) != 1 || got[0].Reason != watchdogReasonAlive {
			t.Fatalf("heal = %+v, want noop/%s", got, watchdogReasonAlive)
		}
	})
}

// TestWatchdogAutohealTickAllowedDebouncesWave is the witness for #3155 fix (2):
// the host-global tick debounce. The watchdogs are host-global scheduled tasks, so
// ONE process healing them per window covers every concurrent guarded session. A
// resume/restart wave of N sessions each reaches watchdogAutohealOnStart; without
// this gate that is the 4xN concurrent cold-probe burst that wedges the host. The
// winner takes a lease it never releases, so every sibling starting within the TTL
// sees the fresh lease and skips the tick — until the lease expires, when healing
// resumes.
func TestWatchdogAutohealTickAllowedDebouncesWave(t *testing.T) {
	dir := t.TempDir()
	now := time.Unix(9000, 0).UTC()
	const ttl = 60 * time.Second

	// The first starter in the window wins the tick.
	if !watchdogAutohealTickAllowed(dir, now, ttl) {
		t.Fatalf("first starter must be allowed to run the tick")
	}
	// Siblings in the same wave (within the TTL) skip — the whole point: one cold-probe
	// pass covers the wave instead of 4xN.
	if watchdogAutohealTickAllowed(dir, now.Add(1*time.Second), ttl) {
		t.Fatalf("a sibling starting within the TTL must skip the tick")
	}
	if watchdogAutohealTickAllowed(dir, now.Add(ttl-time.Nanosecond), ttl) {
		t.Fatalf("a sibling just inside the TTL must still skip the tick")
	}
	// Once the lease expires, healing resumes: the next starter steals the stale lease
	// and runs the tick.
	if !watchdogAutohealTickAllowed(dir, now.Add(ttl+time.Second), ttl) {
		t.Fatalf("after the TTL elapses a fresh starter must run the tick again")
	}
	// ...and that fresh winner re-arms the debounce for its own window.
	if watchdogAutohealTickAllowed(dir, now.Add(ttl+2*time.Second), ttl) {
		t.Fatalf("the post-expiry winner must re-arm the debounce")
	}
}

// TestWatchdogAutohealTickAllowedFailsOpen pins the fail-open rungs: with no state
// dir to lease under, or a non-positive TTL, the debounce cannot be evaluated — and
// an unbraked heal beats a silently-skipped one, so the tick is always allowed.
func TestWatchdogAutohealTickAllowedFailsOpen(t *testing.T) {
	now := time.Unix(9000, 0).UTC()
	if !watchdogAutohealTickAllowed("", now, 60*time.Second) {
		t.Fatalf("an empty state dir must fail open (tick allowed)")
	}
	if !watchdogAutohealTickAllowed("   ", now, 60*time.Second) {
		t.Fatalf("a blank state dir must fail open (tick allowed)")
	}
	if !watchdogAutohealTickAllowed(t.TempDir(), now, 0) {
		t.Fatalf("a zero TTL must fail open (tick allowed)")
	}
	if !watchdogAutohealTickAllowed(t.TempDir(), now, -time.Second) {
		t.Fatalf("a negative TTL must fail open (tick allowed)")
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
		!serviceProjectionHas(win, "taskscheduler", "FleetDOSDispatchWatchdog") {
		t.Fatalf("windows projection missing expected Scheduled Tasks: %+v", win)
	}
	// #5096: the heartbeat/PID supervisor is tombstoned and superseded by DOS.
	// Keeping it in autoheal would turn its deliberate disabled state into restart noise.
	if serviceProjectionHas(win, "taskscheduler", "FleetSupervisorWatchdog") {
		t.Fatalf("windows projection still autoheals tombstoned FleetSupervisorWatchdog: %+v", win)
	}
	if !serviceProjectionHas(win, "taskscheduler", "FleetStaleWorkGarden") {
		t.Fatalf("windows projection missing stale-work garden task: %+v", win)
	}
	// #3324: the proc-resource guard must be in the autoheal target set so a deleted
	// FleetProcResourceGuard task self-reinstalls via the same schtasks /Run path as
	// its siblings, instead of staying gone until an operator notices.
	if !serviceProjectionHas(win, "taskscheduler", "FleetProcResourceGuard") {
		t.Fatalf("windows projection missing proc-resource-guard task (self-heal on deletion, #3324): %+v", win)
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
	if serviceProjectionHas(linux, "systemd", "fleet-supervisor-watchdog.timer") {
		t.Fatalf("linux projection still autoheals tombstoned fleet supervisor timer: %+v", linux)
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

func TestWatchdogAutohealSummaryLineIsUnconditional(t *testing.T) {
	results := []watchdogAutohealResult{
		{Action: "noop", ID: "a"},
		{Action: "noop", ID: "b"},
		{Action: "give_up", ID: "c"},
	}
	b := watchdogAutohealSummaryLine("guard", 123*time.Millisecond, results)
	if b == nil {
		t.Fatalf("summary line = nil, want JSON")
	}
	var s watchdogAutohealSummary
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatalf("unmarshal summary: %v; line = %s", err, b)
	}
	if s.Specs != 3 || s.Noop != 2 || s.GaveUp != 1 || s.ElapsedMS != 123 || s.Verb != "guard" || !s.Summary {
		t.Fatalf("summary = %+v, want specs=3 noop=2 gave_up=1 elapsed_ms=123 verb=guard summary=true", s)
	}
	if got := watchdogAutohealSummaryLine("guard", time.Millisecond, nil); got != nil {
		t.Fatalf("summary line for zero results = %q, want nil", got)
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

func TestRunWatchdogAutohealSerializesProbes(t *testing.T) {
	var active, maxActive atomic.Int32
	specs := make([]watchdogAutohealSpec, 4)
	for i := range specs {
		specs[i] = watchdogAutohealSpec{watchdogService: watchdogService{ID: fmt.Sprintf("s%d", i)}, Restart: func(context.Context) error { return nil }, Probe: func(context.Context) (watchdogProbe, error) {
			n := active.Add(1)
			for {
				m := maxActive.Load()
				if n <= m || maxActive.CompareAndSwap(m, n) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			active.Add(-1)
			return watchdogProbe{Installed: true, Alive: true}, nil
		}}
	}
	opts := watchdogAutohealOptions{Mode: watchdogAutohealOn, Specs: specs, StateDir: t.TempDir(), Clock: time.Now, Sleep: time.Sleep}
	got := runWatchdogAutoheal(context.Background(), opts)
	if len(got) != 4 || maxActive.Load() != 1 {
		t.Fatalf("results=%d max concurrent=%d", len(got), maxActive.Load())
	}
}
