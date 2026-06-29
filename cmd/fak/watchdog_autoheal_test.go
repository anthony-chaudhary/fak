package main

import (
	"context"
	"errors"
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

	darwin := watchdogAutohealServicesForGOOS("darwin")
	if !serviceProjectionHas(darwin, "launchd", "com.fleet.dispatch-supervisor") {
		t.Fatalf("darwin projection missing launchd dispatch supervisor: %+v", darwin)
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
}

func serviceProjectionHas(services []watchdogService, manager, unit string) bool {
	for _, svc := range services {
		if svc.Manager == manager && svc.Unit == unit {
			return true
		}
	}
	return false
}
