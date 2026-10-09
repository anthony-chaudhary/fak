package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=10ms lane=default
func TestRestartLaunchdPreservesConcurrentStart(t *testing.T) {
	const label = "com.fleet.stale-work-garden"
	domain := fmt.Sprintf("gui/%d", os.Getuid())
	target := domain + "/" + label
	for _, tc := range []struct {
		name             string
		loaded           bool
		startsBeforeKick bool
	}{
		{name: "loaded stopped job", loaded: true},
		{name: "starts after list", loaded: true, startsBeforeKick: true},
		{name: "missing stopped job"},
		{name: "starts during bootstrap", startsBeforeKick: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plist := filepath.Join(t.TempDir(), label+".plist")
			if err := os.WriteFile(plist, []byte("fixture"), 0o600); err != nil {
				t.Fatal(err)
			}
			var calls [][]string
			pid, kills := 0, 0
			run := func(_ context.Context, name string, args ...string) (string, error) {
				calls = append(calls, append([]string{name}, args...))
				if name != "launchctl" || len(args) == 0 {
					t.Fatalf("unexpected command: %s %q", name, args)
				}
				switch args[0] {
				case "list":
					if !tc.loaded {
						return "service absent", errors.New("launchctl list failed")
					}
					// Simulate a concurrent start after observing a stopped job.
					if tc.startsBeforeKick {
						pid = 41
					}
				case "bootstrap":
					// RunAtLoad/KeepAlive can start the job before bootstrap returns.
					if tc.startsBeforeKick {
						pid = 41
					}
				case "kickstart":
					// Model the documented -k distinction, not a launchd execution.
					for _, arg := range args[1:] {
						if arg == "-k" && pid != 0 {
							kills++
							pid = 0
						}
					}
					if pid == 0 {
						pid = 42
					}
				default:
					t.Fatalf("unexpected lifecycle command: %q", args)
				}
				return "", nil
			}
			if err := restartLaunchd(context.Background(), run, label, plist); err != nil {
				t.Fatal(err)
			}
			wantPID := 42
			if tc.startsBeforeKick {
				wantPID = 41
			}
			if kills != 0 || pid != wantPID {
				t.Fatalf("kills=%d, pid=%d; want no kills and pid=%d", kills, pid, wantPID)
			}
			wantCalls := [][]string{{"launchctl", "list", label}}
			if !tc.loaded {
				wantCalls = append(wantCalls, []string{"launchctl", "bootstrap", domain, plist})
			}
			wantCalls = append(wantCalls, []string{"launchctl", "kickstart", target})
			if !reflect.DeepEqual(calls, wantCalls) {
				t.Fatalf("commands=%q; want %q", calls, wantCalls)
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestLaunchdObservationDoesNotHealUncertainState(t *testing.T) {
	const label = "com.fleet.stale-work-garden"
	domain := fmt.Sprintf(" in domain for user gui: %d", os.Getuid())
	exitErr := errors.New("launchctl failed")
	for _, tc := range []struct {
		name, output string
		runErr       error
		cancel       bool
		wantErr      bool
		wantAlive    bool
	}{
		{name: "loaded", wantAlive: true},
		{name: "absent quoted service", output: "Bad request.\nCould not find service \"" + label + "\"" + domain, runErr: exitErr},
		{name: "absent unquoted service", output: "Could not find service " + label + domain, runErr: exitErr},
		{name: "GUI domain unavailable", output: "Could not find domain for user gui: 501", runErr: exitErr, wantErr: true},
		{name: "permission denied", output: "Operation not permitted", runErr: exitErr, wantErr: true},
		{name: "wrong absent service", output: "Could not find service \"com.fleet.other\"" + domain, runErr: exitErr, wantErr: true},
		{name: "wrong domain", output: "Could not find service \"" + label + "\" in domain for system", runErr: exitErr, wantErr: true},
		{name: "wrong GUI UID", output: fmt.Sprintf("Could not find service \"%s\" in domain for user gui: %d", label, os.Getuid()+1), runErr: exitErr, wantErr: true},
		{name: "unclassified failure", output: "No such process", runErr: exitErr, wantErr: true},
		{name: "runner canceled", output: "Could not find service \"" + label + "\"" + domain, runErr: context.Canceled, wantErr: true},
		{name: "runner deadline exceeded", output: "Could not find service \"" + label + "\"" + domain, runErr: context.DeadlineExceeded, wantErr: true},
		{name: "already canceled", cancel: true, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			plist := filepath.Join(dir, label+".plist")
			// The existing plist is essential to the regression: previously any
			// failed list probe made the watchdog eligible for a restart.
			if err := os.WriteFile(plist, []byte("fixture"), 0o600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tc.cancel {
				cancel()
			}
			var calls [][]string
			run := func(_ context.Context, name string, args ...string) (string, error) {
				calls = append(calls, append([]string{name}, args...))
				return tc.output, tc.runErr
			}
			probe, err := probeLaunchd(ctx, run, label, plist)
			if (err != nil) != tc.wantErr || probe.Alive != tc.wantAlive {
				t.Fatalf("probe = %+v, %v; want error=%v, alive=%v", probe, err, tc.wantErr, tc.wantAlive)
			}
			if !tc.wantErr && !probe.Installed {
				t.Fatalf("existing plist lost installation state: %+v", probe)
			}
			wantCalls := [][]string{{"launchctl", "print", fmt.Sprintf("gui/%d/%s", os.Getuid(), label)}}
			if tc.cancel {
				wantCalls = nil
			}
			if !reflect.DeepEqual(calls, wantCalls) {
				t.Fatalf("commands = %q, want %q", calls, wantCalls)
			}
			if !tc.wantErr {
				return
			}
			if !tc.cancel && (!errors.Is(err, tc.runErr) || !strings.Contains(err.Error(), tc.output)) {
				t.Fatalf("observation discarded command error/output: %v", err)
			}
			restarts := 0
			spec := watchdogAutohealSpec{
				watchdogService: watchdogService{ID: "garden-observation-test", Manager: "launchd", Unit: label},
				Probe: func(ctx context.Context) (watchdogProbe, error) {
					return probeLaunchd(ctx, run, label, plist)
				},
				Restart: func(context.Context) error { restarts++; return nil },
			}
			now := time.Unix(1000, 0).UTC()
			opts := testWatchdogAutohealOptions(filepath.Join(dir, "state"), &now, spec)
			results := runWatchdogAutoheal(ctx, opts)
			if restarts != 0 || len(results) != 1 || results[0].Action != "probe_failed" {
				t.Fatalf("uncertain observation triggered healing: restarts=%d, results=%+v", restarts, results)
			}
		})
	}
}
