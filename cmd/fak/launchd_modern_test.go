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
