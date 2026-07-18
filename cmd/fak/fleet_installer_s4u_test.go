package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestFleetInstallersAreS4UWithStartWhenAvailable is the reboot-survival doctor
// assertion for the fleet task set (#3322). The 2026-07-08 audit found ~17 fleet
// loops installed as LogonType=Interactive + StartWhenAvailable=false — the
// `schtasks /Create ... /RL LIMITED` default. Such tasks do NOT run at cold boot
// before a user logs on, and skip any tick missed while the machine was off, so the
// fleet's "always-on" posture silently degrades to "on only after logon" — invisible
// until a reboot. The fix is to register each task with an S4U principal
// (windowless, session 0, runs whether logged on or not) + StartWhenAvailable
// (a tick missed while the box slept fires on wake).
//
// This test pins that contract as source: every reboot-survival installer must
// register via Register-ScheduledTask with `-LogonType S4U -RunLevel Limited`
// (S4U, NOT elevation) + `-StartWhenAvailable`, and the ones migrated under #3322
// must not ship the bare `schtasks /Create /TN ...` install invocation that
// re-introduces the Interactive+SWA=false degradation on the next reinstall.
func TestFleetInstallersAreS4UWithStartWhenAvailable(t *testing.T) {
	toolsDir := filepath.Join("..", "..", "tools")

	// Installers migrated to S4U + StartWhenAvailable under #3322.
	migrated := []string{
		"register_dos_dispatch_watchdog.ps1",
		"register_supervisor_watchdog.ps1",
		"register_control_pane_tick.ps1",
		"install_self_update_schedule.ps1",
	}
	// Sibling reboot-survival installers already on the S4U + SWA contract
	// (register_resume_watchdog.ps1 landed it under #3321). Held to the same
	// principal/SWA bar so a future edit cannot silently regress the set.
	alreadyS4U := []string{
		"register_resume_watchdog.ps1",
	}

	// Every reboot-survival installer must carry the S4U principal + SWA settings.
	principalWants := []string{
		"Register-ScheduledTask",
		"-LogonType S4U",
		"-RunLevel Limited", // S4U runs unattended; RunLevel stays Limited (not elevation)
		"-StartWhenAvailable",
	}

	assertPrincipal := func(t *testing.T, name string) string {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(toolsDir, name))
		if err != nil {
			t.Fatalf("read installer %s: %v", name, err)
		}
		s := string(b)
		for _, want := range principalWants {
			if !strings.Contains(s, want) {
				t.Errorf("%s: missing %q (a fresh install would not report S4U + StartWhenAvailable)", name, want)
			}
		}
		return s
	}

	for _, name := range migrated {
		t.Run(name, func(t *testing.T) {
			s := assertPrincipal(t, name)
			// The degraded install invocation the audit flagged. Match the exact
			// `schtasks /Create /TN` call, not a historical prose mention in a
			// header comment, so the negative check pins the behavior, not the docs.
			if strings.Contains(s, "schtasks /Create /TN") {
				t.Errorf("%s: still ships `schtasks /Create /TN` — that install path re-registers "+
					"LogonType=Interactive + StartWhenAvailable=false (no cold-boot-before-logon run, no catch-up)", name)
			}
		})
	}
	for _, name := range alreadyS4U {
		t.Run(name, func(t *testing.T) { assertPrincipal(t, name) })
	}
}
