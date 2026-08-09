package main

import (
	"strings"
	"testing"
	"time"
)

// TestCronTaskSchedulerEmitsS4UPrincipal extends the fleet's reboot-survival contract
// (#3322) from the tools/*.ps1 installers — where fleet_installer_s4u_test.go already
// pins it — to fak's OWN unit emitter. Every scheduled task fak prints is meant to run
// unattended; a Register-ScheduledTask with no -Principal is registered for the
// interactive user and stops firing the moment the box reboots to a lock screen, which
// is the steady state of an unattended machine. That failure is silent: the task still
// EXISTS in Task Scheduler, it just never runs, so `Get-ScheduledTask` looks healthy.
func TestCronTaskSchedulerEmitsS4UPrincipal(t *testing.T) {
	got := cronRenderTaskScheduler("fak-git-daily", "daily git hygiene", 24*time.Hour,
		[]string{"fak", "git-daily"})

	for _, want := range []string{
		"$principal = New-ScheduledTaskPrincipal",
		"-UserId $env:USERNAME",
		"-LogonType S4U",
		"-RunLevel Limited",
		"-Principal $principal",
		"-StartWhenAvailable",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("emitted task missing %q\n---\n%s", want, got)
		}
	}
	// The principal must be DEFINED before Register-ScheduledTask consumes it, or the
	// snippet registers with an empty -Principal and the whole contract is decorative.
	if strings.Index(got, "$principal = New-ScheduledTaskPrincipal") > strings.Index(got, "Register-ScheduledTask -TaskName") {
		t.Errorf("principal defined after it is used:\n%s", got)
	}
	if strings.Contains(got, "-RunLevel Highest") {
		t.Errorf("emitted unit asks for elevation it was not granted:\n%s", got)
	}
}
