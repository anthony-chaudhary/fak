//go:build windows

package sysproc

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The tests in this file are the RUNTIME survival witnesses for the detached /
// durable seam (fak#13468). The flag-assertion tests in sysproc_windows_test.go
// only pin the bit pattern; these prove the behavior that pattern exists for:
// a child configured for durability still runs AFTER the process that launched
// it has exited.
//
// They are deliberately structured as a matched pair:
//
//   - TestConfigureDurableChildSurvivesLauncherExit is the POSITIVE witness.
//   - TestConfigureDetachedChildDiesWithLauncher is the NEGATIVE control, which
//     keeps the hazard real and documented rather than assumed.
//
// If the negative control ever starts passing (a DETACHED_PROCESS child that
// survives), the durable seam's reason to exist has changed and both tests must
// be revisited together.

const (
	survivalHelperEnv   = "SYSPROC_SURVIVAL_HELPER"
	survivalReportEnv   = "SYSPROC_SURVIVAL_REPORT"
	survivalDelayMillis = "3000"
)

// runSurvivalLauncher re-execs this test binary as a separate launcher process
// that applies `configure` to a delayed child and then EXITS immediately. The
// child writes reportPath only if it is still alive after the delay, so the
// existence of reportPath is proof the child outlived its launcher.
//
// The launcher must be a separate process: the property under test is about the
// launcher exiting, which a launcher that keeps running cannot exercise.
func runSurvivalLauncher(t *testing.T, configure func(*exec.Cmd), reportPath string) {
	t.Helper()
	launcher := exec.Command(os.Args[0], "-test.run=^"+t.Name()+"$")
	launcher.Env = append(os.Environ(),
		survivalHelperEnv+"=1",
		survivalReportEnv+"="+reportPath,
	)
	if out, err := launcher.CombinedOutput(); err != nil {
		t.Fatalf("launcher helper failed: %v: %s", err, out)
	}
}

// survivalHelperBody runs inside the re-exec'd launcher: it starts a delayed
// writer child, applies the caller's configuration, and returns WITHOUT waiting,
// so the launcher exits while the child is still scheduled to run.
func survivalHelperBody(t *testing.T, configure func(*exec.Cmd)) {
	reportPath := os.Getenv(survivalReportEnv)
	if reportPath == "" {
		t.Fatal(survivalReportEnv + " not set")
	}
	script := "Start-Sleep -Milliseconds " + survivalDelayMillis +
		"; Set-Content -Path '" + reportPath + "' -Value SURVIVED"
	child := exec.Command("powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	configure(child)

	devnull, err := os.OpenFile(os.DevNull, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer devnull.Close()
	child.Stdin, child.Stdout, child.Stderr = devnull, devnull, devnull

	if err := child.Start(); err != nil {
		t.Fatalf("start child: %v", err)
	}
	// Release every handle to the child so this launcher holds no wait
	// relationship at all: exactly the state after a launcher has exited.
	_ = child.Process.Release()
}

// awaitReport polls for the survival report and returns true when the child
// wrote it inside the budget.
func awaitReport(t *testing.T, reportPath string) bool {
	t.Helper()
	deadline := time.Now().Add(25 * time.Second)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(reportPath); err == nil && strings.Contains(string(b), "SURVIVED") {
			return true
		}
		time.Sleep(200 * time.Millisecond)
	}
	return false
}

// TestConfigureDurableChildSurvivesLauncherExit is the load-bearing acceptance
// witness for fak#13468: a child configured with ConfigureDurableChild must
// still run after the process that launched it has exited. Before the durable
// seam, `fak-flow spawn --mode detached` reported a PID and left 0-byte worker
// logs precisely because the child was torn down with its launcher.
func TestConfigureDurableChildSurvivesLauncherExit(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns helper processes")
	}
	reportPath := filepath.Join(t.TempDir(), "durable-survived.txt")
	if os.Getenv(survivalHelperEnv) == "1" {
		survivalHelperBody(t, ConfigureDurableChild)
		return
	}
	runSurvivalLauncher(t, ConfigureDurableChild, reportPath)
	// The launcher has exited by now.
	if !awaitReport(t, reportPath) {
		t.Fatalf("durable child never wrote %s after its launcher exited: "+
			"ConfigureDurableChild does not actually keep the child alive", reportPath)
	}
}

// TestConfigureDetachedChildDiesWithLauncher is the NEGATIVE control. It runs
// the SAME scenario under ConfigureDetached and asserts the report never
// appears, proving the hazard the durable seam exists to avoid is real.
//
// A failure here means a DETACHED_PROCESS child now survives its launcher, so
// the documented hazard (and possibly the need for a separate durable seam)
// should be re-derived rather than assumed.
func TestConfigureDetachedChildDiesWithLauncher(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns helper processes")
	}
	reportPath := filepath.Join(t.TempDir(), "detached-survived.txt")
	if os.Getenv(survivalHelperEnv) == "1" {
		survivalHelperBody(t, ConfigureDetached)
		return
	}
	runSurvivalLauncher(t, ConfigureDetached, reportPath)
	// Give the detached child well past its own delay to prove it never ran.
	time.Sleep(6 * time.Second)
	if _, err := os.Stat(reportPath); err == nil {
		t.Fatalf("a DETACHED_PROCESS child survived its launcher (%s exists); "+
			"the ConfigureDurableChild hazard documentation is stale", reportPath)
	}
}
