package gateway

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resumemetrics"
)

// TestDebugVarsWatchdogNilWhenCold pins the fail-quiet contract: a process that has recorded no
// resume/heal watchdog signal omits the block entirely rather than fabricating an all-zero one —
// the same nil-when-empty convention the other optional /debug/vars blocks keep (#3803).
func TestDebugVarsWatchdogNilWhenCold(t *testing.T) {
	resumemetrics.Reset()
	t.Cleanup(resumemetrics.Reset)

	srv := newTestServer(t)
	if got := srv.debugVars(time.Now()).Watchdog; got != nil {
		t.Fatalf("watchdog block on a cold process = %+v, want nil (block omitted)", got)
	}
}

// TestDebugVarsWatchdogMirrorsCounters proves the fold: once the watchdog records ticks, per-
// verdict actions, an autoheal result, witnessed progress, and folded health, the /debug/vars
// block carries each one in its published (normalized) form.
func TestDebugVarsWatchdogMirrorsCounters(t *testing.T) {
	resumemetrics.Reset()
	t.Cleanup(resumemetrics.Reset)

	resumemetrics.Tick()
	resumemetrics.Tick()
	resumemetrics.RecordAction("launch")
	resumemetrics.RecordAction("skip_blocked")
	resumemetrics.RecordAutohealResult("WATCHDOG_RESTARTED") // normalized to lower-case
	resumemetrics.ProgressWitnessed()
	resumemetrics.SetMonitorStatus("fleet-resume-watchdog", "HEALTHY")
	resumemetrics.SetHealthRollup("HEALTHY")

	srv := newTestServer(t)
	wd := srv.debugVars(time.Now()).Watchdog
	if wd == nil {
		t.Fatal("watchdog block is nil after recording signal, want the folded counters")
	}
	if wd.Ticks != 2 {
		t.Errorf("ticks = %d, want 2", wd.Ticks)
	}
	if wd.ProgressWitnessed != 1 {
		t.Errorf("progress_witnessed = %d, want 1", wd.ProgressWitnessed)
	}
	if wd.Actions["launch"] != 1 || wd.Actions["skip_blocked"] != 1 {
		t.Errorf("actions = %+v, want launch=1 skip_blocked=1", wd.Actions)
	}
	if wd.AutohealResults["watchdog_restarted"] != 1 {
		t.Errorf("autoheal_results = %+v, want watchdog_restarted=1", wd.AutohealResults)
	}
	if wd.MonitorStatus["fleet-resume-watchdog"] != "healthy" {
		t.Errorf("monitor_status = %+v, want fleet-resume-watchdog=healthy", wd.MonitorStatus)
	}
	if wd.HealthRollup != "healthy" {
		t.Errorf("health_rollup = %q, want healthy", wd.HealthRollup)
	}
}
