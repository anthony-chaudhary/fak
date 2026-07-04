package toolproc

import (
	"errors"
	"testing"
)

// TestMonitorArm_NoFailureCoverageRefused is the arm-time doctrine witness: a
// monitor whose filter matches only its progress pattern is REFUSED with the
// closed MONITOR_NO_FAILURE_COVERAGE token, while the same monitor with a
// failure-signature class in its filter arms cleanly. A caller bug (no cadence)
// is a distinct plain error, not the doctrine refusal.
func TestMonitorArm_NoFailureCoverageRefused(t *testing.T) {
	base := int64(1_000)

	// Progress-only filter: silent through a crashloop -> refused.
	_, err := ArmMonitor(MonitorSpec{
		CallID: "m-happy", Tool: "monitor:train", Filter: "elapsed_steps=",
		HeartbeatEveryMS: 5_000, AtMS: base,
	})
	if err == nil {
		t.Fatal("progress-only filter must be refused")
	}
	if !errors.Is(err, ErrMonitorNoFailureCoverage) {
		t.Fatalf("refusal must cite %s, got %v", ReasonMonitorNoFailureCoverageName, err)
	}

	// Same monitor, failure signatures added -> armed as a spawn with Monitor set.
	ev, err := ArmMonitor(MonitorSpec{
		CallID: "m-covered", Tool: "monitor:train", Session: "s1",
		Filter: "elapsed_steps=|Traceback|Error|FAILED", HeartbeatEveryMS: 5_000, AtMS: base,
	})
	if err != nil {
		t.Fatalf("covered filter must arm, got %v", err)
	}
	if ev.Kind != EvSpawn || !ev.Monitor || ev.HeartbeatEveryMS != 5_000 {
		t.Fatalf("armed monitor spawn malformed: %+v", ev)
	}
	if err := ValidateEvent(ev); err != nil {
		t.Fatalf("armed monitor spawn must validate: %v", err)
	}

	// A monitor with no cadence can never stall: a caller bug, distinct from the
	// coverage refusal (must NOT be ErrMonitorNoFailureCoverage).
	_, err = ArmMonitor(MonitorSpec{
		CallID: "m-nocadence", Filter: "Error|FAILED", HeartbeatEveryMS: 0, AtMS: base,
	})
	if err == nil {
		t.Fatal("zero-cadence monitor must be refused")
	}
	if errors.Is(err, ErrMonitorNoFailureCoverage) {
		t.Fatal("a cadence bug must not masquerade as a coverage refusal")
	}

	// The reason token is part of the closed toolproc verdict vocabulary.
	found := false
	for _, pr := range ReasonPairs() {
		if pr.Code == ReasonMonitorNoFailureCoverage && pr.Name == ReasonMonitorNoFailureCoverageName {
			found = true
		}
	}
	if !found {
		t.Fatalf("%s must be in ReasonPairs()", ReasonMonitorNoFailureCoverageName)
	}
}

// TestMonitorSilence_FoldsStalled is the fold-time doctrine witness: a monitor
// that goes quiet past its cadence folds to STALLED with a TOOL_HEARTBEAT_STALLED
// finding whose advice is KILL (not the generic probe), and it is counted as a
// stalled monitor — silence becomes a typed, actionable verdict.
func TestMonitorSilence_FoldsStalled(t *testing.T) {
	base := int64(1_000)
	now := base + 60_000 // >> 3x the 5s cadence with no pulse

	ev, err := ArmMonitor(MonitorSpec{
		CallID: "m1", Tool: "monitor:deploy", Session: "s1",
		Filter: "Ready in|Traceback|Error|FAILED", HeartbeatEveryMS: 5_000, AtMS: base,
	})
	if err != nil {
		t.Fatal(err)
	}
	tab, err := Fold([]Event{ev}, now, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if len(tab.Procs) != 1 {
		t.Fatalf("want one proc, got %d", len(tab.Procs))
	}
	p := tab.Procs[0]
	if !p.Monitor {
		t.Fatal("folded proc must carry the monitor flag")
	}
	if p.Liveness != LivenessStalled {
		t.Fatalf("silent monitor must fold STALLED, got %s", p.Liveness)
	}
	var stall *Finding
	for i := range p.Findings {
		if p.Findings[i].Reason == ReasonToolHeartbeatStalledName {
			stall = &p.Findings[i]
		}
	}
	if stall == nil {
		t.Fatalf("want a %s finding, got %+v", ReasonToolHeartbeatStalledName, p.Findings)
	}
	if stall.Advice != AdviceKill {
		t.Fatalf("a stalled MONITOR must advise kill, got %s", stall.Advice)
	}
	if tab.Counts.Stalled != 1 || tab.Counts.StalledMonitors != 1 {
		t.Fatalf("counts must record one stalled monitor, got stalled=%d stalled_monitors=%d",
			tab.Counts.Stalled, tab.Counts.StalledMonitors)
	}
	if !tab.AttentionNeeded {
		t.Fatal("a killable stalled monitor needs attention")
	}

	// Contrast: the SAME cadence on a generic (non-monitor) long-runner stays an
	// advisory probe and is not counted as a stalled monitor — the distinction
	// the doctrine turns on.
	generic := Event{Kind: EvSpawn, CallID: "g1", Tool: "bg_tail", Session: "s1",
		AtMS: base, HeartbeatEveryMS: 5_000}
	gtab, err := Fold([]Event{generic}, now, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if gtab.Counts.StalledMonitors != 0 {
		t.Fatalf("a generic stall is not a monitor stall, got stalled_monitors=%d", gtab.Counts.StalledMonitors)
	}
	if len(gtab.Procs[0].Findings) != 1 || gtab.Procs[0].Findings[0].Advice != AdviceProbe {
		t.Fatalf("a generic stall must stay an advisory probe, got %+v", gtab.Procs[0].Findings)
	}
}
