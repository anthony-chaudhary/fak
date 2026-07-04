package main

import (
	"strings"
	"testing"
)

// TestGuardInfoAgentActivityClause pins the agents-pane live-status cell (#2627) at the
// FULL level: an in-flight row reads "hot", an idle row reads "idle" with singular spawn
// grammar, and a pre-activity row is byte-for-byte the old row (no activity clause).
func TestGuardInfoAgentActivityClause(t *testing.T) {
	// In-flight row: last tool + plural spawns + in-flight age, and NOT idle.
	inflight := guardInfoAgentText(guardInfoSession{
		TraceID: "main-trace", Run: "running", LastTool: "Task", SpawnCount: 2, InflightSeconds: 12,
	})
	for _, want := range []string{"tool Task", "2 spawns", "in-flight 12s"} {
		if !strings.Contains(inflight, want) {
			t.Errorf("in-flight row missing %q: %q", want, inflight)
		}
	}
	if strings.Contains(inflight, "idle") {
		t.Errorf("an in-flight row must not also render idle: %q", inflight)
	}

	// Idle row: singular spawn grammar + idle age, and NOT in-flight or plural spawns.
	idle := guardInfoAgentText(guardInfoSession{
		TraceID: "sub", Run: "running", ParentTrace: "main-trace", LastTool: "Read", SpawnCount: 1, IdleSeconds: 45,
	})
	for _, want := range []string{"tool Read", "1 spawn", "idle 45s"} {
		if !strings.Contains(idle, want) {
			t.Errorf("idle row missing %q: %q", want, idle)
		}
	}
	if strings.Contains(idle, "1 spawns") {
		t.Errorf("a single spawn must be singular: %q", idle)
	}
	if strings.Contains(idle, "in-flight") {
		t.Errorf("an idle row must not render in-flight: %q", idle)
	}

	// Pre-activity row: none of the activity clauses appear.
	plain := guardInfoAgentText(guardInfoSession{TraceID: "cold", Run: "running"})
	for _, absent := range []string{"tool ", "spawn", "in-flight", "idle"} {
		if strings.Contains(plain, absent) {
			t.Errorf("a pre-activity row must omit %q: %q", absent, plain)
		}
	}
}

// TestGuardInfoAgentsSummaryInflight pins the cell at the MINI level: the one-row summary
// folds how many sessions hold an open request right now (the compact "who is hot"), and
// omits the clause entirely when none are in flight.
func TestGuardInfoAgentsSummaryInflight(t *testing.T) {
	hot := guardInfoAgentsSummary([]guardInfoSession{
		{TraceID: "a", InflightSeconds: 3},
		{TraceID: "b", ParentTrace: "a", Generation: 1, IdleSeconds: 5},
	})
	if !strings.Contains(hot, "2 active") {
		t.Errorf("summary must count active sessions: %q", hot)
	}
	if !strings.Contains(hot, "1 in-flight") {
		t.Errorf("summary must fold the in-flight count: %q", hot)
	}

	quiet := guardInfoAgentsSummary([]guardInfoSession{{TraceID: "a"}, {TraceID: "b"}})
	if strings.Contains(quiet, "in-flight") {
		t.Errorf("summary must omit the in-flight clause when none are hot: %q", quiet)
	}
}
