package main

import (
	"testing"
	"time"
)

func TestSelfcheck(t *testing.T) {
	if err := runSelfcheck(); err != nil {
		t.Fatal(err)
	}
}

func TestGatePrecedence(t *testing.T) {
	now, _ := time.Parse(time.RFC3339, defaultNow)
	r := Receipt{Schema: "fak-guard-crash/1", Reason: "TERMINAL_CRASH", Producer: "guard", ProducedAt: "2026-08-31T11:59:00Z", EffectKey: "panic:x", Recursion: true, Capacity: 0, ExpectedValue: 0}
	d := Evaluate(r, now, nil)
	if d.Decision != "SKIP" || d.Reason != "RECURSION_SUPPRESSED" {
		t.Fatalf("got %+v", d)
	}
}

func TestHeuristicNeedsEvidence(t *testing.T) {
	now, _ := time.Parse(time.RFC3339, defaultNow)
	r := Receipt{Schema: "fak-intent-cluster/1", Reason: "RECURRING_INTENT", Producer: "trajectory", ProducedAt: "2026-08-31T11:59:00Z", EffectKey: "intent:x", Capacity: 1, ExpectedValue: 2, SampleCount: 2}
	d := Evaluate(r, now, nil)
	if d.Decision != "SKIP" || d.Reason != "INSUFFICIENT_SAMPLE" {
		t.Fatalf("got %+v", d)
	}
}
