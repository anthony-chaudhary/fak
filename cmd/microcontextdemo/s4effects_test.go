package main

import "testing"

func TestEffectsWitness(t *testing.T) {
	r, err := buildEffectsReport()
	if err != nil {
		t.Fatal(err)
	}
	if r.PhysicalApplies != 3 || r.ReplayedAfterRestart != 1 {
		t.Fatalf("bad report: %+v", r)
	}
}
