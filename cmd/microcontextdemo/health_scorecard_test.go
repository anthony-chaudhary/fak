package main

import "testing"

func TestHealthScorecardGradesControlled1K(t *testing.T) {
	r, e := gradeHealth("../../experiments/microcontext/s5-gcp-1000-cuda-outcomes-2026-08-07.json")
	if e != nil {
		t.Fatal(e)
	}
	if r.Grade != "A" || r.Score != 100 || r.Success != 1000 {
		t.Fatalf("score=%+v", r)
	}
}
