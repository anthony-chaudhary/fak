package main

import "testing"

func TestSelfcheck(t *testing.T) {
	got, err := runSelfcheck()
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != schema || got.Verdict != "pass" || got.RichDashboardCount != 9 || !got.RichDashboardsLazy {
		t.Fatalf("unexpected witness: %+v", got)
	}
}
