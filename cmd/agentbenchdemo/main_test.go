package main

import (
	"context"
	"testing"
)

// registrations is blank-imported by main.go, so the test binary already has the
// full adjudicator chain wired.

func TestRunBench_DeterministicCounts(t *testing.T) {
	const iters = 50
	r, err := runBench(context.Background(), benchToolset(), iters)
	if err != nil {
		t.Fatalf("runBench: %v", err)
	}
	if r.Calls != iters*3 {
		t.Errorf("calls = %d, want %d", r.Calls, iters*3)
	}
	if r.Allowed != iters {
		t.Errorf("allowed = %d, want %d (1 allow per iteration)", r.Allowed, iters)
	}
	if r.Denied != iters*2 {
		t.Errorf("denied = %d, want %d (2 denies per iteration)", r.Denied, iters*2)
	}
	if r.NsPerCall <= 0 {
		t.Errorf("ns_per_call = %d, want > 0", r.NsPerCall)
	}
	if r.CheaperBy <= 0 {
		t.Errorf("cheaper_by = %d, want > 0", r.CheaperBy)
	}
}

func TestSelfcheck_InvariantsHold(t *testing.T) {
	if code := selfcheck(context.Background(), benchToolset()); code != 0 {
		t.Fatalf("selfcheck exit = %d, want 0", code)
	}
}

func TestHumanNs(t *testing.T) {
	cases := []struct {
		ns   int64
		want string
	}{
		{500, "~500 ns"},
		{1500, "~1.50 µs"},
		{2_000_000, "~2.00 ms"},
	}
	for _, tc := range cases {
		if got := humanNs(tc.ns); got != tc.want {
			t.Errorf("humanNs(%d) = %q, want %q", tc.ns, got, tc.want)
		}
	}
}

func TestCommas(t *testing.T) {
	cases := map[int64]string{
		0:       "0",
		999:     "999",
		1000:    "1,000",
		1234567: "1,234,567",
	}
	for in, want := range cases {
		if got := commas(in); got != want {
			t.Errorf("commas(%d) = %q, want %q", in, got, want)
		}
	}
}
