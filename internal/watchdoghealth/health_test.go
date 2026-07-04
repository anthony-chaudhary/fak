package watchdoghealth

import (
	"reflect"
	"testing"
)

func TestClassify(t *testing.T) {
	cases := []struct {
		name string
		m    Monitor
		want Status
	}{
		{"probe error wins over everything", Monitor{ProbeErr: true, Installed: true, Alive: true}, StatusUnknown},
		{"alive is healthy", Monitor{Installed: true, Alive: true}, StatusHealthy},
		{"not installed is a no-op", Monitor{Installed: false, Alive: false}, StatusNotInstalled},
		{"installed dead, no heal yet, is down", Monitor{Installed: true, Alive: false}, StatusDown},
		{"installed dead, restart recorded, is healing", Monitor{Installed: true, Alive: false, LastRestartUnixNano: 42}, StatusHealing},
		{"installed dead, mid streak, is healing", Monitor{Installed: true, Alive: false, Attempts: 1, MaxAttempts: 3}, StatusHealing},
		{"installed dead, streak exhausted, gave up", Monitor{Installed: true, Alive: false, Attempts: 3, MaxAttempts: 3}, StatusGaveUp},
		{"installed dead, streak over cap, gave up", Monitor{Installed: true, Alive: false, Attempts: 9, MaxAttempts: 3}, StatusGaveUp},
		{"give-up undecidable without cap reads healing", Monitor{Installed: true, Alive: false, Attempts: 9, MaxAttempts: 0}, StatusHealing},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Classify(c.m); got != c.want {
				t.Fatalf("Classify(%+v) = %q, want %q", c.m, got, c.want)
			}
		})
	}
}

func TestNeedsAttentionByStatus(t *testing.T) {
	// HEALING self-corrects; the three at/above the floor need a closer look.
	attn := map[Status]bool{
		StatusHealthy:      false,
		StatusNotInstalled: false,
		StatusHealing:      false,
		StatusDown:         true,
		StatusUnknown:      true,
		StatusGaveUp:       true,
	}
	for s, want := range attn {
		if got := (Health{Status: s}).NeedsAttention(); got != want {
			t.Fatalf("NeedsAttention(%s) = %v, want %v", s, got, want)
		}
	}
}

func TestFoldRollupIsWorstOf(t *testing.T) {
	cases := []struct {
		name       string
		monitors   []Monitor
		wantRollup Status
		wantAttn   bool
	}{
		{
			name:       "empty is not-installed, no attention",
			monitors:   nil,
			wantRollup: StatusNotInstalled,
			wantAttn:   false,
		},
		{
			name: "healthy beats not-installed in the rollup",
			monitors: []Monitor{
				{ID: "a", Installed: false},
				{ID: "b", Installed: true, Alive: true},
			},
			wantRollup: StatusHealthy,
			wantAttn:   false,
		},
		{
			name: "all not installed rolls up to not-installed",
			monitors: []Monitor{
				{ID: "a", Installed: false},
				{ID: "b", Installed: false},
			},
			wantRollup: StatusNotInstalled,
			wantAttn:   false,
		},
		{
			name: "one healing keeps the layer self-correcting (no attention)",
			monitors: []Monitor{
				{ID: "a", Installed: true, Alive: true},
				{ID: "b", Installed: true, Alive: false, Attempts: 1, MaxAttempts: 3},
			},
			wantRollup: StatusHealing,
			wantAttn:   false,
		},
		{
			name: "a gave-up monitor dominates and demands attention",
			monitors: []Monitor{
				{ID: "a", Installed: true, Alive: true},
				{ID: "b", Installed: true, Alive: false, Attempts: 1, MaxAttempts: 3},
				{ID: "c", Installed: true, Alive: false, Attempts: 3, MaxAttempts: 3},
			},
			wantRollup: StatusGaveUp,
			wantAttn:   true,
		},
		{
			name: "a probe error demands attention even amid healthy peers",
			monitors: []Monitor{
				{ID: "a", Installed: true, Alive: true},
				{ID: "b", ProbeErr: true},
			},
			wantRollup: StatusUnknown,
			wantAttn:   true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := Fold(c.monitors)
			if d.Rollup != c.wantRollup {
				t.Fatalf("rollup = %q, want %q", d.Rollup, c.wantRollup)
			}
			if d.NeedsAttention != c.wantAttn {
				t.Fatalf("needs_attention = %v, want %v", d.NeedsAttention, c.wantAttn)
			}
			if len(d.Monitors) != len(c.monitors) {
				t.Fatalf("monitors = %d, want %d", len(d.Monitors), len(c.monitors))
			}
			if d.Schema != Schema {
				t.Fatalf("schema = %q, want %q", d.Schema, Schema)
			}
		})
	}
}

func TestFoldPreservesOrderAndCounts(t *testing.T) {
	monitors := []Monitor{
		{ID: "z", Installed: true, Alive: true},
		{ID: "a", Installed: false},
		{ID: "m", Installed: true, Alive: false, Attempts: 3, MaxAttempts: 3},
	}
	d := Fold(monitors)
	gotIDs := []string{d.Monitors[0].ID, d.Monitors[1].ID, d.Monitors[2].ID}
	wantIDs := []string{"z", "a", "m"}
	if !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("monitor order = %v, want %v (input order must be preserved)", gotIDs, wantIDs)
	}
	wantCounts := map[Status]int{StatusHealthy: 1, StatusNotInstalled: 1, StatusGaveUp: 1}
	if !reflect.DeepEqual(d.Counts, wantCounts) {
		t.Fatalf("counts = %v, want %v", d.Counts, wantCounts)
	}
}

func TestStatusesAreSeverityOrderedAndComplete(t *testing.T) {
	got := Statuses()
	want := []Status{
		StatusNotInstalled, StatusHealthy, StatusHealing,
		StatusDown, StatusUnknown, StatusGaveUp,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Statuses() = %v, want %v", got, want)
	}
	// Severity must be strictly non-decreasing across the returned order.
	for i := 1; i < len(got); i++ {
		if severity(got[i]) < severity(got[i-1]) {
			t.Fatalf("Statuses() not severity-ordered at %d: %s(%d) < %s(%d)",
				i, got[i], severity(got[i]), got[i-1], severity(got[i-1]))
		}
	}
}
