package main

import "testing"

// TestRegressionDirection locks the track-mode alert rule: for a higher-is-better
// KPI a DROP is a regression; for a lower-is-better KPI a RISE is. This is the
// signal the ongoing benchmark-vs-main emits (exit 3) so a cron can alert.
func TestRegressionDirection(t *testing.T) {
	cases := []struct {
		name        string
		prev, cur   float64
		lowerBetter bool
		want        bool
	}{
		{"hit-rate dropped -> regression", 0.30, 0.20, false, true},
		{"hit-rate rose -> ok", 0.20, 0.30, false, false},
		{"hit-rate flat -> ok", 0.20, 0.20, false, false},
		{"latency rose -> regression", 5, 8, true, true},
		{"latency fell -> ok", 8, 5, true, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := regression(c.prev, c.cur, c.lowerBetter); got != c.want {
				t.Errorf("regression(%v,%v,lower=%v)=%v, want %v", c.prev, c.cur, c.lowerBetter, got, c.want)
			}
		})
	}
}

// TestParseInts locks the candidate-list flag parsing (tolerant of spaces/blanks).
func TestParseInts(t *testing.T) {
	got := parseInts(" 6, 8,8 , ,10,")
	want := []int{6, 8, 8, 10}
	if len(got) != len(want) {
		t.Fatalf("parseInts len=%d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("parseInts[%d]=%d, want %d", i, got[i], want[i])
		}
	}
	if len(parseInts("")) != 0 || len(parseInts("  ")) != 0 {
		t.Fatal("empty candidate list should parse to no candidates")
	}
}
