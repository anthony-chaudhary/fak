package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

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

func TestSelectHarnessSessionObs(t *testing.T) {
	h, err := selectHarness("sessionobs", ".", "main", "", "", "")
	if err != nil {
		t.Fatalf("selectHarness(sessionobs): %v", err)
	}
	if h.MetricName != "s0_loop_index" || h.LowerBetter {
		t.Fatalf("sessionobs harness should expose higher-better S0 loop-index, got %q lower=%v",
			h.MetricName, h.LowerBetter)
	}
}

func TestScoreSuffixSummarizesStructuredScore(t *testing.T) {
	score := &rsiloop.Scorecard{
		Name:  "attention_sn",
		Value: 0.9,
		Grade: "lean",
		Components: []rsiloop.ScoreComponent{
			{Name: "mean_ratio", Value: 0.9, Unit: "ratio"},
			{Name: "mean_fault_ratio", Value: 0, Unit: "ratio"},
			{Name: "signal_tokens", Value: 9, Unit: "tokens"},
			{Name: "caught", Value: 2, Unit: "calls"},
			{Name: "regressed", Value: 0, Unit: "calls"},
			{Name: "cache_size", Value: 8, Unit: "entries"},
			{Name: "scored_turns", Value: 1, Unit: "turns"},
		},
	}
	got := scoreSuffix(score)
	for _, want := range []string{
		"score attention_sn=0.900",
		"grade=lean",
		"mean_ratio=0.900",
		"mean_fault_ratio=0",
		"signal_tokens=9",
		"caught=2",
		"regressed=0",
		"cache_size=8",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("score suffix missing %q: %s", want, got)
		}
	}
	if strings.Contains(got, "scored_turns") {
		t.Fatalf("score suffix should stay compact and omit non-summary components: %s", got)
	}
}

func TestScoreSuffixNilIsEmpty(t *testing.T) {
	if got := scoreSuffix(nil); got != "" {
		t.Fatalf("nil score suffix = %q, want empty", got)
	}
}
