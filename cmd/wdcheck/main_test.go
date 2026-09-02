// Tests for wdcheck's pure watchdog mappers: token normalization, the glyph and
// rollup-word vocabularies, the attention floor, and the monitor/count folds
// (attention-first ordering, count-desc/key-asc ordering, zero dropping, and
// the "+K more" cap). These are the exact behaviors main() asserts with ad-hoc
// print checks; here they run as real Go tests.
package main

import (
	"strings"
	"testing"
)

func TestWatchdogStatusNormalizesToken(t *testing.T) {
	cases := []struct{ in, want string }{
		{"healthy", "HEALTHY"},
		{"  HEALTHY ", "HEALTHY"},
		{"gave_up", "GAVE_UP"},
		{"", ""},
		{"   ", ""},
	}
	for _, c := range cases {
		if got := string(watchdogStatus(c.in)); got != c.want {
			t.Errorf("watchdogStatus(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestWatchdogGlyphVocabulary(t *testing.T) {
	cases := []struct {
		token string
		want  string
	}{
		{"healthy", guardWatchdogChipHealthy},
		{" Healthy ", guardWatchdogChipHealthy},
		{"not_installed", guardWatchdogChipAbsent},
		{"healing", guardWatchdogChipHealing},
		{"down", guardWatchdogChipAttention},
		{"unknown", guardWatchdogChipAttention},
		{"gave_up", guardWatchdogChipGaveUp},
		{"", guardWatchdogChipAbsent},
		{"bogus", guardWatchdogChipAttention},
	}
	for _, c := range cases {
		if got := watchdogGlyph(c.token); got != c.want {
			t.Errorf("watchdogGlyph(%q) = %q, want %q", c.token, got, c.want)
		}
	}
}

func TestWatchdogRollupWord(t *testing.T) {
	cases := []struct{ token, want string }{
		{"healthy", "healthy"},
		{"not_installed", "not installed"},
		{"healing", "healing"},
		{"down", "down"},
		{"unknown", "unknown"},
		{"gave_up", "gave up (needs a human)"},
		{"", "no verdict yet"},
		{" Wonky State ", "wonky state"},
	}
	for _, c := range cases {
		if got := watchdogRollupWord(c.token); got != c.want {
			t.Errorf("watchdogRollupWord(%q) = %q, want %q", c.token, got, c.want)
		}
	}
}

func TestWatchdogNeedsAttentionFloor(t *testing.T) {
	cases := []struct {
		token string
		want  bool
	}{
		{"down", true},
		{"unknown", true},
		{"gave_up", true},
		{"bogus", true},
		{"healthy", false},
		{"not_installed", false},
		{"healing", false},
		{"", false},
	}
	for _, c := range cases {
		if got := watchdogNeedsAttention(c.token); got != c.want {
			t.Errorf("watchdogNeedsAttention(%q) = %v, want %v", c.token, got, c.want)
		}
	}
}

func TestGuardInfoWatchdogMonitorsText(t *testing.T) {
	monitors := map[string]string{
		"a-healthy": "healthy", "b-healthy": "healthy",
		"c-down": "down", "d-gaveup": "gave_up",
	}
	cases := []struct {
		name  string
		limit int
		want  string
	}{
		{
			name:  "attention rows first, capped with +K more",
			limit: 2,
			want:  guardWatchdogChipAttention + " c-down  " + guardWatchdogChipGaveUp + " d-gaveup  +2 more",
		},
		{
			name:  "no cap renders every monitor",
			limit: 0,
			want: guardWatchdogChipAttention + " c-down  " + guardWatchdogChipGaveUp + " d-gaveup  " +
				guardWatchdogChipHealthy + " a-healthy  " + guardWatchdogChipHealthy + " b-healthy",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := guardInfoWatchdogMonitorsText(monitors, tc.limit); got != tc.want {
				t.Fatalf("guardInfoWatchdogMonitorsText(limit=%d) =\n  %q\nwant\n  %q", tc.limit, got, tc.want)
			}
		})
	}
	if got := guardInfoWatchdogMonitorsText(map[string]string{}, 3); got != "" {
		t.Fatalf("empty monitor map rendered %q, want \"\"", got)
	}
}

func TestGuardInfoWatchdogCountMap(t *testing.T) {
	counts := map[string]int64{"a": 1, "b": 5, "c": 5, "d": 2, "e": 9, "zero": 0}
	cases := []struct {
		name  string
		limit int
		want  string
	}{
		{"count desc then key asc, capped", 3, "e 9 · b 5 · c 5 · +2 more"},
		{"no cap keeps deterministic tail", 0, "e 9 · b 5 · c 5 · d 2 · a 1"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardInfoWatchdogCountMap(counts, tc.limit)
			if got != tc.want {
				t.Fatalf("guardInfoWatchdogCountMap(limit=%d) = %q, want %q", tc.limit, got, tc.want)
			}
			if strings.Contains(got, "zero") {
				t.Fatalf("zero-count key leaked into render: %q", got)
			}
		})
	}
	if got := guardInfoWatchdogCountMap(map[string]int64{}, 3); got != "" {
		t.Fatalf("empty count map rendered %q, want \"\"", got)
	}
	if got := guardInfoWatchdogCountMap(map[string]int64{"only": 0}, 3); got != "" {
		t.Fatalf("all-zero count map rendered %q, want \"\"", got)
	}
}
