package main

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"
)

// TestDispatchAgingFailOnStarvedGate is the #3590 loop-gate witness: `--fail-on-starved N` turns
// the folded starved count into an exit code (mirrors dispatch-conservation --fail-on-leak). The
// fixture pins the clock so one candidate has waited past the 6h hard starvation deadline; the
// gate fails when the starved count exceeds N and passes otherwise. Without the flag the command
// stays report-only (exit 0) regardless of starvation.
func TestDispatchAgingFailOnStarvedGate(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	// #42 ready 7h ago (starved, past the 6h deadline); #7 ready 30s ago (fresh).
	starved := now.Add(-7 * time.Hour).Unix()
	fresh := now.Add(-30 * time.Second).Unix()
	withStarved := fmt.Sprintf(
		`[{"id":"42","base_weight":150,"ready_since":%d},{"id":"7","base_weight":1000,"ready_since":%d}]`,
		starved, fresh)
	noStarved := fmt.Sprintf(`[{"id":"7","base_weight":1000,"ready_since":%d}]`, fresh)

	cases := []struct {
		name     string
		candJSON string
		argv     []string
		wantCode int
	}{
		{"starved trips the 0 gate", withStarved, []string{"--fail-on-starved", "0"}, 1},
		{"starved within the 1 gate", withStarved, []string{"--fail-on-starved", "1"}, 0},
		{"no starved clears the 0 gate", noStarved, []string{"--fail-on-starved", "0"}, 0},
		{"report-only without the gate", withStarved, nil, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runDispatchAging(tc.argv, strings.NewReader(tc.candJSON), &stdout, &stderr, now)
			if code != tc.wantCode {
				t.Fatalf("exit = %d, want %d (out: %s / err: %s)",
					code, tc.wantCode, stdout.String(), stderr.String())
			}
		})
	}
}
