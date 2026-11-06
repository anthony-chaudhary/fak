package modver

import (
	"fmt"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestCoverageScoresRepeatedBlockStatementCounts(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name         string
		first, next  int
		executions   int
		wantError    bool
		wantCoverage float64
	}{
		{name: "conflicting counts", first: 1, next: 9, executions: 7, wantError: true},
		{name: "reversed conflict", first: 9, next: 1, executions: 7, wantError: true},
		{name: "consistent covered", first: 2, next: 2, executions: 7, wantCoverage: 50},
		{name: "consistent uncovered", first: 2, next: 2, executions: 0, wantCoverage: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			profile := fmt.Sprintf("mode: count\n"+
				"example.com/repo/internal/leaf/a.go:1.1,2.2 %d 0\n"+
				"example.com/repo/internal/leaf/a.go:1.1,2.2 %d %d\n"+
				"example.com/repo/internal/leaf/b.go:1.1,2.2 2 0\n",
				tc.first, tc.next, tc.executions)
			got, err := CoverageScores([]byte(profile), "example.com/repo")
			if tc.wantError {
				if got != nil || err == nil || !strings.Contains(err.Error(),
					"line 3: inconsistent statement count for example.com/repo/internal/leaf/a.go:1.1,2.2") {
					t.Fatalf("CoverageScores = %v, %v; want no scores and a conflicting-block error", got, err)
				}
				return
			}
			if err != nil || len(got) != 1 || got["internal/leaf"] != tc.wantCoverage {
				t.Fatalf("CoverageScores = %v, %v; want only internal/leaf=%v", got, err, tc.wantCoverage)
			}
		})
	}
}
