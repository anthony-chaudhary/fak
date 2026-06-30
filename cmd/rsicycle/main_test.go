package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// TestDecideContract locks the driver's witness -> (decision, exit-code) mapping:
// the keep-bit is non-forgeable, so KEEP requires a STRICT measured gain AND a green
// suite AND a clean truth syscall; every other combination REVERTs (exit 3). These
// are the same control cases the live RSI cycle exercised — a no-op, a regression
// with a huge claimed gain, and a dirty truth syscall all REVERT; only the real
// strict-gain candidate KEEPs.
func TestDecideContract(t *testing.T) {
	cases := []struct {
		name     string
		w        shipgate.Witness
		wantKeep bool
		wantCode int
	}{
		{"strict gain + green + clean -> KEEP",
			shipgate.Witness{Before: 0, After: 1, SuiteGreen: true, TruthClean: true}, true, 0},
		{"no-op (metric unchanged) -> REVERT",
			shipgate.Witness{Before: 1, After: 1, SuiteGreen: true, TruthClean: true}, false, 3},
		{"regression: huge claimed gain but suite RED -> REVERT",
			shipgate.Witness{Before: 0, After: 999, SuiteGreen: false, TruthClean: true}, false, 3},
		{"gain + green but truth NOT clean -> REVERT",
			shipgate.Witness{Before: 0, After: 1, SuiteGreen: true, TruthClean: false}, false, 3},
		{"lower-better strict gain + green + clean -> KEEP",
			shipgate.Witness{Before: 10, After: 5, LowerBetter: true, SuiteGreen: true, TruthClean: true}, true, 0},
		{"lower-better wrong direction -> REVERT",
			shipgate.Witness{Before: 5, After: 10, LowerBetter: true, SuiteGreen: true, TruthClean: true}, false, 3},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, kept, code := decide(c.w)
			if kept != c.wantKeep {
				t.Errorf("kept=%v, want %v (decision=%s)", kept, c.wantKeep, d)
			}
			if code != c.wantCode {
				t.Errorf("exit code=%d, want %d (decision=%s)", code, c.wantCode, d)
			}
			wantKeepDecision := c.wantKeep
			if (d == shipgate.KEEP) != wantKeepDecision {
				t.Errorf("decision=%s inconsistent with wantKeep=%v", d, c.wantKeep)
			}
		})
	}
}

func TestScoreLineReportsDirectionalDelta(t *testing.T) {
	kept := shipgate.Witness{Metric: "hit_rate", Before: 0.2, After: 0.5, SuiteGreen: true, TruthClean: true}
	line := scoreLine(kept, true)
	for _, want := range []string{"score=rsicycle", "grade=kept", "value=0.500", "metric_delta=0.300", "truth_clean=true"} {
		if !strings.Contains(line, want) {
			t.Fatalf("kept score line missing %q:\n%s", want, line)
		}
	}

	noGain := shipgate.Witness{Metric: "latency", Before: 5, After: 7, LowerBetter: true, SuiteGreen: true, TruthClean: true}
	line = scoreLine(noGain, false)
	for _, want := range []string{"grade=no-gain", "metric_delta=-2.000"} {
		if !strings.Contains(line, want) {
			t.Fatalf("lower-better score line missing %q:\n%s", want, line)
		}
	}
}
