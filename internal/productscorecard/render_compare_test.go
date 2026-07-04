package productscorecard

import (
	"strings"
	"testing"
)

// TestRenderCompareReductionVerdictBoundary pins the integer reduction-verdict
// thresholds in RenderCompare. A ">=Nx reduction" claim means baseline/current >= N,
// i.e. (integer) current <= floor(baseline/N). The ">=2x" sibling already uses the
// correct floor (bd/2); the ">=3x" branch must too, or a ~2.5x drop is over-claimed
// as ">=3x reduction achieved". The existing renderer test only exercises the
// bd=cd=0 payload, so this nonzero boundary is otherwise unwitnessed.
func TestRenderCompareReductionVerdictBoundary(t *testing.T) {
	const achieved3x = "VERDICT: >=3x reduction achieved"
	const only2x = "VERDICT: >=2x (not yet 3x)"
	const not2x = "VERDICT: not yet 2x"

	cases := []struct {
		name    string
		bd, cd  int
		want    string // verdict line the render MUST contain
		notWant string // verdict line the render must NOT contain ("" = no negative check)
	}{
		// 10 -> 4 is a 2.5x drop: >=2x but NOT >=3x. A ceil threshold (ceil(10/3)=4)
		// wrongly admits cd=4 as ">=3x".
		{"two-and-a-half-x is not 3x", 10, 4, only2x, achieved3x},
		// 5 -> 2 is a 2.5x drop: >=2x, not 3x. ceil(5/3)=2 wrongly admits cd=2 as ">=3x".
		{"small non-divisible drop", 5, 2, only2x, achieved3x},
		// 9 -> 3 is exactly 3x: the verdict must STILL say >=3x (no over-correction).
		{"exactly 3x still counts", 9, 3, achieved3x, ""},
		// 10 -> 3 is 3.33x: comfortably >=3x.
		{"more than 3x", 10, 3, achieved3x, ""},
		// 10 -> 6 is 1.67x: not yet 2x.
		{"below 2x", 10, 6, not2x, only2x},
	}

	for _, tc := range cases {
		baseline := map[string]any{"corpus": map[string]any{"product_debt": tc.bd}}
		current := Payload{Corpus: map[string]any{"product_debt": tc.cd}}
		out := RenderCompare(baseline, current)
		if !strings.Contains(out, tc.want) {
			t.Fatalf("%s (bd=%d cd=%d): want verdict %q, got:\n%s", tc.name, tc.bd, tc.cd, tc.want, out)
		}
		if tc.notWant != "" && strings.Contains(out, tc.notWant) {
			t.Fatalf("%s (bd=%d cd=%d): must NOT contain %q, got:\n%s", tc.name, tc.bd, tc.cd, tc.notWant, out)
		}
	}
}
