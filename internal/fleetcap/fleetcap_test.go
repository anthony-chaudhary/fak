package fleetcap

import (
	"strings"
	"testing"
)

// TestTableAt400 is the issue #1820 acceptance witness: at the program's 400
// issues/hour target, the required-live-worker count for 5/10/15/30-minute median
// sessions must be the hand-computed Little's-law ceilings. Each expected value is
// derived by hand from L = lambda * W:
//
//	 5 min: 400 *  5/60 =  33.333... -> ceil  34
//	10 min: 400 * 10/60 =  66.666... -> ceil  67
//	15 min: 400 * 15/60 = 100.0      -> ceil 100
//	30 min: 400 * 30/60 = 200.0      -> ceil 200
func TestTableAt400(t *testing.T) {
	const rate = 400.0
	want := map[float64]int{
		5:  34,
		10: 67,
		15: 100,
		30: 200,
	}
	rows := Table(rate)
	if len(rows) != len(want) {
		t.Fatalf("Table(%g) returned %d rows, want %d", rate, len(rows), len(want))
	}
	for _, r := range rows {
		exp, ok := want[r.MedianSessionMinutes]
		if !ok {
			t.Errorf("unexpected session duration in table: %g min", r.MedianSessionMinutes)
			continue
		}
		if r.RequiredWorkers != exp {
			t.Errorf("%g-min session at %g/hr: got %d workers, want %d (exact load %.4f)",
				r.MedianSessionMinutes, rate, r.RequiredWorkers, exp, r.ExactLoad)
		}
	}
}

// TestRequiredWorkersCeil pins the ceil behavior directly through the exported
// function for the four ticket durations plus a couple of edge magnitudes.
func TestRequiredWorkersCeil(t *testing.T) {
	cases := []struct {
		rate    float64
		minutes float64
		want    int
	}{
		{400, 5, 34},   // 33.33 -> 34: a fractional load always rounds UP
		{400, 10, 67},  // 66.67 -> 67
		{400, 15, 100}, // exactly 100, no rounding
		{400, 30, 200}, // exactly 200, no rounding
		{60, 1, 1},     // 60/hr * 1 min = exactly 1 worker
		{1, 1, 1},      // 1/hr * 1 min = 0.01667 -> ceil 1, never round a positive load to 0
	}
	for _, c := range cases {
		got := RequiredWorkers(c.rate, c.minutes)
		if got != c.want {
			t.Errorf("RequiredWorkers(%g/hr, %g min) = %d, want %d", c.rate, c.minutes, got, c.want)
		}
	}
}

// TestMonotonicInSession asserts the core Little's-law shape: at a fixed arrival
// rate, a longer median session can never require FEWER concurrent workers. The
// table is returned in ascending-duration order, so the worker counts must be
// non-decreasing.
func TestMonotonicInSession(t *testing.T) {
	rows := Table(400)
	if len(rows) < 2 {
		t.Fatalf("need at least two rows to test monotonicity, got %d", len(rows))
	}
	for i := 1; i < len(rows); i++ {
		prev, cur := rows[i-1], rows[i]
		if cur.MedianSessionMinutes <= prev.MedianSessionMinutes {
			t.Fatalf("table not in ascending duration order: row %d (%g) <= row %d (%g)",
				i, cur.MedianSessionMinutes, i-1, prev.MedianSessionMinutes)
		}
		if cur.RequiredWorkers < prev.RequiredWorkers {
			t.Errorf("non-monotonic: %g-min needs %d workers but shorter %g-min needs %d",
				cur.MedianSessionMinutes, cur.RequiredWorkers,
				prev.MedianSessionMinutes, prev.RequiredWorkers)
		}
	}
}

// TestNonPositiveInputs documents the fail-safe: a zero/negative/NaN rate or
// duration describes no sustained work and so requires no standing worker.
func TestNonPositiveInputs(t *testing.T) {
	cases := []struct {
		rate    float64
		minutes float64
	}{
		{0, 10},
		{400, 0},
		{-400, 10},
		{400, -10},
	}
	for _, c := range cases {
		if got := RequiredWorkers(c.rate, c.minutes); got != 0 {
			t.Errorf("RequiredWorkers(%g, %g) = %d, want 0", c.rate, c.minutes, got)
		}
	}
}

// TestComputeExactLoad checks the un-rounded load alongside the ceil'd count, so a
// regression that breaks the ratio but happens to ceil to the same integer is
// still caught.
func TestComputeExactLoad(t *testing.T) {
	c := Compute(400, 10)
	if c.RequiredWorkers != 67 {
		t.Errorf("Compute(400,10).RequiredWorkers = %d, want 67", c.RequiredWorkers)
	}
	if c.ExactLoad < 66.6 || c.ExactLoad > 66.7 {
		t.Errorf("Compute(400,10).ExactLoad = %g, want ~66.67", c.ExactLoad)
	}
}

// TestRenderContainsCounts checks the rendered table is wired to the same numbers
// the API returns, so the operator-facing block can't silently drift from Table.
func TestRenderContainsCounts(t *testing.T) {
	out := Render(400)
	for _, want := range []string{"34", "67", "100", "200", "400"} {
		if !strings.Contains(out, want) {
			t.Errorf("Render(400) missing %q\n%s", want, out)
		}
	}
}
