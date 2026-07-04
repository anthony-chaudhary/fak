package bench

import "testing"

// Unit 84 (#2679): the DollarPerTask cost model must honor the two-rate table its
// own doc declares — $3/Mtok-in AND $15/Mtok-out. A flat $3/Mtok applied to the
// in+out sum under-prices output 5x (a pure-output workload was charged the input
// rate). This pins the two-rate arithmetic so the output rate cannot silently
// collapse back to the input rate.
func TestDollar_TwoRateModel(t *testing.T) {
	cases := []struct {
		name          string
		inTok, outTok int64
		want          float64
	}{
		{"input-only priced at the $3/Mtok input rate", 1_000_000, 0, 3.0},
		{"output-only priced at the $15/Mtok output rate", 0, 1_000_000, 15.0},
		{"split priced additively (3*in + 15*out)", 1_000_000, 1_000_000, 18.0},
		{"zero tokens cost nothing", 0, 0, 0.0},
	}
	for _, c := range cases {
		if got := dollar(c.inTok, c.outTok); got != c.want {
			t.Errorf("%s: dollar(in=%d,out=%d) = %v, want %v", c.name, c.inTok, c.outTok, got, c.want)
		}
	}
}
