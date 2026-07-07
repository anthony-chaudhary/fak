package modelroute

import "testing"

// TestParseWorkTier pins the string -> WorkTier parse: the canonical T0/T1/T2
// tokens, case/whitespace/backtick tolerance and token extraction from a
// "tier/T1-required" wrapper, and the two rejection modes that make the parser
// safe to build tier tagging on — an OUT-OF-RANGE token (T3+) and a non-tier token
// ("P1", a priority) both return ok=false rather than a silent tier.
func TestParseWorkTier(t *testing.T) {
	cases := []struct {
		in   string
		want WorkTier
		ok   bool
	}{
		{"T0", TierT0, true},
		{"t1", TierT1, true},
		{"T2", TierT2, true},
		{"  T1  ", TierT1, true},
		{"`tier/T1-required`", TierT1, true},
		{"tier/t2-optimal", TierT2, true},
		{"T3", 0, false}, // real token, out of range
		{"t9", 0, false}, // real token, out of range
		{"P1", 0, false}, // a priority, not a tier
		{"priority/P1", 0, false},
		{"", 0, false},
		{"routine", 0, false}, // 't' present but not followed by a digit
	}
	for _, c := range cases {
		got, ok := ParseWorkTier(c.in)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("ParseWorkTier(%q) = (%v, %v), want (%v, %v)", c.in, got, ok, c.want, c.ok)
		}
	}
}
