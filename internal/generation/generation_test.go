package generation

import "testing"

func TestNormalizeAndLabel(t *testing.T) {
	for _, tc := range []struct{ in, bare, label string }{
		{"gen/now", "now", "gen/now"},
		{" SECOND-NEXT ", "second-next", "gen/second-next"},
		{"custom", "unclassified", ""},
	} {
		if got := Normalize(tc.in); got != tc.bare {
			t.Errorf("Normalize(%q)=%q want %q", tc.in, got, tc.bare)
		}
		if got := Label(tc.in); got != tc.label {
			t.Errorf("Label(%q)=%q want %q", tc.in, got, tc.label)
		}
	}
}
