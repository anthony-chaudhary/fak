package generation

import "testing"

func TestOrder(t *testing.T) {
	want := []string{"now", "next", "second-next", "future"}
	got := Order()
	if len(got) != len(want) {
		t.Fatalf("Order() len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Order()[%d]=%q want %q", i, got[i], want[i])
		}
	}

	// Verify slice copy isolation
	got[0] = "mutated"
	fresh := Order()
	if fresh[0] != "now" {
		t.Errorf("Order() did not return an isolated copy")
	}
}

func TestNormalizeAndLabel(t *testing.T) {
	for _, tc := range []struct{ in, bare, label string }{
		{"gen/now", "now", "gen/now"},
		{"now", "now", "gen/now"},
		{"gen/next", "next", "gen/next"},
		{"next", "next", "gen/next"},
		{" SECOND-NEXT ", "second-next", "gen/second-next"},
		{"gen/second-next", "second-next", "gen/second-next"},
		{"FUTURE", "future", "gen/future"},
		{"gen/future", "future", "gen/future"},
		{"custom", "unclassified", ""},
		{"", "unclassified", ""},
		{"gen/unknown", "unclassified", ""},
		{"   ", "unclassified", ""},
	} {
		if got := Normalize(tc.in); got != tc.bare {
			t.Errorf("Normalize(%q)=%q want %q", tc.in, got, tc.bare)
		}
		if got := Label(tc.in); got != tc.label {
			t.Errorf("Label(%q)=%q want %q", tc.in, got, tc.label)
		}
	}
}
