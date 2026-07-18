package cacheprice

import "testing"

// TestAdmissionTokens locks the residency-discounted admission cost (#3893): the billable
// token count is the prompt minus the resident prefix, clamped so a miscount can never bill
// negative or discount past the prompt. The two anchor cases the issue names — a fully
// resident prefix billing 0 and a cold prompt billing the whole prompt — pin the endpoints;
// the rest guard the clamps.
func TestAdmissionTokens(t *testing.T) {
	cases := []struct {
		name             string
		prompt, resident int
		want             int
	}{
		{"cold prompt: no residency bills the whole prompt", 1000, 0, 1000},
		{"fully resident prefix bills nothing", 1000, 1000, 0},
		{"warm prefix discounts the resident half", 1000, 400, 600},
		{"over-resident witness caps the discount at the prompt", 1000, 5000, 0},
		{"negative residency clamps to a zero discount", 1000, -50, 1000},
		{"empty prompt bills nothing", 0, 1000, 0},
		{"negative prompt bills nothing", -10, 5, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := AdmissionTokens(tc.prompt, tc.resident); got != tc.want {
				t.Fatalf("AdmissionTokens(%d, %d) = %d, want %d", tc.prompt, tc.resident, got, tc.want)
			}
		})
	}

	// The whole point of the discount: a warmer prefix is never MORE expensive to admit than
	// a colder one. As residency grows the billable count only falls, and it bottoms out at 0
	// (a full discount) rather than going negative.
	prev := 1 << 30
	for r := 0; r <= 1200; r += 100 {
		b := AdmissionTokens(1000, r)
		if b > prev {
			t.Fatalf("billable rose (%d -> %d) as residency grew to %d: a warmer prefix must not cost more", prev, b, r)
		}
		if b < 0 {
			t.Fatalf("billable went negative (%d) at residency %d", b, r)
		}
		prev = b
	}
}
