package cacheprice

import "testing"

func TestDisaggregationDividend(t *testing.T) {
	cases := []struct {
		name               string
		external, overhead int
		want               int
	}{
		{"remote fetch beat local recompute", 1000, 300, 700},
		{"break-even is zero", 500, 500, 0},
		{"fabric hop cost more than recompute — disaggregation lost", 200, 512, -312},
		{"no external transfer, no overhead", 0, 0, 0},
		{"pure toll with nothing served is fully negative", 0, 128, -128},
		{"negative external clamps to 0 (miscount never invents saving)", -50, 100, -100},
		{"negative overhead clamps to 0 (miscount never invents toll)", 900, -40, 900},
		{"both negative clamp to 0", -10, -10, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := DisaggregationDividend(c.external, c.overhead); got != c.want {
				t.Fatalf("DisaggregationDividend(%d, %d) = %d, want %d", c.external, c.overhead, got, c.want)
			}
		})
	}
}

func TestDisaggregationWorthwhile(t *testing.T) {
	cases := []struct {
		name               string
		external, overhead int
		want               bool
	}{
		{"strictly positive dividend is worthwhile", 1000, 300, true},
		{"break-even is NOT worthwhile (prefer local recompute)", 500, 500, false},
		{"negative dividend is not worthwhile", 200, 512, false},
		{"clamped-negative external is not worthwhile", -50, 100, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := DisaggregationWorthwhile(c.external, c.overhead); got != c.want {
				t.Fatalf("DisaggregationWorthwhile(%d, %d) = %v, want %v", c.external, c.overhead, got, c.want)
			}
		})
	}
}

// TestDisaggregationDividendReconcilesWithAdmission ties the two axes: the external-transfer
// tokens that feed the dividend are a subset of the residency that AdmissionTokens discounts, so
// a turn's dividend can never claim to have saved more than admission booked as resident.
func TestDisaggregationDividendReconcilesWithAdmission(t *testing.T) {
	const prompt, localHit, externalTransfer, overhead = 4000, 1500, 1000, 250
	resident := localHit + externalTransfer
	billable := AdmissionTokens(prompt, resident)
	if billable != prompt-resident {
		t.Fatalf("admission sanity: got %d want %d", billable, prompt-resident)
	}
	// The gross external saving is one recompute per transferred token; it cannot exceed the
	// resident tokens admission already credited.
	if externalTransfer > resident {
		t.Fatalf("fixture invalid: external %d > resident %d", externalTransfer, resident)
	}
	if got, want := DisaggregationDividend(externalTransfer, overhead), externalTransfer-overhead; got != want {
		t.Fatalf("dividend = %d, want %d", got, want)
	}
}
