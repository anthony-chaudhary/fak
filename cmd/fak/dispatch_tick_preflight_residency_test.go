package main

import (
	"math"
	"os"
	"testing"
)

// TestDispatchPreflightTurnPricing verifies the vLLM-inspired residency admission discount (#3893).
func TestDispatchPreflightTurnPricing(t *testing.T) {
	const eps = 1e-6

	cases := []struct {
		name             string
		prompt, resident int
		wantBillable     int
		wantDiscount     int
		wantRate         float64
	}{
		{"cold prompt: full billable, zero discount", 1000, 0, 1000, 0, 0.0},
		{"fully resident: zero billable, 100% discount", 1000, 1000, 0, 1000, 1.0},
		{"warm prefix: partial discount", 1000, 400, 600, 400, 0.4},
		{"over-resident: capped at prompt", 1000, 1500, 0, 1000, 1.0},
		{"zero prompt: bills nothing", 0, 500, 0, 0, 0.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := dispatchTurnResidencyPricing(tc.prompt, tc.resident)
			if got := res["billable_tokens"].(int); got != tc.wantBillable {
				t.Fatalf("billable_tokens = %d, want %d", got, tc.wantBillable)
			}
			if got := res["discount_tokens"].(int); got != tc.wantDiscount {
				t.Fatalf("discount_tokens = %d, want %d", got, tc.wantDiscount)
			}
			if rate := res["discount_rate"].(float64); math.Abs(rate-tc.wantRate) > eps {
				t.Fatalf("discount_rate = %v, want %v", rate, tc.wantRate)
			}
		})
	}
}

// TestDispatchPreflightDecorateResidencyPricing verifies that residency pricing is attached
// to preflight output when configured and omitted when unset (#3893).
func TestDispatchPreflightDecorateResidencyPricing(t *testing.T) {
	// When unset, residency_pricing is absent
	os.Unsetenv("FAK_DISPATCH_PROMPT_TOKENS")
	os.Unsetenv("FAK_DISPATCH_RESIDENT_TOKENS")

	pricing := dispatchTurnResidencyDiscountEnv()
	if pricing != nil {
		t.Fatalf("expected nil pricing when env unset, got %+v", pricing)
	}

	// When set, residency_pricing is populated
	os.Setenv("FAK_DISPATCH_PROMPT_TOKENS", "2000")
	os.Setenv("FAK_DISPATCH_RESIDENT_TOKENS", "800")
	defer func() {
		os.Unsetenv("FAK_DISPATCH_PROMPT_TOKENS")
		os.Unsetenv("FAK_DISPATCH_RESIDENT_TOKENS")
	}()

	pricing = dispatchTurnResidencyDiscountEnv()
	if pricing == nil {
		t.Fatalf("expected non-nil pricing when env set")
	}
	if pricing["prompt_tokens"] != 2000 || pricing["resident_tokens"] != 800 {
		t.Fatalf("unexpected prompt/resident tokens: %+v", pricing)
	}
	if pricing["billable_tokens"] != 1200 || pricing["discount_tokens"] != 800 {
		t.Fatalf("unexpected billable/discount tokens: %+v", pricing)
	}
	if pricing["discount_rate"] != 0.4 {
		t.Fatalf("unexpected discount_rate: %v", pricing["discount_rate"])
	}
}
