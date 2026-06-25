package savingsvector

import (
	"math"
	"testing"
)

// TestSelfcheckPasses is the anti-overclaim gate ported whole: over the synthetic
// Report the vector must decompose Net (not inflate), surface local_cpu as measured,
// label gpu_prefill/wall_clock modeled, save 0 on the happy control, and bind by
// profile. A nil error means every hard invariant held.
func TestSelfcheckPasses(t *testing.T) {
	if err := Selfcheck(); err != nil {
		t.Fatalf("Selfcheck reported a violated invariant: %v", err)
	}
}

// TestDollarAxisEqualsNetToTheCent pins the load-bearing invariant: the dollar-axis
// re-projection equals Net.dollars_saved exactly (decompose, never inflate).
func TestDollarAxisEqualsNetToTheCent(t *testing.T) {
	v, err := BuildVector(syntheticReport(), "laptop", nil)
	if err != nil {
		t.Fatalf("BuildVector: %v", err)
	}
	if math.Abs(v.NetDollarsSaved-v.VectorDollarsSaved) >= 1e-9 {
		t.Fatalf("vector dollars %.9f != Net dollars %.9f (inflation bug)",
			v.VectorDollarsSaved, v.NetDollarsSaved)
	}
	if v.NetDollarsSaved != 0.054 {
		t.Fatalf("Net dollars re-projected wrong: got %.9f want 0.054", v.NetDollarsSaved)
	}
}

// TestNoAxisInflatesOrGoesNegative asserts every axis saving is >= 0 and that no
// account carries more than the event allows (the four axes are derived, never
// summed into a fifth claim).
func TestNoAxisInflatesOrGoesNegative(t *testing.T) {
	v, err := BuildVector(syntheticReport(), "laptop", nil)
	if err != nil {
		t.Fatalf("BuildVector: %v", err)
	}
	for _, a := range v.Axes {
		if a.Amount < 0 {
			t.Errorf("axis %s saving is negative: %v", a.Account, a.Amount)
		}
	}
	// local_cpu MEASURED amount = turns x local_serve_ns x (tax-1) / 1e6.
	cpu, ok := axisByAccount(v.Axes, "local_cpu")
	if !ok {
		t.Fatal("local_cpu axis missing")
	}
	want := round(10*362*(SpawnedHookTaxX-1.0)/1e6, 4)
	if cpu.Amount != want {
		t.Errorf("local_cpu amount = %v, want %v", cpu.Amount, want)
	}
	if cpu.Provenance != "measured" {
		t.Errorf("local_cpu provenance = %q, want measured", cpu.Provenance)
	}
	// gpu_prefill MODELED amount = turns x prompt_tokens_per_turn.
	gpu, ok := axisByAccount(v.Axes, "gpu_prefill")
	if !ok {
		t.Fatal("gpu_prefill axis missing")
	}
	if gpu.Amount != float64(10*1200) {
		t.Errorf("gpu_prefill amount = %v, want %v", gpu.Amount, float64(10*1200))
	}
	if gpu.Provenance != "modeled" {
		t.Errorf("gpu_prefill provenance = %q, want modeled", gpu.Provenance)
	}
}

// TestHappyControlSavesZero is the anti-inflation control: zero turns saved => every
// axis is exactly 0 and nothing binds.
func TestHappyControlSavesZero(t *testing.T) {
	v, err := BuildVector(syntheticHappy(), "laptop", nil)
	if err != nil {
		t.Fatalf("BuildVector: %v", err)
	}
	for _, a := range v.Axes {
		if a.Amount != 0 {
			t.Errorf("happy control nonzero on %s: %v", a.Account, a.Amount)
		}
	}
	if v.Binding != "none" {
		t.Errorf("happy control binds an account: %s", v.Binding)
	}
}

// TestBindingFollowsProfile asserts the binding account varies by host shape and is
// never hard-coded to dollars: laptop binds context_window, ci-hook binds local_cpu.
func TestBindingFollowsProfile(t *testing.T) {
	lap, err := BuildVector(syntheticReport(), "laptop", nil)
	if err != nil {
		t.Fatalf("BuildVector laptop: %v", err)
	}
	ci, err := BuildVector(syntheticReport(), "ci-hook", nil)
	if err != nil {
		t.Fatalf("BuildVector ci-hook: %v", err)
	}
	if lap.Binding == ci.Binding {
		t.Fatalf("binding did not vary by profile (laptop=%s, ci=%s)", lap.Binding, ci.Binding)
	}
	if lap.Binding != "context_window" {
		t.Errorf("laptop should bind context_window, got %s", lap.Binding)
	}
	if ci.Binding != "local_cpu" {
		t.Errorf("ci-hook should bind local_cpu, got %s", ci.Binding)
	}
}

// TestPollutionRatePromotesContextAxis asserts a supplied pollution rate flips the
// context_window axis from modeled to measured-rate (the amount is unchanged).
func TestPollutionRatePromotesContextAxis(t *testing.T) {
	rate := 0.37
	v, err := BuildVector(syntheticReport(), "laptop", &rate)
	if err != nil {
		t.Fatalf("BuildVector: %v", err)
	}
	ctx, ok := axisByAccount(v.Axes, "context_window")
	if !ok {
		t.Fatal("context_window axis missing")
	}
	if ctx.Provenance != "measured-rate" {
		t.Errorf("context provenance with rate = %q, want measured-rate", ctx.Provenance)
	}
	if ctx.Amount != 10 {
		t.Errorf("context amount = %v, want 10 (rate must not change the amount)", ctx.Amount)
	}

	// Without a rate, the axis is honestly modeled.
	v2, err := BuildVector(syntheticReport(), "laptop", nil)
	if err != nil {
		t.Fatalf("BuildVector: %v", err)
	}
	ctx2, _ := axisByAccount(v2.Axes, "context_window")
	if ctx2.Provenance != "modeled" {
		t.Errorf("context provenance without rate = %q, want modeled", ctx2.Provenance)
	}
}

// TestUnknownProfileErrors asserts a bad profile is rejected (the CLI exits 2 on it).
func TestUnknownProfileErrors(t *testing.T) {
	if _, err := BuildVector(syntheticReport(), "mainframe", nil); err == nil {
		t.Fatal("expected an error for an unknown profile, got nil")
	}
}

// TestNegativeLocalServeClamps asserts a defensively-clamped negative local_serve_ns
// surfaces the unmeasured local-CPU axis rather than a negative saving.
func TestNegativeLocalServeClamps(t *testing.T) {
	r := syntheticReport()
	r.LocalServeNs = -5
	v, err := BuildVector(r, "laptop", nil)
	if err != nil {
		t.Fatalf("BuildVector: %v", err)
	}
	cpu, _ := axisByAccount(v.Axes, "local_cpu")
	if cpu.Amount != 0 {
		t.Errorf("local_cpu with clamped negative local_serve_ns = %v, want 0", cpu.Amount)
	}
}
