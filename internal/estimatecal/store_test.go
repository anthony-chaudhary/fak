package estimatecal

import (
	"math"
	"testing"
)

func TestStoreIsolatesProviderAndModelKeys(t *testing.T) {
	var store Store
	for range MinSamples {
		store.Observe("anthropic", "opus", 100, 200)
		store.Observe("anthropic", "sonnet", 100, 150)
		store.Observe("openai", "opus", 100, 75)
	}

	assertRatio(t, &store, "anthropic", "opus", 2)
	assertRatio(t, &store, "anthropic", "sonnet", 1.5)
	assertRatio(t, &store, "openai", "opus", 0.75)
	if _, ok := store.Ratio("openai", "sonnet"); ok {
		t.Fatal("unobserved provider/model pair returned a ratio")
	}
}

func TestRatioRequiresThreeSamples(t *testing.T) {
	var store Store
	for sample := 1; sample < MinSamples; sample++ {
		store.Observe("provider", "model", 100, 125)
		if ratio, ok := store.Ratio("provider", "model"); ok {
			t.Fatalf("sample %d: Ratio() = (%v, true), want false below %d samples", sample, ratio, MinSamples)
		}
	}

	store.Observe("provider", "model", 100, 125)
	assertRatio(t, &store, "provider", "model", 1.25)
}

func TestRatioClampsBothBounds(t *testing.T) {
	tests := []struct {
		name string
		real int
		want float64
	}{
		{name: "lower", real: 1, want: MinRatio},
		{name: "upper", real: 1_000, want: MaxRatio},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var store Store
			for range MinSamples {
				store.Observe("provider", "model", 100, tt.real)
			}
			assertRatio(t, &store, "provider", "model", tt.want)
		})
	}
}

func TestObserveUsesDecayingWeightWithFloor(t *testing.T) {
	var early Store
	early.Observe("provider", "model", 100, 100)
	early.Observe("provider", "model", 100, 300)
	early.Observe("provider", "model", 100, 300)
	assertRatio(t, &early, "provider", "model", 7.0/3.0)

	var established Store
	for range 10 {
		established.Observe("provider", "model", 100, 100)
	}
	established.Observe("provider", "model", 100, 300)
	assertRatio(t, &established, "provider", "model", 1.2)
	established.Observe("provider", "model", 100, 300)
	assertRatio(t, &established, "provider", "model", 1.38)
}

func TestObserveCannotAmplifyItsOwnCorrection(t *testing.T) {
	var store Store
	const rawEstimate = 100
	for range MinSamples {
		store.Observe("anthropic", "opus", rawEstimate, 200)
	}

	ratio, ok := store.Ratio("anthropic", "opus")
	if !ok {
		t.Fatal("Ratio() returned false after minimum samples")
	}
	correctedEstimate := int(float64(rawEstimate) * ratio)
	if correctedEstimate != 200 {
		t.Fatalf("corrected estimate = %d, want 200", correctedEstimate)
	}

	// The billed count can equal the corrected forecast, but Observe still divides
	// it by the raw estimate. An implementation that silently corrects the
	// denominator here would learn 1 and pull its own ratio downward.
	store.Observe("anthropic", "opus", rawEstimate, correctedEstimate)
	assertRatio(t, &store, "anthropic", "opus", 2)
}

func assertRatio(t *testing.T, store *Store, provider, model string, want float64) {
	t.Helper()
	got, ok := store.Ratio(provider, model)
	if !ok {
		t.Fatalf("Ratio(%q, %q) returned false", provider, model)
	}
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("Ratio(%q, %q) = %v, want %v", provider, model, got, want)
	}
}
