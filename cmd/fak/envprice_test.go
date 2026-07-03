package main

import (
	"math"
	"testing"
)

const envPriceTestVar = "FAK_TEST_ENVPRICE_PER_MTOK_USD"

func TestEnvPriceUnsetAndEmpty(t *testing.T) {
	for _, wrapper := range []struct {
		name string
		fn   func(string) (float64, bool)
	}{
		{"cachevalue", cachevaluePriceFromEnv},
		{"spend", spendPriceFromEnv},
	} {
		t.Run(wrapper.name+"/unset", func(t *testing.T) {
			if v, set := wrapper.fn(envPriceTestVar); v != 0 || set {
				t.Fatalf("unset: got (%v, %v), want (0, false)", v, set)
			}
		})
		for _, raw := range []string{"", "   "} {
			t.Run(wrapper.name+"/blank", func(t *testing.T) {
				t.Setenv(envPriceTestVar, raw)
				if v, set := wrapper.fn(envPriceTestVar); v != 0 || set {
					t.Fatalf("blank %q: got (%v, %v), want (0, false)", raw, v, set)
				}
			})
		}
	}
}

func TestEnvPriceValid(t *testing.T) {
	for _, wrapper := range []struct {
		name string
		fn   func(string) (float64, bool)
	}{
		{"cachevalue", cachevaluePriceFromEnv},
		{"spend", spendPriceFromEnv},
	} {
		t.Run(wrapper.name, func(t *testing.T) {
			t.Setenv(envPriceTestVar, " 3.75 ")
			if v, set := wrapper.fn(envPriceTestVar); v != 3.75 || !set {
				t.Fatalf("valid: got (%v, %v), want (3.75, true)", v, set)
			}
			t.Setenv(envPriceTestVar, "0")
			if v, set := wrapper.fn(envPriceTestVar); v != 0 || !set {
				t.Fatalf("zero: got (%v, %v), want (0, true)", v, set)
			}
		})
	}
}

// Negative and unparseable values still count as "set" (an explicit override
// suppresses the default table) but price as 0 in both wrappers.
func TestEnvPriceInvalidStaysSet(t *testing.T) {
	for _, wrapper := range []struct {
		name string
		fn   func(string) (float64, bool)
	}{
		{"cachevalue", cachevaluePriceFromEnv},
		{"spend", spendPriceFromEnv},
	} {
		for _, raw := range []string{"-1.5", "not-a-number", "-Inf"} {
			t.Run(wrapper.name+"/"+raw, func(t *testing.T) {
				t.Setenv(envPriceTestVar, raw)
				if v, set := wrapper.fn(envPriceTestVar); v != 0 || !set {
					t.Fatalf("%q: got (%v, %v), want (0, true)", raw, v, set)
				}
			})
		}
	}
}

// The one intentional divergence: spendPriceFromEnv rejects NaN/+Inf to 0,
// while cachevaluePriceFromEnv passes them through unchanged.
func TestEnvPriceNonFiniteDivergence(t *testing.T) {
	t.Setenv(envPriceTestVar, "NaN")
	if v, set := spendPriceFromEnv(envPriceTestVar); v != 0 || !set {
		t.Fatalf("spend NaN: got (%v, %v), want (0, true)", v, set)
	}
	if v, set := cachevaluePriceFromEnv(envPriceTestVar); !math.IsNaN(v) || !set {
		t.Fatalf("cachevalue NaN: got (%v, %v), want (NaN, true)", v, set)
	}

	t.Setenv(envPriceTestVar, "+Inf")
	if v, set := spendPriceFromEnv(envPriceTestVar); v != 0 || !set {
		t.Fatalf("spend +Inf: got (%v, %v), want (0, true)", v, set)
	}
	if v, set := cachevaluePriceFromEnv(envPriceTestVar); !math.IsInf(v, 1) || !set {
		t.Fatalf("cachevalue +Inf: got (%v, %v), want (+Inf, true)", v, set)
	}
}
