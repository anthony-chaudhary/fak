package modelroute

import (
	"math"
	"strings"
	"testing"
)

// dec builds a one-PICK Decision over the given model ids (an ensemble when >1).
func dec(models ...string) Decision {
	mem := make([]Member, len(models))
	for i, m := range models {
		mem[i] = Member{Model: m}
	}
	return Decision{Plan: Plan{Members: mem}}
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// A cheap single PICK saves vs the frontier, and the saved fraction is identical
// on input and output tokens (the ladder is proportional, so it is blend-free).
func TestSavingsCheapPickIsBlendIndependent(t *testing.T) {
	s := EstimateSavings(dec("small"), nil, "")
	if !s.Estimable || !approx(s.SavedOutFrac, s.SavedInFrac) {
		t.Fatalf("expected blend-independent saving, got in=%v out=%v", s.SavedInFrac, s.SavedOutFrac)
	}
	// small out=1.25 vs frontier out=15 -> 1 - 1.25/15 = 0.91666...
	if want := 1 - 1.25/15; !approx(s.SavedOutFrac, want) {
		t.Fatalf("saved out frac = %v, want %v", s.SavedOutFrac, want)
	}
	if h := s.Headline(); !strings.Contains(h, "cheaper than always-frontier") || !strings.Contains(h, "rough") {
		t.Fatalf("headline missing cheaper/rough framing: %q", h)
	}
}

// Routing to the frontier tier itself is the baseline: ~0% saving, no premium.
func TestSavingsFrontierPickIsBaseline(t *testing.T) {
	s := EstimateSavings(dec("large"), nil, "")
	if !approx(s.SavedOutFrac, 0) {
		t.Fatalf("frontier pick should be baseline (0), got %v", s.SavedOutFrac)
	}
	if h := s.Headline(); !strings.Contains(h, "baseline") {
		t.Fatalf("headline should say baseline: %q", h)
	}
}

// An ensemble runs more compute than one frontier call, so it is a PREMIUM
// (negative saving), reported honestly — never dressed up as a saving.
func TestSavingsEnsembleIsPremiumNotSaving(t *testing.T) {
	// Two unpriced guard models -> each charged at the frontier rate (15) -> 30 vs 15.
	s := EstimateSavings(dec("guard-a", "guard-b"), nil, "")
	if s.SavedOutFrac >= 0 {
		t.Fatalf("two-model ensemble should be a premium (negative), got %v", s.SavedOutFrac)
	}
	if !approx(s.SavedOutFrac, (15-30)/15.0) { // -1.0 (+100%)
		t.Fatalf("premium frac = %v, want -1.0", s.SavedOutFrac)
	}
	h := s.Headline()
	if !strings.Contains(h, "+100% vs one frontier call") || !strings.Contains(h, "deliberate") {
		t.Fatalf("premium headline wrong: %q", h)
	}
	if strings.Contains(h, "cheaper") {
		t.Fatalf("a premium must never read as cheaper: %q", h)
	}
}

// An unpriced member is charged at the conservative frontier rate AND listed so an
// operator knows to supply a real price — fak does not invent a cheap number.
func TestSavingsUnpricedIsConservativeAndListed(t *testing.T) {
	s := EstimateSavings(dec("mystery-7b"), nil, "")
	if len(s.Assumed) != 1 || s.Assumed[0] != "mystery-7b" {
		t.Fatalf("unpriced member not recorded: %v", s.Assumed)
	}
	if !approx(s.RoutedOut, 15) || !approx(s.SavedOutFrac, 0) {
		t.Fatalf("unpriced should be charged at frontier (no claimed saving), got out=%v frac=%v", s.RoutedOut, s.SavedOutFrac)
	}
	if h := s.Headline(); !strings.Contains(h, "unpriced, charged at frontier: mystery-7b") {
		t.Fatalf("headline must disclose the assumption: %q", h)
	}
}

// --prices overrides the book; a real cheap price for a custom model yields a real
// saving (and clears the "assumed" disclosure).
func TestSavingsPricesOverride(t *testing.T) {
	over, err := ParsePrices("mystery-7b=0.2/1")
	if err != nil {
		t.Fatalf("ParsePrices: %v", err)
	}
	book := DefaultPrices().Overlay(over)
	s := EstimateSavings(dec("mystery-7b"), book, "")
	if len(s.Assumed) != 0 {
		t.Fatalf("priced model should not be assumed: %v", s.Assumed)
	}
	if !approx(s.RoutedOut, 1) || !approx(s.SavedOutFrac, 1-1.0/15) {
		t.Fatalf("override price not applied: out=%v frac=%v", s.RoutedOut, s.SavedOutFrac)
	}
}

// A named --frontier baseline reprices the comparison; comparing small to the mid
// tier yields a smaller (still positive) saving than vs the frontier anchor.
func TestSavingsNamedFrontier(t *testing.T) {
	vsFrontier := EstimateSavings(dec("small"), nil, "").SavedOutFrac
	vsMid := EstimateSavings(dec("small"), nil, "medium").SavedOutFrac // baseline out=5
	if !(vsMid > 0 && vsMid < vsFrontier) {
		t.Fatalf("vs mid (%v) should be a smaller positive saving than vs frontier (%v)", vsMid, vsFrontier)
	}
	if !approx(vsMid, 1-1.25/5) {
		t.Fatalf("vs mid frac = %v, want %v", vsMid, 1-1.25/5)
	}
}

// A $0 baseline (all-local book) is honestly "not estimated", never a divide.
func TestSavingsZeroBaselineNotEstimated(t *testing.T) {
	s := EstimateSavings(dec("local"), nil, "local")
	if s.Estimable {
		t.Fatalf("a $0 baseline must not be estimable")
	}
	if h := s.Headline(); !strings.Contains(h, "not estimated") {
		t.Fatalf("headline should decline to estimate: %q", h)
	}
}

func TestParsePricesErrors(t *testing.T) {
	for _, bad := range []string{"small", "=1/2", "small=x", "small=1/y"} {
		if _, err := ParsePrices(bad); err == nil {
			t.Fatalf("ParsePrices(%q) should error", bad)
		}
	}
	if pb, err := ParsePrices("small=0.25/1.25, large=3"); err != nil || len(pb) != 2 {
		t.Fatalf("good spec failed: pb=%v err=%v", pb, err)
	}
}

func TestMoneyFormat(t *testing.T) {
	cases := map[float64]string{15: "15", 1.25: "1.25", 1.2: "1.2", 0: "0", 13.75: "13.75"}
	for v, want := range cases {
		if got := money(v); got != want {
			t.Fatalf("money(%v) = %q, want %q", v, got, want)
		}
	}
}
