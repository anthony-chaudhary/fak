package modelroute

import (
	"math"
	"strings"
	"testing"
)

func nearly(t *testing.T, got, want float64, what string) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("%s: got %v, want %v", what, got, want)
	}
}

func TestEffortMultiplierOf_DefaultsAndAliases(t *testing.T) {
	b := DefaultEffortMultipliers()
	// blank and unknown fall back to the medium anchor, never inflate/deflate.
	nearly(t, b.Of(""), 1.0, "blank effort")
	nearly(t, b.Of("   "), 1.0, "whitespace effort")
	nearly(t, b.Of("wat-9000"), 1.0, "unknown effort")
	// known rungs, case/space-insensitive.
	nearly(t, b.Of("medium"), 1.0, "medium anchor")
	nearly(t, b.Of("  HIGH  "), 1.6, "high case/space-insensitive")
	nearly(t, b.Of("Max"), 3.2, "max")
	// ultracode is the xhigh posture (accounts_launch --settings ultracode).
	nearly(t, b.Of("ultracode"), b.Of("xhigh"), "ultracode == xhigh")
}

func TestDefaultEffortMultipliers_MonotoneAroundMediumAnchor(t *testing.T) {
	b := DefaultEffortMultipliers()
	ladder := []string{"minimal", "low", "medium", "high", "xhigh", "max"}
	for i := 1; i < len(ladder); i++ {
		lo, hi := b.Of(ladder[i-1]), b.Of(ladder[i])
		if !(hi > lo) {
			t.Fatalf("effort ladder not strictly increasing at %s(%v) -> %s(%v)", ladder[i-1], lo, ladder[i], hi)
		}
	}
	nearly(t, b.Of("medium"), 1.0, "medium is the 1.0 anchor")
}

func TestParseEffortCosts(t *testing.T) {
	over, err := ParseEffortCosts("high=1.8, xhigh=3")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	nearly(t, over["high"], 1.8, "parsed high")
	nearly(t, over["xhigh"], 3, "parsed xhigh")

	// empty spec => non-nil empty map (mirrors ParsePrices).
	empty, err := ParseEffortCosts("   ")
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("empty spec: map=%v err=%v", empty, err)
	}

	for _, bad := range []string{"high", "high=", "high=abc", "high=0", "high=-2"} {
		if _, err := ParseEffortCosts(bad); err == nil {
			t.Fatalf("expected error for %q, got nil", bad)
		}
	}
}

func TestEffortMultiplier_Overlay(t *testing.T) {
	merged := DefaultEffortMultipliers().Overlay(EffortMultiplier{"high": 9})
	nearly(t, merged.Of("high"), 9, "overlaid high wins")
	nearly(t, merged.Of("low"), 0.7, "un-overlaid low untouched")
	// overlay must not mutate the default book.
	nearly(t, DefaultEffortMultipliers().Of("high"), 1.6, "default book unmutated")
}

func TestEstimateEffortCost_EqualEffortsCancel(t *testing.T) {
	base := EstimateSavings(dec("small"), nil, "")
	e := EstimateEffortCost(dec("small"), nil, "", "high", "high", nil)
	// Equal efforts on both sides => the multiplier cancels; effort-adjusted fraction
	// equals the pure rate fraction exactly.
	nearly(t, e.SavedOutFrac, base.SavedOutFrac, "equal efforts cancel to rate fraction")
	// units are rate x volume on the OUT axis only.
	nearly(t, e.FrontierUnits, base.FrontierOut*1.6, "frontier units = out x mult")
	nearly(t, e.RoutedUnits, base.RoutedOut*1.6, "routed units = out x mult")
	if !e.Estimable {
		t.Fatal("expected estimable")
	}
}

func TestEstimateEffortCost_LowerRoutedEffortCompoundsSaving(t *testing.T) {
	base := EstimateSavings(dec("small"), nil, "")
	// Baseline runs high, routed runs low => the routed plan spends fewer output tokens
	// too, so the saving must exceed the pure-rate saving.
	e := EstimateEffortCost(dec("small"), nil, "", "high", "low", nil)
	if !(e.SavedOutFrac > base.SavedOutFrac) {
		t.Fatalf("lower routed effort should compound saving: got %v, base %v", e.SavedOutFrac, base.SavedOutFrac)
	}
	// IN axis is never scaled by effort — only OUT.
	nearly(t, e.RoutedUnits, base.RoutedOut*0.7, "routed uses OUT rate, not IN")
}

func TestEstimateEffortCost_HigherRoutedEffortErodesSaving(t *testing.T) {
	base := EstimateSavings(dec("small"), nil, "")
	e := EstimateEffortCost(dec("small"), nil, "", "low", "max", nil)
	if !(e.SavedOutFrac < base.SavedOutFrac) {
		t.Fatalf("higher routed effort should erode saving: got %v, base %v", e.SavedOutFrac, base.SavedOutFrac)
	}
}

func TestEstimateEffortCost_ZeroBaselineNotEstimable(t *testing.T) {
	e := EstimateEffortCost(dec("local"), nil, "local", "high", "low", nil)
	if e.Estimable {
		t.Fatal("a $0 baseline rate must not be estimable")
	}
	if !strings.Contains(e.Headline(), "not estimated") {
		t.Fatalf("headline should note not-estimated, got %q", e.Headline())
	}
}

func TestEffortCost_Headline(t *testing.T) {
	// Cheaper route, differing efforts: SAVED reading + effort annotation, tagged not-a-bill.
	saved := EstimateEffortCost(dec("small"), nil, "", "high", "low", nil).Headline()
	if !strings.Contains(saved, "not a bill") || !strings.Contains(saved, "cheaper than always-frontier") {
		t.Fatalf("saved headline unexpected: %q", saved)
	}
	if !strings.Contains(saved, "effort:") {
		t.Fatalf("saved headline should carry effort annotation: %q", saved)
	}
	// Equal efforts: notes the pure-rate saving.
	same := EstimateEffortCost(dec("small"), nil, "", "high", "high", nil).Headline()
	if !strings.Contains(same, "pure rate saving") {
		t.Fatalf("equal-effort headline should note pure rate saving: %q", same)
	}
	// Ensemble premium: PREMIUM reading.
	prem := EstimateEffortCost(dec("frontier", "frontier"), nil, "", "high", "high", nil).Headline()
	if !strings.Contains(prem, "deliberate compute spend") {
		t.Fatalf("premium headline unexpected: %q", prem)
	}
}
