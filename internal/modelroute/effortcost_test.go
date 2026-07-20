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

// TestSaturateEffort clamps a canonical tier to the nearest supported one (#4069): the
// DeepSeek-shaped {high,max} set is the motivating case, plus the neutral "" degradation
// that keeps an unplaceable tier from silently becoming some other tier.
func TestSaturateEffort(t *testing.T) {
	ds := []string{"high", "max"} // DeepSeek's documented two-level set
	for _, tc := range []struct {
		name      string
		canonical string
		supported []string
		want      string
	}{
		// Over-range: xhigh is one rung from BOTH high and max; the tie breaks upward.
		{"xhigh saturates up to max", "xhigh", ds, "max"},
		{"ultracode aliases xhigh", "ultracode", ds, "max"},
		// Under-range: everything below high collapses to the nearest, high.
		{"low saturates up to high", "low", ds, "high"},
		{"medium saturates up to high", "medium", ds, "high"},
		{"minimal saturates up to high", "minimal", ds, "high"},
		{"none saturates up to high", "none", ds, "high"},
		// Already supported: an exact rung is returned unchanged, normalized.
		{"high is already supported", "high", ds, "high"},
		{"max is already supported", "max", ds, "max"},
		{"maximum aliases max", "maximum", ds, "max"},
		{"casing and spacing normalized", "  XHigh  ", ds, "max"},
		// Unplaceable inputs degrade to the neutral "", never to a guess.
		{"blank is neutral", "", ds, ""},
		{"whitespace is neutral", "   ", ds, ""},
		{"unknown tier is neutral (a typo, not a rung)", "extreem", ds, ""},
		{"empty supported set is neutral", "xhigh", nil, ""},
		{"unplaceable supported set is neutral", "xhigh", []string{"wat-9000"}, ""},
		// A supported set fak can only partly place clamps to the placeable rungs.
		{"unplaceable supported entries are skipped", "low", []string{"wat-9000", "max"}, "max"},
		// A single-tier provider always collapses to that tier.
		{"single supported tier absorbs everything", "none", []string{"medium"}, "medium"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := SaturateEffort(tc.canonical, tc.supported); got != tc.want {
				t.Errorf("SaturateEffort(%q, %v) = %q, want %q", tc.canonical, tc.supported, got, tc.want)
			}
		})
	}
}

// TestSaturateEffortIsIdempotent: saturating an already-saturated tier is a no-op, so a
// value cannot drift further up or down each time it crosses the seam.
func TestSaturateEffortIsIdempotent(t *testing.T) {
	ds := []string{"high", "max"}
	for _, in := range []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"} {
		once := SaturateEffort(in, ds)
		if twice := SaturateEffort(once, ds); twice != once {
			t.Errorf("SaturateEffort not idempotent for %q: %q -> %q", in, once, twice)
		}
	}
}

// TestEffortRankAgreesWithMultiplierOrder pins the ORDER lens (effortRank, which decides
// which tier a request saturates onto) against the VOLUME lens (the multipliers, which
// only price it). A rung that ranks higher must never cost less, or the two lenses have
// drifted and "nearest supported tier" would stop meaning what the ladder says.
func TestEffortRankAgreesWithMultiplierOrder(t *testing.T) {
	b := DefaultEffortMultipliers()
	ladder := []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}
	for i := 1; i < len(ladder); i++ {
		loRank, ok := effortRank(ladder[i-1])
		if !ok {
			t.Fatalf("ladder rung %q missing from effortRanks", ladder[i-1])
		}
		hiRank, ok := effortRank(ladder[i])
		if !ok {
			t.Fatalf("ladder rung %q missing from effortRanks", ladder[i])
		}
		if !(hiRank > loRank) {
			t.Fatalf("rank ladder not increasing at %s(%d) -> %s(%d)", ladder[i-1], loRank, ladder[i], hiRank)
		}
		// Multipliers are non-decreasing (none/minimal deliberately share 0.4).
		if lo, hi := b.Of(ladder[i-1]), b.Of(ladder[i]); hi < lo {
			t.Fatalf("rank rises but multiplier falls at %s(%v) -> %s(%v)", ladder[i-1], lo, ladder[i], hi)
		}
	}
	// Every alias the volume lens knows must also be placeable by the order lens,
	// otherwise an alias would saturate to "" while still pricing fine.
	for label := range b {
		if _, ok := effortRank(label); !ok {
			t.Errorf("multiplier label %q has no rank; the two ladders have drifted", label)
		}
	}
}
