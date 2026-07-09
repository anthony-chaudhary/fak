package modelroute

import (
	"fmt"
	"strconv"
	"strings"
)

// effortcost.go — the reasoning-EFFORT lens on a routing Decision, layered on top of
// the per-model rate lens in cost.go (EstimateSavings/Savings).
//
// WHY EFFORT IS A SEPARATE LENS, NEVER A RATE MULTIPLIER. A model's $/Mtok price is
// fixed by the model, not by how hard it is told to think: Opus at `xhigh` and Opus at
// `low` bill the SAME dollars per output token. What reasoning effort changes is the
// number of output (thinking + completion) tokens a task SPENDS — a higher effort emits
// more of them. So effort belongs in a FORWARD estimate as an expected-output-token
// VOLUME multiplier, applied on top of the rate, and it must never be folded into the
// per-Mtok rate itself (that would double-count: the extra tokens are already the whole
// effect). This is the same OBSERVED-vs-modeled discipline the billed lenses keep — an
// OBSERVED billed row already carries effort in its token counts, so only a FORWARD
// projection, which has no token counts yet, needs this multiplier.
//
// The two lenses stay structurally distinct here: EstimateSavings prices the RATE (cost.go)
// and is left untouched; EffortCost scales its OUTPUT axis by a per-side volume multiplier
// and reports a distinct effort-adjusted fraction. The IN (prompt) axis is deliberately
// NOT scaled — reasoning effort grows the completion, not the task's fixed input prompt.
//
// Pure and deterministic: same Decision + same books + same efforts => same numbers, with
// no clock, no I/O. The multipliers are rough and overridable (ParseEffortCosts /
// EffortMultiplier.Overlay), exactly like the rough price ladder in cost.go.

// EffortMultiplier maps a reasoning-effort label to a rough EXPECTED-OUTPUT-TOKEN VOLUME
// multiplier, anchored so that "medium" (the neutral default) == 1.0. A value of 2.0 means
// "at this effort a task emits ~2x the output tokens it would at medium effort" — NOT that
// the token costs twice as much. Unknown/blank efforts resolve to 1.0 (see Of), so an
// un-annotated plan is priced exactly as the rate lens alone would price it.
type EffortMultiplier map[string]float64

// DefaultEffortMultipliers is the built-in rough effort ladder, keyed by the labels fak's
// routing and launch surfaces use (and their common synonyms). The curve is monotone and
// anchored at medium=1.0; the exact values are deliberately rough — override any of them
// with `--effort-cost effort=mult,...` (ParseEffortCosts) the same way `--prices` overrides
// the rate ladder. "ultracode" maps to the xhigh rung because that is what the launcher's
// `--settings '{"ultracode":true}'` posture selects (xhigh reasoning; see
// cmd/fak/accounts_launch.go).
func DefaultEffortMultipliers() EffortMultiplier {
	return EffortMultiplier{
		"none": 0.4, "minimal": 0.4, // barely-reasoning: a fraction of a medium task's output
		"low": 0.7,
		"medium": 1.0, "default": 1.0, "normal": 1.0, // the neutral anchor
		"high":      1.6,
		"xhigh":     2.4, "very-high": 2.4, "ultra": 2.4, "ultracode": 2.4, // xhigh == ultracode posture
		"max": 3.2, "maximum": 3.2,
	}
}

// Of returns the volume multiplier for an effort label, defaulting to 1.0 (the medium
// anchor) for a blank or unrecognized effort — so an un-annotated or novel effort never
// silently inflates or deflates a forward estimate, it just leaves it at the rate lens's
// own numbers. Case- and space-insensitive.
func (b EffortMultiplier) Of(effort string) float64 {
	key := strings.ToLower(strings.TrimSpace(effort))
	if key == "" {
		return 1.0
	}
	if m, ok := b[key]; ok {
		return m
	}
	return 1.0
}

// Overlay returns a copy of b with every entry of over applied on top — the same
// override discipline as PriceBook.Overlay.
func (b EffortMultiplier) Overlay(over EffortMultiplier) EffortMultiplier {
	return overlayMaps(b, over)
}

// ParseEffortCosts reads an --effort-cost spec into an EffortMultiplier overlay:
// comma-separated "effort=mult" pairs (e.g. "high=1.8,xhigh=3"). Fails loud on a
// malformed pair or a non-positive multiplier, mirroring ParsePrices. The caller layers
// the result on top of DefaultEffortMultipliers so a spec need only name the efforts it
// overrides.
func ParseEffortCosts(spec string) (EffortMultiplier, error) {
	return parseBook(spec, "effort-cost", "effort=mult", func(s string) (float64, error) {
		v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil {
			return 0, fmt.Errorf("effort multiplier %q: %w", s, err)
		}
		if v <= 0 {
			return 0, fmt.Errorf("effort multiplier %q: must be > 0", s)
		}
		return v, nil
	})
}

// EffortCost is the effort-adjusted forward cost lens on a routing Decision: it takes the
// rate lens's Savings (cost.go) and scales the OUTPUT axis by the expected-output-token
// volume each side runs at, so a plan that routes to a cheaper model AND at a lower effort
// shows the two savings compounded, while an aspect deliberately run at a HIGHER effort
// than the baseline shows the resulting premium. The IN axis is not scaled (effort grows
// the completion, not the fixed prompt), so only the OUT-side fraction is re-expressed.
//
// When FrontierEffort == RoutedEffort the multipliers cancel and SavedOutFrac equals the
// underlying Savings.SavedOutFrac exactly — effort only ever moves the number when the two
// sides run at DIFFERENT efforts, which is the honest claim: effort is a per-run volume
// knob, not a routing saving on its own.
type EffortCost struct {
	Base           Savings `json:"base"`            // the un-scaled rate lens (cost.go)
	FrontierEffort string  `json:"frontier_effort"` // effort the baseline is assumed to run at
	RoutedEffort   string  `json:"routed_effort"`   // effort the routed plan is assumed to run at
	FrontierMult   float64 `json:"frontier_mult"`   // volume multiplier for FrontierEffort
	RoutedMult     float64 `json:"routed_mult"`     // volume multiplier for RoutedEffort
	// FrontierUnits/RoutedUnits are effort-adjusted OUTPUT cost in units of one
	// medium-effort frontier-out $/Mtok (rate x volume) — comparable to each other, not a
	// literal $/Mtok. SavedOutFrac is their normalized difference.
	FrontierUnits float64 `json:"frontier_units"`
	RoutedUnits   float64 `json:"routed_units"`
	SavedOutFrac  float64 `json:"saved_out_frac"`
	Estimable     bool    `json:"estimable"`
}

// EstimateEffortCost prices a routing Decision through the rate lens (EstimateSavings) and
// then layers the effort-volume adjustment on the OUTPUT axis. book/frontier are the rate
// lens's inputs (nil book => DefaultPrices); eff supplies the effort ladder (nil =>
// DefaultEffortMultipliers); frontierEffort/routedEffort name the effort each side runs at
// (blank => the medium anchor, 1.0). Pure and deterministic.
func EstimateEffortCost(d Decision, book PriceBook, frontier string, frontierEffort, routedEffort string, eff EffortMultiplier) EffortCost {
	if eff == nil {
		eff = DefaultEffortMultipliers()
	}
	base := EstimateSavings(d, book, frontier)
	fMult, rMult := eff.Of(frontierEffort), eff.Of(routedEffort)
	e := EffortCost{
		Base:           base,
		FrontierEffort: frontierEffort,
		RoutedEffort:   routedEffort,
		FrontierMult:   fMult,
		RoutedMult:     rMult,
		FrontierUnits:  base.FrontierOut * fMult,
		RoutedUnits:    base.RoutedOut * rMult,
		Estimable:      base.Estimable,
	}
	if e.FrontierUnits > 0 {
		e.SavedOutFrac = (e.FrontierUnits - e.RoutedUnits) / e.FrontierUnits
	} else {
		e.Estimable = false
	}
	return e
}

// Headline renders the one-line effort-adjusted rough usage note, mirroring
// Savings.Headline's SAVED/PREMIUM/BASELINE reading but on the effort-adjusted output
// fraction. Always tagged rough + overridable so it cannot be misread as a bill; ASCII
// only. When both sides run the same effort it notes that the number is the pure rate
// saving (effort cancelled).
func (e EffortCost) Headline() string {
	const tag = "usage (rough list prices x effort volume, overridable; not a bill): "
	if !e.Estimable {
		return tag + fmt.Sprintf("not estimated (baseline %s has a $0 rate in this price book)", e.Base.Frontier)
	}
	same := effortLabel(e.FrontierEffort) == effortLabel(e.RoutedEffort)
	effortNote := fmt.Sprintf(" [effort: baseline %s x%s vs routed %s x%s]",
		effortLabel(e.FrontierEffort), money(e.FrontierMult), effortLabel(e.RoutedEffort), money(e.RoutedMult))
	if same {
		effortNote = fmt.Sprintf(" [effort %s both sides -- pure rate saving]", effortLabel(e.FrontierEffort))
	}
	frac := e.SavedOutFrac
	var msg string
	switch {
	case frac > 0.005:
		msg = fmt.Sprintf("~%.0f%% cheaper than always-%s -- routed effort-out ~%sx vs %sx baseline units",
			frac*100, e.Base.Frontier, money(e.RoutedUnits), money(e.FrontierUnits))
	case frac < -0.005:
		msg = fmt.Sprintf("+%.0f%% vs one %s call -- effort-out ~%sx vs %sx baseline units (a deliberate compute spend)",
			-frac*100, e.Base.Frontier, money(e.RoutedUnits), money(e.FrontierUnits))
	default:
		msg = fmt.Sprintf("~ the %s baseline -- no effort-adjusted saving on this aspect", e.Base.Frontier)
	}
	return tag + msg + effortNote
}

// effortLabel renders an effort for display, showing "medium" for the blank/default anchor
// so the headline never prints an empty effort.
func effortLabel(effort string) string {
	if strings.TrimSpace(effort) == "" {
		return "medium"
	}
	return strings.ToLower(strings.TrimSpace(effort))
}
