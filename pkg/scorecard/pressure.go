package scorecard

// The unbounded, dynamic-baseline scoring layer that sits ALONGSIDE the legacy 0-100
// score/grade. A 0-100 grade saturates: once a KPI is "0/100 bad" it can get no worse, so a
// surface twice past its ceiling reads the same as one just over it, and the fold cannot tell
// a small regression from a blowout. This layer measures each KPI as a SIGNED distance from a
// baseline it may move over time, sums those distances with the same weights the composite
// uses, and never clamps -- so the headline keeps discriminating past the point a capped score
// stops. It is the kernel generalization of internal/heavinessscore's `heaviness_pressure`,
// promoted here so every card that folds through Fold gets the same continuous gauge for free.

// DefaultPassLine is the baseline a KPI is measured against when it sets no PassLine of its
// own: 100, the perfect-score anchor (value 1.0). A KPI sitting exactly on the pass-line
// contributes zero pressure and zero slack -- it is neither in debt nor ahead.
const DefaultPassLine = 100.0

// passLineOf returns a KPI's dynamic baseline: its PassLine when set, else DefaultPassLine.
// Zero is treated as "unset" so the common card never fills it in; a card that wants "every
// score passes" (no pressure term) simply leaves the KPI's Score at/above 100, which already
// yields zero deficit, rather than needing a zero baseline.
func passLineOf(k KPI) float64 {
	if k.PassLine != 0 {
		return k.PassLine
	}
	return DefaultPassLine
}

// Deficit is the UNBOUNDED shortfall of score below a moving baseline, in score points, and
// never clamped at the -100 floor a raw 0-100 Score would imply. A score at or above the
// baseline has zero deficit. Because Score itself is unclamped (ValueFromScore never clamps),
// a KPI twice as far past its floor reports twice the deficit -- the discrimination a 0-100
// grade throws away when it saturates at 0.
func Deficit(score, passLine float64) float64 {
	if d := passLine - score; d > 0 {
		return d
	}
	return 0
}

// Surplus is the UNBOUNDED credit of score above the baseline: how far a KPI clears its
// pass-line. Symmetric to Deficit and likewise unclamped, so a score of 130 against a
// pass-line of 100 reports 30 of headroom a capped 100 ceiling would hide.
func Surplus(score, passLine float64) float64 {
	if s := score - passLine; s > 0 {
		return s
	}
	return 0
}

// Pressure is the continuous, unbounded quality-debt headline: the weighted sum of every
// KPI's Deficit against its own (possibly moving) baseline. Zero == every KPI at or above its
// pass-line; it grows without bound as debt deepens, so unlike a 0-100 grade it keeps
// discriminating past the point a clamped score saturates. weights are the SAME Group/Key
// weights Fold applies to the composite mean, so pressure and score weight identically --
// this is the "continuous weighting" the gauge is built on. Lower is lighter.
func Pressure(kpis []KPI, weights map[string]float64) float64 {
	var total float64
	for _, k := range kpis {
		total += weightOf(k, weights) * Deficit(k.Score, passLineOf(k))
	}
	return total
}

// Slack is the weighted sum of every KPI's Surplus -- the unbounded credit side of the same
// ledger. Reported alongside Pressure so a card exposes both the debt below and the room above
// each baseline instead of collapsing them into one clamped number that can only fall to 0.
func Slack(kpis []KPI, weights map[string]float64) float64 {
	var total float64
	for _, k := range kpis {
		total += weightOf(k, weights) * Surplus(k.Score, passLineOf(k))
	}
	return total
}

// weightOf is the Group-then-Key weight lookup shared by the composite mean and the
// pressure/slack sums so all three weight a KPI identically. A KPI with no matching entry
// (or a nil/empty weights map) weighs 1.
func weightOf(k KPI, weights map[string]float64) float64 {
	if len(weights) > 0 {
		if wv, ok := weights[k.Group]; ok {
			return wv
		}
		if wv, ok := weights[k.Key]; ok {
			return wv
		}
	}
	return 1
}
