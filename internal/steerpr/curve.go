package steerpr

// curve.go — issue #5038 (child of epic #5015): carry a bound trajctl objective's
// curve signal + witness rung onto an overlay unit, so an operator reading a
// CLEARED unit can still see that the objective behind it has been DRIFTing. A
// CLEARED band means "the machine confirmed each claim" — it does NOT mean "this
// was worth building". The curve is the only signal that carries direction.
//
// The two axes stay separately legible and this file NEVER folds one into the
// other: Band (steerpr.go) answers "was each claim confirmed"; Curve answers "is
// the objective progressing". Collapsing them makes both illegible, so a DRIFT
// curve is surfaced independently of the band — even when the band is CLEARED.
//
// steerpr is stdlib-only and imports nothing internal (architest tier-1,
// stdlib-only set), so it cannot import internal/trajctl. The curve is carried as
// a PURE MIRROR value type: the CLI maps trajctl.Signal / trajctl.WitnessRung
// onto these verbatim, exactly the way the Band mirrors dispatchtick's supplied
// verdicts. Mirroring — not importing — keeps this leaf a pure VIEW: it DISPLAYS
// what trajctl already folded and never re-scores, nudges, or re-anchors (that
// regime gate belongs to trajctl, out of scope here).

import "fmt"

// CurveSignal mirrors trajctl's closed curve vocabulary (fak-trajctl-curve/1),
// carried as a string so steerpr stays import-free. It is exactly four values; a
// consumer that sees any other string is reading a foreign or newer schema.
type CurveSignal string

const (
	// CurveHealthy: the witnessed progress curve is rising or steady.
	CurveHealthy CurveSignal = "HEALTHY"
	// CurveStall: flat progress while an activity-divergence signal fired.
	CurveStall CurveSignal = "STALL"
	// CurveDrift: the witnessed progress curve is declining — alignment is falling.
	CurveDrift CurveSignal = "DRIFT"
	// CurveDetourOverrun: a detour child ran past its turn budget while its parent
	// is paused — the return-to-main signal.
	CurveDetourOverrun CurveSignal = "DETOUR_OVERRUN"
)

// CurveRung mirrors trajctl.WitnessRung: the height of the evidence behind the
// curve. Doctrine: trust the curve to the height of its rung — a bare W0
// self-report is rendered but never acted on.
type CurveRung string

const (
	RungW3 CurveRung = "W3" // deterministic evidence; may gate automation
	RungW2 CurveRung = "W2" // structured activity evidence
	RungW1 CurveRung = "W1" // judge/rubric verdict
	RungW0 CurveRung = "W0" // self-report; recorded, never actionable alone
)

// Curve is a bound trajctl objective's curve carried onto an overlay unit — the
// objective-progress axis, orthogonal to the unit's attention Band. A unit with
// no bound objective simply has no Curve (a nil *Curve): the common case today,
// and NOT a warning to paper over.
type Curve struct {
	ObjectiveID string      `json:"objective_id"`
	Signal      CurveSignal `json:"signal"`
	Rung        CurveRung   `json:"rung"`
	Latest      float64     `json:"latest"`
	Delta       float64     `json:"delta"`
	Detail      string      `json:"detail,omitempty"`
}

// NeedsAttention reports whether the curve pulls the objective off course —
// DRIFT or DETOUR_OVERRUN. It is TRUE independently of the unit's band, which is
// exactly how a drifting objective becomes visible on a CLEARED unit. A nil curve
// (no bound objective) needs no attention — degrade cleanly, never warn.
func (c *Curve) NeedsAttention() bool {
	if c == nil {
		return false
	}
	return c.Signal == CurveDrift || c.Signal == CurveDetourOverrun
}

// Actionable reports whether the curve may gate an automated action. A bare W0
// self-report never can (trajctl doctrine: trust the curve to the height of its
// rung); an unset rung is treated as not-actionable, because absent evidence
// never clears the bar. W1/W2/W3 carry corroborable evidence and may act.
func (c *Curve) Actionable() bool {
	if c == nil {
		return false
	}
	switch c.Rung {
	case RungW3, RungW2, RungW1:
		return true
	default: // RungW0 or unset
		return false
	}
}

// Annotate renders the curve as one operator-visible line, or "" when there is no
// curve (degrade cleanly: no objective → no line → no warning). The line is
// emitted regardless of the unit's band, so a DRIFT curve shows on a CLEARED
// unit; a NeedsAttention signal is flagged with a leading marker. A curve whose
// rung is W0 (or unset) is rendered but tagged "self-report; not actionable",
// honoring the rung doctrine — the operator sees the signal but is told not to
// act on it.
func (c *Curve) Annotate() string {
	if c == nil {
		return ""
	}
	rung := string(c.Rung)
	if rung == "" {
		rung = "W?"
	}
	if !c.Actionable() {
		rung += " self-report; not actionable"
	}
	marker := "curve"
	if c.NeedsAttention() {
		marker = "⚠ curve"
	}
	line := fmt.Sprintf("%s: %s [%s]", marker, c.Signal, rung)
	if c.Detail != "" {
		line += " — " + c.Detail
	}
	return line
}

// WithCurve returns the unit with curve c bound to it. A zero ObjectiveID clears
// any curve (degrade to curveless). It is a value method returning a copy so the
// pure fold stays free of caller state.
func (u Unit) WithCurve(c Curve) Unit {
	if c.ObjectiveID == "" {
		u.Curve = nil
		return u
	}
	cc := c
	u.Curve = &cc
	return u
}

// AttachCurves binds each unit to the curve returned by lookup, in place. A unit
// whose lookup returns ok=false (or a zero ObjectiveID) is left curveless — the
// common no-objective case, and never a warning. lookup receives the unit so the
// caller can bind by whatever key it owns (a resolved issue, the leaf); steerpr
// stays free of the join key because it stays free of trajctl. The caller is the
// only place that touches both worlds, keeping this leaf a pure VIEW.
func AttachCurves(units []Unit, lookup func(Unit) (Curve, bool)) {
	for i := range units {
		if c, ok := lookup(units[i]); ok && c.ObjectiveID != "" {
			cc := c
			units[i].Curve = &cc
		} else {
			units[i].Curve = nil
		}
	}
}

// DriftHiddenByBand returns the units whose attention band is CLEARED yet whose
// bound curve NeedsAttention (DRIFT / DETOUR_OVERRUN) — the exact "individually
// correct, collectively wrong" set. It is the overlay's acceptance gate made
// legible: if a drifting objective could hide behind a clean band, this set would
// be silently invisible. Naming the set lets the caller render it emphatically
// and lets a test assert a drift on a CLEARED unit is never dropped.
func DriftHiddenByBand(units []Unit) []Unit {
	var out []Unit
	for _, u := range units {
		if u.Band == BandCleared && u.Curve.NeedsAttention() {
			out = append(out, u)
		}
	}
	return out
}
