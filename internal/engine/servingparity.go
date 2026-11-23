package engine

// The single-GPU single-stream serving parity bar against llama.cpp: decode
// throughput, cold prefill throughput at P=2048, and warm multi-turn TTFT. A pure
// value/contract type, so the request path can carry and grade it without importing
// benchmark or harness packages.
//
// Fail-closed is the whole point: an axis with no trustworthy reading grades
// unmeasured, never a fabricated pass, so an unevidenced claim cannot read as parity.

import "math"

// ParityAxis names one dimension the serving parity bar grades. The zero value is the
// empty string, which is NOT a member (Valid() == false): an unset axis is fail-closed,
// never silently graded as some default axis.
type ParityAxis string

const (
	// ParityAxisDecode is steady-state single-stream decode throughput.
	ParityAxisDecode ParityAxis = "decode"
	// ParityAxisPrefill is cold prefill throughput at prompt length 2048.
	ParityAxisPrefill ParityAxis = "prefill"
	// ParityAxisWarmTTFT is time-to-first-token on a warm multi-turn conversation.
	ParityAxisWarmTTFT ParityAxis = "warm-ttft"
)

// ParityAxes returns the graded axes in a stable order. Grading emits exactly one grade
// per entry, so this order is also the report's order.
func ParityAxes() []ParityAxis {
	return []ParityAxis{ParityAxisDecode, ParityAxisPrefill, ParityAxisWarmTTFT}
}

// Valid reports whether a is a member of the closed axis vocabulary.
func (a ParityAxis) Valid() bool {
	switch a {
	case ParityAxisDecode, ParityAxisPrefill, ParityAxisWarmTTFT:
		return true
	default:
		return false
	}
}

// HigherIsBetter reports the ratio polarity: throughput axes want fak/llama.cpp >= bar,
// latency wants <= bar. An invalid axis returns false so a caller cannot assume a
// polarity and grade a value with the wrong comparison.
func (a ParityAxis) HigherIsBetter() bool {
	switch a {
	case ParityAxisDecode, ParityAxisPrefill:
		return true
	default:
		return false
	}
}

// Unit returns the unit a reading on this axis must carry. An invalid axis returns the
// empty ParityUnit (itself invalid) so a unit check against it can only fail closed.
func (a ParityAxis) Unit() ParityUnit {
	switch a {
	case ParityAxisDecode, ParityAxisPrefill:
		return ParityUnitTokensPerSecond
	case ParityAxisWarmTTFT:
		return ParityUnitMilliseconds
	default:
		return ParityUnit("")
	}
}

// ParityUnit is the measurement unit of a reading. It is coupled to the axis so a
// milliseconds latency reading can never be graded as a tokens/s throughput one; a
// mismatched unit grades unmeasured rather than passing a meaningless ratio.
type ParityUnit string

const (
	ParityUnitTokensPerSecond ParityUnit = "tokens/s"
	ParityUnitMilliseconds    ParityUnit = "ms"
)

// ParityUnits returns the closed unit vocabulary in a stable order.
func ParityUnits() []ParityUnit {
	return []ParityUnit{ParityUnitTokensPerSecond, ParityUnitMilliseconds}
}

// Valid reports whether u is a member of the closed unit vocabulary.
func (u ParityUnit) Valid() bool {
	switch u {
	case ParityUnitTokensPerSecond, ParityUnitMilliseconds:
		return true
	default:
		return false
	}
}

// ParityVerdict is the outcome of grading one axis. ParityUnmeasured is the explicit
// fail-closed outcome for a missing or untrustworthy reading — distinct from
// ParityMissed, which means a real reading fell short of the bar.
type ParityVerdict string

const (
	ParityUnmeasured ParityVerdict = "unmeasured"
	ParityMet        ParityVerdict = "met"
	ParityMissed     ParityVerdict = "missed"
)

// ParityVerdicts returns the closed verdict vocabulary in a stable order.
func ParityVerdicts() []ParityVerdict {
	return []ParityVerdict{ParityUnmeasured, ParityMet, ParityMissed}
}

// Valid reports whether v is a member of the closed verdict vocabulary.
func (v ParityVerdict) Valid() bool {
	switch v {
	case ParityUnmeasured, ParityMet, ParityMissed:
		return true
	default:
		return false
	}
}

// ParityRequirement is the bar for one axis: a unit and the fak/llama.cpp ratio the
// axis must reach (throughput) or stay under (latency). The ratio is compared against
// the sample ratio, never against raw values, so one bar works across axes.
type ParityRequirement struct {
	Axis  ParityAxis
	Unit  ParityUnit
	Ratio float64
}

// ParityRequirements returns the bar for every axis in ParityAxes() order.
func ParityRequirements() []ParityRequirement {
	return []ParityRequirement{
		{Axis: ParityAxisDecode, Unit: ParityUnitTokensPerSecond, Ratio: 0.8},
		{Axis: ParityAxisPrefill, Unit: ParityUnitTokensPerSecond, Ratio: 0.5},
		{Axis: ParityAxisWarmTTFT, Unit: ParityUnitMilliseconds, Ratio: 2.0},
	}
}

// ParityRequirementFor returns the bar for a single axis and whether the axis is known.
func ParityRequirementFor(a ParityAxis) (ParityRequirement, bool) {
	for _, r := range ParityRequirements() {
		if r.Axis == a {
			return r, true
		}
	}
	return ParityRequirement{}, false
}

// ParitySample is one measured reading to grade: fak's value against the rival's on a
// single axis. RivalLabel and Evidence are carried into the receipt so a graded report
// names its rival and can be traced back to the run; grading itself uses only the
// numbers, so a caller that omits them still gets a truthful verdict, just a less
// auditable one. Measured must be set explicitly — an unset sample is unmeasured, never
// a zero-valued pass.
type ParitySample struct {
	Axis       ParityAxis
	Unit       ParityUnit
	Fak        float64
	Rival      float64
	Measured   bool
	RivalLabel string
	Evidence   string
}

// ParityGrade is the graded outcome for one axis, carrying the bar it was compared to,
// the observed ratio, the verdict, and a short reason for any non-met outcome.
type ParityGrade struct {
	Requirement ParityRequirement
	Ratio       float64
	Verdict     ParityVerdict
	Reason      string
}

// Met reports whether the axis cleared its bar. Only ParityMet is a pass; unmeasured
// and missed are both not met.
func (g ParityGrade) Met() bool {
	return g.Verdict == ParityMet
}

// ServingParityReport is the graded parity report: exactly one grade per axis once
// graded. A zero-value report has no grades and is neither complete nor all-met.
type ServingParityReport struct {
	Grades []ParityGrade
}

// GradeParity grades every axis in ParityAxes() order, using the first sample whose
// axis matches. A missing axis, or one whose reading is untrustworthy, grades
// unmeasured so an absent measurement can never read as parity. The input slice is not
// mutated.
func GradeParity(samples []ParitySample) ServingParityReport {
	report := ServingParityReport{}
	for _, axis := range ParityAxes() {
		req, _ := ParityRequirementFor(axis)
		grade := ParityGrade{Requirement: req, Verdict: ParityUnmeasured, Reason: "no reading"}

		if s, ok := firstSampleFor(samples, axis); ok {
			grade = gradeSample(axis, req, s)
		}
		report.Grades = append(report.Grades, grade)
	}
	return report
}

// firstSampleFor returns the first sample matching axis. Grading takes the first
// reading so a caller cannot append a second, passing reading to mask a failing one.
func firstSampleFor(samples []ParitySample, axis ParityAxis) (ParitySample, bool) {
	for _, s := range samples {
		if s.Axis == axis {
			return s, true
		}
	}
	return ParitySample{}, false
}

// gradeSample applies the fail-closed trust checks, then the polarity-aware comparison.
// Any untrustworthy reading grades unmeasured rather than producing a ratio.
func gradeSample(axis ParityAxis, req ParityRequirement, s ParitySample) ParityGrade {
	unmeasured := func(reason string) ParityGrade {
		return ParityGrade{Requirement: req, Verdict: ParityUnmeasured, Reason: reason}
	}
	switch {
	case !s.Measured:
		return unmeasured("not measured")
	case s.Unit != axis.Unit():
		return unmeasured("unit mismatch")
	case math.IsNaN(s.Fak) || math.IsInf(s.Fak, 0):
		return unmeasured("non-finite fak reading")
	case math.IsNaN(s.Rival) || math.IsInf(s.Rival, 0):
		return unmeasured("non-finite rival reading")
	case s.Fak <= 0:
		return unmeasured("non-positive fak reading")
	case s.Rival <= 0:
		return unmeasured("non-positive rival reading")
	}

	ratio := s.Fak / s.Rival
	if math.IsInf(ratio, 0) || math.IsNaN(ratio) {
		return unmeasured("non-finite ratio")
	}
	met := ratio >= req.Ratio
	if !axis.HigherIsBetter() {
		met = ratio <= req.Ratio
	}
	if met {
		return ParityGrade{Requirement: req, Ratio: ratio, Verdict: ParityMet}
	}
	reason := "ratio below bar"
	if !axis.HigherIsBetter() {
		reason = "latency ratio above bar"
	}
	return ParityGrade{Requirement: req, Ratio: ratio, Verdict: ParityMissed, Reason: reason}
}

// Grade returns the grade for axis and whether the report holds one.
func (r ServingParityReport) Grade(a ParityAxis) (ParityGrade, bool) {
	for _, g := range r.Grades {
		if g.Requirement.Axis == a {
			return g, true
		}
	}
	return ParityGrade{}, false
}

// AllMet reports whether every axis has a grade and every grade is met. A zero-value
// report is not all-met: absent evidence is not a pass.
func (r ServingParityReport) AllMet() bool {
	for _, axis := range ParityAxes() {
		g, ok := r.Grade(axis)
		if !ok || !g.Met() {
			return false
		}
	}
	return true
}

// Complete reports whether every axis was actually measured (verdict is not
// unmeasured), regardless of whether it cleared its bar. Completeness and passing are
// separate: a complete report may still miss, and only a complete one can be trusted.
func (r ServingParityReport) Complete() bool {
	for _, axis := range ParityAxes() {
		g, ok := r.Grade(axis)
		if !ok || g.Verdict == ParityUnmeasured || !g.Verdict.Valid() {
			return false
		}
	}
	return true
}

// MetAxes returns the met axes in ParityAxes() order; nil for a zero-value report.
func (r ServingParityReport) MetAxes() []ParityAxis {
	return r.axesWithVerdict(ParityMet)
}

// MissedAxes returns the missed axes in ParityAxes() order; nil for a zero-value report.
func (r ServingParityReport) MissedAxes() []ParityAxis {
	return r.axesWithVerdict(ParityMissed)
}

// UnmeasuredAxes returns the unmeasured axes in ParityAxes() order. A missing axis
// counts as unmeasured, so a zero-value report yields all three axes.
func (r ServingParityReport) UnmeasuredAxes() []ParityAxis {
	var axes []ParityAxis
	for _, axis := range ParityAxes() {
		g, ok := r.Grade(axis)
		if !ok || g.Verdict == ParityUnmeasured || !g.Verdict.Valid() {
			axes = append(axes, axis)
		}
	}
	return axes
}

// axesWithVerdict collects the axes whose grade carries verdict, in ParityAxes() order.
// A missing axis is never included: only a present grade can be met or missed.
func (r ServingParityReport) axesWithVerdict(verdict ParityVerdict) []ParityAxis {
	var axes []ParityAxis
	for _, axis := range ParityAxes() {
		if g, ok := r.Grade(axis); ok && g.Verdict == verdict {
			axes = append(axes, axis)
		}
	}
	return axes
}
