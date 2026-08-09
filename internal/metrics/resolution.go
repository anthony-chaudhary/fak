package metrics

import (
	"errors"
	"math"
)

// resolution.go — the instrument-resolution ceiling every score delta is read
// against (#5667, a leaf of benchmark-portfolio epic #1063).
//
// Before a score is interpreted as system quality, the measurement instrument
// must be shown capable of resolving the threshold the conclusion depends on.
// This file is that check, and it runs BEFORE statistical uncertainty is
// considered: a confidence interval prices sampling noise, this prices the
// OUTPUT GRID. They are different objects and this one comes first — a claim
// smaller than one output quantum is not a measured zero no matter how many
// samples back it.
//
//   - ResolutionVerdict is the CLOSED vocabulary the contract speaks:
//     MEASURABLE / BELOW_RESOLUTION / MIXED / UNMEASURED. A named token, never a
//     bare float — the same discipline Report.GatePrimary ("pass"/"fail") and
//     Report.LiveSeam ("live_seam_unverified") already carry.
//   - ResolveThreshold answers the contract's headline question for one slice:
//     could this population's grid have shown an effect of the required size?
//   - ResolveClaim is the general form, pricing any magnitude a conclusion
//     depends on against the slice's grid.
//   - ResolvePercentile is the same engine applied to a percentile request: an
//     n-sample set exposes only n distinct percentile points, so Hist.pct's P99
//     on a 12-sample arm is the index-10 sample — the value a p91 request also
//     returns — wearing a p99 label.
//   - FoldResolution reduces a set of readouts to the artifact-level verdict and
//     the per-slice floors.
//   - Report.InstrumentResolution is the fold over the A/B report shape, and
//     Report.JSON stamps it into report.json beside the numbers it qualifies.
//
// UNMEASURED IS NOT A MEASURED ZERO. An empty denominator means nothing was
// observed; internal/bench's rate() returns 0 for that case (bench.go:253-258),
// which makes "we never measured this" byte-identical to "we measured this and
// it was zero" in report.json. A downstream comparison then reads an instrument
// artifact as a subject property. Splitting those two states is the defect this
// file exists to kill — and splitting them BOTH ways: an exact zero on a
// populated slice IS a measurement (0 hits out of 1,000 calls is real evidence)
// and must stay MEASURABLE, or the contract would trade one collapse for
// another.
//
// BELOW_RESOLUTION IS NOT ZERO AND NOT FAILURE. It means the instrument could
// not REPRESENT a change of the size the conclusion needs; the effect may well
// be real. Mapping it onto 0.0, onto a gate failure, or onto "no regression"
// reintroduces the exact collapse the contract forbids. Nothing here moves a
// measured value or flips a gate — ComputeGate, the identical-workload guard,
// and every published scalar keep their current semantics. The readout qualifies
// a number; it never rewrites one.
//
// THE FLOOR IS PER SLICE. Populations differ, so each arm derives its own
// quantum from its OWN denominator. A single global floor taken from the
// better-populated arm would pass a naive test while hiding precisely the
// distortion the contract names — a coarse slice's ratio read against a fine
// slice's grid.
//
// FAIL CLOSED. Whenever the denominator or the grid cannot be established, the
// verdict is a refusing one, never MEASURABLE by default: an empty fold refuses
// rather than reporting a vacuous pass, an unnamed metric refuses rather than
// qualifying an anonymous number, and a readout with no claim to price refuses
// rather than assuming zero. A contract that defaults to MEASURABLE on missing
// input is a silent fail-open no test would catch.
//
// The package stays pure — no engine, no kernel import, stdlib arithmetic only
// — so the seam is unit-testable and any bench, scorecard, or gate consumer can
// fold its own slices through it.
//
// Generation intent: gen/now. It qualifies scalars the current `fak bench`
// artifact already publishes, using denominators the report already carries, and
// depends on no future architecture bet.
//   - Promotion evidence (already held): the readout is derivable from
//     Arm.Calls / Hist.Count() with no new instrumentation, the fold is stamped
//     into the emitted report.json, and the committed receipt shows the accepted
//     and refused cases side by side.
//   - Demotion / retirement evidence: if every real bench arm turns out to run
//     denominators large enough that no published scalar ever lands on
//     BELOW_RESOLUTION or UNMEASURED, the per-claim readout earns nothing over a
//     single "min denominator" line on the report, and this should retire down
//     to that one field.
//   - Invalidating assumption: that 1/Denominator is the true output quantum.
//     It holds for the integer-count-over-Calls rates the KPI set publishes, but
//     a metric whose grid is set by something other than its denominator (a
//     bucketed histogram, a truncated token counter, a sampled trace) has a
//     COARSER quantum than this derives, so passing such a metric through
//     ResolveClaim without an explicit Grid understates its floor.

// InstrumentResolutionSchema versions the readout on the wire.
const InstrumentResolutionSchema = "fak.instrument_resolution.v1"

// ResolutionVerdict is the closed vocabulary of instrument-resolution states.
// The four states are mutually exclusive and jointly exhaustive; a caller that
// invents a fifth token has left the contract.
type ResolutionVerdict string

const (
	// ResolutionMeasurable: the slice was observed AND the magnitude the
	// conclusion depends on is at least one output quantum (or is an exact zero
	// on a populated slice, which is a real measurement). The number may be read
	// as a measurement.
	ResolutionMeasurable ResolutionVerdict = "MEASURABLE"
	// ResolutionBelowResolution: the slice was observed, but the magnitude the
	// conclusion depends on is smaller than the smallest non-zero change the
	// instrument can represent. NOT a measured zero — the effect may be real and
	// simply invisible at this grid.
	ResolutionBelowResolution ResolutionVerdict = "BELOW_RESOLUTION"
	// ResolutionMixed: an artifact-level verdict only — the slices straddle the
	// ceiling, so the artifact as a whole supports no single conclusion.
	ResolutionMixed ResolutionVerdict = "MIXED"
	// ResolutionUnmeasured: the evidence the contract needs is absent (no
	// denominator, no grid, no name, or no claim to price). The refusing default.
	ResolutionUnmeasured ResolutionVerdict = "UNMEASURED"
)

// ResolutionVerdicts returns the closed vocabulary in a stable order. Exported
// so a consumer can assert exhaustiveness rather than hard-coding four strings.
func ResolutionVerdicts() []ResolutionVerdict {
	return []ResolutionVerdict{
		ResolutionMeasurable,
		ResolutionBelowResolution,
		ResolutionMixed,
		ResolutionUnmeasured,
	}
}

// ResolutionClaim is one magnitude offered for pricing against a slice's own
// instrument grid.
type ResolutionClaim struct {
	// Metric is the published scalar's name (its report.json key), so the
	// readout can be matched back to the number it qualifies.
	Metric string
	// Slice names the population the grid comes from ("vdso_on", "vdso_off",
	// "delta"). Populations differ, so floors differ.
	Slice string
	// Denominator is the observed population size that sets the grid. <= 0 means
	// nothing was observed: UNMEASURED, never a measured zero.
	Denominator int
	// Grid is an EXPLICIT output quantum in the metric's own units, for a metric
	// whose grid is not 1/Denominator (a percentile tail, a token delta carried
	// in percentage points). <= 0 derives the quantum as 1/Denominator.
	Grid float64
	// Claim is the magnitude the conclusion depends on being visible: a claimed
	// effect, a published level, or a required detection threshold. Its sign is
	// irrelevant to resolution; the magnitude is what the grid must cover.
	Claim float64
	// ClaimKnown reports whether Claim carries a real magnitude. False refuses:
	// an absent claim is not a zero claim.
	ClaimKnown bool
}

// ResolutionReadout is the typed, machine-readable result for one claim: what
// the instrument could resolve, what was claimed, and the verdict that follows.
type ResolutionReadout struct {
	Schema string `json:"schema"`
	Metric string `json:"metric"`
	Slice  string `json:"slice"`
	// Denominator is the population that set the grid, preserved so the verdict
	// can be audited without the producer.
	Denominator int `json:"denominator"`
	// Quantum is the smallest non-zero change this instrument can represent for
	// this slice, in the metric's own units. 0 when no grid could be established.
	Quantum float64 `json:"quantum"`
	// Claim is the priced magnitude (absolute value).
	Claim      float64           `json:"claim"`
	ClaimKnown bool              `json:"claim_known"`
	Verdict    ResolutionVerdict `json:"verdict"`
	Reason     string            `json:"reason"`
}

// Published reports whether this scalar may be read as a measurement. Only
// MEASURABLE qualifies; BELOW_RESOLUTION and UNMEASURED are both refusals, for
// different reasons that must not be merged.
func (r ResolutionReadout) Published() bool { return r.Verdict == ResolutionMeasurable }

// ResolveClaim prices one magnitude against one slice's instrument grid. Pure:
// identical input always yields identical output.
//
// It fails closed at every missing input — an unnamed metric or slice, an empty
// denominator, a non-derivable or non-finite grid, an absent claim and a
// non-finite claim all land on UNMEASURED rather than defaulting to MEASURABLE.
func ResolveClaim(c ResolutionClaim) ResolutionReadout {
	out := ResolutionReadout{
		Schema:      InstrumentResolutionSchema,
		Metric:      c.Metric,
		Slice:       c.Slice,
		Denominator: c.Denominator,
		ClaimKnown:  c.ClaimKnown,
		Verdict:     ResolutionUnmeasured,
	}
	if c.Metric == "" || c.Slice == "" {
		out.Reason = "metric or slice unnamed: a readout that cannot say what it qualifies proves nothing"
		return out
	}
	if c.Denominator <= 0 {
		out.Reason = "denominator " + itoa(int64(c.Denominator)) + ": nothing was observed for this slice, so any scalar it carries is unmeasured, not zero"
		return out
	}

	quantum := c.Grid
	if quantum <= 0 {
		quantum = 1 / float64(c.Denominator)
	}
	if math.IsNaN(quantum) || math.IsInf(quantum, 0) || quantum <= 0 {
		out.Reason = "output grid could not be established for this slice"
		return out
	}
	out.Quantum = quantum

	if !c.ClaimKnown {
		out.Reason = "no claim supplied: an absent magnitude is not a zero magnitude and is not priced against the grid"
		return out
	}
	claim := math.Abs(c.Claim)
	if math.IsNaN(claim) || math.IsInf(claim, 0) {
		out.Reason = "claimed magnitude is not a finite number"
		return out
	}
	out.Claim = claim

	switch {
	case claim == 0:
		// An exact zero on a POPULATED slice is a real measurement: the grid
		// could have shown one quantum and did not. Calling this
		// BELOW_RESOLUTION would trade the unmeasured/zero collapse for an
		// equally wrong zero/invisible collapse.
		out.Verdict = ResolutionMeasurable
		out.Reason = "exact zero on a denominator of " + itoa(int64(c.Denominator)) +
			" whose grid resolves " + ftoa(quantum) + ": a measured zero, not an absent measurement"
	case claim < quantum:
		out.Verdict = ResolutionBelowResolution
		out.Reason = "claim " + ftoa(claim) + " is smaller than one output quantum " + ftoa(quantum) +
			" on a denominator of " + itoa(int64(c.Denominator)) +
			": the instrument cannot represent a change this small (this is NOT a measured zero)"
	default:
		out.Verdict = ResolutionMeasurable
		out.Reason = "claim " + ftoa(claim) + " is at least one output quantum " + ftoa(quantum) +
			" on a denominator of " + itoa(int64(c.Denominator))
	}
	return out
}

// ResolveThreshold is the contract's headline question in one call: could a
// population of this size have shown an effect of the required size at all?
//
// This is the check that must run BEFORE a score is read as system quality. A
// 12-call slice resolves nothing finer than 8.3 points, so a conclusion that
// depends on detecting a 5-point effect there was predetermined by the
// instrument, not by the subject.
func ResolveThreshold(metric, slice string, denominator int, threshold float64) ResolutionReadout {
	return ResolveClaim(ResolutionClaim{
		Metric:      metric,
		Slice:       slice,
		Denominator: denominator,
		Claim:       threshold,
		ClaimKnown:  true,
	})
}

// ResolvePercentile prices a percentile REQUEST against the sample count that
// has to answer it.
//
// Hist.pct indexes a sorted slice at int(p/100*(n-1)) (metrics.go:32-46), so an
// n-sample set exposes exactly n distinct percentile points spaced
// percentileGrid(n) = 100/(n-1) apart. A request for p is representable only
// when its tail mass (100-p) is at least that spacing. Below it the index
// arithmetic hands back the sample a much LOWER percentile would have returned:
// a 12-sample "p99" is the index-10 sample — the 90.9th point — wearing a p99
// label, and p99 only lands on its own grid point from 101 samples up.
//
// This DISCLOSES the ceiling; it never interpolates, clamps, or otherwise
// changes the published percentile, because the existing latency gates read that
// value and moving it would be a different change.
func ResolvePercentile(metric, slice string, sampleCount int, p float64) ResolutionReadout {
	if sampleCount <= 0 {
		return ResolveClaim(ResolutionClaim{Metric: metric, Slice: slice, Denominator: sampleCount})
	}
	if math.IsNaN(p) || p <= 0 || p >= 100 {
		out := ResolveClaim(ResolutionClaim{
			Metric: metric, Slice: slice, Denominator: sampleCount,
			Grid: percentileGrid(sampleCount),
		})
		out.Reason = "percentile " + ftoa(p) + " is outside the open interval (0,100): no tail mass to resolve"
		return out
	}
	return ResolveClaim(ResolutionClaim{
		Metric:      metric,
		Slice:       slice,
		Denominator: sampleCount,
		Grid:        percentileGrid(sampleCount),
		Claim:       100 - p,
		ClaimKnown:  true,
	})
}

// percentileGrid is the spacing, in percentile points, between the distinct
// values an n-sample set can return. A single sample resolves no percentile at
// all, so its grid is the whole scale.
func percentileGrid(sampleCount int) float64 {
	if sampleCount <= 1 {
		return 100
	}
	return 100 / float64(sampleCount-1)
}

// ResolutionFloor is one slice's instrument floor: the population it observed
// and the smallest non-zero rate that population can represent. Emitted per
// slice because slice populations differ, which is the whole point — one report
// carrying a 1,000-call arm and a 6-call arm carries two different floors.
type ResolutionFloor struct {
	Slice       string  `json:"slice"`
	Denominator int     `json:"denominator"`
	RateQuantum float64 `json:"rate_quantum"`
	// Resolvable is false when the slice observed nothing, so no grid exists.
	Resolvable bool `json:"resolvable"`
}

// SliceFloor derives one slice's floor from its own denominator.
func SliceFloor(slice string, denominator int) ResolutionFloor {
	f := ResolutionFloor{Slice: slice, Denominator: denominator}
	if denominator > 0 {
		f.RateQuantum = 1 / float64(denominator)
		f.Resolvable = true
	}
	return f
}

// ResolutionReport is the artifact-level typed result: every per-claim readout,
// the per-slice floors, and the one verdict that says whether the artifact
// supports a conclusion at all.
type ResolutionReport struct {
	Schema  string            `json:"schema"`
	Verdict ResolutionVerdict `json:"verdict"`
	Reason  string            `json:"reason"`
	// Conclusive is false whenever ANY published scalar is unbacked by its
	// instrument. It is the fail-closed bit a consumer reads before treating the
	// artifact as a comparison.
	Conclusive      bool                `json:"conclusive"`
	Measurable      int                 `json:"measurable"`
	BelowResolution int                 `json:"below_resolution"`
	Unmeasured      int                 `json:"unmeasured"`
	Floors          []ResolutionFloor   `json:"floors"`
	Readouts        []ResolutionReadout `json:"readouts"`
}

// ErrResolutionRefused is the sentinel every instrument-resolution refusal
// wraps, so a consumer can branch on the contract rather than on a string.
var ErrResolutionRefused = errors.New("metrics: instrument resolution refused the conclusion")

// Refusal returns nil when the artifact supports a conclusion, and an error
// wrapping ErrResolutionRefused when it does not. This is the "refuse a
// conclusion when the evidence the contract needs is absent" half of the
// contract: a caller that wants to read the artifact as a comparison must check
// it, and an empty artifact refuses rather than passing vacuously.
func (rr ResolutionReport) Refusal() error {
	if rr.Conclusive {
		return nil
	}
	return refusalError{verdict: rr.Verdict, reason: rr.Reason}
}

type refusalError struct {
	verdict ResolutionVerdict
	reason  string
}

func (e refusalError) Error() string {
	return "metrics: instrument resolution refused the conclusion (" + string(e.verdict) + "): " + e.reason
}
func (e refusalError) Unwrap() error { return ErrResolutionRefused }

// FoldResolution reduces per-claim readouts and per-slice floors to the
// artifact verdict.
//
// An EMPTY fold refuses. That is the non-vacuity guard: a report that priced
// nothing must not read as a report that priced everything and found it fine,
// which is exactly how a contract passes vacuously.
func FoldResolution(floors []ResolutionFloor, readouts ...ResolutionReadout) ResolutionReport {
	rr := ResolutionReport{
		Schema:   InstrumentResolutionSchema,
		Verdict:  ResolutionUnmeasured,
		Floors:   floors,
		Readouts: readouts,
	}
	if len(readouts) == 0 {
		rr.Reason = "no readouts: the instrument grid was never established, so this artifact supports no conclusion"
		return rr
	}
	for _, r := range readouts {
		switch r.Verdict {
		case ResolutionMeasurable:
			rr.Measurable++
		case ResolutionBelowResolution:
			rr.BelowResolution++
		default:
			// Anything not in the closed vocabulary counts as unmeasured: an
			// unrecognised token must refuse, never pass.
			rr.Unmeasured++
		}
	}
	switch {
	case rr.Measurable == len(readouts):
		rr.Verdict = ResolutionMeasurable
		rr.Conclusive = true
		rr.Reason = "every published scalar is at or above its slice's instrument floor"
	case rr.Unmeasured == len(readouts):
		rr.Verdict = ResolutionUnmeasured
		rr.Reason = "no slice was observed: every scalar in this artifact is unmeasured, not zero"
	case rr.BelowResolution == len(readouts):
		rr.Verdict = ResolutionBelowResolution
		rr.Reason = "every claim is smaller than its slice's output quantum"
	default:
		rr.Verdict = ResolutionMixed
		rr.Reason = "slices straddle the instrument ceiling (" + itoa(int64(rr.Measurable)) + " measurable, " +
			itoa(int64(rr.BelowResolution)) + " below resolution, " + itoa(int64(rr.Unmeasured)) +
			" unmeasured): the artifact supports no single conclusion"
	}
	return rr
}

// InstrumentResolution folds the A/B report shape into its resolution readout:
// one floor per arm from that arm's OWN Calls, and one readout per published
// scalar priced against the slice that produced it.
//
// The rate KPIs are integer-count-over-Calls (internal/bench/bench.go:221-231),
// so their output grid is genuinely discrete at 1/Calls — a 12-call arm cannot
// represent any change below 8.3 points. TokensPerTask moves in whole tokens, so
// its grid is 1/Calls tokens-per-task. TokenDeltaPct's smallest non-zero step is
// one token of difference, i.e. 100/offTok percentage points.
func (r *Report) InstrumentResolution() ResolutionReport {
	onSlice := sliceName(r.On.Label, "vdso_on")
	floors := []ResolutionFloor{
		SliceFloor(onSlice, r.On.Calls),
		SliceFloor(sliceName(r.Off.Label, "vdso_off"), r.Off.Calls),
	}

	readouts := []ResolutionReadout{
		ResolveClaim(ResolutionClaim{
			Metric: "vdso_hit_rate", Slice: onSlice, Denominator: r.On.Calls,
			Claim: r.KPIs.VDSOHitRate, ClaimKnown: true,
		}),
		ResolveClaim(ResolutionClaim{
			Metric: "preflight_catch_rate", Slice: onSlice, Denominator: r.On.Calls,
			Claim: r.KPIs.PreflightCatchRate, ClaimKnown: true,
		}),
		ResolveClaim(ResolutionClaim{
			Metric: "context_pollution_rate", Slice: onSlice, Denominator: r.On.Calls,
			Claim: r.KPIs.ContextPollutionRate, ClaimKnown: true,
		}),
		ResolveClaim(ResolutionClaim{
			Metric: "tokens_per_task", Slice: onSlice, Denominator: r.On.Calls,
			Claim: r.KPIs.TokensPerTask, ClaimKnown: true,
		}),
		ResolvePercentile("tool_call_p50_ns", onSlice, r.On.Calls, 50),
		ResolvePercentile("tool_call_p99_ns", onSlice, r.On.Calls, 99),
		r.tokenDeltaReadout(),
	}
	return FoldResolution(floors, readouts...)
}

// tokenDeltaReadout prices the soft on-vs-off token delta. Its denominator is
// the OFF arm's token total (the delta's own denominator, bench.go:234-238) and
// its quantum is the one-token step expressed in the percentage points the field
// is published in — so a zero-token off arm refuses instead of publishing the
// 0.00% that bench's `if offTok > 0` guard leaves behind.
func (r *Report) tokenDeltaReadout() ResolutionReadout {
	offTok := r.Off.InTokens + r.Off.OutTokens
	c := ResolutionClaim{
		Metric: "token_delta_pct", Slice: "delta",
		Denominator: int(offTok),
		Claim:       r.TokenDeltaPct, ClaimKnown: true,
	}
	if offTok > 0 {
		c.Grid = 100 / float64(offTok)
	}
	return ResolveClaim(c)
}

// sliceName falls back to a stable label when an arm carries none, so a readout
// never names an empty slice (which would itself refuse).
func sliceName(label, fallback string) string {
	if label == "" {
		return fallback
	}
	return label
}

// ftoa renders a float compactly and deterministically for the reason strings,
// without pulling fmt into this arithmetic path.
func ftoa(f float64) string {
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "+Inf"
	case math.IsInf(f, -1):
		return "-Inf"
	}
	neg := f < 0
	if neg {
		f = -f
	}
	// Six fractional digits: enough to show a 1/1000-call quantum distinctly.
	const scale = 1e6
	units := int64(f)
	frac := int64(math.Round((f - float64(units)) * scale))
	if frac >= int64(scale) {
		units++
		frac -= int64(scale)
	}
	fracStr := itoa(frac)
	for len(fracStr) < 6 {
		fracStr = "0" + fracStr
	}
	// Trim trailing zeros, keeping at least one fractional digit.
	for len(fracStr) > 1 && fracStr[len(fracStr)-1] == '0' {
		fracStr = fracStr[:len(fracStr)-1]
	}
	s := itoa(units) + "." + fracStr
	if neg {
		s = "-" + s
	}
	return s
}
