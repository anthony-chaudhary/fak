package computetrace

// activation_profile.go -- a per-(layer, stage) error profile over two bounded
// activation-sample traces.
//
// CompareActivationTraces answers "do these traces agree?" and stops at the
// first divergence. A bring-up against an independent reference needs the whole
// picture instead: how far each layer and stage is from the reference, so error
// growth can be read layer by layer rather than as one pass/fail bit.
// ActivationErrorProfile never short-circuits. It applies the same per-sample
// tolerance rule and the same provenance and shape refusals, but per row, and it
// reports the first divergent (layer, stage) in reference execution order.

import (
	"errors"
	"math"
	"slices"
	"sort"
)

// ActivationSampleIndices is the deterministic logical sample rule every
// activation producer and reference converter shares: n = min(max, width)
// points at j = i*(width/n). It returns nil for a non-positive width or max.
func ActivationSampleIndices(width, max int) []int {
	if width <= 0 || max <= 0 {
		return nil
	}
	n := min(width, max)
	stride := width / n
	idx := make([]int, 0, n)
	for i := 0; i < n; i++ {
		idx = append(idx, i*stride)
	}
	return idx
}

// ActivationStage is one (layer, stage) row identity of an error profile.
type ActivationStage struct {
	Layer int    `json:"layer"`
	Stage string `json:"stage"`
}

// ActivationStageProfile is the aggregated error of one (layer, stage) across
// every token both traces sampled.
//
// MaxRelError is the symmetric relative difference |ref-got| / max(|ref|,|got|)
// (0 when both are 0), bounded in [0, 2] so a near-zero reference cannot report
// an unbounded ratio. A non-finite operand is counted in NonFinite and is always
// a mismatch; it never enters MaxAbsError or MaxRelError, which stay finite and
// JSON-encodable.
type ActivationStageProfile struct {
	Layer           int     `json:"layer"`
	Stage           string  `json:"stage"`
	Tokens          int     `json:"tokens"`
	ComparedSamples int     `json:"compared_samples"`
	ExcludedSamples int     `json:"excluded_samples,omitempty"`
	Mismatches      int     `json:"mismatches"`
	NonFinite       int     `json:"non_finite,omitempty"`
	MissingTokens   int     `json:"missing_tokens,omitempty"`
	Refusals        int     `json:"refusals,omitempty"`
	MaxAbsError     float64 `json:"max_abs_error"`
	MaxRelError     float64 `json:"max_rel_error"`
	Comparable      bool    `json:"comparable"`
	Match           bool    `json:"match"`
	Reason          string  `json:"reason,omitempty"`
}

// ActivationProfile is the result of ActivationErrorProfile.
//
// Verdict is DIVERGENT when any comparable row mismatched (a measured
// divergence stands even beside incomparable rows), INCOMPARABLE when nothing
// diverged but agreement is unproven (a dropped sample, a missing identity, a
// refused row, or no compared sample), and EQUAL otherwise.
type ActivationProfile struct {
	Verdict           Verdict                  `json:"verdict"`
	Rows              []ActivationStageProfile `json:"rows"`
	FirstDivergent    *ActivationStage         `json:"first_divergent,omitempty"`
	ComparedSamples   int                      `json:"compared_samples"`
	MissingIdentities int                      `json:"missing_identities,omitempty"`
	ExtraIdentities   int                      `json:"extra_identities,omitempty"`
	Incomplete        bool                     `json:"incomplete,omitempty"`
	Reason            string                   `json:"reason,omitempty"`
}

// ActivationErrorProfile profiles got against ref per (layer, stage). Unlike
// CompareActivationTraces it never returns early: every aligned identity is
// measured. Rows are ordered by layer, then by the stage's first appearance in
// ref. Within an aligned event pair, samples align by SampleIndices when both
// sides carry them; an index either side lists in ExcludedSampleIndices is
// skipped and counted. Provenance (input/weight digest), shape, or index-set
// disagreement refuses that token for its row (Comparable=false) rather than
// being measured.
func ActivationErrorProfile(ref, got Artifact, tol ActivationTolerance) ActivationProfile {
	var out ActivationProfile
	if ref.DroppedSampleValues != 0 || got.DroppedSampleValues != 0 {
		out.Incomplete = true
		out.Reason = "a trace dropped activation samples; rows cover only retained samples"
	}
	refIdx, reason := sampleIndex(ref)
	if reason != "" {
		return ActivationProfile{Verdict: TraceIncomparable, Reason: "reference: " + reason}
	}
	gotIdx, reason := sampleIndex(got)
	if reason != "" {
		return ActivationProfile{Verdict: TraceIncomparable, Reason: "actual: " + reason}
	}
	if len(refIdx) == 0 {
		return ActivationProfile{Verdict: TraceIncomparable, Reason: "no sampled activation events; a sampleless trace proves nothing"}
	}

	rows := map[ActivationStage]*ActivationStageProfile{}
	stageRank := map[string]int{}
	for _, id := range refOrder(ref) {
		if _, ok := stageRank[id.Stage]; !ok {
			stageRank[id.Stage] = len(stageRank)
		}
		key := ActivationStage{Layer: id.Layer, Stage: id.Stage}
		row := rows[key]
		if row == nil {
			row = &ActivationStageProfile{Layer: id.Layer, Stage: id.Stage}
			rows[key] = row
		}
		ge, ok := gotIdx[id]
		if !ok {
			row.MissingTokens++
			out.MissingIdentities++
			continue
		}
		profilePair(row, refIdx[id], ge, tol)
	}
	for id := range gotIdx {
		if _, ok := refIdx[id]; !ok {
			out.ExtraIdentities++
		}
	}

	out.Rows = make([]ActivationStageProfile, 0, len(rows))
	for _, row := range rows {
		row.Comparable = row.ComparedSamples > 0 && row.Refusals == 0
		row.Match = row.Comparable && row.Mismatches == 0 && row.NonFinite == 0
		out.Rows = append(out.Rows, *row)
		out.ComparedSamples += row.ComparedSamples
	}
	sort.Slice(out.Rows, func(i, j int) bool {
		a, b := out.Rows[i], out.Rows[j]
		if a.Layer != b.Layer {
			return a.Layer < b.Layer
		}
		return stageRank[a.Stage] < stageRank[b.Stage]
	})

	unproven := out.Incomplete || out.MissingIdentities > 0 || out.ComparedSamples == 0
	for i := range out.Rows {
		row := &out.Rows[i]
		if row.ComparedSamples > 0 && (row.Mismatches > 0 || row.NonFinite > 0) && out.FirstDivergent == nil {
			out.FirstDivergent = &ActivationStage{Layer: row.Layer, Stage: row.Stage}
		}
		if !row.Comparable {
			unproven = true
		}
	}
	switch {
	case out.FirstDivergent != nil:
		out.Verdict = TraceDivergent
	case unproven:
		out.Verdict = TraceIncomparable
		if out.Reason == "" {
			out.Reason = "agreement unproven: a missing identity, refused row, or uncompared row"
		}
	default:
		out.Verdict = TraceEqual
	}
	return out
}

// profilePair folds one aligned (layer, token, stage) event pair into its row.
func profilePair(row *ActivationStageProfile, re, ge Event, tol ActivationTolerance) {
	refuse := func(reason string) {
		row.Refusals++
		if row.Reason == "" {
			row.Reason = reason
		}
	}
	if re.InputDigest != ge.InputDigest || re.WeightDigest != ge.WeightDigest {
		refuse("provenance mismatch (input or weight digest)")
		return
	}
	if !sameShape(re.Shapes, ge.Shapes) {
		refuse("incompatible sampled shape")
		return
	}
	refVals, okR := indexedValues(re)
	gotVals, okG := indexedValues(ge)
	if !okR || !okG {
		refuse("sample indices and values are not parallel")
		return
	}
	excluded := map[int]bool{}
	for _, j := range re.ExcludedSampleIndices {
		excluded[j] = true
	}
	for _, j := range ge.ExcludedSampleIndices {
		excluded[j] = true
	}
	keys := make([]int, 0, len(refVals))
	for j := range refVals {
		if excluded[j] {
			continue
		}
		if _, ok := gotVals[j]; !ok {
			refuse("sample index sets differ")
			return
		}
		keys = append(keys, j)
	}
	for j := range gotVals {
		if _, ok := refVals[j]; !ok && !excluded[j] {
			refuse("sample index sets differ")
			return
		}
	}
	slices.Sort(keys)
	row.Tokens++
	for j := range refVals {
		if excluded[j] {
			row.ExcludedSamples++
		}
	}
	for _, j := range keys {
		rv, gv := refVals[j], gotVals[j]
		row.ComparedSamples++
		if !finite(rv) || !finite(gv) {
			row.NonFinite++
			continue
		}
		d := math.Abs(float64(rv) - float64(gv))
		if d > row.MaxAbsError {
			row.MaxAbsError = d
		}
		if den := math.Max(math.Abs(float64(rv)), math.Abs(float64(gv))); den > 0 {
			if rel := d / den; rel > row.MaxRelError {
				row.MaxRelError = rel
			}
		}
		if d > tol.Atol+tol.Rtol*math.Abs(float64(gv)) {
			row.Mismatches++
		}
	}
}

// indexedValues maps logical sample index -> value. An event without indices
// uses positional indices. It reports false when indices are present but not
// parallel to the values, or repeat.
func indexedValues(e Event) (map[int]float32, bool) {
	out := make(map[int]float32, len(e.SampleValues))
	if len(e.SampleIndices) == 0 {
		for i, v := range e.SampleValues {
			out[i] = v
		}
		return out, true
	}
	if len(e.SampleIndices) != len(e.SampleValues) {
		return nil, false
	}
	for i, j := range e.SampleIndices {
		if _, dup := out[j]; dup {
			return nil, false
		}
		out[j] = e.SampleValues[i]
	}
	return out, true
}

// Errors returned by AdoptActivationDigests.
var (
	ErrAdoptTokensMissing  = errors.New("computetrace: a trace does not record its input tokens")
	ErrAdoptTokensMismatch = errors.New("computetrace: the traces consumed different input tokens")
	ErrAdoptDigestsMixed   = errors.New("computetrace: the source trace carries no single input/weight digest pair")
)

// AdoptActivationDigests rewrites every sampled event of ref to carry the
// single input/weight digest pair of src, after proving both traces recorded
// the same InputTokens. It is the documented route for an external reference
// (for example ConvertLlamaCppDump), which cannot compute fak's digests, to
// become comparable: the token check is what the digest would have proven about
// the input. The weight digest is adopted on the operator's assertion that both
// runs loaded the same checkpoint; the comparators themselves are unchanged and
// still refuse any remaining mismatch.
func AdoptActivationDigests(ref *Artifact, src Artifact) error {
	if len(ref.InputTokens) == 0 || len(src.InputTokens) == 0 {
		return ErrAdoptTokensMissing
	}
	if !slices.Equal(ref.InputTokens, src.InputTokens) {
		return ErrAdoptTokensMismatch
	}
	var input, weight string
	for _, e := range src.Events {
		if len(e.SampleValues) == 0 {
			continue
		}
		if e.InputDigest == "" || e.WeightDigest == "" {
			return ErrAdoptDigestsMixed
		}
		if input == "" {
			input, weight = e.InputDigest, e.WeightDigest
			continue
		}
		if e.InputDigest != input || e.WeightDigest != weight {
			return ErrAdoptDigestsMixed
		}
	}
	if input == "" {
		return ErrAdoptDigestsMixed
	}
	for i := range ref.Events {
		if len(ref.Events[i].SampleValues) == 0 {
			continue
		}
		ref.Events[i].InputDigest = input
		ref.Events[i].WeightDigest = weight
	}
	return nil
}

// NonFiniteClampedStatus marks an event whose non-finite samples were clamped.
const NonFiniteClampedStatus = "non_finite_clamped"

// ClampNonFinite makes a trace JSON-encodable: encoding/json refuses NaN and
// Inf, which a diverging bring-up forward produces. Each non-finite sample
// becomes a signed math.MaxFloat32 (NaN becomes +MaxFloat32) and its event's
// Status becomes NonFiniteClampedStatus, so the value still reads as a gross
// divergence. It returns the number of samples clamped.
func (a *Artifact) ClampNonFinite() int {
	n := 0
	for i := range a.Events {
		e := &a.Events[i]
		cloned := false
		for j, v := range e.SampleValues {
			if finite(v) {
				continue
			}
			if !cloned {
				e.SampleValues = append([]float32(nil), e.SampleValues...)
				e.Status = NonFiniteClampedStatus
				cloned = true
			}
			if math.IsInf(float64(v), -1) {
				e.SampleValues[j] = -math.MaxFloat32
			} else {
				e.SampleValues[j] = math.MaxFloat32
			}
			n++
		}
	}
	return n
}
