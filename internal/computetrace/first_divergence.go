package computetrace

// first_divergence.go -- the bring-up frontier: the first (layer, stage) where
// fak leaves a reference, with a full per-layer x stage error profile.
//
// A bring-up probe must report how far the forward agrees with the reference,
// not how far it ran. FirstDivergence measures every aligned (layer, stage) row
// by relative L2 and cosine over all its sampled tokens, never stops at the
// first bad row, and folds the rows into closed per-layer verdicts.
// FirstDivergent.Layer is the frontier a bring-up funnel records; ParityLayers
// counts the layers at parity before it.

import (
	"math"
	"slices"
	"sort"
)

// Row and layer verdict tokens.
const (
	RowMatch        = "MATCH"
	RowDivergent    = "DIVERGENT"
	RowIncomparable = "INCOMPARABLE"
)

// DivergenceTolerance gates a row on whole-row error.
type DivergenceTolerance struct {
	// RelL2 is the largest accepted ||ref-got|| / max(||ref||, ||got||).
	RelL2 float64 `json:"rel_l2"`
	// MinCosine is the smallest accepted cosine similarity; 0 disables it.
	MinCosine float64 `json:"min_cosine,omitempty"`
}

// DivergenceRow is one (layer, stage) of the profile. A measured divergence
// stands even when other tokens of the row were refused; otherwise a refusal,
// a missing token or an uncompared row makes it INCOMPARABLE.
//
// RelL2 is symmetric, ||ref-got|| / max(||ref||, ||got||), so it stays in
// [0, 2] when the reference row is near zero; it is 0 when both rows are zero.
// Cosine is 1 when both rows are zero and 0 when exactly one is.
type DivergenceRow struct {
	Layer           int     `json:"layer"`
	Stage           string  `json:"stage"`
	Verdict         string  `json:"verdict"`
	Tokens          int     `json:"tokens"`
	ComparedSamples int     `json:"compared_samples"`
	ExcludedSamples int     `json:"excluded_samples,omitempty"`
	MissingTokens   int     `json:"missing_tokens,omitempty"`
	Refusals        int     `json:"refusals,omitempty"`
	NonFinite       int     `json:"non_finite,omitempty"`
	RelL2           float64 `json:"rel_l2"`
	Cosine          float64 `json:"cosine"`
	Reason          string  `json:"reason,omitempty"`

	diff2, ref2, got2, dot float64
	refusal                string
}

// DivergenceLayer folds the rows of one layer.
type DivergenceLayer struct {
	Layer          int     `json:"layer"`
	Verdict        string  `json:"verdict"`
	Stages         int     `json:"stages"`
	WorstStage     string  `json:"worst_stage,omitempty"`
	MaxRelL2       float64 `json:"max_rel_l2"`
	MinCosine      float64 `json:"min_cosine"`
	FirstDivergent string  `json:"first_divergent_stage,omitempty"`
}

// DivergenceReport is the result of FirstDivergence.
//
// Verdict is DIVERGENT when any row diverged, INCOMPARABLE when nothing
// diverged but agreement is unproven, and EQUAL otherwise. FirstDivergent is
// the first DIVERGENT row in layer order, then the reference's execution order
// within the layer. ParityLayers is the number of leading MATCH layers 0, 1, 2,
// ... with no gap in numbering, and 0 when a pre-stack layer
// (numbered < 0, such as the embedding) is not MATCH; a stage map's post-stack
// stages (final norm, logits) are numbered one past the last layer, so a fully
// matching n-layer trace that carries them reports n+1. Strict carries
// CompareActivationTraces' element-wise first divergence under the caller's
// atol/rtol; it ignores excluded samples, so it is evidence, not the verdict.
type DivergenceReport struct {
	Verdict        Verdict              `json:"verdict"`
	Tolerance      DivergenceTolerance  `json:"tolerance"`
	FirstDivergent *ActivationStage     `json:"first_divergent,omitempty"`
	ParityLayers   int                  `json:"parity_layers"`
	Layers         []DivergenceLayer    `json:"layers"`
	Rows           []DivergenceRow      `json:"rows"`
	Missing        int                  `json:"missing_identities,omitempty"`
	Extra          int                  `json:"extra_identities,omitempty"`
	Incomplete     bool                 `json:"incomplete,omitempty"`
	Reason         string               `json:"reason,omitempty"`
	Strict         ActivationComparison `json:"strict"`
}

// FirstDivergence profiles got against ref per (layer, stage) and reports the
// first divergent row. It applies the same provenance, shape and index-set
// refusals as ActivationErrorProfile, per row.
func FirstDivergence(ref, got Artifact, tol DivergenceTolerance, strict ActivationTolerance) DivergenceReport {
	out := DivergenceReport{Tolerance: tol, Strict: CompareActivationTraces(ref, got, strict)}
	if !validDivergenceTolerance(tol) {
		out.Verdict, out.Reason = TraceIncomparable, "rel_l2 and min_cosine must be finite, rel_l2 >= 0 and min_cosine <= 1"
		return out
	}
	if ref.Dropped != 0 || got.Dropped != 0 || ref.DroppedSampleValues != 0 || got.DroppedSampleValues != 0 {
		out.Incomplete = true
		out.Reason = "a trace dropped events or activation samples; rows cover only retained samples"
	}
	refIdx, reason := sampleIndex(ref)
	if reason != "" {
		out.Verdict, out.Reason = TraceIncomparable, "reference: "+reason
		return out
	}
	gotIdx, reason := sampleIndex(got)
	if reason != "" {
		out.Verdict, out.Reason = TraceIncomparable, "actual: "+reason
		return out
	}
	if len(refIdx) == 0 {
		out.Verdict, out.Reason = TraceIncomparable, "no sampled activation events; a sampleless trace proves nothing"
		return out
	}

	rows := map[ActivationStage]*DivergenceRow{}
	firstSeen := map[ActivationStage]int{}
	for _, id := range refOrder(ref) {
		key := ActivationStage{Layer: id.Layer, Stage: id.Stage}
		row := rows[key]
		if row == nil {
			row = &DivergenceRow{Layer: id.Layer, Stage: id.Stage}
			rows[key] = row
			firstSeen[key] = len(firstSeen)
		}
		ge, ok := gotIdx[id]
		if !ok {
			row.MissingTokens++
			out.Missing++
			continue
		}
		divergencePair(row, refIdx[id], ge)
	}
	for id := range gotIdx {
		if _, ok := refIdx[id]; !ok {
			out.Extra++
		}
	}

	out.Rows = make([]DivergenceRow, 0, len(rows))
	for _, row := range rows {
		finishDivergenceRow(row, tol)
		out.Rows = append(out.Rows, *row)
	}
	sort.Slice(out.Rows, func(i, j int) bool {
		a, b := out.Rows[i], out.Rows[j]
		if a.Layer != b.Layer {
			return a.Layer < b.Layer
		}
		return firstSeen[ActivationStage{Layer: a.Layer, Stage: a.Stage}] < firstSeen[ActivationStage{Layer: b.Layer, Stage: b.Stage}]
	})

	unproven := out.Incomplete || out.Missing > 0 || out.Extra > 0
	for i := range out.Rows {
		row := &out.Rows[i]
		switch row.Verdict {
		case RowDivergent:
			if out.FirstDivergent == nil {
				out.FirstDivergent = &ActivationStage{Layer: row.Layer, Stage: row.Stage}
			}
		case RowIncomparable:
			unproven = true
		}
		if n := len(out.Layers); n == 0 || out.Layers[n-1].Layer != row.Layer {
			out.Layers = append(out.Layers, DivergenceLayer{Layer: row.Layer, Verdict: RowMatch, MinCosine: 1})
		}
		foldDivergenceLayer(&out.Layers[len(out.Layers)-1], row)
	}
	for _, l := range out.Layers {
		if l.Verdict != RowMatch || (l.Layer >= 0 && l.Layer != out.ParityLayers) {
			break
		}
		if l.Layer >= 0 {
			out.ParityLayers++
		}
	}
	switch {
	case out.FirstDivergent != nil:
		out.Verdict = TraceDivergent
	case unproven:
		out.Verdict = TraceIncomparable
		if out.Reason == "" {
			out.Reason = "agreement unproven: a missing or extra identity, refused row, or uncompared row"
		}
	default:
		out.Verdict = TraceEqual
	}
	return out
}

func validDivergenceTolerance(t DivergenceTolerance) bool {
	ok := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	return ok(t.RelL2) && ok(t.MinCosine) && t.RelL2 >= 0 && t.MinCosine <= 1
}

// divergencePair folds one aligned (layer, token, stage) event pair into row.
func divergencePair(row *DivergenceRow, re, ge Event) {
	refuse := func(reason string) {
		row.Refusals++
		if row.refusal == "" {
			row.refusal = reason
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
	if re.Status == NonFiniteClampedStatus || ge.Status == NonFiniteClampedStatus {
		row.Tokens++
		row.NonFinite++
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
	for j := range gotVals {
		if _, ok := refVals[j]; !ok && !excluded[j] {
			refuse("sample index sets differ")
			return
		}
	}
	for j := range refVals {
		if _, ok := gotVals[j]; !ok && !excluded[j] {
			refuse("sample index sets differ")
			return
		}
	}
	keys := make([]int, 0, len(refVals))
	for j := range refVals {
		keys = append(keys, j)
	}
	slices.Sort(keys)
	row.Tokens++
	for _, j := range keys {
		if excluded[j] {
			row.ExcludedSamples++
			continue
		}
		rv, gv := refVals[j], gotVals[j]
		row.ComparedSamples++
		if !finite(rv) || !finite(gv) {
			row.NonFinite++
			continue
		}
		r, g := float64(rv), float64(gv)
		row.diff2 += (r - g) * (r - g)
		row.ref2 += r * r
		row.got2 += g * g
		row.dot += r * g
	}
}

func finishDivergenceRow(row *DivergenceRow, tol DivergenceTolerance) {
	rn, gn := math.Sqrt(row.ref2), math.Sqrt(row.got2)
	if den := math.Max(rn, gn); den > 0 {
		row.RelL2 = math.Sqrt(row.diff2) / den
	}
	switch {
	case rn == 0 && gn == 0:
		row.Cosine = 1
	case rn == 0 || gn == 0:
		row.Cosine = 0
	default:
		row.Cosine = math.Max(-1, math.Min(1, row.dot/(rn*gn)))
	}
	switch {
	case row.NonFinite > 0:
		row.Verdict, row.Reason = RowDivergent, "non-finite sample"
	case row.ComparedSamples > 0 && row.RelL2 > tol.RelL2:
		row.Verdict, row.Reason = RowDivergent, "rel_l2 above tolerance"
	case row.ComparedSamples > 0 && tol.MinCosine > 0 && row.Cosine < tol.MinCosine:
		row.Verdict, row.Reason = RowDivergent, "cosine below tolerance"
	case row.Refusals > 0:
		row.Verdict, row.Reason = RowIncomparable, row.refusal
	case row.ComparedSamples == 0:
		row.Verdict, row.Reason = RowIncomparable, "no compared sample"
	case row.MissingTokens > 0:
		row.Verdict, row.Reason = RowIncomparable, "reference tokens missing from the actual trace"
	default:
		row.Verdict = RowMatch
	}
}

func foldDivergenceLayer(l *DivergenceLayer, row *DivergenceRow) {
	l.Stages++
	if row.RelL2 >= l.MaxRelL2 && (l.WorstStage == "" || row.RelL2 > l.MaxRelL2) {
		l.MaxRelL2, l.WorstStage = row.RelL2, row.Stage
	}
	l.MinCosine = math.Min(l.MinCosine, row.Cosine)
	switch row.Verdict {
	case RowDivergent:
		if l.FirstDivergent == "" {
			l.FirstDivergent = row.Stage
		}
		l.Verdict = RowDivergent
	case RowIncomparable:
		if l.Verdict == RowMatch {
			l.Verdict = RowIncomparable
		}
	}
}
