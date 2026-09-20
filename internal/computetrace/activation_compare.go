package computetrace

// activation_compare.go -- first-divergence comparison for bounded activation
// sample traces (#13324).
//
// CompareActivationTraces is a pure function over two Artifacts. It aligns the
// sampled events by stable (layer, token, stage) identity, verifies the
// provenance that makes the comparison meaningful (input and weight digests),
// and reports the FIRST divergent identity plus the maximum sampled error. Any
// condition that makes agreement unprovable -- a dropped sample, a missing
// identity, a duplicate identity, a conflicting provenance, or an incompatible
// shape -- fails closed to INCOMPARABLE rather than being reported as a match.

import "math"

// Verdict is the closed result of a trace comparison.
type Verdict string

const (
	// TraceEqual: every aligned sample matched within tolerance.
	TraceEqual Verdict = "EQUAL"
	// TraceDivergent: an aligned sample exceeded tolerance; Divergent names the
	// first such identity.
	TraceDivergent Verdict = "DIVERGENT"
	// TraceIncomparable: the traces cannot be meaningfully compared (dropped
	// samples, conflicting provenance, missing/duplicate identities, or
	// incompatible shapes). Never treat this as agreement.
	TraceIncomparable Verdict = "INCOMPARABLE"
)

// ActivationTolerance is the caller-declared comparison tolerance. A sample
// value pair (a, b) matches when |a-b| <= Atol + Rtol*|b|.
type ActivationTolerance struct {
	Atol float64
	Rtol float64
}

// ActivationIdentity is the stable identity of one sampled stage within a forward.
type ActivationIdentity struct {
	Layer int    `json:"layer"`
	Token int    `json:"token"`
	Stage string `json:"stage"`
}

// ActivationComparison is the result of CompareActivationTraces.
type ActivationComparison struct {
	Verdict         Verdict            `json:"verdict"`
	ComparedSamples int                `json:"compared_samples"`
	Divergent       ActivationIdentity `json:"divergent,omitempty"`
	DivergentIndex  int                `json:"divergent_index,omitempty"`
	MaxAbsError     float64            `json:"max_abs_error"`
	Reason          string             `json:"reason,omitempty"`
}

func incomparable(reason string) ActivationComparison {
	return ActivationComparison{Verdict: TraceIncomparable, Reason: reason}
}

// CompareActivationTraces compares a reference trace against an actual trace.
//
// It is intentionally strict: it compares only events that carry activation
// samples (SampleValues), requires both traces to be complete (no dropped
// samples), and requires identical provenance and shape for every aligned
// identity. A trace with no sampled events at all is INCOMPARABLE -- a
// sampleless trace is not evidence of agreement.
func CompareActivationTraces(ref, got Artifact, tol ActivationTolerance) ActivationComparison {
	if ref.DroppedSampleValues != 0 || got.DroppedSampleValues != 0 {
		return incomparable("a trace dropped activation samples; agreement cannot be proven")
	}

	refIndex, reason := sampleIndex(ref)
	if reason != "" {
		return incomparable("reference: " + reason)
	}
	gotIndex, reason := sampleIndex(got)
	if reason != "" {
		return incomparable("actual: " + reason)
	}
	if len(refIndex) == 0 {
		return incomparable("no sampled activation events; a sampleless trace proves nothing")
	}
	if len(refIndex) != len(gotIndex) {
		return incomparable("trace event counts differ; sample sets are not aligned")
	}

	res := ActivationComparison{Verdict: TraceEqual}
	// Compare in the reference's stable (sequence) order so "first divergence"
	// is deterministic and matches the producer's execution order.
	for _, key := range refOrder(ref) {
		re := refIndex[key]
		ge, ok := gotIndex[key]
		if !ok {
			return incomparable("actual trace is missing activation identity " + key.String())
		}
		if re.InputDigest != ge.InputDigest || re.WeightDigest != ge.WeightDigest {
			return incomparable("provenance mismatch at " + key.String())
		}
		if !sameShape(re.Shapes, ge.Shapes) || len(re.SampleValues) != len(ge.SampleValues) {
			return incomparable("incompatible sampled shape at " + key.String())
		}
		for i, rv := range re.SampleValues {
			gv := ge.SampleValues[i]
			if len(re.SampleIndices) != 0 && len(ge.SampleIndices) != 0 && re.SampleIndices[i] != ge.SampleIndices[i] {
				return incomparable("sample index mismatch at " + key.String())
			}
			err := sampleError(float64(rv), float64(gv))
			if err > res.MaxAbsError || math.IsNaN(err) {
				res.MaxAbsError = err
			}
			res.ComparedSamples++
			// A non-finite operand is always a divergence: no tolerance can make
			// an Inf/NaN sample agree with the reference. Checking the operands
			// (not just the difference) is required because |Inf - Inf| is NaN
			// and |Inf - x| is Inf, either of which can slip past a permissive
			// limit when the limit itself is Inf.
			if !finite(rv) || !finite(gv) || math.IsNaN(err) || err > tol.Atol+tol.Rtol*math.Abs(float64(gv)) {
				res.Verdict = TraceDivergent
				res.Divergent = key
				res.DivergentIndex = i
				return res
			}
		}
	}
	return res
}

// sampleError returns |a-b|, propagating NaN/Inf as a non-comparable maximum so
// a non-finite sample can never be reported as a small finite error.
// finite reports whether v is neither NaN nor an infinity.
func finite(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func sampleError(a, b float64) float64 {
	d := a - b
	if math.IsNaN(d) {
		return math.NaN()
	}
	return math.Abs(d)
}

func (id ActivationIdentity) String() string {
	return "layer=" + itoa(id.Layer) + " token=" + itoa(id.Token) + " stage=" + id.Stage
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// sampleIndex builds identity -> event for every event that carries samples,
// rejecting duplicate identities. Events without samples are ignored so a mixed
// trace (some stages sampled, some not) is still comparable on the sampled set.
func sampleIndex(a Artifact) (map[ActivationIdentity]Event, string) {
	out := make(map[ActivationIdentity]Event)
	for _, e := range a.Events {
		if len(e.SampleValues) == 0 {
			continue
		}
		key := ActivationIdentity{Layer: e.Layer, Token: e.Token, Stage: stageOf(e)}
		if _, dup := out[key]; dup {
			return nil, "duplicate activation identity " + key.String()
		}
		out[key] = e
	}
	return out, ""
}

// refOrder returns the sampled identities in the reference's execution order.
func refOrder(ref Artifact) []ActivationIdentity {
	order := make([]ActivationIdentity, 0, len(ref.Events))
	for _, e := range ref.Events {
		if len(e.SampleValues) == 0 {
			continue
		}
		order = append(order, ActivationIdentity{Layer: e.Layer, Token: e.Token, Stage: stageOf(e)})
	}
	return order
}

// stageOf derives the stable stage label for an event, preferring the explicit
// Phase and falling back to the operation.
func stageOf(e Event) string {
	if e.Phase != "" {
		return e.Phase
	}
	return e.Operation
}

func sameShape(a, b [][]int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}
