// SPDX-License-Identifier: Apache-2.0
// Golden acceptance-length gate adapted from SemiAnalysisAI/InferenceX
// (Apache-2.0), inferencex-e2e/infx/bench/al_gate.py and
// inferencex-e2e/infx/golden_al_distribution/README.md, revision

package benchscore

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// The ratio band a realized acceptance length must fall in to match its golden
// curve. Both ends are inclusive.
const (
	GoldenALMinRatio = 0.95
	GoldenALMaxRatio = 1.05
)

// ErrGoldenALCurve marks a curve identity or curve body that cannot be used to
// judge a run.
var ErrGoldenALCurve = errors.New("benchscore: invalid golden AL curve")

// GoldenALCurveID is the standardized identity of one golden curve. Two runs
// are comparable only when all five fields agree.
type GoldenALCurveID struct {
	Model       string
	Method      string
	Draft       string
	DraftSample string
	Verify      string
}

const goldenALFieldSep = "__"

// Stem renders the identity as the curve's file stem:
// <model>__<method>__<draft>__<draft_sample>-<verify>.
func (id GoldenALCurveID) Stem() string {
	return strings.Join([]string{id.Model, id.Method, id.Draft, id.DraftSample + "-" + id.Verify}, goldenALFieldSep)
}

// Validate refuses an identity whose stem would not parse back to itself.
func (id GoldenALCurveID) Validate() error {
	for _, f := range []struct{ name, value string }{
		{"model", id.Model}, {"method", id.Method}, {"draft", id.Draft},
		{"draft_sample", id.DraftSample}, {"verify", id.Verify},
	} {
		if f.value == "" {
			return fmt.Errorf("%w: empty %s", ErrGoldenALCurve, f.name)
		}
		if strings.Contains(f.value, goldenALFieldSep) || strings.ContainsAny(f.value, " \t\r\n/\\") {
			return fmt.Errorf("%w: %s %q is not a stem field", ErrGoldenALCurve, f.name, f.value)
		}
	}
	// The sampler and verifier share one stem field split on its last hyphen.
	if strings.Contains(id.Verify, "-") {
		return fmt.Errorf("%w: verify %q contains a hyphen", ErrGoldenALCurve, id.Verify)
	}
	// A field that ends with an underscore merges with the separator after it
	// and splits back at a different place.
	if parts := strings.Split(id.Stem(), goldenALFieldSep); len(parts) != 4 ||
		parts[0] != id.Model || parts[1] != id.Method || parts[2] != id.Draft {
		return fmt.Errorf("%w: stem %q does not split back into its fields", ErrGoldenALCurve, id.Stem())
	}
	return nil
}

// ParseGoldenALCurveStem is the inverse of Stem.
func ParseGoldenALCurveStem(stem string) (GoldenALCurveID, error) {
	parts := strings.Split(stem, goldenALFieldSep)
	if len(parts) != 4 {
		return GoldenALCurveID{}, fmt.Errorf("%w: stem %q has %d fields, want 4", ErrGoldenALCurve, stem, len(parts))
	}
	cut := strings.LastIndex(parts[3], "-")
	if cut < 0 {
		return GoldenALCurveID{}, fmt.Errorf("%w: stem %q has no <draft_sample>-<verify> field", ErrGoldenALCurve, stem)
	}
	id := GoldenALCurveID{
		Model:       parts[0],
		Method:      parts[1],
		Draft:       parts[2],
		DraftSample: parts[3][:cut],
		Verify:      parts[3][cut+1:],
	}
	if err := id.Validate(); err != nil {
		return GoldenALCurveID{}, err
	}
	return id, nil
}

// GoldenALCurve is one committed curve: the golden mean acceptance length per
// thinking mode and draft length, plus the run that produced it.
type GoldenALCurve struct {
	ID GoldenALCurveID
	// SourceRun is the provenance key: the collection run the values came from.
	SourceRun string
	// Acceptance maps thinking mode -> draft tokens -> golden acceptance length.
	Acceptance map[string]map[int]float64
}

// Validate refuses a curve without provenance or with an acceptance length
// outside [1, draft tokens + 1], the range one verification step can produce.
func (c GoldenALCurve) Validate() error {
	if err := c.ID.Validate(); err != nil {
		return err
	}
	if strings.TrimSpace(c.SourceRun) == "" {
		return fmt.Errorf("%w: %s has no source_run", ErrGoldenALCurve, c.ID.Stem())
	}
	if len(c.Acceptance) == 0 {
		return fmt.Errorf("%w: %s has no acceptance values", ErrGoldenALCurve, c.ID.Stem())
	}
	for mode, byTokens := range c.Acceptance {
		if len(byTokens) == 0 {
			return fmt.Errorf("%w: %s mode %q has no acceptance values", ErrGoldenALCurve, c.ID.Stem(), mode)
		}
		for tokens, al := range byTokens {
			if tokens < 1 || math.IsNaN(al) || al < 1 || al > float64(tokens)+1 {
				return fmt.Errorf("%w: %s mode %q draft_tokens=%d acceptance_length=%v", ErrGoldenALCurve, c.ID.Stem(), mode, tokens, al)
			}
		}
	}
	return nil
}

// Golden returns the curve's acceptance length for a thinking mode and draft
// length. A curve never interpolates: an unrecorded point is refused.
func (c GoldenALCurve) Golden(thinkingMode string, draftTokens int) (float64, error) {
	if err := c.Validate(); err != nil {
		return 0, err
	}
	al, ok := c.Acceptance[thinkingMode][draftTokens]
	if !ok {
		return 0, fmt.Errorf("%w: %s records no %s point at draft_tokens=%d", ErrGoldenALCurve, c.ID.Stem(), thinkingMode, draftTokens)
	}
	return al, nil
}

// SpecDecodeCounters is one reading of an engine's cumulative speculative
// decoding counters.
type SpecDecodeCounters struct {
	AcceptedDraftTokens float64
	VerificationSteps   float64
}

// ReadSpecDecodeCounters sums the two named counters across every label set of
// a Prometheus text exposition. ok is false when either counter is absent, so
// an engine that exports neither is told apart from one that never drafted.
func ReadSpecDecodeCounters(exposition, acceptedName, verificationsName string) (SpecDecodeCounters, bool) {
	accepted, okAccepted := sumPrometheusSeries(exposition, acceptedName)
	verifications, okVerifications := sumPrometheusSeries(exposition, verificationsName)
	if !okAccepted || !okVerifications {
		return SpecDecodeCounters{}, false
	}
	return SpecDecodeCounters{AcceptedDraftTokens: accepted, VerificationSteps: verifications}, true
}

func sumPrometheusSeries(exposition, name string) (float64, bool) {
	if name == "" {
		return 0, false
	}
	var sum float64
	found := false
	for _, line := range strings.Split(exposition, "\n") {
		line = strings.TrimSpace(line)
		rest, ok := strings.CutPrefix(line, name)
		if !ok || rest == "" {
			continue
		}
		switch rest[0] {
		case '{':
			end := prometheusLabelsEnd(rest)
			if end < 0 {
				continue
			}
			rest = rest[end+1:]
		case ' ', '\t':
		default:
			// A longer metric name that only shares this prefix.
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		v, err := strconv.ParseFloat(fields[0], 64)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		sum += v
		found = true
	}
	return sum, found
}

// prometheusLabelsEnd returns the index of the brace closing the label set
// that opens s. Braces inside a quoted label value do not count, and neither
// does anything after the set, such as an exemplar's own labels.
func prometheusLabelsEnd(s string) int {
	quoted := false
	for i := 1; i < len(s); i++ {
		switch {
		case quoted && s[i] == '\\':
			i++
		case s[i] == '"':
			quoted = !quoted
		case !quoted && s[i] == '}':
			return i
		}
	}
	return -1
}

// SpecDecodeIncrease is the counter movement across one replay. ok is false
// when a counter went backwards: the engine restarted mid-replay and the
// difference measures nothing.
func SpecDecodeIncrease(before, after SpecDecodeCounters) (SpecDecodeCounters, bool) {
	d := SpecDecodeCounters{
		AcceptedDraftTokens: after.AcceptedDraftTokens - before.AcceptedDraftTokens,
		VerificationSteps:   after.VerificationSteps - before.VerificationSteps,
	}
	if d.AcceptedDraftTokens < 0 || d.VerificationSteps < 0 {
		return SpecDecodeCounters{}, false
	}
	return d, true
}

// GoldenALStatus is the gate's verdict vocabulary.
type GoldenALStatus string

const (
	GoldenALPass  GoldenALStatus = "pass"
	GoldenALBelow GoldenALStatus = "below"
	GoldenALAbove GoldenALStatus = "above"
	// GoldenALNoCounters: the engine exposes no usable counters; gate skipped.
	GoldenALNoCounters GoldenALStatus = "no-counters"
	// GoldenALNoDrafting: counters exist but no verification step ran; gate skipped.
	GoldenALNoDrafting GoldenALStatus = "no-drafting"
)

// GoldenALVerdict is the gate's result document.
type GoldenALVerdict struct {
	Curve                  string         `json:"curve"`
	SourceRun              string         `json:"source_run"`
	ThinkingMode           string         `json:"thinking_mode"`
	DraftTokens            int            `json:"draft_tokens"`
	GoldenAcceptanceLength float64        `json:"golden_acceptance_length"`
	AcceptanceLength       float64        `json:"acceptance_length,omitempty"`
	Ratio                  float64        `json:"ratio,omitempty"`
	Status                 GoldenALStatus `json:"status"`
	AcceptedTokens         float64        `json:"accepted_tokens"`
	Verifications          float64        `json:"verifications"`
	ReportOnly             bool           `json:"report_only"`
}

// Measured reports whether the verdict compares a realized acceptance length
// with the golden one, as opposed to a skipped gate.
func (v GoldenALVerdict) Measured() bool {
	return v.Status == GoldenALPass || v.Status == GoldenALBelow || v.Status == GoldenALAbove
}

// ExitCode is 0 for every verdict. The gate is report-only: neither an
// out-of-band measurement nor a skipped gate fails the run that carries it.
// Only a measurement that could not run at all is an error, and that is the
// error return of EvaluateGoldenALGate, not a verdict.
func (v GoldenALVerdict) ExitCode() int { return 0 }

// EvaluateGoldenALGate judges one replay against a curve. increase is the
// counter movement across the replay, or nil when the engine exposes no usable
// counters. The realized acceptance length is
// 1 + accepted draft tokens / verification steps.
func EvaluateGoldenALGate(curve GoldenALCurve, thinkingMode string, draftTokens int, increase *SpecDecodeCounters) (GoldenALVerdict, error) {
	golden, err := curve.Golden(thinkingMode, draftTokens)
	if err != nil {
		return GoldenALVerdict{}, err
	}
	v := GoldenALVerdict{
		Curve:                  curve.ID.Stem(),
		SourceRun:              curve.SourceRun,
		ThinkingMode:           thinkingMode,
		DraftTokens:            draftTokens,
		GoldenAcceptanceLength: golden,
		ReportOnly:             true,
	}
	if increase == nil || increase.AcceptedDraftTokens < 0 || increase.VerificationSteps < 0 ||
		math.IsNaN(increase.AcceptedDraftTokens) || math.IsNaN(increase.VerificationSteps) {
		v.Status = GoldenALNoCounters
		return v, nil
	}
	v.AcceptedTokens = increase.AcceptedDraftTokens
	v.Verifications = increase.VerificationSteps
	if increase.VerificationSteps == 0 {
		v.Status = GoldenALNoDrafting
		return v, nil
	}
	length := 1 + increase.AcceptedDraftTokens/increase.VerificationSteps
	// The band is applied to the rounded ratio so the verdict always agrees
	// with the ratio the document reports.
	v.Ratio = roundTo4(length / golden)
	v.AcceptanceLength = roundTo4(length)
	switch {
	case v.Ratio < GoldenALMinRatio:
		v.Status = GoldenALBelow
	case v.Ratio <= GoldenALMaxRatio:
		v.Status = GoldenALPass
	default:
		v.Status = GoldenALAbove
	}
	return v, nil
}

// roundTo4 rounds the exact binary value to four decimals, so a value just
// under a midpoint is not pushed over it by a scaling multiply.
func roundTo4(x float64) float64 {
	r, err := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)
	if err != nil {
		return x
	}
	return r
}
