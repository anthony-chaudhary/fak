package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

const (
	profileComparisonSchema       = "fak.modelbench.profile-comparison/1"
	nativeProfileSequenceSelector = "FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE"
	nativeProfileSelectorOff      = "OFF"
	nativeProfileSelectorOn       = "ON"
	nativeProfileCampaignRuns     = 6
	nativeProfileArmRuns          = 3
	nativeProfileMinimumGain      = 15.0
	nativeProfileWelchAlpha       = 0.05
)

type profileComparisonPhase string

const (
	profileComparisonPhasePrefill      profileComparisonPhase = "prefill"
	profileComparisonPhaseSteadyDecode profileComparisonPhase = "steady-decode"
	profileComparisonPhaseEndToEnd     profileComparisonPhase = "end-to-end"
)

var nativeProfileComparisonPhaseSelection = profileComparisonPhasePrefill

type profileComparisonAxis string

const (
	profileComparisonAxisSequence        profileComparisonAxis = "sequence"
	profileComparisonAxisM3DecodeHandoff profileComparisonAxis = "m3-decode-handoff"
)

var nativeProfileComparisonAxisSelection = profileComparisonAxisSequence

func (a profileComparisonAxis) String() string { return string(a) }

func (a *profileComparisonAxis) Set(value string) error {
	axis := profileComparisonAxis(strings.TrimSpace(value))
	if axis != profileComparisonAxisSequence && axis != profileComparisonAxisM3DecodeHandoff {
		return fmt.Errorf("comparison axis %q is invalid; want sequence or m3-decode-handoff", value)
	}
	*a = axis
	return nil
}

func (p profileComparisonPhase) String() string { return string(p) }

func (p *profileComparisonPhase) Set(value string) error {
	phase := profileComparisonPhase(strings.TrimSpace(value))
	if !phase.valid() {
		return fmt.Errorf("comparison phase %q is invalid; want prefill, steady-decode, or end-to-end", value)
	}
	*p = phase
	return nil
}

func (p profileComparisonPhase) valid() bool {
	return p == profileComparisonPhasePrefill || p == profileComparisonPhaseSteadyDecode || p == profileComparisonPhaseEndToEnd
}

type profileComparison struct {
	Schema                           string                   `json:"schema"`
	Verdict                          string                   `json:"verdict"`
	Reason                           string                   `json:"reason"`
	Phase                            profileComparisonPhase   `json:"phase"`
	EnvelopeID                       string                   `json:"envelope_id,omitempty"`
	Selector                         string                   `json:"selector"`
	ControlSelector                  string                   `json:"control_selector"`
	CandidateSelector                string                   `json:"candidate_selector"`
	ControlForwardPath               string                   `json:"control_forward_path,omitempty"`
	CandidateForwardPath             string                   `json:"candidate_forward_path,omitempty"`
	ControlPrefillMilliseconds       []float64                `json:"control_prefill_milliseconds,omitempty"`
	CandidatePrefillMilliseconds     []float64                `json:"candidate_prefill_milliseconds,omitempty"`
	ControlPhaseMilliseconds         []float64                `json:"control_phase_milliseconds,omitempty"`
	CandidatePhaseMilliseconds       []float64                `json:"candidate_phase_milliseconds,omitempty"`
	ControlMedianMilliseconds        float64                  `json:"control_median_milliseconds,omitempty"`
	CandidateMedianMilliseconds      float64                  `json:"candidate_median_milliseconds,omitempty"`
	MedianImprovementPercent         float64                  `json:"median_improvement_percent,omitempty"`
	WelchSignificance                profileWelchSignificance `json:"welch_significance"`
	MinimumMedianImprovementPercent  float64                  `json:"minimum_median_improvement_percent"`
	EveryCandidateBelowControlMedian bool                     `json:"every_candidate_below_control_median"`
}

// profileWelchSignificance is advisory. The quality, operating-envelope, and
// median acceptance gates remain the only inputs to profileComparison.Verdict.
type profileWelchSignificance struct {
	Verdict   string  `json:"verdict"`
	Direction string  `json:"direction"`
	PValue    float64 `json:"p_value"`
	Alpha     float64 `json:"alpha"`
	Reason    string  `json:"reason"`
}

type nativeProfileComparisonInput struct {
	profile  nativeperf.ProfileBundle
	receipt  nativeProfileReceipt
	duration float64
	controls map[string]string
}

func newProfileComparisonForPhase(phase profileComparisonPhase) profileComparison {
	return newProfileComparisonForPhaseAxis(phase, profileComparisonAxisSequence)
}

func newProfileComparisonForPhaseAxis(phase profileComparisonPhase, axis profileComparisonAxis) profileComparison {
	selector, control, candidate := nativeProfileSequenceSelector, nativeProfileSelectorOff, nativeProfileSelectorOn
	if axis == profileComparisonAxisM3DecodeHandoff {
		selector = nativeProfileDecodeHandoffControl
		control = model.Qwen35DecodeHandoffControl.String()
		candidate = model.Qwen35DecodeHandoffMixer.String()
	}
	return profileComparison{
		Schema:                          profileComparisonSchema,
		Verdict:                         "HOLD",
		Phase:                           phase,
		Selector:                        selector,
		ControlSelector:                 control,
		CandidateSelector:               candidate,
		WelchSignificance:               newProfileWelchSignificance(),
		MinimumMedianImprovementPercent: nativeProfileMinimumGain,
	}
}

func newProfileWelchSignificance() profileWelchSignificance {
	return profileWelchSignificance{
		Verdict:   "HOLD",
		Direction: "none",
		PValue:    1,
		Alpha:     nativeProfileWelchAlpha,
		Reason:    "not evaluated",
	}
}

// compareNativeProfileCampaign grades exactly three OFF captures followed by three ON
// captures. Receipt paths are derived from each profile path, so a profile can never be
// admitted without the raw event/source/binary companion produced alongside it.
func compareNativeProfileCampaign(spec string) profileComparison {
	return compareNativeProfileCampaignPhaseAxis(spec, nativeProfileComparisonPhaseSelection, nativeProfileComparisonAxisSelection)
}

func compareNativeProfileCampaignPhase(spec string, phase profileComparisonPhase) profileComparison {
	return compareNativeProfileCampaignPhaseAxis(spec, phase, profileComparisonAxisSequence)
}

func compareNativeProfileCampaignPhaseAxis(spec string, phase profileComparisonPhase, axis profileComparisonAxis) profileComparison {
	r := newProfileComparisonForPhaseAxis(phase, axis)
	if !phase.valid() {
		r.Reason = fmt.Sprintf("comparison phase %q is invalid", phase)
		return r
	}
	if axis != profileComparisonAxisSequence && axis != profileComparisonAxisM3DecodeHandoff {
		r.Reason = fmt.Sprintf("comparison axis %q is invalid", axis)
		return r
	}
	if axis == profileComparisonAxisM3DecodeHandoff && phase == profileComparisonPhasePrefill {
		r.Reason = "m3-decode-handoff comparison requires steady-decode or end-to-end phase"
		return r
	}
	paths := splitProfileComparisonPaths(spec)
	if len(paths) != nativeProfileCampaignRuns {
		r.Reason = fmt.Sprintf("campaign requires exactly 6 ordered profile paths (3 OFF then 3 ON), got %d", len(paths))
		return r
	}

	inputs := make([]nativeProfileComparisonInput, 0, len(paths))
	profileDigests := make(map[string]int, len(paths))
	receiptBindings := make(map[string]int, len(paths))
	for i, path := range paths {
		input, err := loadNativeProfileComparisonInput(path, phase, axis)
		if err != nil {
			r.Reason = fmt.Sprintf("pair %d is invalid: %v", i+1, err)
			return r
		}
		wantSelector := r.ControlSelector
		if i >= nativeProfileArmRuns {
			wantSelector = r.CandidateSelector
		}
		if axis == profileComparisonAxisM3DecodeHandoff && input.receipt.Controls[nativeProfileSequenceSelector] != nativeProfileSelectorOn {
			r.Reason = fmt.Sprintf("pair %d sequence selector must be ON for M3 decode handoff", i+1)
			return r
		}
		if got := input.receipt.Controls[r.Selector]; got != wantSelector {
			r.Reason = fmt.Sprintf("pair %d selector = %q, want %q", i+1, got, wantSelector)
			return r
		}
		if firstAt, reused := receiptBindings[input.receipt.BindingSHA256]; reused {
			r.Reason = fmt.Sprintf("pair %d reuses bound receipt capture from pair %d", i+1, firstAt)
			return r
		}
		receiptBindings[input.receipt.BindingSHA256] = i + 1
		if firstAt, reused := profileDigests[input.receipt.ProfileSHA256]; reused {
			r.Reason = fmt.Sprintf("pair %d reuses profile capture from pair %d", i+1, firstAt)
			return r
		}
		profileDigests[input.receipt.ProfileSHA256] = i + 1
		if input.profile.Override != nil {
			r.Reason = fmt.Sprintf("pair %d carries a selection override", i+1)
			return r
		}
		inputs = append(inputs, input)
	}

	first := inputs[0]
	r.EnvelopeID = first.profile.EnvelopeID
	r.ControlForwardPath = first.profile.Execution.ForwardPath
	r.CandidateForwardPath = inputs[nativeProfileArmRuns].profile.Execution.ForwardPath
	for i := 1; i < len(inputs); i++ {
		if !sameNativeProfileComparisonIdentity(first, inputs[i]) {
			r.Reason = fmt.Sprintf("pair %d does not match source/binary/artifact/host/execution identity", i+1)
			return r
		}
	}

	control := make([]float64, nativeProfileArmRuns)
	candidate := make([]float64, nativeProfileArmRuns)
	for i := range nativeProfileArmRuns {
		control[i] = inputs[i].duration
		candidate[i] = inputs[i+nativeProfileArmRuns].duration
	}
	result := compareProfilePhase(phase, control, candidate)
	result.Selector = r.Selector
	result.ControlSelector = r.ControlSelector
	result.CandidateSelector = r.CandidateSelector
	result.EnvelopeID = first.profile.EnvelopeID
	result.ControlForwardPath = r.ControlForwardPath
	result.CandidateForwardPath = r.CandidateForwardPath
	return result
}

func splitProfileComparisonPaths(spec string) []string {
	parts := strings.Split(spec, ",")
	paths := make([]string, 0, len(parts))
	for _, part := range parts {
		path := strings.TrimSpace(part)
		if path != "" {
			paths = append(paths, path)
		}
	}
	return paths
}

func loadNativeProfileComparisonInput(profilePath string, phase profileComparisonPhase, axis profileComparisonAxis) (nativeProfileComparisonInput, error) {
	profileBytes, err := os.ReadFile(profilePath)
	if err != nil {
		return nativeProfileComparisonInput{}, fmt.Errorf("read profile: %w", err)
	}
	profile, err := nativeperf.DecodeProfile(profileBytes)
	if err != nil {
		return nativeProfileComparisonInput{}, err
	}
	receiptBytes, err := os.ReadFile(nativeReceiptPath(profilePath))
	if err != nil {
		return nativeProfileComparisonInput{}, fmt.Errorf("read companion receipt: %w", err)
	}
	var receipt nativeProfileReceipt
	if err := decodeExactJSON(receiptBytes, &receipt); err != nil {
		return nativeProfileComparisonInput{}, fmt.Errorf("decode companion receipt: %w", err)
	}
	if err := validateNativeProfileReceipt(profileBytes, profile, receipt); err != nil {
		return nativeProfileComparisonInput{}, err
	}
	envelope, err := profileEnvelopeByID(nativeperf.ActiveGraph(), profile.EnvelopeID)
	if err != nil {
		return nativeProfileComparisonInput{}, err
	}
	if envelope.PromptTokens != 32 || envelope.DecodeTokens != 64 || envelope.Repetitions != nativeProfileArmRuns || envelope.Engine != "fak-native" || envelope.Backend != "metal" {
		return nativeProfileComparisonInput{}, fmt.Errorf("envelope is not the canonical fak-native Metal P32/T64 three-repetition lineage")
	}
	duration, err := profileComparisonPhaseDuration(profile.Phases, phase)
	if err != nil {
		return nativeProfileComparisonInput{}, err
	}
	return nativeProfileComparisonInput{
		profile: profile, receipt: receipt, duration: duration,
		controls: controlsWithoutSelector(receipt.Controls, newProfileComparisonForPhaseAxis(phase, axis).Selector),
	}, nil
}

type profileComparisonPhaseError struct {
	Phase  profileComparisonPhase
	Detail string
}

func (e profileComparisonPhaseError) Error() string {
	return fmt.Sprintf("selected comparison phase %q %s", e.Phase, e.Detail)
}

func profileComparisonPhaseDuration(phases []nativeperf.ProfilePhase, selected profileComparisonPhase) (float64, error) {
	if !selected.valid() {
		return 0, profileComparisonPhaseError{Phase: selected, Detail: "is invalid"}
	}
	if selected != profileComparisonPhaseEndToEnd {
		var duration float64
		matches := 0
		for _, phase := range phases {
			if phase.Name == selected.String() {
				matches++
				duration = phase.DurationMilliseconds
			}
		}
		if matches != 1 {
			return 0, profileComparisonPhaseError{Phase: selected, Detail: fmt.Sprintf("must appear exactly once, got %d", matches)}
		}
		if !positiveFinite(duration) {
			return 0, profileComparisonPhaseError{Phase: selected, Detail: "duration must be finite and positive"}
		}
		return duration, nil
	}

	// End-to-end is the full canonical capture wall. Including load setup,
	// verification, and teardown prevents a phase-local gain from hiding the
	// setup or proof overhead needed to obtain it.
	wanted := [...]string{"load-setup", "prefill", "first-token", "steady-decode", "verification", "teardown"}
	if len(phases) != len(wanted) {
		return 0, profileComparisonPhaseError{Phase: selected, Detail: fmt.Sprintf("requires all %d canonical phases exactly once, got %d", len(wanted), len(phases))}
	}
	for i, name := range wanted {
		phase := phases[i]
		if phase.Name != name {
			return 0, profileComparisonPhaseError{Phase: selected, Detail: fmt.Sprintf("requires canonical phase %q at position %d, got %q", name, i+1, phase.Name)}
		}
		if i > 0 {
			previous := phases[i-1]
			previousEnd := previous.StartMilliseconds + previous.DurationMilliseconds
			if phase.StartMilliseconds != previousEnd {
				return 0, profileComparisonPhaseError{Phase: selected, Detail: fmt.Sprintf("requires contiguous canonical phases; %q ends at %g ms but %q starts at %g ms", previous.Name, previousEnd, phase.Name, phase.StartMilliseconds)}
			}
		}
	}
	start := phases[0].StartMilliseconds
	last := phases[len(phases)-1]
	end := last.StartMilliseconds + last.DurationMilliseconds
	duration := end - start
	if !positiveFinite(duration) {
		return 0, profileComparisonPhaseError{Phase: selected, Detail: "interval must be finite and positive"}
	}
	return duration, nil
}

func decodeExactJSON(data []byte, out any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(out); err != nil {
		return err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return fmt.Errorf("trailing data: %w", err)
	}
	return nil
}

func controlsWithoutSelector(controls map[string]string, selector string) map[string]string {
	out := make(map[string]string, len(controls)-1)
	for key, value := range controls {
		if key != selector {
			out[key] = value
		}
	}
	return out
}

func sameNativeProfileComparisonIdentity(a, b nativeProfileComparisonInput) bool {
	return a.profile.EnvelopeID == b.profile.EnvelopeID &&
		a.profile.Execution.Engine == b.profile.Execution.Engine &&
		a.profile.Execution.FallbackCount == b.profile.Execution.FallbackCount &&
		a.receipt.Artifact == b.receipt.Artifact &&
		a.receipt.Host == b.receipt.Host &&
		a.receipt.Source == b.receipt.Source &&
		a.receipt.Binary == b.receipt.Binary &&
		a.receipt.ModelConfigSHA256 == b.receipt.ModelConfigSHA256 &&
		reflect.DeepEqual(a.controls, b.controls)
}

func compareProfiles(control, candidate []float64) profileComparison {
	return compareProfilePhase(profileComparisonPhasePrefill, control, candidate)
}

func compareProfilePhase(phase profileComparisonPhase, control, candidate []float64) profileComparison {
	r := newProfileComparisonForPhase(phase)
	if !phase.valid() {
		r.Reason = fmt.Sprintf("comparison phase %q is invalid", phase)
		return r
	}
	if len(control) != nativeProfileArmRuns || len(candidate) != nativeProfileArmRuns {
		r.Reason = "campaign requires exactly three control and three candidate repetitions"
		return r
	}
	for _, duration := range append(append([]float64(nil), control...), candidate...) {
		if !positiveFinite(duration) {
			r.Reason = fmt.Sprintf("%s durations must be finite and positive", phase)
			return r
		}
	}
	// Keep the original prefill keys byte-for-byte available to existing readers;
	// non-prefill selectors use neutral keys so decode evidence is never mislabeled.
	if phase == profileComparisonPhasePrefill {
		r.ControlPrefillMilliseconds = append([]float64(nil), control...)
		r.CandidatePrefillMilliseconds = append([]float64(nil), candidate...)
	} else {
		r.ControlPhaseMilliseconds = append([]float64(nil), control...)
		r.CandidatePhaseMilliseconds = append([]float64(nil), candidate...)
	}
	r.ControlMedianMilliseconds = median(control)
	r.CandidateMedianMilliseconds = median(candidate)
	r.MedianImprovementPercent = (r.ControlMedianMilliseconds - r.CandidateMedianMilliseconds) / r.ControlMedianMilliseconds * 100
	// Significance is deliberately reported but never consulted below: quality,
	// envelope eligibility, and the existing median gates stay authoritative.
	r.WelchSignificance = welchSignificance(control, candidate)
	r.EveryCandidateBelowControlMedian = true
	for _, duration := range candidate {
		if duration >= r.ControlMedianMilliseconds {
			r.EveryCandidateBelowControlMedian = false
			break
		}
	}
	r.Verdict = "REJECT"
	if !r.EveryCandidateBelowControlMedian {
		r.Reason = fmt.Sprintf("not every candidate %s improved on the control median", phase)
		return r
	}
	if r.MedianImprovementPercent < nativeProfileMinimumGain {
		r.Reason = "median improvement below gate"
		return r
	}
	r.Verdict = "KEEP"
	r.Reason = "acceptance gate passed"
	return r
}

func welchSignificance(control, candidate []float64) profileWelchSignificance {
	r := newProfileWelchSignificance()
	if len(control) < 2 || len(candidate) < 2 {
		r.Reason = "Welch significance requires at least two samples per arm"
		return r
	}
	controlMean, controlVariance, ok := sampleMeanVariance(control)
	if !ok {
		r.Reason = "control samples must be finite and positive"
		return r
	}
	candidateMean, candidateVariance, ok := sampleMeanVariance(candidate)
	if !ok {
		r.Reason = "candidate samples must be finite and positive"
		return r
	}
	// Welch's degrees-of-freedom estimate is undefined when either arm has no
	// observed variance. Treat that as absent evidence rather than significance.
	if controlVariance == 0 || candidateVariance == 0 {
		r.Reason = "Welch significance requires non-degenerate variance in both arms"
		return r
	}

	controlTerm := controlVariance / float64(len(control))
	candidateMeanVariance := candidateVariance / float64(len(candidate))
	standardErrorSquared := controlTerm + candidateMeanVariance
	degreesFreedomDenominator := controlTerm*controlTerm/float64(len(control)-1) +
		candidateMeanVariance*candidateMeanVariance/float64(len(candidate)-1)
	if !positiveFinite(standardErrorSquared) || !positiveFinite(degreesFreedomDenominator) {
		r.Reason = "Welch significance could not establish finite variance"
		return r
	}
	degreesFreedom := standardErrorSquared * standardErrorSquared / degreesFreedomDenominator
	tStatistic := math.Abs(controlMean-candidateMean) / math.Sqrt(standardErrorSquared)
	if math.IsNaN(tStatistic) || math.IsInf(tStatistic, 0) || !positiveFinite(degreesFreedom) {
		r.Reason = "Welch significance could not establish a finite statistic"
		return r
	}
	p, ok := studentTTwoSidedP(tStatistic, degreesFreedom)
	if !ok {
		r.Reason = "Welch significance could not establish a finite p-value"
		return r
	}
	r.PValue = p
	if p >= nativeProfileWelchAlpha {
		r.Reason = "Welch difference is not statistically significant"
		return r
	}
	if candidateMean < controlMean {
		r.Direction = "candidate-faster"
	} else if candidateMean > controlMean {
		r.Direction = "candidate-slower"
	} else {
		r.Reason = "Welch significance has no directional mean difference"
		return r
	}
	r.Verdict = "SIGNIFICANT"
	r.Reason = "Welch difference is statistically significant"
	return r
}

func sampleMeanVariance(xs []float64) (mean, variance float64, ok bool) {
	var sumSquares float64
	for i, x := range xs {
		if !positiveFinite(x) {
			return 0, 0, false
		}
		delta := x - mean
		mean += delta / float64(i+1)
		sumSquares += delta * (x - mean)
	}
	variance = sumSquares / float64(len(xs)-1)
	if variance < 0 || math.IsNaN(variance) || math.IsInf(variance, 0) {
		return 0, 0, false
	}
	return mean, variance, true
}

// studentTTwoSidedP evaluates the two-sided Student-t tail using the regularized
// incomplete beta identity. Its boolean result lets callers fail closed if the
// bounded continued fraction cannot produce finite evidence.
func studentTTwoSidedP(tStatistic, degreesFreedom float64) (float64, bool) {
	if tStatistic < 0 || math.IsNaN(tStatistic) || math.IsInf(tStatistic, 0) || !positiveFinite(degreesFreedom) {
		return 0, false
	}
	if tStatistic == 0 {
		return 1, true
	}
	x := degreesFreedom / (degreesFreedom + tStatistic*tStatistic)
	p, ok := regularizedIncompleteBeta(x, degreesFreedom/2, 0.5)
	if !ok || p < 0 || p > 1 || math.IsNaN(p) || math.IsInf(p, 0) {
		return 0, false
	}
	return p, true
}

func regularizedIncompleteBeta(x, a, b float64) (float64, bool) {
	if x < 0 || x > 1 || !positiveFinite(a) || !positiveFinite(b) {
		return 0, false
	}
	if x == 0 {
		return 0, true
	}
	if x == 1 {
		return 1, true
	}
	logGammaAB, _ := math.Lgamma(a + b)
	logGammaA, _ := math.Lgamma(a)
	logGammaB, _ := math.Lgamma(b)
	factor := math.Exp(logGammaAB - logGammaA - logGammaB + a*math.Log(x) + b*math.Log1p(-x))
	if math.IsNaN(factor) || math.IsInf(factor, 0) {
		return 0, false
	}
	if x < (a+1)/(a+b+2) {
		fraction, ok := incompleteBetaContinuedFraction(a, b, x)
		return factor * fraction / a, ok
	}
	fraction, ok := incompleteBetaContinuedFraction(b, a, 1-x)
	return 1 - factor*fraction/b, ok
}

func incompleteBetaContinuedFraction(a, b, x float64) (float64, bool) {
	const (
		maxIterations = 200
		epsilon       = 3e-14
		minimum       = 1e-300
	)
	qab := a + b
	qap := a + 1
	qam := a - 1
	c := 1.0
	d := 1 - qab*x/qap
	if math.Abs(d) < minimum {
		d = minimum
	}
	d = 1 / d
	h := d
	for iteration := 1; iteration <= maxIterations; iteration++ {
		m := float64(iteration)
		m2 := 2 * m
		aa := m * (b - m) * x / ((qam + m2) * (a + m2))
		c, d = stepContinuedFraction(c, d, aa, minimum)
		h *= d * c

		aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
		c, d = stepContinuedFraction(c, d, aa, minimum)
		delta := d * c
		h *= delta
		if math.Abs(delta-1) < epsilon {
			return h, !math.IsNaN(h) && !math.IsInf(h, 0)
		}
	}
	return 0, false
}

func stepContinuedFraction(c, d, aa, minimum float64) (float64, float64) {
	d = 1 + aa*d
	if math.Abs(d) < minimum {
		d = minimum
	}
	c = 1 + aa/c
	if math.Abs(c) < minimum {
		c = minimum
	}
	return c, 1 / d
}

func positiveFinite(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func median(xs []float64) float64 {
	v := append([]float64(nil), xs...)
	sort.Float64s(v)
	n := len(v)
	if n%2 == 1 {
		return v[n/2]
	}
	return (v[n/2-1] + v[n/2]) / 2
}

func writeProfileComparison(f *benchFlags, comparison profileComparison) error {
	if *f.out != "" {
		return benchcli.WriteReport(*f.out, comparison)
	}
	b, err := benchcli.MarshalReport(comparison)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_, err = os.Stdout.Write(b)
	return err
}
