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
)

type profileComparison struct {
	Schema                           string    `json:"schema"`
	Verdict                          string    `json:"verdict"`
	Reason                           string    `json:"reason"`
	EnvelopeID                       string    `json:"envelope_id,omitempty"`
	Selector                         string    `json:"selector"`
	ControlSelector                  string    `json:"control_selector"`
	CandidateSelector                string    `json:"candidate_selector"`
	ControlForwardPath               string    `json:"control_forward_path,omitempty"`
	CandidateForwardPath             string    `json:"candidate_forward_path,omitempty"`
	ControlPrefillMilliseconds       []float64 `json:"control_prefill_milliseconds,omitempty"`
	CandidatePrefillMilliseconds     []float64 `json:"candidate_prefill_milliseconds,omitempty"`
	ControlMedianMilliseconds        float64   `json:"control_median_milliseconds,omitempty"`
	CandidateMedianMilliseconds      float64   `json:"candidate_median_milliseconds,omitempty"`
	MedianImprovementPercent         float64   `json:"median_improvement_percent,omitempty"`
	MinimumMedianImprovementPercent  float64   `json:"minimum_median_improvement_percent"`
	EveryCandidateBelowControlMedian bool      `json:"every_candidate_below_control_median"`
}

type nativeProfileComparisonInput struct {
	profile  nativeperf.ProfileBundle
	receipt  nativeProfileReceipt
	prefill  float64
	controls map[string]string
}

func newProfileComparison() profileComparison {
	return profileComparison{
		Schema:                          profileComparisonSchema,
		Verdict:                         "HOLD",
		Selector:                        nativeProfileSequenceSelector,
		ControlSelector:                 nativeProfileSelectorOff,
		CandidateSelector:               nativeProfileSelectorOn,
		MinimumMedianImprovementPercent: nativeProfileMinimumGain,
	}
}

// compareNativeProfileCampaign grades exactly three OFF captures followed by three ON
// captures. Receipt paths are derived from each profile path, so a profile can never be
// admitted without the raw event/source/binary companion produced alongside it.
func compareNativeProfileCampaign(spec string) profileComparison {
	r := newProfileComparison()
	paths := splitProfileComparisonPaths(spec)
	if len(paths) != nativeProfileCampaignRuns {
		r.Reason = fmt.Sprintf("campaign requires exactly 6 ordered profile paths (3 OFF then 3 ON), got %d", len(paths))
		return r
	}

	inputs := make([]nativeProfileComparisonInput, 0, len(paths))
	profileDigests := make(map[string]int, len(paths))
	receiptBindings := make(map[string]int, len(paths))
	for i, path := range paths {
		input, err := loadNativeProfileComparisonInput(path)
		if err != nil {
			r.Reason = fmt.Sprintf("pair %d is invalid: %v", i+1, err)
			return r
		}
		wantSelector := nativeProfileSelectorOff
		if i >= nativeProfileArmRuns {
			wantSelector = nativeProfileSelectorOn
		}
		if got := input.receipt.Controls[nativeProfileSequenceSelector]; got != wantSelector {
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
		control[i] = inputs[i].prefill
		candidate[i] = inputs[i+nativeProfileArmRuns].prefill
	}
	result := compareProfiles(control, candidate)
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

func loadNativeProfileComparisonInput(profilePath string) (nativeProfileComparisonInput, error) {
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
	if len(profile.Phases) != 6 || profile.Phases[1].Name != "prefill" || !positiveFinite(profile.Phases[1].DurationMilliseconds) {
		return nativeProfileComparisonInput{}, fmt.Errorf("canonical prefill phase is missing or invalid")
	}
	return nativeProfileComparisonInput{
		profile: profile, receipt: receipt, prefill: profile.Phases[1].DurationMilliseconds,
		controls: controlsWithoutSequenceSelector(receipt.Controls),
	}, nil
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

func controlsWithoutSequenceSelector(controls map[string]string) map[string]string {
	out := make(map[string]string, len(controls)-1)
	for key, value := range controls {
		if key != nativeProfileSequenceSelector {
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
	r := newProfileComparison()
	if len(control) != nativeProfileArmRuns || len(candidate) != nativeProfileArmRuns {
		r.Reason = "campaign requires exactly three control and three candidate repetitions"
		return r
	}
	for _, duration := range append(append([]float64(nil), control...), candidate...) {
		if !positiveFinite(duration) {
			r.Reason = "prefill durations must be finite and positive"
			return r
		}
	}
	r.ControlPrefillMilliseconds = append([]float64(nil), control...)
	r.CandidatePrefillMilliseconds = append([]float64(nil), candidate...)
	r.ControlMedianMilliseconds = median(control)
	r.CandidateMedianMilliseconds = median(candidate)
	r.MedianImprovementPercent = (r.ControlMedianMilliseconds - r.CandidateMedianMilliseconds) / r.ControlMedianMilliseconds * 100
	r.EveryCandidateBelowControlMedian = true
	for _, duration := range candidate {
		if duration >= r.ControlMedianMilliseconds {
			r.EveryCandidateBelowControlMedian = false
			break
		}
	}
	r.Verdict = "REJECT"
	if !r.EveryCandidateBelowControlMedian {
		r.Reason = "not every candidate prefill improved on the control median"
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
	b, err := json.MarshalIndent(comparison, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if *f.out == "" {
		_, err = os.Stdout.Write(b)
		return err
	}
	return os.WriteFile(*f.out, b, 0o644)
}
