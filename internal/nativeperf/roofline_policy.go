package nativeperf

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// rejectDuplicateRooflineFields checks fields consumed by the receipt decoder
// before encoding/json can silently overwrite them. Matching follows its
// case-insensitive field lookup; opaque metadata contents remain untouched.
func rejectDuplicateRooflineFields(data []byte, object string) error {
	fields := map[string][]string{
		"": {"schema", "receipt_id", "device", "target_device", "model_architecture",
			"theoretical_peak_bandwidth_gbps", "measured_peak_bandwidth_gbps", "roofline_denominator",
			"bandwidth_metric", "bandwidth_units", "bandwidth_scope", "achieved_bandwidth_gbps",
			"target_80pct_bandwidth_gbps", "logit_cosine_similarity", "simulated", "is_simulated",
			"proof_token", "signature", "execution_time_ms", "latency_ms", "tokens_per_second",
			"prompt_tokens", "decode_tokens", "total_tokens", "peak_memory_bytes", "timestamp",
			"git_revision", "environment", "metadata", "hardware", "workload", "numerical_parity",
			"roofline_attainment", "timing"},
		"roofline_denominator": {"value", "kind", "target_device", "metric", "units", "scope"},
		"hardware":             {"target_isa", "platform", "peak_dram_bandwidth_gbps"},
		"workload":             {"model", "workload_type"},
		"numerical_parity":     {"logit_cosine_similarity"},
		"roofline_attainment":  {"measured_roofline", "useful_throughput", "attainment_ratio"},
		"timing":               {"execution_time_ms", "latency_ms", "duration_ms"},
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	start, err := decoder.Token()
	if err != nil {
		return err
	}
	if start == nil { // JSON null is an absent object; validation still requires a ceiling.
		return nil
	}
	if start != json.Delim('{') {
		return fmt.Errorf("%w: expected roofline receipt object %q", ErrInvalidMetrics, object)
	}
	seen := make(map[string]bool)
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return err
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return err
		}
		for _, field := range fields[object] {
			if !strings.EqualFold(key.(string), field) {
				continue
			}
			if seen[field] {
				return fmt.Errorf("%w: duplicate field %q in %q", ErrIncompatibleDenominator, field, object)
			}
			seen[field] = true
			if object == "" && fields[field] != nil {
				if err := rejectDuplicateRooflineFields(value, field); err != nil {
					return err
				}
			}
			break
		}
	}
	_, err = decoder.Token()
	return err
}

// RooflineDenominatorKind distinguishes a full ceiling from an attainment target.
type RooflineDenominatorKind string

const (
	RooflineMeasuredCeiling    RooflineDenominatorKind = "measured"
	RooflineTheoreticalCeiling RooflineDenominatorKind = "theoretical"
	RooflineBandwidthMetric                            = "memory_bandwidth"
	RooflineBandwidthUnits                             = "GB/s"
	RooflineDeviceDRAMScope                            = "device-dram"
)

var (
	ErrInvalidRooflinePolicy   = errors.New("nativeperf: invalid roofline policy")
	ErrIncompatibleDenominator = errors.New("nativeperf: incompatible or ambiguous roofline denominator")
	ErrSubRooflineAttainment   = errors.New("nativeperf: roofline attainment below policy threshold")
)

// RooflinePolicy selects the device and comparable bandwidth domain explicitly.
// Scope identifies the same memory domain and workload envelope for numerator and
// denominator. A caller must independently qualify any measured ceiling; passing
// this arithmetic policy is not evidence of physical execution or measurement.
type RooflinePolicy struct {
	TargetDevice                 string                  `json:"target_device"`
	DeviceAliases                []string                `json:"device_aliases,omitempty"`
	Metric                       string                  `json:"metric"`
	Units                        string                  `json:"units"`
	Scope                        string                  `json:"scope"`
	DenominatorKind              RooflineDenominatorKind `json:"denominator_kind"`
	MinimumAttainmentRatio       float64                 `json:"minimum_attainment_ratio"`
	MinimumLogitCosineSimilarity float64                 `json:"minimum_logit_cosine_similarity"`
}

// RooflineDenominator is a full ceiling, never an already-discounted target.
// Its dimensions must match both the policy and the receipt's achieved bandwidth.
type RooflineDenominator struct {
	Value        float64                 `json:"value"`
	Kind         RooflineDenominatorKind `json:"kind"`
	TargetDevice string                  `json:"target_device"`
	Metric       string                  `json:"metric"`
	Units        string                  `json:"units"`
	Scope        string                  `json:"scope"`
}

// LegacyStrixHaloRooflinePolicy preserves the original theoretical-ceiling
// campaign. It supplies no bandwidth value. Measured ceilings require an explicit
// policy and cannot be represented by the legacy reproducibility packet schema.
func LegacyStrixHaloRooflinePolicy() RooflinePolicy {
	return RooflinePolicy{
		TargetDevice: DefaultAMDStrixHaloDevice,
		DeviceAliases: []string{
			"strix-halo", "strix halo", "strix_halo", "strixhalo",
			"AMD Strix Halo (Ryzen AI Max+ 395)", "AMD Radeon 8060S",
			"Radeon 8060S", "8060s", "ryzen ai max+ 395",
		},
		Metric:                       RooflineBandwidthMetric,
		Units:                        RooflineBandwidthUnits,
		Scope:                        RooflineDeviceDRAMScope,
		DenominatorKind:              RooflineTheoreticalCeiling,
		MinimumAttainmentRatio:       MinimumRooflineAttainmentRatio,
		MinimumLogitCosineSimilarity: MinimumLogitCosineSimilarity,
	}
}

func finiteRooflineValue(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func (p RooflinePolicy) validate() error {
	if strings.TrimSpace(p.TargetDevice) == "" || strings.TrimSpace(p.Scope) == "" ||
		p.Metric != RooflineBandwidthMetric || p.Units != RooflineBandwidthUnits ||
		(p.DenominatorKind != RooflineMeasuredCeiling && p.DenominatorKind != RooflineTheoreticalCeiling) ||
		!finiteRooflineValue(p.MinimumAttainmentRatio) || p.MinimumAttainmentRatio <= 0 || p.MinimumAttainmentRatio > 1 ||
		!finiteRooflineValue(p.MinimumLogitCosineSimilarity) || p.MinimumLogitCosineSimilarity <= 0 || p.MinimumLogitCosineSimilarity > 1 {
		return ErrInvalidRooflinePolicy
	}
	for _, alias := range p.DeviceAliases {
		if strings.TrimSpace(alias) == "" {
			return ErrInvalidRooflinePolicy
		}
	}
	return nil
}

func (p RooflinePolicy) matchesDevice(device string) bool {
	device = strings.TrimSpace(device)
	if device == "" {
		return false
	}
	if strings.EqualFold(device, strings.TrimSpace(p.TargetDevice)) {
		return true
	}
	for _, alias := range p.DeviceAliases {
		if strings.EqualFold(device, strings.TrimSpace(alias)) {
			return true
		}
	}
	return false
}

func (p RooflinePolicy) denominator(r *BenchmarkReceipt) (RooflineDenominator, error) {
	d := RooflineDenominator{
		TargetDevice: r.GetDevice(), Metric: r.BandwidthMetric,
		Units: r.BandwidthUnits, Scope: r.BandwidthScope,
	}
	switch {
	case r.RooflineDenominator != nil:
		if r.TheoreticalPeakBandwidthGBps != 0 || r.MeasuredPeakBandwidthGBps != 0 {
			return d, fmt.Errorf("%w: explicit and legacy ceilings both supplied", ErrIncompatibleDenominator)
		}
		d = *r.RooflineDenominator
	case r.TheoreticalPeakBandwidthGBps != 0 && r.MeasuredPeakBandwidthGBps != 0:
		return d, fmt.Errorf("%w: measured and theoretical ceilings both supplied", ErrIncompatibleDenominator)
	case r.TheoreticalPeakBandwidthGBps != 0:
		d.Value, d.Kind = r.TheoreticalPeakBandwidthGBps, RooflineTheoreticalCeiling
	case r.MeasuredPeakBandwidthGBps != 0:
		d.Value, d.Kind = r.MeasuredPeakBandwidthGBps, RooflineMeasuredCeiling
	default:
		return d, fmt.Errorf("%w: missing denominator; an attainment target is not a full ceiling", ErrMissingDenominator)
	}
	if !finiteRooflineValue(d.Value) || d.Value <= 0 {
		return d, fmt.Errorf("%w: missing denominator; full ceiling must be finite and positive", ErrMissingDenominator)
	}
	if d.Kind != p.DenominatorKind || !p.matchesDevice(d.TargetDevice) ||
		d.Metric != p.Metric || d.Units != p.Units || d.Scope != p.Scope ||
		r.BandwidthMetric != p.Metric || r.BandwidthUnits != p.Units || r.BandwidthScope != p.Scope {
		return d, fmt.Errorf("%w: ceiling kind, device, metric, units and scope must match the selected policy and achieved bandwidth", ErrIncompatibleDenominator)
	}
	return d, nil
}
