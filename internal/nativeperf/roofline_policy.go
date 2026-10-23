package nativeperf

import (
	"fmt"
	"math"
	"strings"
)

// RooflineDenominator describes a full ceiling, never an attainment target.
// Its declaration supports arithmetic only, not proof of physical execution.
type RooflineDenominator struct {
	Value  float64 `json:"value"`
	Kind   string  `json:"kind"`
	Device string  `json:"device"`
	Metric string  `json:"metric"`
	Units  string  `json:"units"`
	Scope  string  `json:"scope"`
}

// RooflinePolicy selects the compatible ceiling and campaign acceptance floors.
// A caller-supplied policy does not qualify a device or authorize performance claims.
type RooflinePolicy struct {
	TargetDevice                 string
	Metric                       string
	Units                        string
	Scope                        string
	MinimumAttainmentRatio       float64
	MinimumLogitCosineSimilarity float64
}

// HaloRooflinePolicy preserves the legacy Halo bandwidth and numerical gates.
func HaloRooflinePolicy() RooflinePolicy {
	return RooflinePolicy{DefaultAMDStrixHaloDevice, "bandwidth", "GB/s", "device-dram", MinimumRooflineAttainmentRatio, MinimumLogitCosineSimilarity}
}

func finitePositive(v float64) bool { return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }

func (p RooflinePolicy) validate() error {
	if strings.TrimSpace(p.TargetDevice) == "" || p.Metric != "bandwidth" || p.Units != "GB/s" || strings.TrimSpace(p.Scope) == "" ||
		!finitePositive(p.MinimumAttainmentRatio) || p.MinimumAttainmentRatio > 1 ||
		!finitePositive(p.MinimumLogitCosineSimilarity) || p.MinimumLogitCosineSimilarity > 1 {
		return fmt.Errorf("nativeperf: invalid roofline policy")
	}
	return nil
}

func (p RooflinePolicy) matchesDevice(device string) bool {
	return strings.EqualFold(strings.TrimSpace(device), p.TargetDevice)
}

func resolveRooflineDenominator(r *BenchmarkReceipt, p RooflinePolicy, legacyHalo bool) (*RooflineDenominator, error) {
	d := r.RooflineDenominator
	if d != nil && r.TheoreticalPeakBandwidthGBps != 0 {
		return nil, fmt.Errorf("missing denominator: ambiguous full ceiling sources")
	}
	if d == nil && legacyHalo && r.TheoreticalPeakBandwidthGBps != 0 {
		d = &RooflineDenominator{r.TheoreticalPeakBandwidthGBps, "theoretical", p.TargetDevice, p.Metric, p.Units, p.Scope}
	}
	if d == nil {
		return nil, fmt.Errorf("missing denominator: a full ceiling is required; an attainment target is not a ceiling")
	}
	// The v1 packet consumer only has a theoretical ceiling field. Keep its
	// legacy entry point fail-closed until that schema can preserve measured provenance.
	if legacyHalo && d.Kind == "measured" {
		return nil, fmt.Errorf("missing denominator: legacy Halo packet path requires a theoretical ceiling; use explicit policy validation for measured ceilings")
	}
	if !finitePositive(d.Value) || (d.Kind != "measured" && d.Kind != "theoretical") ||
		!p.matchesDevice(d.Device) || d.Metric != p.Metric || d.Units != p.Units || d.Scope != p.Scope {
		return nil, fmt.Errorf("missing denominator: invalid or incompatible full ceiling kind, value, device, metric, units or scope")
	}
	copy := *d
	return &copy, nil
}
