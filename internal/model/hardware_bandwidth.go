package model

import "fmt"

// HardwareSample is one time-aligned software/hardware observation during native decode.
type HardwareSample struct {
	TimestampNS       int64   `json:"timestamp_ns"`
	AcceptedTokens    int     `json:"accepted_tokens"`
	SoftwareBytes     int64   `json:"software_bytes"`
	DRAMBytes         int64   `json:"dram_bytes"`
	ClockMHz          float64 `json:"clock_mhz"`
	PowerWatts        float64 `json:"power_watts"`
	TemperatureC      float64 `json:"temperature_c"`
	ECCErrors         uint64  `json:"ecc_errors"`
	Throttled         bool    `json:"throttled"`
	ConcurrentTenants int     `json:"concurrent_tenants"`
}

type SustainableBandwidthReceipt struct {
	Schema               string  `json:"schema"`
	Engine               string  `json:"engine"`
	Samples              int     `json:"samples"`
	AcceptedTokens       int     `json:"accepted_tokens"`
	DurationNS           int64   `json:"duration_ns"`
	DRAMBytes            int64   `json:"dram_bytes"`
	SoftwareBytes        int64   `json:"software_bytes"`
	ByteAmplification    float64 `json:"byte_amplification"`
	SustainableGBps      float64 `json:"sustainable_gbps"`
	JoulesPerAccepted    float64 `json:"joules_per_accepted_token"`
	MinClockMHz          float64 `json:"min_clock_mhz"`
	MaxTemperatureC      float64 `json:"max_temperature_c"`
	ECCErrors            uint64  `json:"ecc_errors"`
	ThrottledSamples     int     `json:"throttled_samples"`
	MaxConcurrentTenants int     `json:"max_concurrent_tenants"`
	QualityConstraint    string  `json:"quality_constraint"`
	StopRule             string  `json:"stop_rule"`
}

// MeasureSustainableBandwidth integrates time-aligned samples and keeps hardware
// derating separate from software byte amplification.
func MeasureSustainableBandwidth(samples []HardwareSample) (SustainableBandwidthReceipt, error) {
	r := SustainableBandwidthReceipt{Schema: "fak-sustainable-bandwidth/1", Engine: "fak-native", Samples: len(samples), MinClockMHz: 1e100, QualityConstraint: "same accepted-token output and model artifact", StopRule: "invalidate samples with non-monotonic time or negative counters"}
	if len(samples) < 2 {
		return SustainableBandwidthReceipt{}, fmt.Errorf("model: need at least two hardware samples")
	}
	var joules float64
	for i, s := range samples {
		if s.TimestampNS < 0 || s.AcceptedTokens < 0 || s.SoftwareBytes < 0 || s.DRAMBytes < 0 || s.ClockMHz < 0 || s.PowerWatts < 0 || s.TemperatureC < 0 || s.ConcurrentTenants < 0 {
			return SustainableBandwidthReceipt{}, fmt.Errorf("model: invalid hardware sample")
		}
		if i > 0 && s.TimestampNS <= samples[i-1].TimestampNS {
			return SustainableBandwidthReceipt{}, fmt.Errorf("model: non-monotonic hardware samples")
		}
		r.AcceptedTokens += s.AcceptedTokens
		r.SoftwareBytes += s.SoftwareBytes
		r.DRAMBytes += s.DRAMBytes
		if s.ClockMHz < r.MinClockMHz {
			r.MinClockMHz = s.ClockMHz
		}
		if s.TemperatureC > r.MaxTemperatureC {
			r.MaxTemperatureC = s.TemperatureC
		}
		r.ECCErrors += s.ECCErrors
		if s.Throttled {
			r.ThrottledSamples++
		}
		if s.ConcurrentTenants > r.MaxConcurrentTenants {
			r.MaxConcurrentTenants = s.ConcurrentTenants
		}
		if i > 0 {
			dt := float64(s.TimestampNS-samples[i-1].TimestampNS) / 1e9
			joules += dt * (s.PowerWatts + samples[i-1].PowerWatts) / 2
		}
	}
	r.DurationNS = samples[len(samples)-1].TimestampNS - samples[0].TimestampNS
	if r.SoftwareBytes > 0 {
		r.ByteAmplification = float64(r.DRAMBytes) / float64(r.SoftwareBytes)
	}
	if r.DurationNS > 0 {
		r.SustainableGBps = float64(r.DRAMBytes) / float64(r.DurationNS)
	}
	if r.AcceptedTokens > 0 {
		r.JoulesPerAccepted = joules / float64(r.AcceptedTokens)
	}
	return r, nil
}
