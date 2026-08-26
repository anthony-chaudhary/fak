package nativeperf

import (
	digestsha256 "crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const AmbientEvidenceSchema = "fak-native-ambient-evidence/v1"

type MetricAvailability string

const (
	MetricMeasured    MetricAvailability = "measured"
	MetricUnavailable MetricAvailability = "unavailable"
	MetricUnknown     MetricAvailability = "unknown"
)

type AmbientVerdict string

const (
	AmbientClean       AmbientVerdict = "clean"
	AmbientInvestigate AmbientVerdict = "investigate"
	AmbientInvalid     AmbientVerdict = "invalid"
)

type AmbientMetric struct {
	Availability MetricAvailability `json:"availability"`
	Value        float64            `json:"value,omitempty"`
}

type AmbientEvidence struct {
	Schema                      string         `json:"schema"`
	StartedAt                   string         `json:"started_at"`
	EndedAt                     string         `json:"ended_at"`
	ElapsedMilliseconds         int64          `json:"elapsed_milliseconds"`
	SampleIntervalMilliseconds  int64          `json:"sample_interval_milliseconds"`
	Source                      string         `json:"source"`
	Platform                    string         `json:"platform"`
	SamplerOverheadMilliseconds float64        `json:"sampler_overhead_milliseconds"`
	HostCPUPercent              AmbientMetric  `json:"host_cpu_percent"`
	AvailableMemoryBytes        AmbientMetric  `json:"available_memory_bytes"`
	ProcessChurn                AmbientMetric  `json:"process_churn"`
	NonSUTCPUPercent            AmbientMetric  `json:"non_sut_cpu_percent"`
	CommandExitCode             int            `json:"command_exit_code"`
	Verdict                     AmbientVerdict `json:"verdict"`
	Digest                      string         `json:"digest"`
}

func (e AmbientEvidence) canonicalDigest() (string, error) {
	e.Digest = ""
	b, err := json.Marshal(e)
	if err != nil {
		return "", err
	}
	sum := digestsha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

func SealAmbientEvidence(e *AmbientEvidence) error {
	if e == nil {
		return fmt.Errorf("ambient evidence is nil")
	}
	digest, err := e.canonicalDigest()
	if err != nil {
		return err
	}
	e.Digest = digest
	return nil
}

func ValidateAmbientEvidence(e AmbientEvidence) error {
	if e.Schema != AmbientEvidenceSchema {
		return fmt.Errorf("ambient evidence schema must be %q", AmbientEvidenceSchema)
	}
	if e.StartedAt == "" || e.EndedAt == "" || e.ElapsedMilliseconds <= 0 || e.SampleIntervalMilliseconds <= 0 {
		return fmt.Errorf("ambient evidence must declare a positive sampled window")
	}
	if e.Source == "" || (e.Platform != "windows" && e.Platform != "linux") {
		return fmt.Errorf("ambient evidence must name a Windows or Linux source")
	}
	if e.SamplerOverheadMilliseconds < 0 {
		return fmt.Errorf("ambient sampler overhead cannot be negative")
	}
	for name, m := range map[string]AmbientMetric{"host_cpu_percent": e.HostCPUPercent, "available_memory_bytes": e.AvailableMemoryBytes, "process_churn": e.ProcessChurn, "non_sut_cpu_percent": e.NonSUTCPUPercent} {
		if m.Availability != MetricMeasured && m.Availability != MetricUnavailable && m.Availability != MetricUnknown {
			return fmt.Errorf("ambient metric %s has invalid availability", name)
		}
		if m.Availability != MetricMeasured && m.Value != 0 {
			return fmt.Errorf("ambient metric %s cannot carry a value when unmeasured", name)
		}
	}
	if e.CommandExitCode != 0 && e.Verdict != AmbientInvalid {
		return fmt.Errorf("failed command ambient evidence must be invalid")
	}
	if e.Verdict != AmbientClean && e.Verdict != AmbientInvestigate && e.Verdict != AmbientInvalid {
		return fmt.Errorf("ambient evidence has invalid verdict")
	}
	expected, err := e.canonicalDigest()
	if err != nil {
		return err
	}
	if e.Digest == "" || e.Digest != expected {
		return fmt.Errorf("ambient evidence digest mismatch")
	}
	return nil
}
