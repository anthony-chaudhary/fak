package benchauthority

import (
	"encoding/json"
	"errors"
	"strings"
)

const (
	benchmarkArtifactSchema     = "fak-benchmark-artifact/1"
	evidenceHardwareMeasurement = "hardware_measurement"
)

type benchmarkEvidence struct {
	Schema  string `json:"schema"`
	RunID   string `json:"run_id"`
	Results struct {
		Metrics map[string]float64 `json:"metrics"`
	} `json:"results"`
	SimulationEvidence *simulationEvidence `json:"simulation_evidence,omitempty"`
}

type simulationEvidence struct {
	Schema       string `json:"schema"`
	EvidenceType string `json:"evidence_type"`
	ClaimCeiling string `json:"claim_ceiling"`
	Engine       struct {
		Name         string `json:"name"`
		Revision     string `json:"revision"`
		ConfigDigest string `json:"config_digest"`
	} `json:"engine"`
	Workload struct {
		Name   string `json:"name"`
		Source string `json:"source"`
		Digest string `json:"digest"`
	} `json:"workload_provenance"`
	ValidityEnvelope struct {
		Description string            `json:"description"`
		Dimensions  map[string]string `json:"dimensions"`
	} `json:"validity_envelope"`
	ExcludedEffects []string `json:"excluded_effects"`
	Replay          struct {
		Stream             string `json:"stream"`
		Repetitions        int    `json:"repetitions"`
		IndependentStreams int    `json:"independent_streams"`
	} `json:"replay"`
	Cost struct {
		HostWallTimeMS float64 `json:"host_wall_time_ms"`
		HostCPUTimeMS  float64 `json:"host_cpu_time_ms"`
		Bytes          int64   `json:"bytes"`
	} `json:"simulator_cost"`
}

func decodeBenchmarkEvidence(raw []byte) (benchmarkEvidence, bool) {
	var root map[string]json.RawMessage
	if json.Unmarshal(raw, &root) != nil {
		return benchmarkEvidence{}, false
	}
	payload, ok := root["benchmark_artifact"]
	if !ok {
		return benchmarkEvidence{}, false
	}
	var art benchmarkEvidence
	if json.Unmarshal(payload, &art) != nil || art.Schema != benchmarkArtifactSchema || strings.TrimSpace(art.RunID) == "" {
		return benchmarkEvidence{}, false
	}
	if validateBenchmarkEvidence(art) != nil {
		return benchmarkEvidence{}, false
	}
	return art, true
}

func validateBenchmarkEvidence(art benchmarkEvidence) error {
	ev := art.SimulationEvidence
	if ev == nil {
		return nil
	}
	var problems []string
	if ev.Schema != "fak-simulation-evidence/1" {
		problems = append(problems, "invalid simulation evidence schema")
	}
	validTypes := map[string]bool{"structural_count": true, "analytical_bound": true, "trace_sim": true, "cycle_sim": true, "learned_estimate": true, "calibrated_sim": true, evidenceHardwareMeasurement: true}
	validClaims := map[string]bool{"correctness_only": true, "bottleneck_only": true, "relative_rank": true, "absolute_estimate": true, "measured_absolute": true}
	if !validTypes[ev.EvidenceType] {
		problems = append(problems, "unknown evidence_type")
	}
	if !validClaims[ev.ClaimCeiling] {
		problems = append(problems, "unknown claim_ceiling")
	}
	if ev.EvidenceType == "structural_count" && ev.ClaimCeiling != "correctness_only" && ev.ClaimCeiling != "bottleneck_only" {
		problems = append(problems, "structural_count exceeds its claim ceiling")
	}
	if ev.EvidenceType != evidenceHardwareMeasurement && ev.ClaimCeiling == "measured_absolute" {
		problems = append(problems, "non-hardware evidence cannot claim measured_absolute")
	}
	if strings.TrimSpace(ev.Engine.Name) == "" || strings.TrimSpace(ev.Engine.Revision) == "" || strings.TrimSpace(ev.Engine.ConfigDigest) == "" {
		problems = append(problems, "simulation engine identity is incomplete")
	}
	if strings.TrimSpace(ev.Workload.Name) == "" || strings.TrimSpace(ev.Workload.Source) == "" || strings.TrimSpace(ev.Workload.Digest) == "" {
		problems = append(problems, "workload provenance is incomplete")
	}
	if strings.TrimSpace(ev.ValidityEnvelope.Description) == "" || len(ev.ValidityEnvelope.Dimensions) == 0 {
		problems = append(problems, "validity envelope is incomplete")
	}
	if len(ev.ExcludedEffects) == 0 {
		problems = append(problems, "excluded effects are required")
	}
	if strings.TrimSpace(ev.Replay.Stream) == "" || ev.Replay.Repetitions <= 0 || ev.Replay.IndependentStreams <= 0 {
		problems = append(problems, "replay provenance is incomplete")
	}
	if ev.Cost.HostWallTimeMS <= 0 || ev.Cost.HostCPUTimeMS <= 0 || ev.Cost.Bytes <= 0 {
		problems = append(problems, "simulator cost must be positive")
	}
	if len(problems) > 0 {
		return errors.New(strings.Join(problems, "; "))
	}
	return nil
}
