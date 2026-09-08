package benchauthority

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
)

const (
	benchmarkArtifactSchema     = "fak-benchmark-artifact/1"
	evidenceHardwareMeasurement = benchcli.EvidenceHardwareMeasurement
)

type benchmarkEvidence struct {
	Schema  string `json:"schema"`
	RunID   string `json:"run_id"`
	Results struct {
		Metrics map[string]float64 `json:"metrics"`
	} `json:"results"`
	SimulationEvidence *benchcli.SimulationEvidence `json:"simulation_evidence,omitempty"`
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
	return benchcli.ValidateSimulationEvidence(*ev)
}
