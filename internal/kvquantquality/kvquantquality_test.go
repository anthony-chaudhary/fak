package kvquantquality

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSeededLongContextQualityWitness(t *testing.T) {
	files, err := filepath.Glob("testdata/*.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) < 3 {
		t.Fatalf("fixture count = %d, want >= 3", len(files))
	}
	seenContexts := map[int]bool{}
	outcomes := map[Outcome]bool{}
	for _, path := range files {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var fixture struct {
			Request
			ExpectedOutcome Outcome    `json:"expected_outcome"`
			ExpectedReason  ReasonCode `json:"expected_reason"`
		}
		if err := json.Unmarshal(raw, &fixture); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		req := fixture.Request
		if req.Evidence != EvidenceModeled {
			t.Fatalf("%s: fixture evidence = %q, want modeled", path, req.Evidence)
		}
		report := Evaluate(req)
		encoded, err := json.Marshal(report)
		if err != nil {
			t.Fatalf("%s report: %v", path, err)
		}
		for _, field := range []string{`"contract":"kvquantquality/v1"`, `"outcome":`, `"reason":`, `"artifact":`, `"recipe":`, `"runtime":`, `"evidence":"modeled"`} {
			if !strings.Contains(string(encoded), field) {
				t.Errorf("%s report missing %s: %s", path, field, encoded)
			}
		}
		seenContexts[req.TokenCount] = true
		outcomes[report.Outcome] = true
		if report.Outcome != fixture.ExpectedOutcome || report.Reason != fixture.ExpectedReason {
			t.Errorf("%s: got %s/%s, want %s/%s (%+v)", path, report.Outcome, report.Reason, fixture.ExpectedOutcome, fixture.ExpectedReason, report.Metrics)
		}
		adapterJSON, err := EvaluateJSON(raw)
		if err != nil {
			t.Fatalf("%s adapter: %v", path, err)
		}
		var adapterReport Report
		if err := json.Unmarshal(adapterJSON, &adapterReport); err != nil {
			t.Fatalf("%s adapter report: %v", path, err)
		}
		if adapterReport.Outcome != report.Outcome || adapterReport.Reason != report.Reason {
			t.Errorf("%s adapter mismatch: %+v != %+v", path, adapterReport, report)
		}
	}
	for _, length := range []int{4096, 16384, 32768} {
		if !seenContexts[length] {
			t.Errorf("missing context length %d", length)
		}
	}
	if !outcomes[OutcomeSupported] || !outcomes[OutcomeRefused] {
		t.Errorf("outcomes = %v, want supported and unsupported", outcomes)
	}
}

func TestTypedDelegationAndRefusal(t *testing.T) {
	malformed, err := EvaluateJSON([]byte(`{"contract_version":`))
	if err != nil {
		t.Fatal(err)
	}
	var malformedReport Report
	if err := json.Unmarshal(malformed, &malformedReport); err != nil {
		t.Fatal(err)
	}
	if malformedReport.Outcome != OutcomeDelegate || malformedReport.Reason != ReasonMalformedJSON {
		t.Fatalf("malformed: %+v", malformedReport)
	}

	req := validRequest()
	req.ContractVersion = "kvquantquality/v99"
	if got := Evaluate(req); got.Outcome != OutcomeDelegate || got.Reason != ReasonUnknownContract {
		t.Fatalf("unknown version: %+v", got)
	}

	req = validRequest()
	req.Runtime.SHA256 = "mutable-latest"
	if got := Evaluate(req); got.Outcome != OutcomeDelegate || got.Reason != ReasonIncompletePin {
		t.Fatalf("unpinned runtime: %+v", got)
	}

	req = validRequest()
	req.Evidence = EvidenceObservedHardware
	if got := Evaluate(req); got.Outcome != OutcomeDelegate || got.Reason != ReasonMissingHardware {
		t.Fatalf("unidentified observed hardware: %+v", got)
	}

	req = validRequest()
	req.Baseline.Precision = "q8_0"
	if got := Evaluate(req); got.Outcome != OutcomeRefused || got.Reason != ReasonInvalidBaseline {
		t.Fatalf("quantized baseline: %+v", got)
	}
}

func validRequest() Request {
	pin := Pin{Name: "fixture", Version: "1", Provenance: "test", SHA256: strings.Repeat("a", 64)}
	return Request{
		ContractVersion: ContractVersion, FixtureID: "unit", Seed: 6257, TokenCount: 4096,
		Evidence: EvidenceModeled, Artifact: pin, Recipe: pin, Runtime: pin,
		Baseline:  Measurement{Precision: "fp16", Attention: [][]float64{{.5, .5}}, Output: [][]float64{{.5, .5}}, TaskScore: .9},
		Candidate: Measurement{Precision: "q8_0", Attention: [][]float64{{.5, .5}}, Output: [][]float64{{.5, .5}}, TaskScore: .9},
		Budget:    Budget{MaxRowJSD: .01, MaxOutputJSD: .01, MaxTaskDrop: .01},
	}
}
