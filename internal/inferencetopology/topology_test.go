package inferencetopology

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
)

func fixture() Manifest {
	return Manifest{Schema: Schema, ID: "synthetic-fixture", AsOf: "2026-10-22", Sources: []Source{{ID: "assumption", URL: "https://example.org/assumptions", Title: "Synthetic assumptions, not hardware specifications", Kind: "assumption", Accessed: "2026-10-22", ReviewBy: "2026-10-22"}}, Claims: []Claim{{ID: "c", Text: "Resident weights must still be read for matmul", Status: "derived", SourceIDs: []string{"assumption"}, ReviewBy: "2026-11-01"}}, Scenarios: []Scenario{{ID: "dense", Label: "Hypothetical 70B BF16 / 80 GB", Mode: "tensor_parallel", SourceIDs: []string{"assumption"}, Assumptions: []string{"Synthetic rates and 128 bytes per token KV; no measured performance"}, Model: Model{DenseParameters: 70e9, BytesPerWeight: 2, KVBytesPerToken: 128, BatchSize: 1, ContextTokens: 1000, HiddenSize: 8192, Layers: 80, BytesPerActivation: 2}, Hardware: Hardware{GPUs: 1, TPRanks: 1, KVShards: 1, MemoryBytesPerGPU: 80e9, HBMBytesPerSecondPerGPU: 1e12, FLOPsPerSecondPerGPU: 1e14}}}}
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeStepResourcesAndCapacity(t *testing.T) {
	m := fixture()
	a, err := Analyze(m)
	if err != nil {
		t.Fatal(err)
	}
	r := a.Scenarios[0]
	if r.Fits || r.UniformFormatLogicalWeightBytes != 140e9 || r.PerRank.KVResidentBytes != 128128 || r.PerRank.KVReadBytesPerStep != 128000 || r.PerRank.KVWriteBytesPerStep != 128 {
		t.Fatalf("incorrect physical ledger: %+v", r)
	}
	if math.Abs(r.HBMSeconds-(140e9+128128)/1e12) > 1e-12 || r.ComputeSeconds != 0.0014 || r.IdealOverlapLowerBoundSeconds != r.HBMSeconds {
		t.Fatalf("resource arithmetic: %+v", r)
	}
	m.Scenarios[0].Hardware.GPUs = 2
	m.Scenarios[0].Hardware.TPRanks = 2
	m.Scenarios[0].Hardware.KVShards = 2
	a, err = Analyze(m)
	if err != nil {
		t.Fatal(err)
	}
	r = a.Scenarios[0]
	if !r.Fits || r.PerRank.ResidentWeightBytes != 70e9 || r.PerRank.KVResidentBytes != 64064 {
		t.Fatalf("partition/capacity: %+v", r)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestMoEUnionAndIndependentCommunicationBounds(t *testing.T) {
	m := fixture()
	s := &m.Scenarios[0]
	s.Model = Model{DenseParameters: 10, MoELayers: 2, ExpertsPerLayer: 4, ActiveExpertsPerToken: 1, ParametersPerExpert: 100, BytesPerWeight: 2, BatchSize: 2, HiddenSize: 8, Layers: 2, BytesPerActivation: 2}
	s.Hardware = Hardware{GPUs: 2, TPRanks: 2, KVShards: 1, MemoryBytesPerGPU: 1e6, HBMBytesPerSecondPerGPU: 1e9, FLOPsPerSecondPerGPU: 1e12, AllReducesPerStep: 1, CollectivePayloadBytes: 100, LinkBytesPerSecondPerRankOneWay: 100, RingAlphaSeconds: 0, ExtraNetworkBytesPerRankPerStep: 200}
	a, err := Analyze(m)
	if err != nil {
		t.Fatal(err)
	}
	r := a.Scenarios[0]
	if r.ExpectedDistinctExpertsPerLayer != 1.75 || r.PerRank.WeightStreamBytesPerStep != 360 || r.PerRank.FLOPsPerStep != 420 {
		t.Fatalf("incorrect reuse arithmetic: %+v", r)
	}
	if r.RingSeconds != 1 || r.ExtraNetworkSeconds != 2 || r.IdealOverlapLowerBoundSeconds != 2 || r.SerialResourceSumSeconds <= 3 {
		t.Fatalf("ring/extra overlap assumptions: %+v", r)
	}
	s.Mode = "explicit_per_rank"
	s.Hardware.GPUs = 72
	s.Hardware.TPRanks = 1
	s.PerRank = &Resources{ResidentWeightBytes: 123, WeightStreamBytesPerStep: 45, FLOPsPerStep: 60}
	a, err = Analyze(m)
	if err != nil {
		t.Fatal(err)
	}
	if a.Scenarios[0].PerRank != *s.PerRank {
		t.Fatal("explicit peak rank was implicitly divided by GPU count")
	}
	s.PerRank.WeightStreamBytesPerStep = 0
	a, err = Analyze(m)
	if err != nil {
		t.Fatal(err)
	}
	if a.Scenarios[0].ArithmeticIntensityFLOPsPerByte != nil {
		t.Fatal("zero HBM traffic has undefined intensity, not zero FLOPs/byte")
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestRejectInvalidAndUnboundedInputs(t *testing.T) {
	for _, change := range []func(*Manifest){
		func(m *Manifest) { m.Scenarios[0].Hardware.TPRanks = 2 },
		func(m *Manifest) { m.Scenarios[0].Hardware.GPUs = 72; m.Scenarios[0].Hardware.TPRanks = 72 },
		func(m *Manifest) { m.Scenarios[0].Hardware.HBMBytesPerSecondPerGPU = math.NaN() },
		func(m *Manifest) { m.Scenarios[0].Hardware.ReservedBytesPerGPU = -1 },
		func(m *Manifest) { m.Scenarios[0].Model.DenseParameters = math.Inf(1) },
		func(m *Manifest) { m.Scenarios[0].Model.MoELayers = 81 },
		func(m *Manifest) { m.Scenarios[0].SourceIDs = []string{"missing"} },
		func(m *Manifest) { m.Sources = append(m.Sources, m.Sources[0]) },
	} {
		m := fixture()
		change(&m)
		if _, err := Analyze(m); !errors.Is(err, ErrInvalid) {
			t.Fatalf("wanted typed input refusal: %v", err)
		}
	}
	m := fixture()
	raw, _ := json.Marshal(m)
	for _, input := range []string{string(raw) + " {}", strings.Replace(string(raw), `"as_of":`, `"unknown":true,"as_of":`, 1), string(raw) + strings.Repeat(" ", 16<<20)} {
		if _, err := Decode(strings.NewReader(input)); !errors.Is(err, ErrInvalid) {
			t.Fatalf("strict decoder admitted invalid input: %v", err)
		}
	}
	m.Scenarios[0].Model.DenseParameters = math.MaxFloat64
	if _, err := Analyze(m); !errors.Is(err, ErrInvalid) {
		t.Fatalf("arithmetic overflow admitted: %v", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestReviewInventoryAndRender(t *testing.T) {
	m := fixture()
	i, err := Reinventory(m, "2026-10-23")
	if err != nil {
		t.Fatal(err)
	}
	if len(i.StaleSources) != 1 || len(i.StaleClaims) != 1 || len(i.AffectedScenarios) != 1 {
		t.Fatalf("stale evidence not propagated: %+v", i)
	}
	var b bytes.Buffer
	if err := Render(&b, m, "A prose argument."); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), "A prose argument.") || !strings.Contains(b.String(), "ANALYTICAL_NOT_MEASURED") || !strings.Contains(b.String(), "CAPACITY_INFEASIBLE") || strings.Index(b.String(), "CAPACITY_INFEASIBLE") > strings.Index(b.String(), "| Scenario |") {
		t.Fatal("report omitted prose, evidence class or early feasibility warning")
	}
}
