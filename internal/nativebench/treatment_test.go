package nativebench

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestTreatmentTopology(t *testing.T) {
	t.Run("RejectionOfAdditiveAgainstReplacementBaseline", testTreatmentTopologyRejection)
	t.Run("AcceptanceOfMatchedPairs", testTreatmentTopologyMatchedPairs)
	t.Run("CanonicalIdentityChanges", testTreatmentTopologyCanonicalIdentity)
	t.Run("NoLlamaCppFallbackIntroduced", testTreatmentTopologyNoLlamaCppFallback)
	t.Run("JSONAndHumanRendering", testTreatmentTopologyRendering)
}

func testTreatmentTopologyRejection(t *testing.T) {
	// 1. ValidateTreatmentTopology rejects Additive vs Replacement
	err := ValidateTreatmentTopology(AdditiveTreatment, ReplacementTreatment)
	if err == nil {
		t.Fatal("expected error for additive candidate vs replacement baseline, got nil")
	}
	if !errors.Is(err, ErrTopologyMismatch) {
		t.Fatalf("err = %v, want ErrTopologyMismatch", err)
	}
	if err.Error() != "TOPOLOGY_MISMATCH: cannot evaluate additive treatment against replacement-only baseline" {
		t.Fatalf("unexpected error message: %q", err.Error())
	}

	// 2. ValidateTreatmentTopology rejects Replacement vs Additive
	errRev := ValidateTreatmentTopology(ReplacementTreatment, AdditiveTreatment)
	if errRev == nil {
		t.Fatal("expected error for replacement candidate vs additive baseline, got nil")
	}
	if !errors.Is(errRev, ErrTopologyMismatch) {
		t.Fatalf("errRev = %v, want ErrTopologyMismatch", errRev)
	}

	// 3. Compare rejects mismatched arms
	cand := CandidateArm{
		Name:              "fak-radixkv-additive",
		CandidateTopology: AdditiveTreatment,
		NativePath:        "internal/radixkv/radixkv.go",
	}
	base := BaselineArm{
		Name:                "vllm-replacement-baseline",
		BaselineTopology:    ReplacementTreatment,
		BaselineComposition: "vllm-engine",
	}
	report, err := Compare(cand, base)
	if !errors.Is(err, ErrTopologyMismatch) {
		t.Fatalf("Compare err = %v, want ErrTopologyMismatch", err)
	}
	if report.Valid {
		t.Fatal("expected report.Valid to be false")
	}
	if report.Reason != ErrTopologyMismatch.Error() {
		t.Fatalf("report.Reason = %q, want %q", report.Reason, ErrTopologyMismatch.Error())
	}

	// 4. Invalid topology rejected
	if err := ValidateTreatmentTopology("unknown", AdditiveTreatment); err == nil {
		t.Fatal("expected error for unknown topology, got nil")
	}
	if err := ValidateTreatmentTopology(AdditiveTreatment, "unknown"); err == nil {
		t.Fatal("expected error for unknown topology, got nil")
	}
}

func testTreatmentTopologyMatchedPairs(t *testing.T) {
	// 1. Matched Additive vs Additive
	if err := ValidateTreatmentTopology(AdditiveTreatment, AdditiveTreatment); err != nil {
		t.Fatalf("expected nil for matched additive pair, got %v", err)
	}
	candAdd := CandidateArm{
		Name:              "fak-radixkv",
		CandidateTopology: AdditiveTreatment,
		NativePath:        "internal/radixkv/radixkv.go",
	}
	baseAdd := BaselineArm{
		Name:                "incumbent-serving-stack",
		BaselineTopology:    AdditiveTreatment,
		BaselineComposition: "qwen-host-serving",
	}
	reportAdd, err := Compare(candAdd, baseAdd)
	if err != nil {
		t.Fatalf("Compare err = %v, want nil", err)
	}
	if !reportAdd.Valid {
		t.Fatal("expected reportAdd.Valid = true")
	}
	if reportAdd.TreatmentTopology != AdditiveTreatment {
		t.Fatalf("reportAdd.TreatmentTopology = %q, want %q", reportAdd.TreatmentTopology, AdditiveTreatment)
	}
	if !strings.HasPrefix(reportAdd.CanonicalDigest, "sha256:") {
		t.Fatalf("reportAdd.CanonicalDigest = %q, want sha256: prefix", reportAdd.CanonicalDigest)
	}

	// 2. Matched Replacement vs Replacement
	if err := ValidateTreatmentTopology(ReplacementTreatment, ReplacementTreatment); err != nil {
		t.Fatalf("expected nil for matched replacement pair, got %v", err)
	}
	candRep := CandidateArm{
		Name:              "fak-native-tokenizer",
		CandidateTopology: ReplacementTreatment,
		NativePath:        "internal/tokenizer/tokenizer.go",
	}
	baseRep := BaselineArm{
		Name:                "huggingface-tokenizers",
		BaselineTopology:    ReplacementTreatment,
		BaselineComposition: "huggingface/tokenizers",
	}
	reportRep, err := Compare(candRep, baseRep)
	if err != nil {
		t.Fatalf("Compare err = %v, want nil", err)
	}
	if !reportRep.Valid {
		t.Fatal("expected reportRep.Valid = true")
	}
	if reportRep.TreatmentTopology != ReplacementTreatment {
		t.Fatalf("reportRep.TreatmentTopology = %q, want %q", reportRep.TreatmentTopology, ReplacementTreatment)
	}
	if !strings.HasPrefix(reportRep.CanonicalDigest, "sha256:") {
		t.Fatalf("reportRep.CanonicalDigest = %q, want sha256: prefix", reportRep.CanonicalDigest)
	}
}

func testTreatmentTopologyCanonicalIdentity(t *testing.T) {
	d1 := CanonicalIdentityDigest(AdditiveTreatment, "composition-alpha", "internal/radixkv/radixkv.go")
	d2 := CanonicalIdentityDigest(ReplacementTreatment, "composition-alpha", "internal/radixkv/radixkv.go")
	d3 := CanonicalIdentityDigest(AdditiveTreatment, "composition-beta", "internal/radixkv/radixkv.go")
	d4 := CanonicalIdentityDigest(AdditiveTreatment, "composition-alpha", "internal/ctxmmu/mmu.go")

	// Must be sha256 prefix with 64 hex chars
	if !strings.HasPrefix(d1, "sha256:") || len(d1) != len("sha256:")+64 {
		t.Fatalf("malformed canonical digest: %q", d1)
	}

	// Topology change must change digest
	if d1 == d2 {
		t.Fatalf("digest collision across topology change: %q == %q", d1, d2)
	}

	// Baseline composition change must change digest
	if d1 == d3 {
		t.Fatalf("digest collision across baseline composition change: %q == %q", d1, d3)
	}

	// Native path change must change digest
	if d1 == d4 {
		t.Fatalf("digest collision across native path change: %q == %q", d1, d4)
	}

	// Deterministic / reproducible
	d1Repeat := CanonicalIdentityDigest(AdditiveTreatment, "composition-alpha", "internal/radixkv/radixkv.go")
	if d1 != d1Repeat {
		t.Fatalf("non-deterministic digest: %q != %q", d1, d1Repeat)
	}
}

func testTreatmentTopologyNoLlamaCppFallback(t *testing.T) {
	// 1. Candidate arms across all contracts must remain fak-native under internal/
	// and never reference llama.cpp as an execution engine or fallback.
	for _, contract := range All() {
		if strings.Contains(strings.ToLower(contract.NativePath), "llama") {
			t.Fatalf("contract %s native path %q contains llama reference", contract.Capability, contract.NativePath)
		}
		if !strings.HasPrefix(contract.NativePath, "internal/") {
			t.Fatalf("contract %s native path %q is not under internal/", contract.Capability, contract.NativePath)
		}

		candArm := contract.CandidateArm()
		if candArm.CandidateTopology != AdditiveTreatment && candArm.CandidateTopology != ReplacementTreatment {
			t.Fatalf("contract %s candidate topology %q is invalid", contract.Capability, candArm.CandidateTopology)
		}

		// 2. llama.cpp is permitted only as an explicitly selected external comparison arm (NextBest)
		for _, alt := range contract.Alternatives {
			if strings.Contains(strings.ToLower(alt.Name), "llama") {
				if alt.Class != NextBest && alt.Class != TunedBaseline {
					t.Fatalf("contract %s alternative %q has class %q; llama.cpp must be an explicit reference arm only",
						contract.Capability, alt.Name, alt.Class)
				}
			}
		}
	}

	// 3. Topology validation fails closed on mismatch without attempting any fallback
	err := ValidateTreatmentTopology(AdditiveTreatment, ReplacementTreatment)
	if !errors.Is(err, ErrTopologyMismatch) {
		t.Fatalf("expected ErrTopologyMismatch without fallback, got %v", err)
	}
}

func testTreatmentTopologyRendering(t *testing.T) {
	// 1. Contract JSON includes treatment_topology
	contract := Contract{
		Capability:        "test_capability",
		NativePath:        "internal/radixkv/radixkv.go",
		Workload:          "workload",
		Metrics:           []string{"metric"},
		TreatmentTopology: AdditiveTreatment,
	}
	contractJSON, err := json.Marshal(contract)
	if err != nil {
		t.Fatalf("marshal contract: %v", err)
	}
	if !strings.Contains(string(contractJSON), `"treatment_topology":"additive"`) {
		t.Fatalf("contract JSON missing treatment_topology: %s", string(contractJSON))
	}

	// 2. Contract Human rendering includes treatment_topology
	humanContract := contract.Human()
	if !strings.Contains(humanContract, "treatment_topology=additive") {
		t.Fatalf("contract Human rendering missing treatment_topology: %s", humanContract)
	}

	// 3. ComparisonReport JSON and Human rendering
	cand := CandidateArm{
		Name:              "fak-radixkv",
		CandidateTopology: AdditiveTreatment,
		NativePath:        "internal/radixkv/radixkv.go",
	}
	base := BaselineArm{
		Name:                "vllm-baseline",
		BaselineTopology:    AdditiveTreatment,
		BaselineComposition: "vllm-stack",
	}
	report, err := Compare(cand, base)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	reportJSON, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	if !strings.Contains(string(reportJSON), `"treatment_topology":"additive"`) {
		t.Fatalf("report JSON missing treatment_topology: %s", string(reportJSON))
	}
	if !strings.Contains(string(reportJSON), `"baseline_composition":"vllm-stack"`) {
		t.Fatalf("report JSON missing baseline_composition: %s", string(reportJSON))
	}

	humanReport := report.Human()
	if !strings.Contains(humanReport, "treatment_topology=additive") {
		t.Fatalf("report Human rendering missing treatment_topology: %s", humanReport)
	}
	if !strings.Contains(humanReport, "composition=vllm-stack") {
		t.Fatalf("report Human rendering missing baseline composition: %s", humanReport)
	}
}

// Ensure standalone test runners also execute via specific names.
func TestTreatmentTopologyRejection(t *testing.T) {
	testTreatmentTopologyRejection(t)
}

func TestTreatmentTopologyMatchedPairs(t *testing.T) {
	testTreatmentTopologyMatchedPairs(t)
}

func TestTreatmentTopologyCanonicalIdentity(t *testing.T) {
	testTreatmentTopologyCanonicalIdentity(t)
}

func TestTreatmentTopologyNoLlamaCppFallback(t *testing.T) {
	testTreatmentTopologyNoLlamaCppFallback(t)
}

func TestTreatmentTopologyRendering(t *testing.T) {
	testTreatmentTopologyRendering(t)
}
