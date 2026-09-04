package kvvectoreval

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func validRequest() Request {
	return Request{
		ContractID: ContractID, PaperID: PaperID,
		PaperPDFSHA256: PaperPDFSHA256, PaperSourceSHA256: PaperSourceSHA256,
		RecipeID: RecipeID, RuntimeID: RuntimeID, RuntimeAvailable: true,
	}
}

func TestEvaluateTypedOutcomes(t *testing.T) {
	tests := []struct {
		name     string
		mutate   func(*Request)
		outcome  Outcome
		reason   Reason
		delegate string
	}{
		{name: "supported", mutate: func(*Request) {}, outcome: OutcomeSupported, reason: ReasonExactMatch},
		{name: "delegate unavailable runtime", mutate: func(r *Request) { r.RuntimeAvailable = false }, outcome: OutcomeDelegate, reason: ReasonExternalRuntime, delegate: RuntimeID},
		{name: "malformed", mutate: func(r *Request) { r.RecipeID = "" }, outcome: OutcomeRefused, reason: ReasonMalformed},
		{name: "unknown contract", mutate: func(r *Request) { r.ContractID = "fak.kvvectoreval/v2" }, outcome: OutcomeRefused, reason: ReasonUnknownContract},
		{name: "unknown paper", mutate: func(r *Request) { r.PaperID = "arxiv:2608.04074v2" }, outcome: OutcomeRefused, reason: ReasonUnknownPaper},
		{name: "artifact mismatch", mutate: func(r *Request) { r.PaperPDFSHA256 = strings.Repeat("0", 64) }, outcome: OutcomeRefused, reason: ReasonArtifactMismatch},
		{name: "unknown recipe", mutate: func(r *Request) { r.RecipeID += "-dirty" }, outcome: OutcomeRefused, reason: ReasonUnknownRecipe},
		{name: "unknown runtime", mutate: func(r *Request) { r.RuntimeID = "sglang@latest" }, outcome: OutcomeRefused, reason: ReasonUnknownRuntime},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := validRequest()
			tt.mutate(&req)
			got := Evaluate(req)
			if got.Outcome != tt.outcome || got.Reason != tt.reason || got.Delegate != tt.delegate {
				t.Fatalf("Evaluate() = (%q, %q, %q), want (%q, %q, %q)", got.Outcome, got.Reason, got.Delegate, tt.outcome, tt.reason, tt.delegate)
			}
		})
	}
}

func TestLedgerPreservesNamedEvaluationAndEvidenceKinds(t *testing.T) {
	got := Evaluate(validRequest())
	if got.Recipe != RecipeID || got.Runtime != RuntimeID || len(got.Artifacts) != 3 {
		t.Fatalf("pins missing: recipe=%q runtime=%q artifacts=%d", got.Recipe, got.Runtime, len(got.Artifacts))
	}
	want := map[string]EvidenceKind{
		"qwen3-8b.niah.aggregate.oscar-2bit":            EvidenceObserved,
		"gpt-oss-20b.quality.aggregate.turboquant-int2": EvidenceObserved,
		"qwen3-8b.decode.throughput-vs-bf16":            EvidenceObserved,
		"metadata.amortized-rate":                       EvidenceModeled,
		"gpt-oss-20b.shipped-codebook":                  EvidenceObserved,
	}
	for _, metric := range got.Metrics {
		if kind, ok := want[metric.Name]; ok {
			if metric.Kind != kind || metric.Provenance == "" || metric.Envelope == "" {
				t.Fatalf("metric %q lost evidence: %+v", metric.Name, metric)
			}
			delete(want, metric.Name)
		}
	}
	if len(want) != 0 {
		t.Fatalf("named evaluation metrics missing: %v", want)
	}
}

func TestVerifyArtifactUsesPinnedDigest(t *testing.T) {
	fixture := []byte("independently readable fixture")
	sumID := pinnedArtifacts()[0].ID
	if err := VerifyArtifact(sumID, fixture); err == nil {
		t.Fatal("fixture unexpectedly matched pinned paper digest")
	}
	if err := VerifyArtifact("unknown", fixture); err == nil {
		t.Fatal("unknown artifact accepted")
	}
}

func TestResearchDocumentCarriesContractAndNoObservedLabClaim(t *testing.T) {
	path := filepath.Join("..", "..", "docs", "research", "quantization", "attention-vq.md")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, needle := range []string{ContractID, PaperID, RecipeID, "v0.5.10+nova-kv.d81c77b", "Modeled", "Observed (paper/repository)", "No fak lab run is claimed"} {
		if !strings.Contains(text, needle) {
			t.Errorf("research witness missing %q", needle)
		}
	}
}
