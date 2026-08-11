package residualquant

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestNamedResearchWitness(t *testing.T) {
	d := PinnedPaperDescriptor()
	cases := []struct {
		name string
		req  Request
		want Verdict
		why  Code
	}{
		{"inspect-pinned-metadata", Request{Descriptor: d, Operation: "inspect", TierBits: 6}, CaseSupported, CodeMetadata},
		{"unknown-contract", Request{Descriptor: d, Operation: "inspect"}, CaseUnsupported, ReasonUnknownContract},
		{"invalid-tier", Request{Descriptor: d, Operation: "inspect", TierBits: 3}, CaseUnsupported, ReasonInvalidTiers},
		{"execute-without-weights", Request{Descriptor: d, Operation: "execute", TierBits: 8}, CaseUnsupported, ReasonArtifactUnpinned},
	}
	cases[1].req.Contract = "fak.residualquant/v2"
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Adjudicate(tc.req)
			if got.Verdict != tc.want || got.Reason != tc.why {
				t.Fatalf("Adjudicate() = %s/%s, want %s/%s (%s)", got.Verdict, got.Reason, tc.want, tc.why, got.Detail)
			}
		})
	}

	exec := d
	exec.Artifact.WeightsURI = "https://example.invalid/rrq.safetensors"
	exec.Artifact.WeightsDigest = "sha256:fixture"
	exec.Runtime = RuntimePin{ID: "fixture-runtime", Version: "1.0.0", Device: "cuda-sm80"}
	got := Adjudicate(Request{Descriptor: exec, Operation: "execute", TierBits: 4})
	if got.Verdict != CaseDelegate || got.Reason != CodeExecution || got.Delegate != "fixture-runtime" {
		t.Fatalf("pinned execution = %#v, want typed delegate", got)
	}
}

func TestResearchEvaluationSeparatesEvidence(t *testing.T) {
	e := NamedResearchEvaluation()
	if e.Disposition != "abstain" || e.Name == "" {
		t.Fatalf("evaluation = %#v", e)
	}
	seen := map[EvidenceKind]bool{}
	for _, f := range e.Findings {
		seen[f.Evidence] = true
		if f.Claim == "" || f.Envelope == "" || f.Source == "" {
			t.Fatalf("incomplete finding: %#v", f)
		}
	}
	for _, kind := range []EvidenceKind{Observed, SourceReported, Modeled} {
		if !seen[kind] {
			t.Fatalf("missing %s evidence", kind)
		}
	}
}

func TestResearchDocumentPinsContract(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(file), "..", "..", "docs", "research", "quantization", "recurrent-residual.md")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, pin := range []string{PaperID, PaperPDFSHA256, RecipeID, string(CodeExecution), "**Disposition: ABSTAIN**"} {
		if !strings.Contains(string(body), pin) {
			t.Errorf("research document missing %q", pin)
		}
	}
}
