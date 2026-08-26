package witnessprocess

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func completeBlock(context Context) Block {
	return Block{Context: context, Envelope: "pinned envelope", BaselineArtifact: "baseline.ref", Lever: "one controlled change", CandidateArtifact: "candidate.ref", PromotionGate: "gate command and threshold", DurableWitness: "regression fixture"}
}

func TestAdaptersRequestContextSpecificArtifactClasses(t *testing.T) {
	wants := map[Context][]string{
		Logic:             {"failing behavior repro", "focused Go test", "regression test"},
		Visual:            {"captured failing render", "render-witness assertion", "before/after screenshot"},
		Security:          {"preflight or red-team receipt", "policy/attestation gate", "policy fixture"},
		Reliability:       {"soak artifact", "including recovery cost", "fault test"},
		Cost:              {"net-true usage receipt", "quality-constrained end-to-end", "usage fixture"},
		NativePerformance: {"native baseline benchmark receipt", "matched-envelope", "native benchmark/receipt"},
	}
	for _, context := range Contexts() {
		t.Run(string(context), func(t *testing.T) {
			adapter, ok := AdapterFor(context)
			if !ok {
				t.Fatal("adapter missing")
			}
			raw, _ := json.Marshal(adapter)
			for _, want := range wants[context] {
				if !strings.Contains(string(raw), want) {
					t.Errorf("adapter %s missing %q: %s", context, want, raw)
				}
			}
		})
	}
}

func TestEnforcedLaneRejectsMissingBaselineOrPromotionGate(t *testing.T) {
	for _, field := range []string{"baseline", "promotion"} {
		block := completeBlock(Logic)
		block.Policy = Enforce
		if field == "baseline" {
			block.BaselineArtifact = ""
		} else {
			block.PromotionGate = ""
		}
		if _, err := block.Validate(); err == nil || !strings.Contains(err.Error(), field) {
			t.Fatalf("%s: err=%v", field, err)
		}
	}
}

func TestDefaultLaneWarnsBeforeDispatch(t *testing.T) {
	block := completeBlock(Security)
	block.BaselineArtifact = ""
	warnings, err := block.Validate()
	if err != nil || len(warnings) != 1 || !strings.Contains(warnings[0], "baseline artifact") {
		t.Fatalf("warnings=%v err=%v", warnings, err)
	}
}

func TestDocumentedExceptionsBypassArtifactRequirements(t *testing.T) {
	for _, exception := range []Exception{ReadOnlyTriage, TrivialChange, UrgentResponse, ExternalUnavailable} {
		t.Run(string(exception), func(t *testing.T) {
			block := Block{Policy: Enforce, Exception: exception, ExceptionReason: "documented constraint"}
			if _, err := block.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestNegativeResultRendersRejectedLever(t *testing.T) {
	block := completeBlock(Reliability)
	block.NegativeResults = []NegativeResult{{Lever: "retry interval", Artifact: "soak-17.json", Falsifier: "p99 recovery regressed"}}
	got, err := block.RenderMarkdown()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Rejected lever", "retry interval", "soak-17.json", "p99 recovery regressed"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q", want)
		}
	}
}

func TestDogfoodIssueRenders(t *testing.T) {
	entries, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != len(Contexts()) {
		t.Fatalf("dogfood fixtures=%d want=%d", len(entries), len(Contexts()))
	}
	for _, entry := range entries {
		data, err := os.ReadFile(filepath.Join("testdata", entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		var block Block
		if err := json.Unmarshal(data, &block); err != nil {
			t.Fatal(err)
		}
		got, err := block.RenderMarkdown()
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(got, string(block.Context)) {
			t.Fatalf("%s missing context", entry.Name())
		}
		if block.Context != NativePerformance && strings.Contains(strings.ToLower(got), "gpu receipt") {
			t.Fatalf("%s gained irrelevant GPU field", entry.Name())
		}
		if block.Context != Cost && block.Context != Reliability && block.Context != NativePerformance && strings.Contains(strings.ToLower(got), "net-true") {
			t.Fatalf("%s gained irrelevant performance field", entry.Name())
		}
	}
}
