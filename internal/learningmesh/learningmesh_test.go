package learningmesh

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studylink"
)

func TestCompileFixtureDeterministicAndDeduplicated(t *testing.T) {
	ledger := loadFixture(t)
	first, err := Compile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("compile is not deterministic")
	}
	if first.CandidateCount != len(first.Candidates) {
		t.Fatalf("candidate_count=%d len=%d", first.CandidateCount, len(first.Candidates))
	}
	ids := map[string]bool{}
	axes := map[string]bool{}
	dispositions := map[Disposition]bool{}
	sourceFrameworks := map[string]bool{}
	hardware := map[string]bool{}
	for _, candidate := range first.Candidates {
		if ids[candidate.ID] {
			t.Fatalf("duplicate candidate id %q", candidate.ID)
		}
		ids[candidate.ID] = true
		for _, axis := range candidate.TransferAxes {
			axes[axis] = true
		}
		dispositions[candidate.Disposition] = true
		sourceFrameworks[candidate.Source.Framework] = true
		hardware[candidate.Source.Hardware] = true
		hardware[candidate.Target.Hardware] = true
		if candidate.LearningObservation.ID == "" || candidate.LearningObservation.Kind != "candidate" {
			t.Fatalf("candidate %s missing learning observation: %+v", candidate.ID, candidate.LearningObservation)
		}
		if len(candidate.Provenance.Artifacts) == 0 && candidate.Provenance.Borrow == nil {
			t.Fatalf("candidate %s lost provenance", candidate.ID)
		}
		if candidate.Target.Engine != "fak-native" && candidate.Disposition != BenchmarkOnly && candidate.Disposition != Reject {
			t.Fatalf("candidate %s silently selected comparator runtime: %+v", candidate.ID, candidate)
		}
	}
	for _, want := range []string{"hardware", "framework", "baseline"} {
		if !axes[want] {
			t.Errorf("missing transfer axis %q", want)
		}
	}
	for _, want := range []Disposition{Copy, Adapt, BenchmarkOnly, Reject, Unknown} {
		if !dispositions[want] {
			t.Errorf("missing disposition %q", want)
		}
	}
	for _, want := range []string{"fak-native", "llama.cpp", "vllm", "dynamo"} {
		if !sourceFrameworks[want] {
			t.Errorf("missing source framework %q", want)
		}
	}
	if !hardware["amd"] || !hardware["nvidia"] {
		t.Fatalf("hardware coverage=%v", hardware)
	}
}

func TestCompileMatchesCapturedWitness(t *testing.T) {
	result, err := Compile(loadFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	want, err := os.ReadFile(filepath.Join("..", "..", "docs", "_witnesses", "issue-9839", "candidates.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("captured witness is stale; regenerate with fak learning-mesh compile\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

func TestExactArtifactRequiresVerifiableIdentity(t *testing.T) {
	ledger := Ledger{
		Schema: InputSchema,
		Mechanisms: []Mechanism{{
			ID: "m", Mechanism: "shared mechanism", Source: Envelope{ID: "s", Hardware: "amd", Framework: "fak-native", Engine: "fak-native"},
			Provenance: Provenance{Artifacts: []studylink.Artifact{{Kind: "study", ID: "placeholder", Revision: "r1+g1111111", Path: "docs/example.md", Exact: true, RecordDigest: "sha256:not-real"}}},
		}},
		Targets: []Envelope{{ID: "t", Hardware: "nvidia", Framework: "fak-native", Engine: "fak-native"}},
	}
	if _, err := Compile(ledger); err == nil {
		t.Fatal("expected exact placeholder provenance to be rejected")
	}
}
func TestComparatorCannotBecomeProductExecutionPath(t *testing.T) {
	ledger := Ledger{
		Schema: InputSchema,
		Mechanisms: []Mechanism{{
			ID: "m", Mechanism: "paged scheduling", Source: Envelope{ID: "s", Hardware: "nvidia", Framework: "vllm", Engine: "vllm"},
			Provenance: Provenance{Artifacts: []studylink.Artifact{{Kind: "study", ID: "study-1"}}},
			Rules:      []Rule{{Target: Selector{Framework: "vllm"}, Disposition: Copy, Reason: "unsafe"}},
		}},
		Targets: []Envelope{{ID: "bad", Hardware: "nvidia", Framework: "vllm", Engine: "vllm", Purpose: "production"}},
	}
	result, err := Compile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Candidates[0].Disposition; got != Reject {
		t.Fatalf("disposition=%q want reject", got)
	}
}

func loadFixture(t *testing.T) Ledger {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "docs", "_witnesses", "issue-9839", "mechanisms.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ledger Ledger
	if err := json.Unmarshal(b, &ledger); err != nil {
		t.Fatal(err)
	}
	return ledger
}
