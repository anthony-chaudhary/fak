package learningmesh

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/studylink"
)

func TestLedgerFromReceiptsMapsHardwareAndDeduplicates(t *testing.T) {
	root := filepath.Join("..", "..", "docs", "_witnesses", "issue-9886")
	paths := []string{"amd-vulkan.json", "nvidia-cuda.json", "apple-metal.json", "amd-vulkan.json"}
	var inputs []ReceiptInput
	for _, name := range paths {
		path := filepath.Join(root, name)
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		inputs = append(inputs, ReceiptInput{Path: filepath.ToSlash(filepath.Join("docs", "_witnesses", "issue-9886", name)), Bytes: raw})
	}
	targets := []Envelope{{ID: "amd", Hardware: "amd", Backend: "vulkan", Framework: "fak-native", Engine: "fak-native"}, {ID: "nvidia", Hardware: "nvidia", Backend: "cuda", Framework: "fak-native", Engine: "fak-native"}, {ID: "apple", Hardware: "apple", Backend: "metal", Framework: "fak-native", Engine: "fak-native"}}
	ledger, err := LedgerFromReceipts(inputs, targets)
	if err != nil {
		t.Fatal(err)
	}
	if len(ledger.Mechanisms) != 3 {
		t.Fatalf("mechanisms=%d want 3", len(ledger.Mechanisms))
	}
	hardware := map[string]bool{}
	for _, mechanism := range ledger.Mechanisms {
		hardware[mechanism.Source.Hardware] = true
		if mechanism.Source.Engine != "fak-native" {
			t.Fatalf("engine=%q", mechanism.Source.Engine)
		}
		if len(mechanism.Provenance.Artifacts) != 2 {
			t.Fatalf("%s artifacts=%d want receipt plus model artifact", mechanism.ID, len(mechanism.Provenance.Artifacts))
		}
		var receiptArtifact, modelArtifact studylink.Artifact
		for _, artifact := range mechanism.Provenance.Artifacts {
			switch artifact.Kind {
			case "nativeperf-receipt":
				receiptArtifact = artifact
			case "model-artifact":
				modelArtifact = artifact
			}
		}
		if !receiptArtifact.Exact || len(receiptArtifact.Revision) != 40 || len(receiptArtifact.RecordDigest) != 64 { //boundarylint:ignore CHANGE_DETECTOR_TEST git sha1 is 40 and sha256 is 64
			t.Fatalf("weak receipt provenance: %+v", receiptArtifact)
		}
		if !strings.HasPrefix(modelArtifact.ID, "sha256:") || len(modelArtifact.RecordDigest) != 64 || modelArtifact.RecordDigest == receiptArtifact.RecordDigest { //boundarylint:ignore CHANGE_DETECTOR_TEST sha256 hex digest is a fixed 64-character invariant
			t.Fatalf("model and receipt identities not separated: receipt=%+v model=%+v", receiptArtifact, modelArtifact)
		}
	}
	for _, want := range []string{"amd", "nvidia", "apple"} {
		if !hardware[want] {
			t.Errorf("missing %s", want)
		}
	}
	first, err := Compile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if first.InputDigest != second.InputDigest || first.CandidateCount != 9 {
		t.Fatalf("nondeterministic or count=%d", first.CandidateCount)
	}
}

func TestReceiptWitnessMatchesCapturedOutput(t *testing.T) {
	root := filepath.Join("..", "..", "docs", "_witnesses", "issue-9886")
	var inputs []ReceiptInput
	for _, name := range []string{"amd-vulkan.json", "nvidia-cuda.json", "apple-metal.json"} {
		raw, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		inputs = append(inputs, ReceiptInput{Path: filepath.ToSlash(filepath.Join("docs", "_witnesses", "issue-9886", name)), Bytes: raw})
	}
	var targetLedger Ledger
	rawTargets, err := os.ReadFile(filepath.Join(root, "targets.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(rawTargets, &targetLedger); err != nil {
		t.Fatal(err)
	}
	ledger, err := LedgerFromReceipts(inputs, targetLedger.Targets)
	if err != nil {
		t.Fatal(err)
	}
	result, err := Compile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	want, err := os.ReadFile(filepath.Join(root, "candidates.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatal("captured receipt witness is stale")
	}
}
func TestLedgerFromReceiptsRejectsComparatorEngine(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "docs", "_witnesses", "issue-9886", "nvidia-cuda.json"))
	if err != nil {
		t.Fatal(err)
	}
	r, err := nativeperf.DecodeReceipt(raw)
	if err != nil {
		t.Fatal(err)
	}
	r.Execution.Engine = "llama.cpp"
	_, err = LedgerFromReceipts([]ReceiptInput{{Path: "docs/x.json", Bytes: mustReceiptJSON(t, r)}}, []Envelope{{ID: "t", Hardware: "nvidia", Framework: "fak-native", Engine: "fak-native"}})
	if err == nil || !strings.Contains(err.Error(), "fak-native") {
		t.Fatalf("expected comparator rejection, got %v", err)
	}
}

func mustReceiptJSON(t *testing.T, receipt nativeperf.ExperimentReceipt) []byte {
	t.Helper()
	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestLedgerFromReceiptsCanonicalizesQwen38AndIgnoresInputOrder(t *testing.T) {
	root := filepath.Join("..", "..", "docs", "_witnesses", "issue-9886")
	load := func(names ...string) []ReceiptInput {
		t.Helper()
		out := make([]ReceiptInput, 0, len(names))
		for _, name := range names {
			raw, err := os.ReadFile(filepath.Join(root, name))
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, ReceiptInput{Path: filepath.ToSlash(filepath.Join("docs", "_witnesses", "issue-9886", name)), Bytes: raw})
		}
		return out
	}
	targets := []Envelope{{ID: "target", Hardware: "amd", Backend: "vulkan", Framework: "fak-native", Engine: "fak-native"}}
	first, err := LedgerFromReceipts(load("amd-vulkan.json", "nvidia-cuda.json", "apple-metal.json", "amd-vulkan.json"), targets)
	if err != nil {
		t.Fatal(err)
	}
	second, err := LedgerFromReceipts(load("apple-metal.json", "amd-vulkan.json", "nvidia-cuda.json"), targets)
	if err != nil {
		t.Fatal(err)
	}
	firstJSON, _ := json.Marshal(first)
	secondJSON, _ := json.Marshal(second)
	if string(firstJSON) != string(secondJSON) {
		t.Fatalf("input order changed ledger:\n%s\n%s", firstJSON, secondJSON)
	}
	for _, mechanism := range first.Mechanisms {
		if mechanism.Source.Model != "qwen3.8" {
			t.Fatalf("model=%q want canonical qwen3.8", mechanism.Source.Model)
		}
	}
}

func TestLedgerFromReceiptsRejectsReceiptStructMismatch(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "docs", "_witnesses", "issue-9886", "amd-vulkan.json"))
	if err != nil {
		t.Fatal(err)
	}
	r, err := nativeperf.DecodeReceipt(raw)
	if err != nil {
		t.Fatal(err)
	}
	r.Memory.PeakBytes = 0
	_, err = LedgerFromReceipts([]ReceiptInput{{Path: "docs/x.json", Bytes: mustReceiptJSON(t, r)}}, []Envelope{{ID: "t", Hardware: "amd", Framework: "fak-native", Engine: "fak-native"}})
	if err == nil || !strings.Contains(err.Error(), "memory") {
		t.Fatalf("expected structural rejection, got %v", err)
	}
}
