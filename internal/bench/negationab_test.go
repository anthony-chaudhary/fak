package bench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func loadNegationABCorpus(t *testing.T) []NegationABItem {
	t.Helper()
	f, err := os.Open("testdata/negation_ab_corpus.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	items, err := ReadNegationABCorpus(f)
	if err != nil {
		t.Fatal(err)
	}
	return items
}

func TestNegationAB(t *testing.T) {
	art, err := BuildNegationAB(loadNegationABCorpus(t), "offline-witness")
	if err != nil {
		t.Fatal(err)
	}
	if art.Provenance != NegationABObservedProvenance || art.AffectsAcceptance {
		t.Fatalf("metadata=%+v", art)
	}
	if len(art.Rungs) != 3 {
		t.Fatalf("rungs=%d want 3", len(art.Rungs))
	}
	for _, r := range art.Rungs {
		if r.Naive.Label != "naive_negation" || r.Operator.Label != "operator" {
			t.Fatalf("%s arm labels: %q/%q", r.Rung, r.Naive.Label, r.Operator.Label)
		}
		if r.Naive.WorkloadHash == "" || r.Naive.WorkloadHash != r.Operator.WorkloadHash {
			t.Fatalf("%s workload mismatch: %q/%q", r.Rung, r.Naive.WorkloadHash, r.Operator.WorkloadHash)
		}
		if r.Naive.Total != r.Operator.Total || r.Naive.Calls != r.Operator.Calls {
			t.Fatalf("%s non-identical workload counts", r.Rung)
		}
		if r.InversionTokensSaved <= 0 {
			t.Fatalf("%s inversion tokens saved=%d want positive", r.Rung, r.InversionTokensSaved)
		}
		if r.ClassificationDelta < 0 {
			t.Fatalf("%s classification delta=%f", r.Rung, r.ClassificationDelta)
		}
	}
}

func TestNegationABRejectsMissingHost(t *testing.T) {
	if _, err := BuildNegationAB(loadNegationABCorpus(t), ""); err == nil {
		t.Fatal("missing OBSERVED host accepted")
	}
}

// TestNegationABArtifact writes the OBSERVED, non-gating witness on request.
//
//	FAK_NEGATION_AB_OUT=/tmp/negation-ab.json FAK_NEGATION_AB_HOST=$(hostname) \
//	  go test ./internal/bench -run TestNegationABArtifact -count=1 -v
func TestNegationABArtifact(t *testing.T) {
	out := os.Getenv("FAK_NEGATION_AB_OUT")
	if out == "" {
		t.Skip("set FAK_NEGATION_AB_OUT to write the observed A/B witness")
	}
	host := os.Getenv("FAK_NEGATION_AB_HOST")
	if host == "" {
		var err error
		host, err = os.Hostname()
		if err != nil {
			t.Fatal(err)
		}
	}
	art, err := BuildNegationAB(loadNegationABCorpus(t), host)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := json.MarshalIndent(art, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, append(enc, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var round NegationABArtifact
	if err := json.Unmarshal(raw, &round); err != nil {
		t.Fatal(err)
	}
	if round.Provenance != NegationABObservedProvenance || round.Host != host || round.AffectsAcceptance {
		t.Fatalf("round trip metadata=%+v", round)
	}
	for _, r := range art.Rungs {
		t.Logf("NEGATION_AB rung=%s naive_tokens=%d operator_tokens=%d saved=%d class_delta=%+.3f", r.Rung, r.Naive.InversionTokens, r.Operator.InversionTokens, r.InversionTokensSaved, r.ClassificationDelta)
	}
	t.Logf("NEGATION_AB artifact=%s host=%s affects_acceptance=false", filepath.Clean(out), host)
}
