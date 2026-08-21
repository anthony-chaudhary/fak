package qwen38quant

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSelfcheck(t *testing.T) {
	if err := Selfcheck(); err != nil {
		t.Fatal(err)
	}
}
func TestRequiredTenArms(t *testing.T) {
	if len(RequiredArms) != 10 {
		t.Fatalf("arms=%d", len(RequiredArms))
	}
}
func TestCheckedInCorpus(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "docs", "benchmarks", "qwen38-quant", "corpus.json"))
	if err != nil {
		t.Fatal(err)
	}
	c, err := DecodeCorpus(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Fixtures) != 6 || len(CorpusDigest(c)) != 64 {
		t.Fatalf("fixtures=%d digest=%q", len(c.Fixtures), CorpusDigest(c))
	}
	for _, f := range c.Fixtures {
		if f.Prompt == "" {
			t.Fatalf("fixture %s is not executable", f.ID)
		}
	}
}
func TestCorpusSemanticDriftRejected(t *testing.T) {
	c := testCorpus()
	r := validFixture(c)
	c.Fixtures[0].Prompt += " drift"
	if err := Validate(r, c); err == nil || !strings.Contains(err.Error(), "corpus drift") {
		t.Fatalf("err=%v", err)
	}
}
func TestCorpusRejectsMissingFixture(t *testing.T) {
	c := testCorpus()
	c.Fixtures = c.Fixtures[:5]
	if err := c.Validate(); err == nil {
		t.Fatal("accepted missing fixture")
	}
}
func TestCorpusRejectsUnknownJSON(t *testing.T) {
	_, err := DecodeCorpus([]byte(`{"schema":"fak.qwen38-quant-corpus/1","unknown":true}`))
	if err == nil {
		t.Fatal("accepted unknown field")
	}
}
func TestFailedTrialRetained(t *testing.T) {
	c := testCorpus()
	r := validFixture(c)
	r.Trials[0].Quality = "FAIL"
	r.Trials[0].Failure = ""
	r.Verdict = "HOLD"
	if err := Validate(r, c); err == nil || !strings.Contains(err.Error(), "retained") {
		t.Fatalf("err=%v", err)
	}
}
func TestAcceptanceOnlyNeverCampaignReady(t *testing.T) {
	c := testCorpus()
	for _, a := range []string{"fp8", "q4_k_m"} {
		r := LegacyAcceptance(a, c.ID, "Qwen3.8-27B", []float64{1, 2, 3})
		if err := Validate(r, c); err == nil {
			t.Fatal("accepted", a)
		}
	}
}
