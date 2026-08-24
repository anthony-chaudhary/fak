package qwen38quant

import (
	"crypto/sha256"
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
func TestRequiredArmsCoverSupportedFormats(t *testing.T) {
	want := map[string]bool{
		"bf16": true, "fp8": true, "q8_0": true, "q6_k": true, "q5_k_m": true,
		"q4_k_m": true, "iq4_xs": true, "awq_int4": true, "gptq_int4": true, "exl2": true,
	}
	for _, arm := range RequiredArms {
		if seen := !want[arm]; seen {
			t.Fatalf("unexpected or duplicate required arm %q", arm)
		}
		delete(want, arm)
	}
	if len(want) != 0 {
		t.Fatalf("missing required arms: %v", want)
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
	digest := CorpusDigest(c)
	if len(c.Fixtures) != len(RequiredWorkloads) || len(digest) != sha256.Size*2 {
		t.Fatalf("fixtures=%d workloads=%d digest=%q", len(c.Fixtures), len(RequiredWorkloads), digest)
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

func TestValidateRejectsCampaignWithoutSuccessfulAPICompletion(t *testing.T) {
	c := testCorpus()
	r := validFixture(c)
	for i := range r.Trials {
		r.Trials[i].CompletionTokens = 0
		r.Trials[i].Quality = "FAIL"
		r.Trials[i].Failure = "usage.completion_tokens: cannot unmarshal object into Go value of type int"
	}
	r.Verdict = "HOLD"
	if err := Validate(r, c); err == nil || !strings.Contains(err.Error(), "no successful API completions") {
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

func TestValidateRequiresNativeExecutionEngineForPromotion(t *testing.T) {
	c := testCorpus()
	r := validFixture(c)
	r.ExecutionEngine = ""
	if err := Validate(r, c); err == nil || !strings.Contains(err.Error(), "execution_engine is required") {
		t.Fatalf("missing engine err=%v", err)
	}
	r = validFixture(c)
	r.ExecutionEngine = "vllm"
	if err := Validate(r, c); err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("unknown engine err=%v", err)
	}
	r = validFixture(c)
	r.ExecutionEngine = EngineLlamaCpp
	if err := Validate(r, c); err == nil || !strings.Contains(err.Error(), "comparison-only") {
		t.Fatalf("external promote err=%v", err)
	}
	r = validFixture(c)
	r.Environment.DenyFallback = false
	if err := Validate(r, c); err == nil || !strings.Contains(err.Error(), "deny_fallback") {
		t.Fatalf("fallback err=%v", err)
	}
}

func TestValidateAllowsExplicitLlamaCppComparisonHold(t *testing.T) {
	c := testCorpus()
	r := validFixture(c)
	r.ExecutionEngine = EngineLlamaCpp
	r.Verdict = "HOLD"
	if err := Validate(r, c); err != nil {
		t.Fatal(err)
	}
}
