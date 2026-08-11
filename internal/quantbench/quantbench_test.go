package quantbench

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestSelfTestMatrix(t *testing.T) {
	r := SelfTest()
	if !r.Pass {
		t.Fatalf("self-test failed: %+v", r)
	}
	if len(r.Cases) < 3 {
		t.Fatalf("fixtures=%d, want >=3", len(r.Cases))
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range r.RequiredDimensions {
		if !strings.Contains(string(b), key) {
			t.Errorf("matrix does not cover %q", key)
		}
	}
}

func TestUnknownAndUnsupportedAreTyped(t *testing.T) {
	tests := []struct {
		in     Input
		d      Outcome
		reason string
	}{
		{fixture("futureq", "1", "vllm"), OutcomeAbstain, ReasonUnknownFormat},
		{fixture("gguf", "v3", "future"), OutcomeDelegate, ReasonUnknownRuntime},
		{fixture("awq", "1", "llama.cpp"), OutcomeRefuse, ReasonPairRejected},
	}
	for _, tt := range tests {
		got := Evaluate(tt.in)
		if got.Outcome != tt.d || got.ReasonCode != tt.reason {
			t.Errorf("got %s/%s want %s/%s", got.Outcome, got.ReasonCode, tt.d, tt.reason)
		}
	}
}

func TestPerformanceClaimRequiresEnvelopeAndSeparation(t *testing.T) {
	in := fixture("gptq", "1", "vllm")
	in.Baseline.Name = ""
	in.Provenance.Source = ""
	in.Hardware.OS = ""
	in.Native.Quality.Dataset = ""
	got := Evaluate(in)
	if got.Outcome != OutcomeRefuse || got.ReasonCode != ReasonInvalidEvidence {
		t.Fatalf("got %+v", got)
	}
	for _, want := range []string{"tuned_baseline.name", "provenance.source", "hardware_envelope.os", "quality.dataset"} {
		if !strings.Contains(got.Detail, want) {
			t.Errorf("detail missing %q: %s", want, got.Detail)
		}
	}
	good := Evaluate(fixture("gptq", "1", "vllm"))
	if good.Native == nil || good.FAK == nil || good.ClaimGrade == nil || good.ClaimGrade.Verdict != "net-true" {
		t.Fatal("native and fak measurements must remain separate")
	}
}
