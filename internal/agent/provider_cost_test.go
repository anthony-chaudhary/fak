package agent

import "testing"

func TestAttachProviderReportedCostRoundTripsTopLevel(t *testing.T) {
	comp := &Completion{}
	attachProviderReportedCost(comp, []byte(`{"id":"x","cost_usd":0.0125,"usage":{"prompt_tokens":3}}`))
	if comp.Usage.CostUSD == nil || *comp.Usage.CostUSD != 0.0125 {
		t.Fatalf("cost=%v", comp.Usage.CostUSD)
	}
	if comp.Usage.CostStatus != "provider-reported" || comp.Usage.CostProvenance != "response.cost_usd" {
		t.Fatalf("usage=%+v", comp.Usage)
	}
}

func TestAttachProviderReportedCostRoundTripsNestedUsage(t *testing.T) {
	comp := &Completion{}
	attachProviderReportedCost(comp, []byte(`{"response":{"usage":{"cost_usd":0}}}`))
	if comp.Usage.CostUSD == nil || *comp.Usage.CostUSD != 0 {
		t.Fatalf("provider-reported zero must remain distinguishable from absent: %+v", comp.Usage)
	}
	if comp.Usage.CostProvenance != "response.response.usage.cost_usd" {
		t.Fatalf("provenance=%q", comp.Usage.CostProvenance)
	}
}

func TestAttachProviderReportedCostAbsentStaysNull(t *testing.T) {
	comp := &Completion{}
	attachProviderReportedCost(comp, []byte(`{"usage":{"prompt_tokens":3}}`))
	if comp.Usage.CostUSD != nil || comp.Usage.CostStatus != "" || comp.Usage.CostProvenance != "" {
		t.Fatalf("absent cost became a claim: %+v", comp.Usage)
	}
}

func TestAttachProviderReportedCostRejectsNegative(t *testing.T) {
	comp := &Completion{}
	attachProviderReportedCost(comp, []byte(`{"total_cost_usd":-1}`))
	if comp.Usage.CostUSD != nil {
		t.Fatalf("negative cost became evidence: %+v", comp.Usage)
	}
}

func TestAttachProviderReportedCostNormalizesAdapterDecodedUsage(t *testing.T) {
	cost := 0.004
	comp := &Completion{Usage: Usage{CostUSD: &cost}}
	attachProviderReportedCost(comp, nil)
	if comp.Usage.CostStatus != "provider-reported" || comp.Usage.CostProvenance != "response.usage.cost_usd" {
		t.Fatalf("usage=%+v", comp.Usage)
	}
}

func TestAttachProviderReportedCostReadsLiteLLMHiddenParams(t *testing.T) {
	comp := &Completion{}
	attachProviderReportedCost(comp, []byte(`{"_hidden_params":{"response_cost":0.005}}`))
	if comp.Usage.CostUSD == nil || *comp.Usage.CostUSD != 0.005 || comp.Usage.CostProvenance != "response._hidden_params.response_cost" {
		t.Fatalf("usage=%+v", comp.Usage)
	}
}
