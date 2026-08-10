package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestVerifyLiveSemanticMatrix(t *testing.T) {
	if err := verifyLiveSemanticMatrix(filepath.Join("..", "..", "experiments", "microcontext", "s8j-live-matrix-openai-2026-08-10.json")); err != nil {
		t.Fatal(err)
	}
}
func TestVerifyLiveSemanticMatrixRejectsModeledTokens(t *testing.T) {
	r := liveMatrixReport{Schema: liveSemanticMatrixSchema, Trials: 2, Endpoint: endpointProvenance{Model: "m", NativeBatch: "unsupported", PrefixCache: "observed", PricingSnapshot: "unavailable"}}
	for _, n := range []string{"retrieval-rerank", "long-context", "chunk-map-reduce", "micro-context"} {
		r.Pipelines = append(r.Pipelines, livePipelineResult{Pipeline: n, TrialResults: make([]liveTrial, 2), Aggregate: liveAggregate{Successful: 1, PromptTokens: 0, TTFTP50MS: 1}, Grade: semanticGrade{Records: 16}})
	}
	b, _ := json.Marshal(r)
	p := filepath.Join(t.TempDir(), "bad.json")
	_ = os.WriteFile(p, b, 0644)
	if verifyLiveSemanticMatrix(p) == nil {
		t.Fatal("accepted matrix without observed tokens")
	}
}
