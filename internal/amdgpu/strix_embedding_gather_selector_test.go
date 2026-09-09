package amdgpu

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func embeddingGatherTestOutput(t *testing.T, mutate func(*embeddingGatherAblationEvent)) string {
	t.Helper()
	zero, exact := 0, true
	baseSubmissions, candidateSubmissions := uint64(30), uint64(5)
	baseBytes, candidateBytes := uint64(840), uint64(840)
	event := embeddingGatherAblationEvent{
		Schema:        embeddingGatherAblationSchema,
		Feature:       embeddingGatherAblationName,
		Selector:      embeddingGatherSelector,
		Engine:        embeddingGatherAblationEngine,
		FallbackCount: &zero,
		ExactParity:   &exact,
		BaselineArm: embeddingGatherAblationArm{
			Name:        embeddingGatherBaselineArm,
			SamplesUS:   []int64{120, 116, 118, 121, 117},
			Submissions: &baseSubmissions,
			Bytes:       &baseBytes,
		},
		CandidateArm: embeddingGatherAblationArm{
			Name:        embeddingGatherAblationName,
			SamplesUS:   []int64{82, 80, 81, 79, 83},
			Submissions: &candidateSubmissions,
			Bytes:       &candidateBytes,
		},
	}
	if mutate != nil {
		mutate(&event)
	}
	ablationJSON, err := json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	parityJSON := fmt.Sprintf(`{"schema":%q,"selector":%q,"test_name":%q,"oracle_kind":%q,"engine":%q,"device_observed":true,"case_count":6,"passed":true,"observed":{"max_abs_delta":0,"finite_output":true}}`,
		StrixSubkernelParitySchema, embeddingGatherSelector, embeddingGatherTestName, OracleMaxAbs, StrixVulkanEngine)
	return parityJSON + "\n" + string(ablationJSON) + "\n--- PASS: " + embeddingGatherTestName + " (0.01s)\nPASS\n"
}

func TestEmbeddingGatherSelectorRegistration(t *testing.T) {
	specs, err := FilterSubkernelSpecs([]string{embeddingGatherSelector})
	if err != nil {
		t.Fatalf("selector rejected: %v", err)
	}
	if len(specs) != 1 || specs[0].Name != embeddingGatherSelector || specs[0].TestPattern != "^"+embeddingGatherTestName+"$" {
		t.Fatalf("unexpected embedding gather spec: %+v", specs)
	}
	contract, ok := LookupSubkernelParityContract(embeddingGatherSelector)
	if !ok {
		t.Fatal("embedding gather parity contract missing")
	}
	if contract.TestName != embeddingGatherTestName || contract.OracleKind != OracleMaxAbs || contract.Engine != StrixVulkanEngine || !contract.DeviceObserved {
		t.Fatalf("unexpected embedding gather parity contract: %+v", contract)
	}
	if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta != 0 || !contract.Bounds.RequireFinite {
		t.Fatalf("embedding gather contract does not require exact finite parity: %+v", contract.Bounds)
	}
	foundCreditable := false
	for _, selector := range DefaultCreditableSubkernelSelectors {
		foundCreditable = foundCreditable || selector == embeddingGatherSelector
	}
	if foundCreditable {
		t.Fatal("embedding gather selector became default-creditable before its physical compute emitter landed")
	}
	if _, err := FilterSubkernelSpecs([]string{embeddingGatherSelector, "unknown-gather"}); err == nil || !strings.Contains(err.Error(), "unknown-gather") {
		t.Fatalf("mixed unknown selector did not fail closed: %v", err)
	}
}

func TestBatchedCopyAblationAcceptsPairedEvidence(t *testing.T) {
	original := executeStrixAblationCommandFn
	t.Cleanup(func() { executeStrixAblationCommandFn = original })
	executeStrixAblationCommandFn = func(_ context.Context, _ *StrixTarget, envVars, testPattern string) (string, time.Duration, error) {
		if envVars != "FAK_VULKAN_EMBEDDING_GATHER_ABLATION=1" {
			t.Fatalf("envVars=%q", envVars)
		}
		if testPattern != "^"+embeddingGatherTestName+"$" {
			t.Fatalf("testPattern=%q", testPattern)
		}
		return embeddingGatherTestOutput(t, nil), time.Millisecond, nil
	}

	results, err := RunStrixAblations(context.Background(), &StrixTarget{Host: "local", Mode: "local", Reachable: true}, []string{embeddingGatherAblationName})
	if err != nil {
		t.Fatalf("batched-copy selector rejected: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("results=%d, want 1", len(results))
	}
	got := results[0]
	if got.Feature != embeddingGatherAblationName || got.Dimension != "embedding" || got.Verdict != "VERIFIED_LIFT" {
		t.Fatalf("unexpected result: %+v", got)
	}
	if got.BaselineArm.Name != embeddingGatherBaselineArm || got.CandidateArm.Name != embeddingGatherAblationName || got.BaselineArm.Samples != 5 || got.CandidateArm.Samples != 5 {
		t.Fatalf("paired arm identity/sample counts lost: %+v", got)
	}
	if got.BaselineArm.LatencyUS != 118 || got.CandidateArm.LatencyUS != 81 || got.CosineParity != 1 || got.Speedup < 1.05 {
		t.Fatalf("unexpected exact-parity median result: %+v", got)
	}
	count, err := validateAblationSelectors([]string{embeddingGatherAblationName})
	if err != nil || count != 1 {
		t.Fatalf("batched-copy catalog count=%d err=%v", count, err)
	}
	if _, err := validateAblationSelectors([]string{embeddingGatherAblationName, "unknown-ablation"}); err == nil || !strings.Contains(err.Error(), "unknown-ablation") {
		t.Fatalf("mixed unknown ablation did not fail closed: %v", err)
	}
}

func TestBatchedCopyAblationFailsClosedOnIncompleteEvidence(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*embeddingGatherAblationEvent)
	}{
		{name: "fewer than five samples", mutate: func(e *embeddingGatherAblationEvent) { e.CandidateArm.SamplesUS = e.CandidateArm.SamplesUS[:4] }},
		{name: "fallback omitted", mutate: func(e *embeddingGatherAblationEvent) { e.FallbackCount = nil }},
		{name: "fallback observed", mutate: func(e *embeddingGatherAblationEvent) { one := 1; e.FallbackCount = &one }},
		{name: "parity not exact", mutate: func(e *embeddingGatherAblationEvent) { no := false; e.ExactParity = &no }},
		{name: "external engine", mutate: func(e *embeddingGatherAblationEvent) { e.Engine = "llama.cpp" }},
		{name: "submissions omitted", mutate: func(e *embeddingGatherAblationEvent) { e.CandidateArm.Submissions = nil }},
		{name: "submissions not reduced", mutate: func(e *embeddingGatherAblationEvent) { *e.CandidateArm.Submissions = *e.BaselineArm.Submissions }},
		{name: "bytes omitted", mutate: func(e *embeddingGatherAblationEvent) { e.BaselineArm.Bytes = nil }},
		{name: "bytes differ", mutate: func(e *embeddingGatherAblationEvent) { *e.CandidateArm.Bytes = *e.BaselineArm.Bytes - 1 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseEmbeddingGatherAblationEvent(embeddingGatherTestOutput(t, tc.mutate)); err == nil {
				t.Fatal("incomplete or unsafe A/B evidence was accepted")
			}
		})
	}

	original := executeStrixAblationCommandFn
	t.Cleanup(func() { executeStrixAblationCommandFn = original })
	executeStrixAblationCommandFn = func(context.Context, *StrixTarget, string, string) (string, time.Duration, error) {
		lines := strings.SplitN(embeddingGatherTestOutput(t, nil), "\n", 2)
		return lines[1], time.Millisecond, nil
	}
	result, err := runEmbeddingGatherBatchedCopyAblation(context.Background(), &StrixTarget{Reachable: true})
	if err == nil || result.Verdict != "REGRESSION" || !strings.Contains(err.Error(), "parity event") {
		t.Fatalf("missing signed parity event did not fail closed: result=%+v err=%v", result, err)
	}
}
