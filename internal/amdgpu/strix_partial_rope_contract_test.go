package amdgpu

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// partialRoPEParityEvent renders a fak.strix.subkernel-parity/v1 event for the
// qwen35-partial-rope selector, permitting tests to perturb one field at a time
// and prove the registered contract fails closed.
func partialRoPEParityEvent(mutate func(*StrixSubkernelParityEvent)) string {
	event := StrixSubkernelParityEvent{
		Schema:         StrixSubkernelParitySchema,
		Selector:       partialRoPESelector,
		TestName:       partialRoPETestName,
		OracleKind:     OracleMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		CaseCount:      6,
		Passed:         true,
		Observed: StrixSubkernelObservedMetrics{
			MaxAbsDelta:  floatPtr(2e-3),
			FiniteOutput: boolPtr(true),
		},
	}
	if mutate != nil {
		mutate(&event)
	}
	raw, err := json.Marshal(&event)
	if err != nil {
		panic(err)
	}
	return string(raw)
}

func TestQwen35PartialRoPESelectorRegistration(t *testing.T) {
	specs, err := FilterSubkernelSpecs([]string{partialRoPESelector})
	if err != nil {
		t.Fatalf("selector rejected: %v", err)
	}
	if len(specs) != 1 || specs[0].Name != partialRoPESelector || specs[0].TestPattern != "^"+partialRoPETestName+"$" {
		t.Fatalf("unexpected partial-RoPE spec: %+v", specs)
	}
	if specs[0].Category != "positional" {
		t.Fatalf("partial-RoPE spec category = %q, want positional", specs[0].Category)
	}
	contract, ok := LookupSubkernelParityContract(partialRoPESelector)
	if !ok {
		t.Fatal("partial-RoPE parity contract missing")
	}
	if contract.TestName != partialRoPETestName || contract.OracleKind != OracleMaxAbs || contract.Engine != StrixVulkanEngine || !contract.DeviceObserved {
		t.Fatalf("unexpected partial-RoPE parity contract: %+v", contract)
	}
	if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta != 2e-3 || !contract.Bounds.RequireFinite {
		t.Fatalf("partial-RoPE contract bounds do not require finite 2e-3 parity: %+v", contract.Bounds)
	}
	// The partial-RoPE table emitter supplied by #12534 does not yet emit a
	// fak.strix.subkernel-parity/v1 event, so the selector must stay out of the
	// default-creditable set until its source-bound physical emitter lands.
	for _, selector := range DefaultCreditableSubkernelSelectors {
		if selector == partialRoPESelector {
			t.Fatal("partial-RoPE selector became default-creditable before its physical compute emitter landed")
		}
	}
	if _, err := FilterSubkernelSpecs([]string{partialRoPESelector, "unknown-partial-rope"}); err == nil || !strings.Contains(err.Error(), "unknown-partial-rope") {
		t.Fatalf("mixed unknown selector did not fail closed: %v", err)
	}
}

func TestQwen35PartialRoPESelectorParityFailsClosed(t *testing.T) {
	if _, err := ParseStrixSubkernelParity(partialRoPEParityEvent(nil), partialRoPESelector); err != nil {
		t.Fatalf("well-formed partial-RoPE parity event rejected: %v", err)
	}
	cases := []struct {
		name   string
		mutate func(*StrixSubkernelParityEvent)
	}{
		{name: "wrong engine", mutate: func(e *StrixSubkernelParityEvent) { e.Engine = "llama.cpp" }},
		{name: "device not observed", mutate: func(e *StrixSubkernelParityEvent) { e.DeviceObserved = false }},
		{name: "not passed", mutate: func(e *StrixSubkernelParityEvent) { e.Passed = false }},
		{name: "delta out of bounds", mutate: func(e *StrixSubkernelParityEvent) { e.Observed.MaxAbsDelta = floatPtr(2.1e-3) }},
		{name: "delta omitted", mutate: func(e *StrixSubkernelParityEvent) { e.Observed.MaxAbsDelta = nil }},
		{name: "non finite", mutate: func(e *StrixSubkernelParityEvent) { e.Observed.FiniteOutput = boolPtr(false) }},
		{name: "finite omitted", mutate: func(e *StrixSubkernelParityEvent) { e.Observed.FiniteOutput = nil }},
		{name: "wrong test name", mutate: func(e *StrixSubkernelParityEvent) { e.TestName = "TestVulkanSomethingElse" }},
		{name: "wrong oracle kind", mutate: func(e *StrixSubkernelParityEvent) { e.OracleKind = OracleExactArgmax }},
		{name: "wrong selector", mutate: func(e *StrixSubkernelParityEvent) { e.Selector = "qwen35-sequence-kv" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseStrixSubkernelParity(partialRoPEParityEvent(tc.mutate), partialRoPESelector); err == nil {
				t.Fatal("unsafe or incomplete partial-RoPE parity evidence was accepted")
			}
		})
	}
}

func TestPrecomputedTableAblationSelectorRegistration(t *testing.T) {
	count, err := validateAblationSelectors([]string{partialRoPEAblationName})
	if err != nil {
		t.Fatalf("precomputed-table ablation selector rejected: %v", err)
	}
	if count != 1 {
		t.Fatalf("precomputed-table selected %d arms, want 1", count)
	}
	count, err = validateAblationSelectors([]string{"positional"})
	if err != nil {
		t.Fatalf("positional ablation dimension rejected: %v", err)
	}
	if count != 1 {
		t.Fatalf("positional dimension selected %d arms, want 1", count)
	}
	if _, err := validateAblationSelectors([]string{partialRoPEAblationName, "unknown-ablation"}); err == nil || !strings.Contains(err.Error(), "unknown-ablation") {
		t.Fatalf("mixed unknown ablation did not fail closed: %v", err)
	}
	if _, err := RunStrixAblations(context.Background(), nil, []string{partialRoPEAblationName, "unknown-ablation"}); err == nil || !strings.Contains(err.Error(), "unknown-ablation") {
		t.Fatalf("execution did not fail before target on unknown selector: %v", err)
	}
}
