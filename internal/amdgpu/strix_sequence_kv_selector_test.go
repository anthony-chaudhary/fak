package amdgpu

import (
	"encoding/json"
	"strings"
	"testing"
)

// sequenceKVParityEvent renders a fak.strix.subkernel-parity/v1 event for the
// geometric Qwen sequence-KV reservation selector, permitting tests to perturb
// one field at a time and prove the contract fails closed.
func sequenceKVParityEvent(mutate func(*StrixSubkernelParityEvent)) string {
	event := StrixSubkernelParityEvent{
		Schema:         StrixSubkernelParitySchema,
		Selector:       sequenceKVSelector,
		TestName:       sequenceKVTestName,
		OracleKind:     OracleMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		CaseCount:      2,
		Passed:         true,
		Observed: StrixSubkernelObservedMetrics{
			MaxAbsDelta:  floatPtr(0),
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

func TestQwen35SequenceKVSelectorRegistration(t *testing.T) {
	specs, err := FilterSubkernelSpecs([]string{sequenceKVSelector})
	if err != nil {
		t.Fatalf("selector rejected: %v", err)
	}
	if len(specs) != 1 || specs[0].Name != sequenceKVSelector || specs[0].TestPattern != "^"+sequenceKVTestName+"$" {
		t.Fatalf("unexpected sequence-KV spec: %+v", specs)
	}
	contract, ok := LookupSubkernelParityContract(sequenceKVSelector)
	if !ok {
		t.Fatal("sequence-KV parity contract missing")
	}
	if contract.TestName != sequenceKVTestName || contract.OracleKind != OracleMaxAbs || contract.Engine != StrixVulkanEngine || !contract.DeviceObserved {
		t.Fatalf("unexpected sequence-KV parity contract: %+v", contract)
	}
	if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta != 0 || !contract.Bounds.RequireFinite {
		t.Fatalf("sequence-KV contract does not require exact finite parity: %+v", contract.Bounds)
	}
	foundCreditable := false
	for _, selector := range DefaultCreditableSubkernelSelectors {
		foundCreditable = foundCreditable || selector == sequenceKVSelector
	}
	if !foundCreditable {
		t.Fatal("sequence-KV selector is not registered as creditable")
	}
	if _, err := FilterSubkernelSpecs([]string{sequenceKVSelector, "unknown-sequence-kv"}); err == nil || !strings.Contains(err.Error(), "unknown-sequence-kv") {
		t.Fatalf("mixed unknown selector did not fail closed: %v", err)
	}
}

func TestQwen35SequenceKVSelectorParityFailsClosed(t *testing.T) {
	if _, err := ParseStrixSubkernelParity(sequenceKVParityEvent(nil), sequenceKVSelector); err != nil {
		t.Fatalf("well-formed sequence-KV parity event rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*StrixSubkernelParityEvent)
	}{
		{name: "wrong engine", mutate: func(e *StrixSubkernelParityEvent) { e.Engine = "llama.cpp" }},
		{name: "device not observed", mutate: func(e *StrixSubkernelParityEvent) { e.DeviceObserved = false }},
		{name: "not passed", mutate: func(e *StrixSubkernelParityEvent) { e.Passed = false }},
		{name: "delta out of bounds", mutate: func(e *StrixSubkernelParityEvent) { e.Observed.MaxAbsDelta = floatPtr(1e-3) }},
		{name: "delta omitted", mutate: func(e *StrixSubkernelParityEvent) { e.Observed.MaxAbsDelta = nil }},
		{name: "non finite", mutate: func(e *StrixSubkernelParityEvent) { e.Observed.FiniteOutput = boolPtr(false) }},
		{name: "finite omitted", mutate: func(e *StrixSubkernelParityEvent) { e.Observed.FiniteOutput = nil }},
		{name: "wrong test name", mutate: func(e *StrixSubkernelParityEvent) { e.TestName = "TestVulkanSomethingElse" }},
		{name: "wrong oracle kind", mutate: func(e *StrixSubkernelParityEvent) { e.OracleKind = OracleExactArgmax }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseStrixSubkernelParity(sequenceKVParityEvent(tc.mutate), sequenceKVSelector); err == nil {
				t.Fatal("unsafe or incomplete sequence-KV parity evidence was accepted")
			}
		})
	}
}
