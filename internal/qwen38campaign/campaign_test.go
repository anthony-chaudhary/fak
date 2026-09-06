package qwen38campaign

import (
	"encoding/json"
	"testing"
)

func TestCampaignGenerateReceipt(t *testing.T) {
	scenarios := []string{
		ScenarioSharedPrefixForked,
		ScenarioWarmSamePrefix,
		ScenarioCold,
	}

	for _, sc := range scenarios {
		t.Run(sc, func(t *testing.T) {
			receipt, err := GenerateReceipt(sc, 4, 5, 16)
			if err != nil {
				t.Fatalf("GenerateReceipt failed: %v", err)
			}
			if err := receipt.Validate(); err != nil {
				t.Fatalf("receipt validation failed: %v", err)
			}
			if receipt.Schema != SubagentFanoutSchema {
				t.Errorf("schema = %q, want %q", receipt.Schema, SubagentFanoutSchema)
			}
			if receipt.Engine != "fak-native" {
				t.Errorf("engine = %q, want fak-native", receipt.Engine)
			}
			if !receipt.Summary.ParityPassed {
				t.Errorf("parity passed = false")
			}

			// Test round-trip marshal/unmarshal and digest verification
			data, err := json.Marshal(receipt)
			if err != nil {
				t.Fatalf("json.Marshal failed: %v", err)
			}
			var unmarshaled SubagentFanoutReceipt
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Fatalf("json.Unmarshal failed: %v", err)
			}
			if err := unmarshaled.Validate(); err != nil {
				t.Fatalf("unmarshaled Validate failed: %v", err)
			}
		})
	}
}
