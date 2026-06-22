// Tests for inferParams, the pure HF-model-id size extractor used to label
// parity cards. It is deterministic, regex-only, and touches no I/O, so its
// behavior can be pinned by table-driven cases.
package main

import "testing"

func TestInferParams(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  string
	}{
		{"decimal billions followed by B", "Qwen2.5-1.5B-Instruct", "1.5B"},
		{"integer millions", "SmolLM-135M", "135M"},
		{"lowercase b is upcased", "llama-3b", "3B"},
		{"whitespace between number and unit", "model 7 B", "7B"},
		{"plain integer billions", "Qwen2-0.5B", "0.5B"},
		{"first matching number wins, not first number", "v2-7B", "7B"},
		{"no size token present", "no-size-here", "?"},
		{"digits but no b/m unit", "checkpoint-1234", "?"},
		{"empty string", "", "?"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := inferParams(tt.model); got != tt.want {
				t.Errorf("inferParams(%q) = %q, want %q", tt.model, got, tt.want)
			}
		})
	}
}
