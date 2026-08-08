package polymodel

import (
	"reflect"
	"testing"
)

func TestNormalizeRequestProviderParity(t *testing.T) {
	temp := 0.25
	want := CanonicalRequest{Model: "reasoner", System: "be concise", Messages: []CanonicalMessage{{Role: "user", Content: "solve 2+2"}, {Role: "assistant", Content: "4"}}, Temperature: &temp, MaxTokens: 64}
	cases := map[Provider]string{
		ProviderOpenAI:    `{"model":"reasoner","messages":[{"role":"system","content":"be concise"},{"role":"user","content":"solve 2+2"},{"role":"assistant","content":"4"}],"temperature":0.25,"max_tokens":64}`,
		ProviderAnthropic: `{"model":"reasoner","system":[{"type":"text","text":"be concise"}],"messages":[{"role":"user","content":[{"type":"text","text":"solve 2+2"}]},{"role":"assistant","content":"4"}],"temperature":0.25,"max_tokens":64}`,
		ProviderGemini:    `{"model":"reasoner","system_instruction":{"parts":[{"text":"be concise"}]},"contents":[{"role":"user","parts":[{"text":"solve 2+2"}]},{"role":"model","parts":[{"text":"4"}]}],"generation_config":{"temperature":0.25,"max_output_tokens":64}}`,
	}
	for provider, raw := range cases {
		t.Run(string(provider), func(t *testing.T) {
			got, err := NormalizeRequest(provider, []byte(raw))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("got %#v want %#v", got, want)
			}
		})
	}
}

func TestNormalizeRequestFailsClosedOnSemanticLoss(t *testing.T) {
	cases := []struct {
		name     string
		provider Provider
		raw      string
	}{
		{"unknown-provider", "other", `{}`},
		{"unknown-field", ProviderOpenAI, `{"model":"m","messages":[{"role":"user","content":"x"}],"stream":true}`},
		{"tool-content", ProviderAnthropic, `{"model":"m","messages":[{"role":"user","content":[{"type":"tool_result","text":"x"}]}]}`},
		{"gemini-image", ProviderGemini, `{"model":"m","contents":[{"role":"user","parts":[{"inline_data":{}}]}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NormalizeRequest(tc.provider, []byte(tc.raw)); err == nil {
				t.Fatal("expected fail-closed error")
			}
		})
	}
}
