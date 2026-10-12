package agent

import (
	"encoding/json"
	"reflect"
	"testing"
)

// TestOpenAIChatExtensionsYieldToOperatorExtraBody pins how a client's top_k, min_p
// and chat_template_kwargs meet an operator ExtraBody on the OpenAI chat wire: the
// operator's scalar values win, kwargs objects merge with operator keys winning, and
// no duplicate key makes the turn refuse. A TopK without ChatWireTopK stays off.
func TestOpenAIChatExtensionsYieldToOperatorExtraBody(t *testing.T) {
	k, minP := 40, 0.05
	cases := []struct {
		name  string
		req   adapterRequest
		want  map[string]any
		absnt []string
	}{
		{name: "no-extra", req: adapterRequest{ChatWireTopK: &k, MinP: &minP, ChatTemplateKwargs: json.RawMessage(`{"enable_thinking":true}`)},
			want: map[string]any{"top_k": 40.0, "min_p": 0.05, "chat_template_kwargs": map[string]any{"enable_thinking": true}}},
		{name: "operator-wins", req: adapterRequest{ChatWireTopK: &k, MinP: &minP, ChatTemplateKwargs: json.RawMessage(`{"enable_thinking":true,"x":1}`),
			ExtraBody: json.RawMessage(`{"top_k":20,"chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":true}}`)},
			want: map[string]any{"top_k": 20.0, "min_p": 0.05, "chat_template_kwargs": map[string]any{"enable_thinking": false, "preserve_thinking": true, "x": 1.0}}},
		{name: "native-topk-only", req: adapterRequest{TopK: &k}, absnt: []string{"top_k", "min_p", "chat_template_kwargs"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.req.Model = "m"
			tc.req.Messages = []Message{{Role: RoleUser, Content: "hi"}}
			raw, err := openAIAdapter{provider: ProviderOpenAI}.MarshalRequest(tc.req)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("decode: %v", err)
			}
			for key, want := range tc.want {
				if !reflect.DeepEqual(got[key], want) {
					t.Errorf("%s = %v, want %v", key, got[key], want)
				}
			}
			for _, key := range tc.absnt {
				if v, ok := got[key]; ok {
					t.Errorf("%s = %v, want absent", key, v)
				}
			}
		})
	}
}

// TestOpenAISpeculativeNMaxPerRequestWins pins the llama-server speculative.n_max
// carrier on the OpenAI chat wire: emitted only when set (0 included), a per-request
// value replaces an operator ExtraBody default without refusing the turn, an
// ExtraBody-only value still rides, and the xAI wire never carries it.
func TestOpenAISpeculativeNMaxPerRequestWins(t *testing.T) {
	zero, four := 0, 4
	cases := []struct {
		name     string
		provider Provider
		req      adapterRequest
		want     map[string]any
		absnt    []string
	}{
		{name: "unset", provider: ProviderOpenAI, absnt: []string{"speculative.n_max"}},
		{name: "set", provider: ProviderOpenAI, req: adapterRequest{SpeculativeNMax: &four},
			want: map[string]any{"speculative.n_max": 4.0}},
		{name: "zero-disables", provider: ProviderOpenAI, req: adapterRequest{SpeculativeNMax: &zero},
			want: map[string]any{"speculative.n_max": 0.0}},
		{name: "per-request-wins", provider: ProviderOpenAI, req: adapterRequest{SpeculativeNMax: &zero,
			ExtraBody: json.RawMessage(`{"speculative.n_max":8,"top_k":20}`)},
			want: map[string]any{"speculative.n_max": 0.0, "top_k": 20.0}},
		{name: "per-request-wins-sole-key", provider: ProviderOpenAI, req: adapterRequest{SpeculativeNMax: &four,
			ExtraBody: json.RawMessage(`{"speculative.n_max":8}`)},
			want: map[string]any{"speculative.n_max": 4.0}},
		{name: "operator-default", provider: ProviderOpenAI, req: adapterRequest{ExtraBody: json.RawMessage(`{"speculative.n_max":8}`)},
			want: map[string]any{"speculative.n_max": 8.0}},
		{name: "xai-omits", provider: ProviderXAI, req: adapterRequest{SpeculativeNMax: &four}, absnt: []string{"speculative.n_max"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.req.Model = "m"
			tc.req.Messages = []Message{{Role: RoleUser, Content: "hi"}}
			raw, err := openAIAdapter{provider: tc.provider}.MarshalRequest(tc.req)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("decode: %v", err)
			}
			for key, want := range tc.want {
				if v, ok := got[key]; !ok || !reflect.DeepEqual(v, want) {
					t.Errorf("%s = %v (present=%t), want %v", key, v, ok, want)
				}
			}
			for _, key := range tc.absnt {
				if v, ok := got[key]; ok {
					t.Errorf("%s = %v, want absent", key, v)
				}
			}
		})
	}
}

func TestWithSpeculativeNMaxIgnoresNilAndNegative(t *testing.T) {
	neg, three := -1, 3
	for _, tc := range []struct {
		name string
		in   *int
		want *int
	}{
		{name: "nil", in: nil, want: nil},
		{name: "negative", in: &neg, want: nil},
		{name: "positive", in: &three, want: &three},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var sp SampleParams
			WithSpeculativeNMax(tc.in)(&sp)
			if !reflect.DeepEqual(sp.SpeculativeNMax, tc.want) {
				t.Errorf("SpeculativeNMax = %v, want %v", sp.SpeculativeNMax, tc.want)
			}
		})
	}
}
