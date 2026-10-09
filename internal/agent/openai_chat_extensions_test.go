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
