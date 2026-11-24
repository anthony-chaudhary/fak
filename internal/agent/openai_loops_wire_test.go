package agent

import (
	"bytes"
	"encoding/json"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestOpenAILoopsWireVLLMXArgs(t *testing.T) {
	cases := []struct {
		name    string
		req     adapterRequest
		want    string // exact vllm_xargs JSON; "" means the key is absent
		wantErr bool
	}{
		{name: "unset omits", req: adapterRequest{}},
		{name: "set", req: adapterRequest{Loops: 3}, want: `{"loops":3}`},
		{name: "operator xargs merged", req: adapterRequest{Loops: 5, ExtraBody: json.RawMessage(`{"vllm_xargs":{"seed_mode":"x"},"top_k":20}`)},
			want: `{"loops":5,"seed_mode":"x"}`},
		{name: "per-request loops wins", req: adapterRequest{Loops: 2, ExtraBody: json.RawMessage(`{"vllm_xargs":{"loops":9}}`)},
			want: `{"loops":2}`},
		{name: "operator xargs not object", req: adapterRequest{Loops: 2, ExtraBody: json.RawMessage(`{"vllm_xargs":"loops=9"}`)}, wantErr: true},
		{name: "operator xargs without loops rides untouched", req: adapterRequest{ExtraBody: json.RawMessage(`{"vllm_xargs":{"seed_mode":"x"}}`)},
			want: `{"seed_mode":"x"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.req.Model = "m"
			tc.req.Messages = []Message{{Role: RoleUser, Content: "hi"}}
			raw, err := openAIAdapter{provider: ProviderOpenAI}.MarshalRequest(tc.req)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("marshal accepted %s", raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got map[string]json.RawMessage
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("decode %s: %v", raw, err)
			}
			xargs, ok := got["vllm_xargs"]
			if tc.want == "" {
				if ok {
					t.Fatalf("vllm_xargs present: %s", raw)
				}
				return
			}
			var gotX, wantX map[string]any
			_ = json.Unmarshal(xargs, &gotX)
			_ = json.Unmarshal([]byte(tc.want), &wantX)
			gb, _ := json.Marshal(gotX)
			wb, _ := json.Marshal(wantX)
			if !bytes.Equal(gb, wb) {
				t.Fatalf("vllm_xargs = %s, want %s", xargs, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestOpenAILoopsZeroIsByteIdentical(t *testing.T) {
	base := adapterRequest{Model: "m", Messages: []Message{{Role: RoleUser, Content: "hi"}}, ReasoningEffort: "high"}
	withZero := base
	withZero.Loops = 0
	a, err := openAIAdapter{provider: ProviderOpenAI}.MarshalRequest(base)
	if err != nil {
		t.Fatal(err)
	}
	b, err := openAIAdapter{provider: ProviderOpenAI}.MarshalRequest(withZero)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) || bytes.Contains(a, []byte("vllm_xargs")) {
		t.Fatalf("non-looped body changed:\n%s\n%s", a, b)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestWithLoopsIgnoresNonPositive(t *testing.T) {
	for _, n := range []int{-1, 0} {
		var sp SampleParams
		WithLoops(n)(&sp)
		if sp.Loops != 0 {
			t.Fatalf("WithLoops(%d) set Loops=%d", n, sp.Loops)
		}
	}
	var sp SampleParams
	WithLoops(4)(&sp)
	if sp.Loops != 4 {
		t.Fatalf("WithLoops(4) set Loops=%d", sp.Loops)
	}
}
