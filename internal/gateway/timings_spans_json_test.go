package gateway

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// The spans declaration must survive the real response builder, not only the
// Timings struct, or a client reading the response cannot tell what each number timed.
// The non-stream chat route serves buildChatResponse's output unchanged (http.go).
func TestChatResponseCarriesSpanDeclarations(t *testing.T) {
	tm := agent.NewTimings(100, 0, 10, 0.5, 0.2)
	tm.FirstEmitMS = 600
	s := &Server{}
	resp := s.buildChatResponse(&agent.Completion{Timings: tm}, agent.Message{Role: "assistant", Content: "hi"}, "stop", "m", nil, nil, nil, false, false, nil)
	raw, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Timings struct {
			Spans map[string]string `json:"spans"`
		} `json:"timings"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"prompt_ms":     "prefill_start..prefill_end",
		"predicted_ms":  "prefill_end..last_token",
		"first_emit_ms": "request_received..first_byte_sent",
	}
	for field, span := range want {
		if got.Timings.Spans[field] != span {
			t.Fatalf("response spans[%s] = %q, want %q (all: %v)", field, got.Timings.Spans[field], span, got.Timings.Spans)
		}
	}
}
