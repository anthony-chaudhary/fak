package agent

import (
	"strings"
	"testing"
)

// A Gemini-CLI / google-genai-shaped request: systemInstruction, a prior model
// turn with a functionCall, and a user turn carrying the matching functionResponse.
// Schema type names are UPPERCASE per the Gemini convention.
const geminiReq = `{
  "systemInstruction": {"parts":[{"text":"You are a coding agent."}]},
  "tools": [{"functionDeclarations":[{"name":"Bash","description":"run a command","parameters":{"type":"OBJECT","properties":{"command":{"type":"STRING"}}}}]}],
  "contents": [
    {"role":"user","parts":[{"text":"list the files"}]},
    {"role":"model","parts":[{"text":"I'll list them."},{"functionCall":{"name":"Bash","args":{"command":"ls"},"id":"g_01"}}]},
    {"role":"user","parts":[{"functionResponse":{"name":"Bash","id":"g_01","response":{"output":"a.go\nb.go"}}}]}
  ],
  "generationConfig": {"maxOutputTokens":4096,"temperature":1,"topP":0.9,"topK":40,"stopSequences":["X"]}
}`

func TestDecodeGeminiGenerateContentRequest(t *testing.T) {
	req, err := DecodeGeminiGenerateContentRequest([]byte(geminiReq), "gemini-2.5-flash")
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.Model != "gemini-2.5-flash" {
		t.Errorf("model (from path) = %q", req.Model)
	}
	if req.MaxTokens != 4096 {
		t.Errorf("maxOutputTokens = %d", req.MaxTokens)
	}
	if req.Temperature != 1 {
		t.Errorf("temperature = %v", req.Temperature)
	}
	if req.TopP == nil || *req.TopP != 0.9 {
		t.Errorf("topP = %v", req.TopP)
	}
	if req.TopK == nil || *req.TopK != 40 {
		t.Errorf("topK = %v", req.TopK)
	}
	if len(req.StopSequences) != 1 || req.StopSequences[0] != "X" {
		t.Errorf("stopSequences = %v", req.StopSequences)
	}
	if req.System != "You are a coding agent." {
		t.Errorf("systemInstruction not folded: %q", req.System)
	}
	// system (prepended) + user + assistant + tool result = 4 canonical messages.
	if len(req.Messages) != 4 {
		t.Fatalf("messages = %d, want 4: %+v", len(req.Messages), req.Messages)
	}
	if req.Messages[0].Role != RoleSystem || req.Messages[0].Content != "You are a coding agent." {
		t.Errorf("messages[0] not system: %+v", req.Messages[0])
	}
	if req.Messages[1].Role != RoleUser || req.Messages[1].Content != "list the files" {
		t.Errorf("messages[1] (text part) wrong: %+v", req.Messages[1])
	}
	asst := req.Messages[2]
	if asst.Role != RoleAssistant || asst.Content != "I'll list them." {
		t.Errorf("assistant text wrong: %+v", asst)
	}
	if len(asst.ToolCalls) != 1 || asst.ToolCalls[0].ID != "g_01" || asst.ToolCalls[0].Function.Name != "Bash" {
		t.Fatalf("functionCall not decoded (id must survive): %+v", asst.ToolCalls)
	}
	if !strings.Contains(asst.ToolCalls[0].Function.Arguments, `"command":"ls"`) {
		t.Errorf("functionCall args not kept as raw args: %q", asst.ToolCalls[0].Function.Arguments)
	}
	tr := req.Messages[3]
	if tr.Role != RoleTool || tr.ToolCallID != "g_01" || tr.Name != "Bash" {
		t.Errorf("functionResponse not mapped to RoleTool keyed by id: %+v", tr)
	}
	if !strings.Contains(tr.Content, "a.go") {
		t.Errorf("functionResponse content not serialized: %q", tr.Content)
	}
	if len(req.Tools) != 1 || req.Tools[0].Function.Name != "Bash" {
		t.Fatalf("tools not mapped: %+v", req.Tools)
	}
	// The UPPERCASE Gemini schema type names MUST normalize to lowercase so the
	// canonical ToolDef matches the lowercase OpenAI-style JSON Schema every other
	// inbound path produces (regardless of which upstream ultimately serves it).
	if got := string(req.Tools[0].Function.Parameters); !strings.Contains(got, `"type":"object"`) || strings.Contains(got, "OBJECT") || strings.Contains(got, "STRING") {
		t.Errorf("schema type names not lowercased to canonical form: %s", got)
	}
}

// TestDecodeGeminiEmptyArgsNormalized pins that a functionCall with absent/nil args
// decodes to the canonical "{}" the kernel adjudicates — never an empty string that
// would bypass the grammar rung's object expectation.
func TestDecodeGeminiEmptyArgsNormalized(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"no args field", `{"contents":[{"role":"model","parts":[{"functionCall":{"name":"f","id":"x"}}]}]}`},
		{"null args", `{"contents":[{"role":"model","parts":[{"functionCall":{"name":"f","args":null,"id":"x"}}]}]}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req, err := DecodeGeminiGenerateContentRequest([]byte(c.body), "m")
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if len(req.Messages) != 1 || len(req.Messages[0].ToolCalls) != 1 {
				t.Fatalf("messages: %+v", req.Messages)
			}
			if got := req.Messages[0].ToolCalls[0].Function.Arguments; got != "{}" {
				t.Errorf("args = %q, want {}", got)
			}
		})
	}
}

// TestGeminiResponsePartsRendering covers the post-adjudication render: prose +
// surviving tool calls become ordered parts, args normalize to JSON objects, and
// the id survives for the functionResponse round-trip.
func TestGeminiResponsePartsRendering(t *testing.T) {
	parts := GeminiResponseParts(Message{
		Role:    RoleAssistant,
		Content: "checking",
		ToolCalls: []ToolCall{
			{ID: "g1", Type: "function", Function: Func{Name: "allow_a", Arguments: `{"x":1}`}},
			{ID: "g2", Type: "function", Function: Func{Name: "deny_b", Arguments: ``}},        // empty -> {}
			{ID: "g3", Type: "function", Function: Func{Name: "bad_c", Arguments: `not-json`}}, // malformed -> {}
		},
	})
	if len(parts) != 4 || parts[0].Text != "checking" {
		t.Fatalf("parts = %+v", parts)
	}
	wantArgs := []string{`{"x":1}`, `{}`, `{}`}
	wantIDs := []string{"g1", "g2", "g3"}
	for i, w := range wantArgs {
		fc := parts[i+1].FunctionCall
		if fc == nil {
			t.Fatalf("part %d missing functionCall", i)
		}
		if string(fc.Args) != w {
			t.Errorf("part %d args = %s, want %s", i, fc.Args, w)
		}
		if fc.ID != wantIDs[i] {
			t.Errorf("part %d id = %q, want %q", i, fc.ID, wantIDs[i])
		}
	}
	// A pure tool-call turn (no prose) renders only the functionCall parts.
	pure := GeminiResponseParts(Message{Role: RoleAssistant, ToolCalls: []ToolCall{
		{ID: "g", Type: "function", Function: Func{Name: "f", Arguments: `{}`}},
	}})
	if len(pure) != 1 || pure[0].FunctionCall == nil {
		t.Fatalf("pure tool turn parts = %+v", pure)
	}
}

func TestGeminiFinishReason(t *testing.T) {
	cases := []struct{ in, want string }{
		{"tool_calls", "STOP"}, // Gemini signals tool use via the parts, not the reason.
		{"stop", "STOP"},
		{"STOP", "STOP"},
		{"length", "MAX_TOKENS"},
		{"max_tokens", "MAX_TOKENS"},
		{"", "STOP"},
	}
	for _, c := range cases {
		if got := GeminiFinishReason(c.in); got != c.want {
			t.Errorf("GeminiFinishReason(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestGeminiDecodeMarshalRoundTrip pins the SERVER decode against the CLIENT
// adapter marshal: a transcript the outbound adapter would PRODUCE, fak receives
// back as an inbound request (a Gemini client echoing a prior turn), and the decode
// must recover the canonical shape. This is the symmetry that keeps the wire
// faithful in both directions.
func TestGeminiDecodeMarshalRoundTrip(t *testing.T) {
	original := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "hello"},
		{Role: RoleAssistant, Content: "hi", ToolCalls: []ToolCall{
			{ID: "c1", Type: "function", Function: Func{Name: "search", Arguments: `{"q":"x"}`}},
		}},
		{Role: RoleTool, ToolCallID: "c1", Name: "search", Content: `{"hits":3}`},
	}
	a, err := NewTranscriptAdapter(ProviderGemini)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := a.MarshalRequest(adapterRequest{Messages: original})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, err := DecodeGeminiGenerateContentRequest(wire, "m")
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	// The decode re-prepends the system message, so compare the tail.
	if decoded.System != "sys" {
		t.Errorf("system round-trip: %q", decoded.System)
	}
	var calls int
	for _, m := range decoded.Messages {
		if m.Role == RoleAssistant {
			calls += len(m.ToolCalls)
		}
		if m.Role == RoleTool && m.ToolCallID != "c1" {
			t.Errorf("tool id not round-tripped: %+v", m)
		}
	}
	if calls != 1 {
		t.Errorf("assistant tool calls round-tripped = %d, want 1", calls)
	}
}
