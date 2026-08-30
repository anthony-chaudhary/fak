package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestDecodeResponsesFunctionCallOutputStringOrTextArray(t *testing.T) {
	tests := []struct {
		name   string
		output string
		want   string
	}{
		{name: "string", output: `"line one\nline two"`, want: "line one\nline two"},
		{name: "text array", output: `[{"type":"input_text","text":"line one"},{"type":"input_text","text":"line two"}]`, want: "line one\nline two"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw := json.RawMessage(`[{"type":"function_call_output","call_id":"call_1","output":` + tc.output + `}]`)
			messages, err := decodeResponsesInput(raw, "")
			if err != nil {
				t.Fatalf("decodeResponsesInput: %v", err)
			}
			if len(messages) != 1 {
				t.Fatalf("messages = %d, want 1: %+v", len(messages), messages)
			}
			got := messages[0]
			if got.Role != agent.RoleTool || got.ToolCallID != "call_1" || got.Content != tc.want {
				t.Fatalf("tool result = %+v, want role=%q call_id=call_1 content=%q", got, agent.RoleTool, tc.want)
			}
		})
	}
}

func TestDecodeResponsesFunctionCallOutputRejectsEmptyArray(t *testing.T) {
	_, err := decodeResponsesFunctionCallOutput(json.RawMessage(`[]`))
	if err == nil || !strings.Contains(err.Error(), "content array must not be empty") {
		t.Fatalf("err=%v", err)
	}
}
func TestDecodeResponsesFunctionCallOutputRejectsUnsupportedOrMalformedArrays(t *testing.T) {
	tests := []struct {
		name   string
		output string
		want   string
	}{
		{name: "image", output: `[{"type":"input_image","image_url":"https://example.invalid/image.png"}]`, want: `unsupported content type "input_image"`},
		{name: "file", output: `[{"type":"input_file","file_id":"file_1"}]`, want: `unsupported content type "input_file"`},
		{name: "missing type", output: `[{"text":"missing tag"}]`, want: "content type is required"},
		{name: "missing text", output: `[{"type":"input_text"}]`, want: "input_text.text is required"},
		{name: "wrong text type", output: `[{"type":"input_text","text":7}]`, want: "content array is malformed"},
		{name: "object", output: `{"type":"input_text","text":"not an array"}`, want: "must be a string or an array"},
		{name: "null", output: `null`, want: "must be a string or an array"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw := json.RawMessage(`[{"type":"function_call_output","call_id":"call_1","output":` + tc.output + `}]`)
			_, err := decodeResponsesInput(raw, "")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestDecodeResponsesFunctionCallOutputRejectsTooManyParts(t *testing.T) {
	parts := make([]map[string]any, maxResponsesFunctionOutputParts+1)
	for i := range parts {
		parts[i] = map[string]any{"type": "input_text", "text": "x"}
	}
	output, err := json.Marshal(parts)
	if err != nil {
		t.Fatal(err)
	}
	raw := json.RawMessage(`[{"type":"function_call_output","call_id":"call_1","output":` + string(output) + `}]`)
	_, err = decodeResponsesInput(raw, "")
	if err == nil || !strings.Contains(err.Error(), "too many content parts") {
		t.Fatalf("error = %v, want too-many-parts refusal", err)
	}
}

func TestDecodeResponsesFunctionCallOutputRequiresOutput(t *testing.T) {
	raw := json.RawMessage(`[{"type":"function_call_output","call_id":"call_1"}]`)
	_, err := decodeResponsesInput(raw, "")
	if err == nil || !strings.Contains(err.Error(), "function_call_output.output is required") {
		t.Fatalf("error = %v, want required-output refusal", err)
	}
}

type captureResponsesInputPlanner struct {
	messages []agent.Message
}

func (p *captureResponsesInputPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.messages = append([]agent.Message(nil), messages...)
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "continued"},
		FinishReason: "stop",
	}, nil
}

func (*captureResponsesInputPlanner) Model() string { return "capture" }

func TestResponsesFunctionCallOutputTextArrayReachesResultAdmission(t *testing.T) {
	srv := newTestServer(t)
	planner := &captureResponsesInputPlanner{}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, resp := postResponses(t, ts.URL, map[string]any{
		"model": "client",
		"input": []map[string]any{
			{"type": "message", "role": "user", "content": "continue"},
			{"type": "function_call", "call_id": "call_1", "name": "update_plan", "arguments": `{}`},
			{"type": "function_call_output", "call_id": "call_1", "output": []map[string]any{
				{"type": "input_text", "text": "Plan"},
				{"type": "input_text", "text": "updated"},
			}},
		},
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if len(planner.messages) != 3 {
		t.Fatalf("planner messages = %d, want 3: %+v", len(planner.messages), planner.messages)
	}
	tool := planner.messages[2]
	if tool.Role != agent.RoleTool || tool.ToolCallID != "call_1" || tool.Content != "Plan\nupdated" {
		t.Fatalf("planner tool result = %+v", tool)
	}
	if resp.Fak == nil || len(resp.Fak.ResultAdmissions) != 1 || resp.Fak.ResultAdmissions[0].Tool != "update_plan" {
		t.Fatalf("structured tool result did not reach result admission: %+v", resp.Fak)
	}
}

func TestResponsesFunctionCallOutputUnsupportedArrayIsBadRequest(t *testing.T) {
	srv := newTestServer(t)
	planner := &captureResponsesInputPlanner{}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := []byte(`{"model":"client","input":[{"type":"message","role":"user","content":"continue"},{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_image","image_url":"https://example.invalid/image.png"}]}]}`)
	resp, err := http.Post(ts.URL+"/v1/responses", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", resp.StatusCode, got)
	}
	if !strings.Contains(string(got), `unsupported content type \"input_image\"`) {
		t.Fatalf("response = %s, want typed unsupported-content refusal", got)
	}
	if len(planner.messages) != 0 {
		t.Fatalf("planner reached with unsupported tool output: %+v", planner.messages)
	}
}
