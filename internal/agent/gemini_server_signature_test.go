package agent

import (
	"encoding/json"
	"testing"
)

func TestGeminiServerThoughtSignatureRoundTrip(t *testing.T) {
	const signature = "opaque-signed-tool-context"
	// Render after adjudication changes arguments, retaining the opaque signature
	// on the same function-call part. The server must also decode it on replay.
	parts := GeminiResponseParts(Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "g1", Type: "function", ThoughtSignature: signature, Function: Func{Name: "read", Arguments: `{"filePath":"safe.txt"}`}}}})
	encoded, err := json.Marshal(map[string]any{"contents": []any{map[string]any{"role": "model", "parts": parts}}})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Contents []struct {
			Parts []struct {
				Signature string `json:"thoughtSignature"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Contents[0].Parts[0].Signature != signature {
		t.Fatalf("signature lost on response: %s", encoded)
	}
	req, err := DecodeGeminiGenerateContentRequest(encoded, "gemini-test")
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 1 || len(req.Messages[0].ToolCalls) != 1 || req.Messages[0].ToolCalls[0].ThoughtSignature != signature || req.Messages[0].ToolCalls[0].Function.Arguments != `{"filePath":"safe.txt"}` {
		t.Fatalf("signature or repaired arguments lost on replay: %+v", req.Messages)
	}
}
