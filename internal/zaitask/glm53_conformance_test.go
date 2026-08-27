package zaitask

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestGLM53FlashNonStreamingContract(t *testing.T) {
	var request map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(mustFixture(t, "glm53_nonstream.json"))
	}))
	defer srv.Close()

	got, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).RunChat(context.Background(), Request{
		Model:     GLM53FlashModel,
		Messages:  []Message{{Role: "user", Content: "weather"}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	thinking := request["thinking"].(map[string]any)
	if thinking["type"] != "enabled" || request["reasoning_effort"] != "max" {
		t.Fatalf("mandatory reasoning defaults missing: %#v", request)
	}
	if got.ReasoningContent != "reason in order" || got.FinishReason != "tool_calls" || got.Usage.CachedTokens != 5 {
		t.Fatalf("result = %#v", got)
	}
	if got.Provider != "z.ai" || got.Engine != "zai-hosted" || got.FakNative {
		t.Fatalf("hosted identity = %#v", got)
	}
	args, err := got.ToolCalls[0].Function.ArgumentsJSON()
	if err != nil || string(args) != `{"city":"Paris"}` {
		t.Fatalf("args=%s err=%v", args, err)
	}
}

func TestGLM53FlashSSEContract(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(mustFixture(t, "glm53_stream.sse"))
	}))
	defer srv.Close()
	got, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).RunChat(context.Background(), Request{
		Model: GLM53FlashModel, Stream: true,
		Messages: []Message{{Role: "user", Content: "weather"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.ReasoningContent != "first second" || got.Content != "answer" || !got.Streamed || !got.Done || got.Usage.CachedTokens != 3 {
		t.Fatalf("stream result = %#v", got)
	}
	args, err := got.ToolCalls[0].Function.ArgumentsJSON()
	if err != nil || string(args) != `{"city":"Paris"}` {
		t.Fatalf("args=%s err=%v", args, err)
	}
}

func TestGLM53FlashPreservedThinkingHistory(t *testing.T) {
	clear := false
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		io.WriteString(w, `{"model":"glm-5.3-flash","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()
	_, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).RunChat(context.Background(), Request{
		Model: GLM53FlashModel, ClearThinking: &clear,
		Messages: []Message{
			{Role: "user", Content: "first"},
			{Role: "assistant", Content: "visible", ReasoningContent: "exact historical bytes"},
			{Role: "user", Content: "next"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"clear_thinking":false`) || !strings.Contains(string(raw), `"reasoning_content":"exact historical bytes"`) {
		t.Fatalf("history changed or omitted: %s", raw)
	}

	_, err = (Client{BaseURL: srv.URL, APIKey: "secret"}).RunChat(context.Background(), Request{
		Model: GLM53FlashModel, ClearThinking: &clear,
		Messages: []Message{{Role: "user", Content: "first"}, {Role: "assistant", Content: "visible"}, {Role: "user", Content: "next"}},
	})
	if err == nil || !strings.Contains(err.Error(), "complete reasoning_content") {
		t.Fatalf("incomplete history error = %v", err)
	}
}

func TestGLM53FlashMultimodalFormsRoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		parts  []ContentPart
		probes CapabilityProbes
		wants  []string
	}{
		{
			name: "image URL and base64",
			parts: []ContentPart{
				{Type: "image_url", ImageURL: &URLContent{URL: "https://example.test/a.png"}},
				{Type: "image_url", ImageURL: &URLContent{URL: "data:image/png;base64,AAAA"}},
				{Type: "text", Text: "describe"},
			},
			wants: []string{"https://example.test/a.png", "data:image/png;base64,AAAA"},
		},
		{
			name:  "uploaded file",
			parts: []ContentPart{{Type: "file", File: &UploadedFile{FileID: "file-123"}}, {Type: "text", Text: "summarize"}},
			wants: []string{"file-123"},
		},
		{
			name:   "direct file URL probe",
			parts:  []ContentPart{{Type: "file_url", FileURL: &URLContent{URL: "https://example.test/doc.pdf"}}},
			probes: CapabilityProbes{DirectFileURL: true},
			wants:  []string{"https://example.test/doc.pdf"},
		},
		{
			name:   "direct video URL probe",
			parts:  []ContentPart{{Type: "video_url", VideoURL: &URLContent{URL: "https://example.test/v.mp4"}}},
			probes: CapabilityProbes{DirectVideoURL: true},
			wants:  []string{"https://example.test/v.mp4"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var raw string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				raw = string(b)
				io.WriteString(w, `{"model":"glm-5.3-flash","choices":[{"message":{"content":"text only"},"finish_reason":"stop"}]}`)
			}))
			defer srv.Close()
			_, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).RunChat(context.Background(), Request{
				Model: GLM53FlashModel, Probes: tc.probes,
				Messages: []Message{{Role: "user", Content: tc.parts}},
			})
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range tc.wants {
				if !strings.Contains(raw, want) {
					t.Fatalf("request omitted %q: %s", want, raw)
				}
			}
		})
	}
}

func TestGLM53FlashProbeGatesAndReasoningEffort(t *testing.T) {
	base := Request{Model: GLM53FlashModel, Messages: []Message{{Role: "user", Content: "x"}}}
	client := Client{APIKey: "secret", HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { t.Fatal("probe refusal reached network"); return nil, nil })}}
	tests := []struct {
		name   string
		mutate func(*Request)
		want   string
	}{
		{"effort", func(r *Request) { r.ReasoningEffort = "medium" }, "low, high, or max"},
		{"response format", func(r *Request) { r.ResponseFormat = &ResponseFormat{Type: "json_object"} }, "response_format probe"},
		{"tool stream", func(r *Request) { r.ToolStream = true; r.Stream = true }, "tool_stream probe"},
		{"file url", func(r *Request) {
			r.Messages[0].Content = []ContentPart{{Type: "file_url", FileURL: &URLContent{URL: "https://example.test/a.pdf"}}}
		}, "direct file URL probe"},
		{"video url", func(r *Request) {
			r.Messages[0].Content = []ContentPart{{Type: "video_url", VideoURL: &URLContent{URL: "https://example.test/a.mp4"}}}
		}, "direct video URL probe"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := base
			req.Messages = append([]Message(nil), base.Messages...)
			tc.mutate(&req)
			_, err := client.RunChat(context.Background(), req)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want %q", err, tc.want)
			}
		})
	}
}

func TestGLM53FlashDocumentedFinishReasons(t *testing.T) {
	for _, reason := range []string{"stop", "tool_calls", "length", "sensitive", "model_context_window_exceeded", "network_error"} {
		t.Run(reason, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				io.WriteString(w, `{"model":"glm-5.3-flash","choices":[{"message":{"content":"x"},"finish_reason":"`+reason+`"}]}`)
			}))
			defer srv.Close()
			got, err := (Client{BaseURL: srv.URL, APIKey: "secret"}).RunChat(context.Background(), Request{Model: GLM53FlashModel, Messages: []Message{{Role: "user", Content: "x"}}})
			if err != nil || got.FinishReason != reason {
				t.Fatalf("got=%#v err=%v", got, err)
			}
		})
	}
}

func TestGLM53FlashCapturedRequestFixture(t *testing.T) {
	var wire chatRequestWire
	if err := json.Unmarshal(mustFixture(t, "glm53_request.json"), &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Model != GLM53FlashModel || wire.Thinking.Type != "enabled" || wire.ReasoningEffort != "max" || !wire.Stream || !wire.ToolStream {
		t.Fatalf("captured request lost the hosted contract: %#v", wire)
	}
}

func mustFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
