package zaitask

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"
)

// TestLiveGLM53FlashConformance is opt-in because it spends provider tokens and
// needs operator-owned media. When enabled it fails closed unless every required
// input is present, then emits one scrubbed provider/engine receipt to the test log.
func TestLiveGLM53FlashConformance(t *testing.T) {
	if os.Getenv("FAK_ZAI_GLM53_LIVE") != "1" {
		t.Skip("set FAK_ZAI_GLM53_LIVE=1 with the documented media env to run the hosted conformance receipt")
	}
	key := os.Getenv("ZAI_API_KEY")
	image := os.Getenv("FAK_ZAI_GLM53_IMAGE_URL")
	fileID := os.Getenv("FAK_ZAI_GLM53_FILE_ID")
	video := os.Getenv("FAK_ZAI_GLM53_VIDEO_URL")
	if key == "" || image == "" || fileID == "" || video == "" {
		t.Fatal("live conformance requires ZAI_API_KEY, FAK_ZAI_GLM53_IMAGE_URL, FAK_ZAI_GLM53_FILE_ID, and FAK_ZAI_GLM53_VIDEO_URL")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	client := Client{APIKey: key}
	checks := map[string]bool{}

	text, err := client.RunChat(ctx, Request{Model: GLM53FlashModel, Messages: []Message{{Role: "user", Content: "Reply with exactly: conformance"}}, MaxTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	checks["text_reasoning_usage_finish"] = text.Content != "" && text.ReasoningContent != "" && text.Usage.TotalTokens > 0 && text.FinishReason != ""

	stream, err := client.RunChat(ctx, Request{Model: GLM53FlashModel, Stream: true, Messages: []Message{{Role: "user", Content: "Reply with exactly: stream"}}, MaxTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	checks["sse_done"] = stream.Done && stream.Streamed

	tools := []ToolDefinition{{Type: "function", Function: ToolFunction{Name: "echo", Description: "Echo one string", Parameters: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}`)}}}
	tool, err := client.RunChat(ctx, Request{Model: GLM53FlashModel, Messages: []Message{{Role: "user", Content: "Call echo exactly once with value conformance."}}, Tools: tools, ToolChoice: "auto", MaxTokens: 128})
	if err != nil {
		t.Fatal(err)
	}
	checks["tool_call"] = tool.FinishReason == "tool_calls" && len(tool.ToolCalls) > 0

	media, err := client.RunChat(ctx, Request{Model: GLM53FlashModel, Probes: CapabilityProbes{DirectVideoURL: true}, Messages: []Message{{Role: "user", Content: []ContentPart{
		{Type: "image_url", ImageURL: &URLContent{URL: image}},
		{Type: "file", File: &UploadedFile{FileID: fileID}},
		{Type: "video_url", VideoURL: &URLContent{URL: video}},
		{Type: "text", Text: "Name the three input modalities briefly."},
	}}}, MaxTokens: 128})
	if err != nil {
		t.Fatal(err)
	}
	checks["image_file_video_text_output"] = media.Content != "" && media.Engine == HostedEngine && !media.FakNative

	for name, ok := range checks {
		if !ok {
			t.Errorf("live conformance check %s failed", name)
		}
	}
	receipt, _ := json.Marshal(map[string]any{
		"schema": "fak.zaitask.glm53-live-conformance.v1", "model": GLM53FlashModel,
		"provider": HostedProvider, "engine": HostedEngine, "fak_native": false,
		"checks": checks,
	})
	t.Logf("live conformance receipt: %s", receipt)
}
