package tb4bench

import (
	"fmt"
	"strings"
	"testing"
)

func TestOpenCodeAdapterInvocation(t *testing.T) {
	port, err := FindFreePort()
	if err != nil {
		t.Fatalf("failed to find free port: %v", err)
	}
	dynamicURL := fmt.Sprintf("http://127.0.0.1:%d/v1", port)

	// 1. Test configuration synthesis
	jsonBytes, err := GenerateOpenCodeJSON(dynamicURL, "qwen3.8-coder")
	if err != nil {
		t.Fatalf("failed to generate opencode.json: %v", err)
	}

	content := string(jsonBytes)
	if !strings.Contains(content, dynamicURL) {
		t.Errorf("config missing baseURL: %s", content)
	}
	if !strings.Contains(content, "qwen3.8-coder") {
		t.Errorf("config missing model name: %s", content)
	}
	if !strings.Contains(content, `"snapshot": false`) {
		t.Errorf("config missing snapshot: false: %s", content)
	}

	// 2. Test environment setup and teardown
	adapter := NewOpenCodeAdapter(OpenCodeConfig{
		ServerBaseURL: dynamicURL,
		ModelID:       "qwen3.8-coder",
	})
	runtimeDir, err := adapter.SetupEnvironment()
	if err != nil {
		t.Fatalf("failed to setup environment: %v", err)
	}
	if runtimeDir == "" {
		t.Errorf("runtime dir is empty")
	}
	if err := adapter.Teardown(); err != nil {
		t.Errorf("failed to teardown: %v", err)
	}

	// 3. Test transcript parser
	sampleOutput := `
Executing: bash with args: "ls -la"
Executing: edit with args: "main.py"
Tokens: 250 prompt, 45 completion
TASK_COMPLETED
`
	res, err := ParseOpenCodeTranscript([]byte(sampleOutput), nil, "tb4-task-01")
	if err != nil {
		t.Fatalf("failed to parse transcript: %v", err)
	}

	if res.TaskID != "tb4-task-01" {
		t.Errorf("expected task tb4-task-01, got %s", res.TaskID)
	}
	if res.TotalPromptTokens != 250 {
		t.Errorf("expected 250 prompt tokens, got %d", res.TotalPromptTokens)
	}
	if res.TotalCompletionTokens != 45 {
		t.Errorf("expected 45 completion tokens, got %d", res.TotalCompletionTokens)
	}
	if len(res.Turns) == 0 {
		t.Errorf("expected at least 1 turn, got 0")
	}
	if len(res.Turns[0].ToolCalls) < 2 {
		t.Errorf("expected at least 2 tool calls, got %d", len(res.Turns[0].ToolCalls))
	}
}

func TestOpenCodeTranscriptReportedTokensOnly(t *testing.T) {
	for _, tc := range []struct {
		name               string
		usage              string
		prompt, completion int64
		wantErr            string
	}{
		{name: "missing", wantErr: "missing reported token usage"},
		{name: "invalid prompt", usage: "Tokens: 9223372036854775808 prompt, 0 completion", wantErr: "invalid reported token usage"},
		{name: "invalid completion", usage: "Tokens: 0 prompt, 9223372036854775808 completion", wantErr: "invalid reported token usage"},
		{name: "malformed after valid", usage: "Tokens: 1 prompt, 2 completion\nTokens: -1 prompt, 2 completion", wantErr: "invalid reported token usage"},
		{name: "accumulation overflow", usage: "Tokens: 9223372036854775807 prompt, 9223372036854775807 completion\nTokens: 1 prompt, 1 completion", wantErr: "invalid reported token usage"},
		{name: "reported zero", usage: "Tokens: 0 prompt, 0 completion"},
		{name: "reported nonzero", usage: "Tokens: 250 prompt, 45 completion", prompt: 250, completion: 45},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stdout := strings.Repeat("diagnostic output without token usage\n", 100) + tc.usage + "\nTASK_COMPLETED\n"
			result, err := ParseOpenCodeTranscript([]byte(stdout), nil, "reported-tokens")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) || result != nil {
					t.Fatalf("result = %v, err = %v; want nil result and %q", result, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if result.TotalPromptTokens != tc.prompt || result.TotalCompletionTokens != tc.completion {
				t.Fatalf("tokens = (%d, %d), want reported (%d, %d)", result.TotalPromptTokens, result.TotalCompletionTokens, tc.prompt, tc.completion)
			}
		})
	}
}
