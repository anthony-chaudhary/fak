package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

const guardLaunchGrantCrossWireHelperEnv = "GUARD_LAUNCH_GRANT_CROSSWIRE_HELPER"

type guardLaunchGrantMatrixCase struct {
	name       string
	tool       string
	args       json.RawMessage
	wantAllow  bool
	wantReason string
}

var guardLaunchGrantMatrixCases = []guardLaunchGrantMatrixCase{
	{name: "exact benign grant", tool: "deploy_preview", args: json.RawMessage(`{}`), wantAllow: true},
	{name: "explicit deny", tool: "delete_file", args: json.RawMessage(`{"path":"notes.txt"}`), wantReason: "POLICY_BLOCK"},
	{name: "shell danger", tool: "opencode.bash", args: json.RawMessage(`{"command":"rm -rf /"}`), wantReason: "POLICY_BLOCK"},
	{name: "structural self modification", tool: "write_file", args: json.RawMessage(`{"path":".hermes/config.yaml","content":"x"}`), wantReason: "SELF_MODIFY"},
	{name: "ungranted default deny", tool: "unknown_probe", args: json.RawMessage(`{}`), wantReason: "DEFAULT_DENY"},
}

type guardLaunchGrantWire struct {
	name         string
	provider     string
	inboundPath  string
	upstreamPath string
	baseV1       bool
}

var guardLaunchGrantWires = []guardLaunchGrantWire{
	{name: "anthropic-messages", provider: "anthropic", inboundPath: "/v1/messages", upstreamPath: "/v1/messages"},
	{name: "openai-chat-completions", provider: "openai", inboundPath: "/v1/chat/completions", upstreamPath: "/v1/chat/completions", baseV1: true},
	{name: "openai-responses", provider: "openai-responses", inboundPath: "/v1/responses", upstreamPath: "/v1/responses", baseV1: true},
	{name: "gemini-generate-content", provider: "gemini", inboundPath: "/v1beta/models/guard-launch-grant:generateContent", upstreamPath: "/models/guard-launch-grant:generateContent"},
}

func TestGuardLaunchGrantCrossWireHostileMatrix(t *testing.T) {
	if os.Getenv(guardLaunchGrantCrossWireHelperEnv) == "1" {
		runGuardLaunchGrantCrossWireHostileMatrix(t)
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestGuardLaunchGrantCrossWireHostileMatrix$", "-test.v")
	cmd.Env = append(os.Environ(), guardLaunchGrantCrossWireHelperEnv+"=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("cross-wire launch-grant matrix failed: %v\n%s", err, out)
	}
	for _, want := range []string{"initial", "reload", "anthropic-messages", "openai-chat-completions", "openai-responses", "gemini-generate-content"} {
		if !bytes.Contains(out, []byte(want)) {
			t.Errorf("cross-wire witness output missing %q:\n%s", want, out)
		}
	}
}

func runGuardLaunchGrantCrossWireHostileMatrix(t *testing.T) {
	isolateGuardLaunchGrantOverlays(t)
	t.Setenv("FAK_GUARD_POSTURE", "fail_closed")
	useGuardLaunchToolGrant(t, "deploy_preview", "delete_file", "opencode.bash", "write_file")
	preserveGuardDefaultPolicy(t)
	assertGuardLaunchGrantWireCoverage(t)

	_, _, initialDigest, _ := loadGuardCapabilityFloor("")
	runGuardLaunchGrantStage(t, "initial")

	reload, err := guardPolicyReloader("")(context.Background())
	if err != nil {
		t.Fatalf("reload launch-grant floor: %v", err)
	}
	if reload.EffectiveDigest != initialDigest {
		t.Fatalf("reload effective digest = %q, want initial %q", reload.EffectiveDigest, initialDigest)
	}
	runGuardLaunchGrantStage(t, "reload")
}

func assertGuardLaunchGrantWireCoverage(t *testing.T) {
	t.Helper()
	covered := make(map[harnessprofile.Wire]bool, len(guardLaunchGrantWires))
	for _, wire := range guardLaunchGrantWires {
		covered[harnessprofile.Wire(wire.provider)] = true
	}
	for _, profile := range harnessprofile.Builtins() {
		if !covered[profile.Wire] {
			t.Errorf("supported guard wire %q from harness profile %q has no launch-grant matrix row", profile.Wire, profile.Name)
		}
	}
}

func runGuardLaunchGrantStage(t *testing.T, stage string) {
	t.Helper()
	t.Run(stage, func(t *testing.T) {
		for _, wire := range guardLaunchGrantWires {
			wire := wire
			t.Run(wire.name, func(t *testing.T) {
				adjudications := runGuardLaunchGrantWire(t, wire)
				if len(adjudications) != len(guardLaunchGrantMatrixCases) {
					t.Fatalf("adjudications = %d, want %d: %+v", len(adjudications), len(guardLaunchGrantMatrixCases), adjudications)
				}
				byTool := make(map[string]gateway.ToolAdjudication, len(adjudications))
				for _, adjudication := range adjudications {
					if _, duplicate := byTool[adjudication.Tool]; duplicate {
						t.Fatalf("duplicate adjudication for tool %q", adjudication.Tool)
					}
					byTool[adjudication.Tool] = adjudication
				}
				for _, fixture := range guardLaunchGrantMatrixCases {
					fixture := fixture
					t.Run(fixture.name, func(t *testing.T) {
						got, ok := byTool[fixture.tool]
						if !ok {
							t.Fatalf("wire %s fixture %s: no adjudication for %q", wire.name, fixture.name, fixture.tool)
						}
						if got.Admitted != fixture.wantAllow {
							t.Fatalf("wire %s fixture %s: admitted = %v, want %v (%s/%s)", wire.name, fixture.name, got.Admitted, fixture.wantAllow, got.Verdict.Kind, got.Verdict.Reason)
						}
						if fixture.wantAllow {
							if got.Verdict.Kind != "ALLOW" || got.Verdict.Reason != "" {
								t.Fatalf("wire %s fixture %s: verdict = %s/%s, want ALLOW", wire.name, fixture.name, got.Verdict.Kind, got.Verdict.Reason)
							}
							return
						}
						if got.Verdict.Kind != "DENY" || got.Verdict.Reason != fixture.wantReason {
							t.Fatalf("wire %s fixture %s: verdict = %s/%s, want DENY/%s", wire.name, fixture.name, got.Verdict.Kind, got.Verdict.Reason, fixture.wantReason)
						}
					})
				}
			})
		}
	})
}

func runGuardLaunchGrantWire(t *testing.T, wire guardLaunchGrantWire) []gateway.ToolAdjudication {
	t.Helper()
	upstreamBody := guardLaunchGrantUpstreamResponse(t, wire.provider)
	upstreamPath := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamPath <- r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(upstreamBody)
	}))
	defer upstream.Close()

	baseURL := upstream.URL
	if wire.baseV1 {
		baseURL += "/v1"
	}
	srv, err := gateway.New(gateway.Config{
		EngineID:        "inkernel",
		Model:           "guard-launch-grant",
		Provider:        wire.provider,
		BaseURL:         baseURL,
		VDSO:            true,
		ToolFloorDenies: adjudicator.Default.NeverAdmits,
		Logf:            func(string, ...any) {},
	})
	if err != nil {
		t.Fatalf("wire %s: create gateway: %v", wire.name, err)
	}
	defer srv.Close()
	gatewayServer := httptest.NewServer(srv.Handler())
	defer gatewayServer.Close()

	req, err := http.NewRequest(http.MethodPost, gatewayServer.URL+wire.inboundPath, bytes.NewReader(guardLaunchGrantInboundRequest(t, wire.provider)))
	if err != nil {
		t.Fatalf("wire %s: create request: %v", wire.name, err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("wire %s: post: %v", wire.name, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("wire %s: read response: %v", wire.name, err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("wire %s: status = %d, want 200: %s", wire.name, resp.StatusCode, raw)
	}
	if gotUpstreamPath := <-upstreamPath; gotUpstreamPath != wire.upstreamPath {
		t.Fatalf("wire %s: upstream path = %q, want %q", wire.name, gotUpstreamPath, wire.upstreamPath)
	}
	var envelope struct {
		Fak *gateway.FakExt `json:"fak"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatalf("wire %s: decode response: %v (%s)", wire.name, err, raw)
	}
	if envelope.Fak == nil {
		t.Fatalf("wire %s: response has no fak adjudication extension: %s", wire.name, raw)
	}
	return envelope.Fak.Adjudications
}

func guardLaunchGrantInboundRequest(t *testing.T, provider string) []byte {
	t.Helper()
	toolNames := make([]string, 0, len(guardLaunchGrantMatrixCases))
	for _, fixture := range guardLaunchGrantMatrixCases {
		toolNames = append(toolNames, fixture.tool)
	}
	var request any
	switch provider {
	case "anthropic":
		tools := make([]map[string]any, 0, len(toolNames))
		for _, name := range toolNames {
			tools = append(tools, map[string]any{"name": name, "input_schema": map[string]any{"type": "object"}})
		}
		request = map[string]any{"model": "guard-launch-grant", "max_tokens": 128, "messages": []map[string]any{{"role": "user", "content": "exercise the grant floor"}}, "tools": tools}
	case "openai":
		tools := make([]map[string]any, 0, len(toolNames))
		for _, name := range toolNames {
			tools = append(tools, map[string]any{"type": "function", "function": map[string]any{"name": name, "parameters": map[string]any{"type": "object"}}})
		}
		request = map[string]any{"model": "guard-launch-grant", "messages": []map[string]any{{"role": "user", "content": "exercise the grant floor"}}, "tools": tools}
	case "openai-responses":
		tools := make([]map[string]any, 0, len(toolNames))
		for _, name := range toolNames {
			tools = append(tools, map[string]any{"type": "function", "name": name, "parameters": map[string]any{"type": "object"}})
		}
		request = map[string]any{"model": "guard-launch-grant", "input": "exercise the grant floor", "tools": tools}
	case "gemini":
		declarations := make([]map[string]any, 0, len(toolNames))
		for _, name := range toolNames {
			declarations = append(declarations, map[string]any{"name": name, "parameters": map[string]any{"type": "object"}})
		}
		request = map[string]any{"contents": []map[string]any{{"role": "user", "parts": []map[string]any{{"text": "exercise the grant floor"}}}}, "tools": []map[string]any{{"functionDeclarations": declarations}}}
	default:
		t.Fatalf("unknown launch-grant wire provider %q", provider)
	}
	raw, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("provider %s: marshal inbound request: %v", provider, err)
	}
	return raw
}

func guardLaunchGrantUpstreamResponse(t *testing.T, provider string) []byte {
	t.Helper()
	var response any
	switch provider {
	case "anthropic":
		content := []map[string]any{{"type": "text", "text": "checking launch grants"}}
		for i, fixture := range guardLaunchGrantMatrixCases {
			content = append(content, map[string]any{"type": "tool_use", "id": fmt.Sprintf("a%d", i), "name": fixture.tool, "input": fixture.args})
		}
		response = map[string]any{"content": content, "stop_reason": "tool_use", "usage": map[string]any{"input_tokens": 7, "output_tokens": 3}}
	case "openai":
		calls := make([]map[string]any, 0, len(guardLaunchGrantMatrixCases))
		for i, fixture := range guardLaunchGrantMatrixCases {
			calls = append(calls, map[string]any{"id": fmt.Sprintf("o%d", i), "type": "function", "function": map[string]any{"name": fixture.tool, "arguments": string(fixture.args)}})
		}
		response = map[string]any{"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "checking launch grants", "tool_calls": calls}, "finish_reason": "tool_calls"}}, "usage": map[string]any{"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
	case "openai-responses":
		output := []map[string]any{{"id": "msg_1", "type": "message", "status": "completed", "role": "assistant", "content": []map[string]any{{"type": "output_text", "text": "checking launch grants"}}}}
		for i, fixture := range guardLaunchGrantMatrixCases {
			output = append(output, map[string]any{"id": fmt.Sprintf("fc_%d", i), "type": "function_call", "call_id": fmt.Sprintf("r%d", i), "name": fixture.tool, "arguments": string(fixture.args)})
		}
		response = map[string]any{"status": "completed", "output": output, "usage": map[string]any{"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}}
	case "gemini":
		parts := []map[string]any{{"text": "checking launch grants"}}
		for i, fixture := range guardLaunchGrantMatrixCases {
			parts = append(parts, map[string]any{"functionCall": map[string]any{"name": fixture.tool, "args": fixture.args, "id": fmt.Sprintf("g%d", i)}})
		}
		response = map[string]any{"candidates": []map[string]any{{"content": map[string]any{"role": "model", "parts": parts}, "finishReason": "STOP"}}, "usageMetadata": map[string]any{"promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10}}
	default:
		t.Fatalf("unknown launch-grant upstream provider %q", provider)
	}
	raw, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("provider %s: marshal upstream response: %v", provider, err)
	}
	return raw
}
