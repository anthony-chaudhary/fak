package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

func streamFixture(model, result string, toolNames []string, toolErrors []bool) []byte {
	var lines []string
	for _, name := range toolNames {
		b, _ := json.Marshal(map[string]any{"type": "assistant", "message": map[string]any{"model": model, "content": []any{map[string]any{"type": "tool_use", "name": name}}}})
		lines = append(lines, string(b))
	}
	for _, isErr := range toolErrors {
		b, _ := json.Marshal(map[string]any{"type": "user", "message": map[string]any{"content": []any{map[string]any{"type": "tool_result", "is_error": isErr}}}})
		lines = append(lines, string(b))
	}
	b, _ := json.Marshal(map[string]any{"type": "result", "subtype": "success", "result": result, "duration_ms": 12, "total_cost_usd": .01, "usage": map[string]any{"input_tokens": 17}, "modelUsage": map[string]any{model: map[string]any{}}})
	lines = append(lines, string(b))
	return []byte(strings.Join(lines, "\n") + "\n")
}

func TestParseClaudeAcceptanceExactIDAndBehaviors(t *testing.T) {
	p, err := parseClaudeAcceptance(streamFixture("exact-a", "RECOVERED", []string{"mcp__acceptance__flaky_lookup", "mcp__acceptance__flaky_lookup"}, []bool{true, false}), "exact-a", modelaccept.Task{RetryRequired: true, RecoveryRequired: true})
	if err != nil {
		t.Fatal(err)
	}
	if p.actualModel != "exact-a" || p.toolCalls != 2 || p.retryCount != 1 || !p.recovered || p.inputTokens != 17 || p.costUSD != .01 {
		t.Fatalf("parsed=%+v", p)
	}
}

func TestParseClaudeAcceptanceFailsClosed(t *testing.T) {
	for _, tc := range []struct {
		name  string
		raw   []byte
		model string
	}{
		{"wrong model", streamFixture("exact-b", "OK", nil, nil), "exact-a"},
		{"missing requested usage", []byte(`{"type":"result","result":"OK","modelUsage":{"b":{}}}` + "\n"), "a"},
		{"missing result", []byte(`{"type":"assistant","message":{"model":"a","content":[]}}` + "\n"), "a"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseClaudeAcceptance(tc.raw, tc.model, modelaccept.Task{}); err == nil {
				t.Fatal("expected fail-closed error")
			}
		})
	}
}

func TestParseClaudeAcceptanceAllowsAccountedHelperModel(t *testing.T) {
	const model = "claude-opus-4-8"
	lines := []string{
		`{"type":"assistant","message":{"model":"claude-opus-4-8","content":[{"type":"text","text":"IMPLEMENTED"}]}}`,
		`{"type":"result","subtype":"success","is_error":false,"duration_ms":10,"total_cost_usd":0.02,"result":"IMPLEMENTED","usage":{"input_tokens":4},"modelUsage":{"claude-opus-4-8":{},"claude-haiku-4-5-20251001":{}}}`,
	}
	got, err := parseClaudeAcceptance([]byte(strings.Join(lines, "\n")), model, modelaccept.Task{Expected: "IMPLEMENTED"})
	if err != nil {
		t.Fatal(err)
	}
	if got.actualModel != model || got.result != "IMPLEMENTED" {
		t.Fatalf("got=%+v", got)
	}
}

func TestParseClaudeAcceptanceRejectsMissingRequestedUsage(t *testing.T) {
	const model = "claude-opus-4-8"
	lines := []string{
		`{"type":"assistant","message":{"model":"claude-opus-4-8","content":[]}}`,
		`{"type":"result","subtype":"success","is_error":false,"result":"IMPLEMENTED","modelUsage":{"claude-haiku-4-5-20251001":{}}}`,
	}
	if _, err := parseClaudeAcceptance([]byte(strings.Join(lines, "\n")), model, modelaccept.Task{Expected: "IMPLEMENTED"}); err == nil || !strings.Contains(err.Error(), "missing requested model") {
		t.Fatalf("err=%v", err)
	}
}

func TestRunModelAcceptanceFixture(t *testing.T) {
	input := strings.Join([]string{`{"jsonrpc":"2.0","id":1,"method":"initialize"}`, `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"flaky_lookup","arguments":{}}}`, `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"flaky_lookup","arguments":{}}}`}, "\n")
	var out, errout bytes.Buffer
	if code := runModelAcceptanceFixture(strings.NewReader(input), &out, &errout); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	if !strings.Contains(out.String(), "TRANSIENT_RETRYABLE") || !strings.Contains(out.String(), "RECOVERY_VALUE=42") {
		t.Fatalf("out=%s", out.String())
	}
}

func TestRunModelAcceptanceRunWritesRetainedReport(t *testing.T) {
	old := acceptanceClaudeCommand
	defer func() { acceptanceClaudeCommand = old }()
	acceptanceClaudeCommand = func(_ context.Context, _ string, args, env []string) ([]byte, []byte, error) {
		var model string
		for i := range args {
			if args[i] == "--model" {
				model = args[i+1]
			}
		}
		return streamFixture(model, "OK", nil, nil), nil, nil
	}
	dir := t.TempDir()
	configDir := filepath.Join(dir, "claude-config")
	if err := os.Mkdir(configDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(configDir, ".credentials.json"), []byte("{}"), 0600); err != nil {
		t.Fatal(err)
	}
	decl := modelaccept.Input{Schema: modelaccept.Schema, Corpus: modelaccept.Corpus{ID: "run-v1", DeclaredAt: time.Now().Add(-time.Hour).Format(time.RFC3339), Tasks: []modelaccept.Task{{ID: "one", Tier: 2, Repetitions: 1, Prompt: "return OK", Expected: "OK"}}, Thresholds: modelaccept.Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 100, MaxAverageInputTokens: 100, MaxAverageCostUSD: 1}}, Models: []modelaccept.ModelRequest{{Model: "exact-a", RequestedTier: 2}}}
	b, _ := json.Marshal(decl)
	input := filepath.Join(dir, "in.json")
	os.WriteFile(input, b, 0600)
	output := filepath.Join(dir, "out.json")
	rawDir := filepath.Join(dir, "raw")
	var stdout, stderr bytes.Buffer
	code := runModelAcceptanceRun(&stdout, &stderr, []string{"--input", input, "--output", output, "--raw-dir", rawDir, "--fixture-command", os.Args[0], "--claude-config-dir", configDir})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s out=%s", code, stderr.String(), stdout.String())
	}
	var got modelaccept.Input
	b, _ = os.ReadFile(output)
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Runs) != 1 || got.Runs[0].ActualModel != "exact-a" || got.Runs[0].Result != "OK" {
		t.Fatalf("report=%+v", got)
	}
}

func TestParseClaudeAcceptanceReadsTopLevelToolUseResults(t *testing.T) {
	raw := []byte(strings.Join([]string{
		`{"type":"assistant","message":{"model":"claude-opus-4-8","content":[{"type":"tool_use","name":"mcp__acceptance__flaky_lookup"}]}}`,
		`{"type":"user","message":{"content":[]},"tool_use_result":{"isError":true,"content":"TRANSIENT_RETRYABLE"}}`,
		`{"type":"assistant","message":{"model":"claude-opus-4-8","content":[{"type":"tool_use","name":"mcp__acceptance__flaky_lookup"}]}}`,
		`{"type":"user","message":{"content":[]},"tool_use_result":{"isError":false,"content":"RECOVERY_VALUE=42"}}`,
		`{"type":"result","result":"RECOVERED","modelUsage":{"claude-opus-4-8":{}},"usage":{"input_tokens":3}}`,
	}, "\n"))
	got, err := parseClaudeAcceptance(raw, "claude-opus-4-8", modelaccept.Task{Expected: "RECOVERED", RetryRequired: true})
	if err != nil {
		t.Fatal(err)
	}
	if got.retryCount != 1 || !got.recovered {
		t.Fatalf("retryCount=%d recovered=%v, want 1/true", got.retryCount, got.recovered)
	}
}
