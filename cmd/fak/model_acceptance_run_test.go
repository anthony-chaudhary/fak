package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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

// TestParseClaudeAcceptanceCountsToolTurns pins the width denominator the fold
// grades on (#5802). The load-bearing case is the split one: Claude Code writes a
// single assistant response as one stream event per content block under the SAME
// message id, so a batched turn must collapse back to ONE turn — otherwise every
// model looks serialized and no corpus task could ever witness width.
func TestParseClaudeAcceptanceCountsToolTurns(t *testing.T) {
	const model = "exact-a"
	toolUse := func(id string, calls int) string {
		content := []any{}
		for i := 0; i < calls; i++ {
			content = append(content, map[string]any{"type": "tool_use", "name": "mcp__acceptance__lookup"})
		}
		b, _ := json.Marshal(map[string]any{"type": "assistant", "message": map[string]any{"model": model, "id": id, "content": content}})
		return string(b)
	}
	result, _ := json.Marshal(map[string]any{"type": "result", "subtype": "success", "result": "OK", "usage": map[string]any{"input_tokens": 1}, "modelUsage": map[string]any{model: map[string]any{}}})
	for _, tc := range []struct {
		name                 string
		events               []string
		wantCalls, wantTurns int
	}{
		{"two calls in one event", []string{toolUse("msg_1", 2)}, 2, 1},
		{"one response split across events", []string{toolUse("msg_1", 1), toolUse("msg_1", 1)}, 2, 1},
		{"serialized across responses", []string{toolUse("msg_1", 1), toolUse("msg_2", 1)}, 2, 2},
		{"tool result closes the turn", []string{toolUse("msg_1", 1), `{"type":"user","message":{"content":[{"type":"tool_result","is_error":false}]}}`, toolUse("msg_1", 1)}, 2, 2},
		{"text-only response is not a tool turn", []string{`{"type":"assistant","message":{"model":"exact-a","id":"msg_1","content":[{"type":"text","text":"thinking"}]}}`, toolUse("msg_2", 1)}, 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(strings.Join(append(append([]string{}, tc.events...), string(result)), "\n") + "\n")
			p, err := parseClaudeAcceptance(raw, model, modelaccept.Task{Expected: "OK", ToolRequired: true})
			if err != nil {
				t.Fatal(err)
			}
			if p.toolCalls != tc.wantCalls || p.toolTurns != tc.wantTurns {
				t.Fatalf("calls=%d turns=%d, want %d/%d", p.toolCalls, p.toolTurns, tc.wantCalls, tc.wantTurns)
			}
			run := modelaccept.Run{ToolCalls: p.toolCalls, ToolTurns: p.toolTurns}
			if got, want := modelaccept.ToolCallWidth(run), (tc.wantCalls+tc.wantTurns-1)/max(tc.wantTurns, 1); got != want {
				t.Fatalf("width=%d, want %d", got, want)
			}
		})
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
	acceptanceClaudeCommand = func(_ context.Context, _ string, args, env []string, _ string) ([]byte, []byte, error) {
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

func TestParseClaudeAcceptanceReadsObservedTopLevelToolUseResultShapes(t *testing.T) {
	raw := []byte(strings.Join([]string{
		`{"type":"assistant","message":{"model":"claude-opus-4-8","content":[{"type":"tool_use","name":"mcp__acceptance__flaky_lookup"}]}}`,
		`{"type":"user","message":{"content":[]},"tool_use_result":"Error: TRANSIENT_RETRYABLE: retry this read once"}`,
		`{"type":"assistant","message":{"model":"claude-opus-4-8","content":[{"type":"tool_use","name":"mcp__acceptance__flaky_lookup"}]}}`,
		`{"type":"user","message":{"content":[]},"tool_use_result":[{"type":"text","text":"RECOVERY_VALUE=42"}]}`,
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

func TestClassifyAcceptanceFailureClasses(t *testing.T) {
	task := modelaccept.Task{Expected: "FAK_ACCEPTANCE_RESULT=OK", ResultMatch: "sentinel_line", ToolRequired: true, MinToolCalls: 1}
	good := modelaccept.Run{Result: "done\nFAK_ACCEPTANCE_RESULT=OK", ToolValid: true, ToolCalls: 1}
	if class, detail := classifyAcceptanceFailure(task, good, nil, nil); class != "" || detail != "" {
		t.Fatalf("good run classified as %q: %s", class, detail)
	}
	if class, _ := classifyAcceptanceFailure(task, good, errors.New("provider unavailable"), nil); class != "provider_infrastructure" {
		t.Fatalf("command failure class = %q", class)
	}
	if class, _ := classifyAcceptanceFailure(task, good, nil, errors.New("malformed stream")); class != "harness" {
		t.Fatalf("parse failure class = %q", class)
	}
	bad := good
	bad.Result = "missing sentinel"
	if class, _ := classifyAcceptanceFailure(task, bad, nil, nil); class != "capability" {
		t.Fatalf("output failure class = %q", class)
	}
	refusal := modelaccept.Task{Expected: "FAK_ACCEPTANCE_RESULT=REFUSED", ResultMatch: "sentinel_line", ExpectedRefusal: "policy"}
	if class, _ := classifyAcceptanceFailure(refusal, modelaccept.Run{Result: refusal.Expected}, nil, nil); class != "policy_refusal" {
		t.Fatalf("refusal failure class = %q", class)
	}
}

func TestAcceptancePromptTransportMovesLargeWindowsPrompt(t *testing.T) {
	prompt := strings.Repeat("acceptance-task", 4000)
	got, stdin, moved := acceptancePromptTransport("claude.exe", []string{"-p", prompt, "--model", "test"}, "windows")
	if !moved || stdin != prompt {
		t.Fatalf("acceptance prompt transport = moved %v stdin bytes %d", moved, len(stdin))
	}
	if strings.Join(got, "\x00") != strings.Join([]string{"claude.exe", "-p", "--model", "test"}, "\x00") {
		t.Fatalf("acceptance argv = %#v", got)
	}
}
