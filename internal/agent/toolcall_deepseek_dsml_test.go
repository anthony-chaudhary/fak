package agent

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// deepSeekDSMLRendered is the DSML span the V4.1 template renders for an assistant
// turn carrying calls: what the model is taught to emit after its content.
func deepSeekDSMLRendered(t *testing.T, content string, calls []ToolCall) string {
	t.Helper()
	rendered := renderDeepSeekV41Chat([]Message{{Role: RoleAssistant, Content: content, ToolCalls: calls}}, nil, nil)
	start := strings.Index(rendered, deepSeekV41Assistant+deepSeekV41ThinkEnd)
	end := strings.LastIndex(rendered, deepSeekV41EOS)
	if start < 0 || end < 0 {
		t.Fatalf("no assistant turn in %q", rendered)
	}
	return rendered[start+len(deepSeekV41Assistant+deepSeekV41ThinkEnd) : end]
}

func deepSeekDSMLSameArgs(t *testing.T, got, want string) {
	t.Helper()
	var g, w any
	if json.Unmarshal([]byte(got), &g) != nil || json.Unmarshal([]byte(want), &w) != nil || !reflect.DeepEqual(g, w) {
		t.Fatalf("arguments %s, want %s", got, want)
	}
}

// The V4.1 template's own DSML render of assistant tool calls lifts back to the
// same calls (names, typed arguments, order), leaving the visible content; a
// malformed, unoffered or example block stays text.
// fak-test:justify why=contract when=changed:internal/agent/**
// fak-test:runtime fast est=5ms lane=default
func TestDeepSeekDSMLToolCallsRoundTrip(t *testing.T) {
	calls := []ToolCall{
		{Type: "function", Function: Func{Name: "get_weather", Arguments: `{"city":"Paris <now>","days":3,"opts":{"units":["c","k"],"ok":true},"note":"a\nb"}`}},
		{Type: "function", Function: Func{Name: "list_files", Arguments: `{}`}},
	}
	text := deepSeekDSMLRendered(t, "Checking.", calls)
	offered := NewOfferedTools([]ToolDef{{Type: "function", Function: ToolDefFunction{Name: "get_weather"}}, {Type: "function", Function: ToolDefFunction{Name: "list_files"}}})
	lifted := LiftTextToolCalls(Message{Role: RoleAssistant, Content: text}, offered)
	if lifted.Content != "Checking." || len(lifted.ToolCalls) != len(calls) {
		t.Fatalf("lift = %q %+v", lifted.Content, lifted.ToolCalls)
	}
	for i, call := range lifted.ToolCalls {
		if call.Function.Name != calls[i].Function.Name {
			t.Fatalf("call %d name %q, want %q", i, call.Function.Name, calls[i].Function.Name)
		}
		deepSeekDSMLSameArgs(t, call.Function.Arguments, calls[i].Function.Arguments)
	}
	if !hasTextToolCandidate(text) {
		t.Fatal("a DSML tool_calls block is not a tool candidate")
	}

	for name, bad := range map[string]string{
		"unoffered":    strings.Replace(text, "list_files", "rm_rf", 1),
		"bad-kind":     strings.Replace(text, `string="false"`, `string="maybe"`, 1),
		"bad-json":     strings.Replace(text, `>3<`, `>three<`, 1),
		"open-invoke":  strings.Replace(text, deepSeekDSMLInvokeEnd, "", 1),
		"unterminated": strings.Replace(text, deepSeekDSMLCallsClose, "", 1),
		"example":      "For example: " + deepSeekDSMLRendered(t, "", calls[:1]),
	} {
		got := LiftTextToolCalls(Message{Role: RoleAssistant, Content: bad}, offered)
		if len(got.ToolCalls) != 0 || strings.TrimSpace(got.Content) != strings.TrimSpace(bad) {
			t.Fatalf("%s: lifted %+v from %q", name, got.ToolCalls, bad)
		}
	}

	var streamed strings.Builder
	guard := newToolSpanGuard(func(delta string) error { streamed.WriteString(delta); return nil })
	for _, r := range text + " Done." {
		if err := guard.feed(string(r)); err != nil {
			t.Fatal(err)
		}
	}
	if err := guard.flush(); err != nil {
		t.Fatal(err)
	}
	if got := streamed.String(); strings.Contains(got, deepSeekV41DSML) || !strings.HasPrefix(got, "Checking.") || !strings.HasSuffix(got, " Done.") {
		t.Fatalf("streamed text leaked or lost DSML markup: %q", got)
	}
}
