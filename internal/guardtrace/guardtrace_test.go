package guardtrace

import (
	"encoding/json"
	"reflect"
	"testing"
)

// TestParseFixtureValidation covers the load-time validation that makes a typo
// fail loud: empty turns, an empty turn, a call missing a tool, and an unknown
// class must all error; a well-formed fixture must parse.
func TestParseFixtureValidation(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		wantErr bool
	}{
		{"not json", `{`, true},
		{"no turns", `{"turns":[]}`, true},
		{"turn with no calls", `{"turns":[{"calls":[]}]}`, true},
		{"call with no tool", `{"turns":[{"calls":[{"class":"allow"}]}]}`, true},
		{"unknown class", `{"turns":[{"calls":[{"tool":"Bash","class":"maybe"}]}]}`, true},
		{"unknown message role", `{"turns":[{"messages":[{"role":"tool","content":"x"}],"calls":[{"tool":"Read","class":"allow"}]}]}`, true},
		{"empty message content", `{"turns":[{"messages":[{"role":"user","content":" "}],"calls":[{"tool":"Read","class":"allow"}]}]}`, true},
		{"valid allow", `{"turns":[{"calls":[{"tool":"Read","class":"allow"}]}]}`, false},
		{"valid messages", `{"turns":[{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"}],"calls":[{"tool":"Read","class":"allow"}]}]}`, false},
		{"valid deny", `{"turns":[{"calls":[{"tool":"Bash","class":"DENY","reason":"POLICY_BLOCK"}]}]}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := ParseFixture([]byte(tt.raw))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ParseFixture(%q) = nil error, want error", tt.raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseFixture(%q) unexpected error: %v", tt.raw, err)
			}
			if f == nil || len(f.Turns) == 0 {
				t.Fatalf("ParseFixture(%q) returned no turns", tt.raw)
			}
		})
	}
}

// TestExpectAndReasons folds a two-turn fixture and asserts the aggregate the
// gateway's AdjudicationSummary must report: call/allow/deny counts, summed
// token axes, and the sorted deny-reason histogram.
func TestExpectAndReasons(t *testing.T) {
	f := &Fixture{Turns: []Turn{
		{
			Usage: Usage{InputTokens: 100, CacheReadInputTokens: 10, CacheCreationInputTokens: 5},
			Calls: []Call{
				{Tool: "Read", Class: "allow"},
				{Tool: "Bash", Class: "deny", Reason: "POLICY_BLOCK"},
			},
		},
		{
			Usage: Usage{InputTokens: 50, CacheReadInputTokens: 20},
			Calls: []Call{
				{Tool: "Write", Class: "deny", Reason: "SELF_MODIFY"},
				{Tool: "Bash", Class: "deny", Reason: "POLICY_BLOCK"},
			},
		},
	}}

	e := f.Expect()
	if e.TotalCalls != 4 {
		t.Errorf("TotalCalls = %d, want 4", e.TotalCalls)
	}
	if e.Allowed != 1 || e.Denied != 3 {
		t.Errorf("Allowed/Denied = %d/%d, want 1/3", e.Allowed, e.Denied)
	}
	if e.InputTokens != 150 {
		t.Errorf("InputTokens = %d, want 150", e.InputTokens)
	}
	if e.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30", e.CacheReadTokens)
	}
	if e.CacheCreationTokens != 5 {
		t.Errorf("CacheCreationTokens = %d, want 5", e.CacheCreationTokens)
	}
	if e.ByReason["POLICY_BLOCK"] != 2 || e.ByReason["SELF_MODIFY"] != 1 {
		t.Errorf("ByReason = %v, want POLICY_BLOCK:2 SELF_MODIFY:1", e.ByReason)
	}
	if got, want := e.Reasons(), []string{"POLICY_BLOCK", "SELF_MODIFY"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Reasons() = %v, want %v (sorted)", got, want)
	}
}

func TestCallExpectAllow(t *testing.T) {
	for _, tt := range []struct {
		class string
		want  bool
	}{
		{"allow", true},
		{"ALLOW", true},
		{" Allow ", true},
		{"deny", false},
		{"", false},
	} {
		if got := (Call{Class: tt.class}).ExpectAllow(); got != tt.want {
			t.Errorf("ExpectAllow(class=%q) = %v, want %v", tt.class, got, tt.want)
		}
	}
}

func TestCallArgString(t *testing.T) {
	// Empty args render as a tight empty object.
	if got := (Call{}).ArgString(); got != "{}" {
		t.Errorf("empty ArgString() = %q, want {}", got)
	}
	// Valid JSON is re-marshaled to drop fixture whitespace.
	if got := (Call{Args: json.RawMessage(`{ "command" : "ls" }`)}).ArgString(); got != `{"command":"ls"}` {
		t.Errorf("ArgString() = %q, want compact object", got)
	}
	// Invalid JSON falls back to the raw bytes.
	if got := (Call{Args: json.RawMessage(`not-json`)}).ArgString(); got != "not-json" {
		t.Errorf("ArgString() on bad json = %q, want raw passthrough", got)
	}
}

func TestCallArgPreview(t *testing.T) {
	// A 60-char value exceeds the 48-char preview cap, so ArgPreview must return
	// the first 47 bytes plus the ellipsis (truncate's contract).
	long := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567"
	longWant := long[:47] + "…"
	for _, tt := range []struct {
		name string
		args string
		want string
	}{
		{"command key", `{"command":"git status"}`, "git status"},
		{"file_path key", `{"file_path":"main.go"}`, "main.go"},
		{"no salient key", `{"foo":"bar"}`, ""},
		{"not an object", `["a","b"]`, ""},
		{"long value truncates", `{"command":"` + long + `"}`, longWant},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := (Call{Args: json.RawMessage(tt.args)}).ArgPreview(); got != tt.want {
				t.Errorf("ArgPreview(%s) = %q, want %q", tt.args, got, tt.want)
			}
		})
	}
}

func TestInboundRoute(t *testing.T) {
	if got := InboundRoute("anthropic"); got != "/v1/messages" {
		t.Errorf("InboundRoute(anthropic) = %q", got)
	}
	for _, p := range []string{"openai", "", "groq"} {
		if got := InboundRoute(p); got != "/v1/chat/completions" {
			t.Errorf("InboundRoute(%q) = %q, want /v1/chat/completions", p, got)
		}
	}
}

func TestToolDeclsDedupPreservesOrder(t *testing.T) {
	turn := Turn{Calls: []Call{
		{Tool: "Read"}, {Tool: "Bash"}, {Tool: "Read"}, {Tool: "Write"}, {Tool: "Bash"},
	}}
	if got, want := toolDecls(turn), []string{"Read", "Bash", "Write"}; !reflect.DeepEqual(got, want) {
		t.Errorf("toolDecls = %v, want %v (deduped, first-seen order)", got, want)
	}
}

// TestBuildInboundRequest checks each wire produces well-formed JSON advertising
// the turn's tools, and that the route differs by provider.
func TestBuildInboundRequest(t *testing.T) {
	turn := Turn{Calls: []Call{{Tool: "Bash"}, {Tool: "Read"}}}

	for _, provider := range []string{"anthropic", "openai"} {
		body, err := BuildInboundRequest(provider, "test-model", turn)
		if err != nil {
			t.Fatalf("BuildInboundRequest(%s): %v", provider, err)
		}
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("BuildInboundRequest(%s) produced invalid JSON: %v", provider, err)
		}
		if req["model"] != "test-model" {
			t.Errorf("%s: model = %v, want test-model", provider, req["model"])
		}
		tools, ok := req["tools"].([]any)
		if !ok || len(tools) != 2 {
			t.Errorf("%s: want 2 advertised tools, got %v", provider, req["tools"])
		}
	}
}

func TestBuildInboundRequestUsesFixtureMessages(t *testing.T) {
	turn := Turn{
		Messages: []RequestMessage{
			{Role: "system", Content: "system prompt"},
			{Role: "user", Content: "first task"},
			{Role: "assistant", Content: "older assistant context"},
			{Role: "user", Content: "latest task"},
		},
		Calls: []Call{{Tool: "Read"}},
	}

	body, err := BuildInboundRequest("openai", "test-model", turn)
	if err != nil {
		t.Fatalf("BuildInboundRequest(openai): %v", err)
	}
	var openAI struct {
		Messages []RequestMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &openAI); err != nil {
		t.Fatalf("decode openai request: %v", err)
	}
	if !reflect.DeepEqual(openAI.Messages, turn.Messages) {
		t.Fatalf("openai messages = %+v, want fixture messages %+v", openAI.Messages, turn.Messages)
	}

	body, err = BuildInboundRequest("anthropic", "test-model", turn)
	if err != nil {
		t.Fatalf("BuildInboundRequest(anthropic): %v", err)
	}
	var anthropic struct {
		System   string           `json:"system"`
		Messages []RequestMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &anthropic); err != nil {
		t.Fatalf("decode anthropic request: %v", err)
	}
	if anthropic.System != "system prompt" {
		t.Fatalf("anthropic system = %q, want fixture system prompt", anthropic.System)
	}
	if len(anthropic.Messages) != 3 || anthropic.Messages[0].Role != "user" ||
		anthropic.Messages[1].Role != "assistant" || anthropic.Messages[2].Content != "latest task" {
		t.Fatalf("anthropic messages = %+v, want fixture non-system history", anthropic.Messages)
	}
}

func TestDecodeAdjudications(t *testing.T) {
	// No fak extension → no verdicts, no error.
	got, err := DecodeAdjudications([]byte(`{"content":[]}`))
	if err != nil || got != nil {
		t.Fatalf("DecodeAdjudications(no fak) = %v, %v; want nil, nil", got, err)
	}
	// Malformed JSON → error.
	if _, err := DecodeAdjudications([]byte(`{`)); err == nil {
		t.Fatal("DecodeAdjudications(bad json) = nil error, want error")
	}
	// A real verdict decodes into the flat shape.
	raw := `{"fak":{"adjudications":[{"tool":"Bash","admitted":false,"verdict":{"kind":"deny","reason":"POLICY_BLOCK"}}]}}`
	adj, err := DecodeAdjudications([]byte(raw))
	if err != nil {
		t.Fatalf("DecodeAdjudications: %v", err)
	}
	if len(adj) != 1 {
		t.Fatalf("want 1 adjudication, got %d", len(adj))
	}
	want := ResponseAdjudication{Tool: "Bash", Admitted: false, Kind: "deny", Reason: "POLICY_BLOCK"}
	if adj[0] != want {
		t.Errorf("adjudication = %+v, want %+v", adj[0], want)
	}
}

func TestTruncate(t *testing.T) {
	for _, tt := range []struct {
		s    string
		n    int
		want string
	}{
		{"short", 10, "short"},
		{"exactfit12", 10, "exactfit12"[:10]}, // len==n boundary returns unchanged-prefix
		{"abcdef", 4, "abc…"},
		{"abcdef", 1, "a"},
	} {
		if got := truncate(tt.s, tt.n); got != tt.want {
			t.Errorf("truncate(%q,%d) = %q, want %q", tt.s, tt.n, got, tt.want)
		}
	}
}
