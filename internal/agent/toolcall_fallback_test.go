package agent

import "testing"

func TestLiftTextToolCalls_Hermes(t *testing.T) {
	m := Message{
		Role:    RoleAssistant,
		Content: `Let me look. <tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>`,
	}
	got := LiftTextToolCalls(m)
	if len(got.ToolCalls) != 1 {
		t.Fatalf("want 1 lifted tool call, got %d (content=%q)", len(got.ToolCalls), got.Content)
	}
	tc := got.ToolCalls[0]
	if tc.Function.Name != "Bash" {
		t.Errorf("name = %q, want Bash", tc.Function.Name)
	}
	if tc.Function.Arguments != `{"command": "ls"}` {
		t.Errorf("arguments = %q, want object JSON", tc.Function.Arguments)
	}
	if got.Content != "Let me look." {
		t.Errorf("content not stripped of tags: %q", got.Content)
	}
}

func TestLiftTextToolCalls_Multiple(t *testing.T) {
	m := Message{
		Role: RoleAssistant,
		Content: `<tool_call>{"name":"Read","arguments":{"path":"a"}}</tool_call>` +
			`<tool_call>{"name":"Read","arguments":{"path":"b"}}</tool_call>`,
	}
	got := LiftTextToolCalls(m)
	if len(got.ToolCalls) != 2 {
		t.Fatalf("want 2 lifted calls, got %d", len(got.ToolCalls))
	}
	if got.ToolCalls[0].ID == got.ToolCalls[1].ID {
		t.Errorf("lifted calls share an id: %q", got.ToolCalls[0].ID)
	}
}

func TestLiftTextToolCalls_StringifiedArgs(t *testing.T) {
	m := Message{
		Role:    RoleAssistant,
		Content: `<tool_call>{"name":"Bash","arguments":"{\"command\":\"pwd\"}"}</tool_call>`,
	}
	got := LiftTextToolCalls(m)
	if len(got.ToolCalls) != 1 {
		t.Fatalf("want 1 call, got %d", len(got.ToolCalls))
	}
	if got.ToolCalls[0].Function.Arguments != `{"command":"pwd"}` {
		t.Errorf("arguments = %q, want unquoted object", got.ToolCalls[0].Function.Arguments)
	}
}

func TestLiftTextToolCalls_OpenAIStyleFunctionPayload(t *testing.T) {
	m := Message{
		Role:    RoleAssistant,
		Content: `<tool_call>{"type":"function","function":{"name":"Bash","arguments":{"command":"pwd"}}}</tool_call>`,
	}
	got := LiftTextToolCalls(m)
	if len(got.ToolCalls) != 1 {
		t.Fatalf("want 1 call, got %d", len(got.ToolCalls))
	}
	tc := got.ToolCalls[0]
	if tc.Function.Name != "Bash" || tc.Function.Arguments != `{"command":"pwd"}` {
		t.Fatalf("bad OpenAI-style lifted call: %+v", tc)
	}
}

func TestLiftTextToolCalls_NoStructuredClobber(t *testing.T) {
	// A provider that already parsed the call must be left alone.
	m := Message{
		Role:      RoleAssistant,
		Content:   `<tool_call>{"name":"X","arguments":{}}</tool_call>`,
		ToolCalls: []ToolCall{{ID: "call_0", Type: "function", Function: Func{Name: "real"}}},
	}
	got := LiftTextToolCalls(m)
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].Function.Name != "real" {
		t.Errorf("structured calls were clobbered: %+v", got.ToolCalls)
	}
}

func TestLiftTextToolCalls_PreservesMalformedBlocksWhenLiftingOthers(t *testing.T) {
	m := Message{
		Role: RoleAssistant,
		Content: `before <tool_call>{"name":"Read","arguments":{"path":"ok"}}</tool_call>` +
			` middle <tool_call>{not json}</tool_call> after`,
	}
	got := LiftTextToolCalls(m)
	if len(got.ToolCalls) != 1 {
		t.Fatalf("want 1 lifted call, got %d", len(got.ToolCalls))
	}
	if got.Content != `before  middle <tool_call>{not json}</tool_call> after` {
		t.Fatalf("content = %q, want malformed block preserved", got.Content)
	}
}

func TestLiftTextToolCalls_Malformed(t *testing.T) {
	// Malformed JSON or a nameless call must NOT fabricate a tool call.
	for _, c := range []string{
		`<tool_call>{not json}</tool_call>`,
		`<tool_call>{"arguments":{"x":1}}</tool_call>`,
		`plain answer, no tool call here`,
	} {
		got := LiftTextToolCalls(Message{Role: RoleAssistant, Content: c})
		if len(got.ToolCalls) != 0 {
			t.Errorf("content %q wrongly lifted %d calls", c, len(got.ToolCalls))
		}
	}
}

// TestLiftTextToolCalls_Dialects table-tests one captured-shape sample per text
// dialect the registry recognizes (issue #53). Each case asserts the lifted name,
// arguments, and that the dialect's delimiters are stripped from the content — the
// adjudication-bypass these dialects used to slip through is now closed.
func TestLiftTextToolCalls_Dialects(t *testing.T) {
	cases := []struct {
		name        string
		content     string
		wantName    string
		wantArgs    string
		wantContent string
	}{
		{
			name:        "function_call_tag",
			content:     `Sure. <function_call>{"name":"Read","arguments":{"path":"x"}}</function_call>`,
			wantName:    "Read",
			wantArgs:    `{"path":"x"}`,
			wantContent: "Sure.",
		},
		{
			name:        "llama_python_tag_with_eom",
			content:     `<|python_tag|>{"name":"Bash","arguments":{"command":"ls"}}<|eom_id|>`,
			wantName:    "Bash",
			wantArgs:    `{"command":"ls"}`,
			wantContent: "",
		},
		{
			name:        "llama_python_tag_no_terminator",
			content:     `<|python_tag|>{"name":"Bash","arguments":{"command":"pwd"}}`,
			wantName:    "Bash",
			wantArgs:    `{"command":"pwd"}`,
			wantContent: "",
		},
		{
			name:        "mistral_single_call_array",
			content:     `[TOOL_CALLS][{"name":"Read","arguments":{"path":"a"}}]`,
			wantName:    "Read",
			wantArgs:    `{"path":"a"}`,
			wantContent: "",
		},
		{
			name:        "fenced_json",
			content:     "Let me run it:\n```json\n{\"name\":\"Bash\",\"arguments\":{\"command\":\"go test\"}}\n```",
			wantName:    "Bash",
			wantArgs:    `{"command":"go test"}`,
			wantContent: "Let me run it:",
		},
		{
			name:        "bare_json_whole_content",
			content:     `{"name":"Read","arguments":{"path":"only"}}`,
			wantName:    "Read",
			wantArgs:    `{"path":"only"}`,
			wantContent: "",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := LiftTextToolCalls(Message{Role: RoleAssistant, Content: c.content})
			if len(got.ToolCalls) != 1 {
				t.Fatalf("want 1 lifted call, got %d (content=%q)", len(got.ToolCalls), got.Content)
			}
			tc := got.ToolCalls[0]
			if tc.Function.Name != c.wantName {
				t.Errorf("name = %q, want %q", tc.Function.Name, c.wantName)
			}
			if tc.Function.Arguments != c.wantArgs {
				t.Errorf("arguments = %q, want %q", tc.Function.Arguments, c.wantArgs)
			}
			if got.Content != c.wantContent {
				t.Errorf("content = %q, want %q (delimiters not stripped)", got.Content, c.wantContent)
			}
		})
	}
}

// TestLiftTextToolCalls_MistralMultiCall covers Mistral's multi-call array: every
// element is lifted, ids stay unique, and the single [TOOL_CALLS] span is stripped.
func TestLiftTextToolCalls_MistralMultiCall(t *testing.T) {
	m := Message{
		Role:    RoleAssistant,
		Content: `[TOOL_CALLS][{"name":"Read","arguments":{"path":"a"}},{"name":"Read","arguments":{"path":"b"}}]`,
	}
	got := LiftTextToolCalls(m)
	if len(got.ToolCalls) != 2 {
		t.Fatalf("want 2 lifted calls, got %d", len(got.ToolCalls))
	}
	if got.ToolCalls[0].ID == got.ToolCalls[1].ID {
		t.Errorf("lifted calls share an id: %q", got.ToolCalls[0].ID)
	}
	if got.Content != "" {
		t.Errorf("content = %q, want empty (marker stripped)", got.Content)
	}
}

// TestLiftTextToolCalls_DialectPrecedence asserts the registry never mixes
// dialects: when content carries a specific delimiter (a Hermes tag), the bare-JSON
// extractor must NOT also fire on the inner JSON — the call is lifted exactly once.
func TestLiftTextToolCalls_DialectPrecedence(t *testing.T) {
	m := Message{
		Role:    RoleAssistant,
		Content: `<tool_call>{"name":"Bash","arguments":{"command":"ls"}}</tool_call>`,
	}
	got := LiftTextToolCalls(m)
	if len(got.ToolCalls) != 1 {
		t.Fatalf("want exactly 1 lifted call (no dialect double-count), got %d", len(got.ToolCalls))
	}
}

// TestLiftTextToolCalls_FencedNonCallLeftAlone is the conservative guard for the
// fenced/bare dialects: a fenced JSON blob with no "name" is ordinary output, not a
// tool call, and must be left in the content untouched (no fabricated call).
func TestLiftTextToolCalls_FencedNonCallLeftAlone(t *testing.T) {
	content := "Here is the config:\n```json\n{\"port\":8080,\"host\":\"localhost\"}\n```"
	got := LiftTextToolCalls(Message{Role: RoleAssistant, Content: content})
	if len(got.ToolCalls) != 0 {
		t.Fatalf("nameless fenced JSON wrongly lifted %d calls", len(got.ToolCalls))
	}
	if got.Content != content {
		t.Errorf("content was modified: %q", got.Content)
	}
}

// TestLiftTextToolCalls_BareJSONOnlyWholeContent guards the most ambiguous dialect:
// a JSON object EMBEDDED in prose (a model discussing an example) must NOT be lifted
// — bare-JSON fires only when the whole trimmed message is the call object.
func TestLiftTextToolCalls_BareJSONOnlyWholeContent(t *testing.T) {
	content := `You could call it like {"name":"Bash","arguments":{"command":"ls"}} if you wanted.`
	got := LiftTextToolCalls(Message{Role: RoleAssistant, Content: content})
	if len(got.ToolCalls) != 0 {
		t.Fatalf("bare JSON embedded in prose wrongly lifted %d calls", len(got.ToolCalls))
	}
	if got.Content != content {
		t.Errorf("content was modified: %q", got.Content)
	}
}

// TestLiftTextToolCalls_NewDialectsRespectStructuredClobber confirms the
// no-clobber posture holds for the new dialects too: a provider-parsed structured
// call is authoritative even if the content also carries a text-dialect block.
func TestLiftTextToolCalls_NewDialectsRespectStructuredClobber(t *testing.T) {
	m := Message{
		Role:      RoleAssistant,
		Content:   `[TOOL_CALLS][{"name":"X","arguments":{}}]`,
		ToolCalls: []ToolCall{{ID: "call_0", Type: "function", Function: Func{Name: "real"}}},
	}
	got := LiftTextToolCalls(m)
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].Function.Name != "real" {
		t.Errorf("structured calls were clobbered by a new dialect: %+v", got.ToolCalls)
	}
}

func TestNormalizeCompletionToolCallsMintsUsableIDsAndTypes(t *testing.T) {
	comp := normalizeCompletionToolCalls(&Completion{
		Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Function: Func{Name: "a", Arguments: `{}`}},
			{ID: "given", Function: Func{Name: "b", Arguments: `{}`}},
			{ID: "given", Type: "function", Function: Func{Name: "c", Arguments: `{}`}},
		}},
		FinishReason: "stop",
	})
	if comp.FinishReason != "tool_calls" {
		t.Fatalf("finish_reason = %q, want tool_calls", comp.FinishReason)
	}
	seen := map[string]bool{}
	for i, tc := range comp.Message.ToolCalls {
		if tc.ID == "" {
			t.Fatalf("tool call %d has empty id: %+v", i, tc)
		}
		if seen[tc.ID] {
			t.Fatalf("duplicate id %q in %+v", tc.ID, comp.Message.ToolCalls)
		}
		seen[tc.ID] = true
		if tc.Type != "function" {
			t.Fatalf("tool call %d type = %q, want function", i, tc.Type)
		}
	}
	if comp.Message.ToolCalls[1].ID != "given" {
		t.Fatalf("first non-empty provider id was not preserved: %+v", comp.Message.ToolCalls)
	}
}

func TestNormalizeToolArguments(t *testing.T) {
	cases := map[string]string{
		`{"a":1}`:     `{"a":1}`,
		`"{\"a\":1}"`: `{"a":1}`,
		``:            `{}`,
		`null`:        `{}`,
	}
	for in, want := range cases {
		if got := normalizeToolArguments([]byte(in)); got != want {
			t.Errorf("normalize(%q) = %q, want %q", in, got, want)
		}
	}
}
