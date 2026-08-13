package inputtrigger

import "testing"

// TestClassify is the table the classifier's contract lives in: every arm, the empty
// turn, the mixed turns, the precedence between arms, and — the load-bearing half — the
// fail-conservative cases where a nearly-valid turn must NOT be optimistically named.
func TestClassify(t *testing.T) {
	sys := Message{Role: RoleSystem, Content: "you are a careful agent"}
	usr := Message{Role: RoleUser, Content: "ship the fix"}
	asst := Message{Role: RoleAssistant, Content: "Here is the plan:"}
	tool := Message{Role: RoleTool, Content: "exit=0", ToolCallID: "call_1"}

	cases := []struct {
		name string
		turn []Message
		want Trigger
	}{
		// --- empty ---
		{"nil turn", nil, Other},
		{"empty turn", []Message{}, Other},

		// --- the four named shapes, minimal form ---
		{"system only, one message", []Message{sys}, SystemOnly},
		{"system only, several", []Message{sys, sys, sys}, SystemOnly},
		{"user message alone", []Message{usr}, UserMessage},
		{"assistant prefill alone", []Message{asst}, AssistantPrefill},
		{"tool result alone", []Message{tool}, ToolResult},

		// --- precedence: the NEWEST message decides ---
		{"system then user is a user turn", []Message{sys, usr}, UserMessage},
		{"user then assistant is a prefill", []Message{sys, usr, asst}, AssistantPrefill},
		{"assistant call then tool result is a tool result", []Message{sys, usr, asst, tool}, ToolResult},
		{"tool result then a new user message is a user turn", []Message{sys, usr, asst, tool, usr}, UserMessage},
		{"an earlier tool result does not win over a later prefill", []Message{usr, tool, asst}, AssistantPrefill},

		// --- mixed / unknown roles fall back, never forward ---
		{"unknown role in the tail", []Message{sys, usr, {Role: "developer", Content: "x"}}, Other},
		{"unknown role earlier in an otherwise tool-result turn", []Message{{Role: "developer", Content: "x"}, tool}, Other},
		{"empty role", []Message{{Role: "", Content: "x"}}, Other},
		{"unknown role among system messages is not system-only", []Message{sys, {Role: "moderator"}}, Other},

		// --- fail-conservative: a nearly-valid turn is Other, never optimistic ---
		{"tool message with no tool_call_id is not a tool result", []Message{usr, {Role: RoleTool, Content: "exit=0"}}, Other},
		{"tool message with a blank tool_call_id is not a tool result", []Message{usr, {Role: RoleTool, Content: "exit=0", ToolCallID: "   "}}, Other},
		{"empty assistant message has nothing to prefill", []Message{usr, {Role: RoleAssistant}}, Other},
		{"whitespace-only assistant message has nothing to prefill", []Message{usr, {Role: RoleAssistant, Content: " \n\t"}}, Other},
		{"a trailing system message after user content is mixed, not system-only", []Message{usr, sys}, Other},
		{"a trailing system message after a tool result is mixed", []Message{tool, sys}, Other},

		// --- content emptiness only matters where the shape depends on it ---
		{"blank user message is still a user turn", []Message{{Role: RoleUser}}, UserMessage},
		{"tool result with empty content still counts (a tool may return nothing)",
			[]Message{{Role: RoleTool, ToolCallID: "call_9"}}, ToolResult},
		{"blank system message is still system-only", []Message{{Role: RoleSystem}}, SystemOnly},

		// --- role tokens are normalized the same way everywhere ---
		{"role case and padding are normalized", []Message{{Role: " USER ", Content: "hi"}}, UserMessage},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(tc.turn); got != tc.want {
				t.Fatalf("Classify(%+v) = %q, want %q", tc.turn, got, tc.want)
			}
		})
	}
}

// TestClassifyIsDeterministicAndNonMutating pins the two properties an audit trail rests
// on: the same turn always classifies the same way, and classifying does not disturb the
// turn (so a later re-read of the messages sees exactly what was classified).
func TestClassifyIsDeterministicAndNonMutating(t *testing.T) {
	turn := []Message{
		{Role: RoleSystem, Content: "policy"},
		{Role: RoleUser, Content: "run it"},
		{Role: RoleAssistant, Content: "calling the tool"},
		{Role: RoleTool, Content: "ok", ToolCallID: "call_7"},
	}
	before := append([]Message(nil), turn...)
	first := Classify(turn)
	for i := 0; i < 8; i++ {
		if got := Classify(turn); got != first {
			t.Fatalf("Classify is not deterministic: run %d = %q, first = %q", i, got, first)
		}
	}
	if first != ToolResult {
		t.Fatalf("Classify = %q, want %q", first, ToolResult)
	}
	for i := range turn {
		if turn[i] != before[i] {
			t.Fatalf("Classify mutated message %d: %+v, was %+v", i, turn[i], before[i])
		}
	}
}

// TestParseRole covers the one place a wire role token is interpreted.
func TestParseRole(t *testing.T) {
	cases := []struct {
		raw   string
		want  Role
		known bool
	}{
		{"system", RoleSystem, true},
		{"user", RoleUser, true},
		{"assistant", RoleAssistant, true},
		{"tool", RoleTool, true},
		{"  Tool\n", RoleTool, true},
		{"ASSISTANT", RoleAssistant, true},
		{"developer", "", false},
		{"function", "", false},
		{"", "", false},
		{"   ", "", false},
	}
	for _, tc := range cases {
		got, ok := ParseRole(tc.raw)
		if ok != tc.known || got != tc.want {
			t.Fatalf("ParseRole(%q) = (%q, %v), want (%q, %v)", tc.raw, got, ok, tc.want, tc.known)
		}
	}
}

// TestKnownIsTheClosedSet keeps Known and the constant block from drifting: every named
// trigger is known, and the unclassified zero value is deliberately not.
func TestKnownIsTheClosedSet(t *testing.T) {
	for _, tr := range []Trigger{Other, SystemOnly, UserMessage, AssistantPrefill, ToolResult} {
		if !Known(tr) {
			t.Fatalf("Known(%q) = false, want true", tr)
		}
	}
	for _, tr := range []Trigger{"", "tool-result", "toolresult", "TOOL_RESULT", "prefill"} {
		if Known(tr) {
			t.Fatalf("Known(%q) = true, want false", tr)
		}
	}
}

// TestClassifyOnlyEverAnswersAKnownTrigger is the invariant every consumer relies on: a
// route policy switching over the closed set can never be handed a value outside it.
func TestClassifyOnlyEverAnswersAKnownTrigger(t *testing.T) {
	roles := []Role{RoleSystem, RoleUser, RoleAssistant, RoleTool, "developer", ""}
	for _, a := range roles {
		for _, b := range roles {
			for _, id := range []string{"", "call_1"} {
				for _, content := range []string{"", "x"} {
					turn := []Message{{Role: a, Content: content, ToolCallID: id}, {Role: b, Content: content, ToolCallID: id}}
					if got := Classify(turn); !Known(got) {
						t.Fatalf("Classify(%+v) = %q, which is outside the closed set", turn, got)
					}
				}
			}
		}
	}
}
