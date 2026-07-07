package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// openai_prefix_stability_test.go — cache-default[22] (#1540): the OpenAI-compatible
// half of "preserve Anthropic cache_control and OpenAI-compatible stable prefix bytes
// through guard/serve transformations".
//
// OpenAI-compatible providers (OpenAI, xAI, and the vLLM/SGLang/llama.cpp-server
// drop-ins) have no cache_control grammar — provider-side prefix caching is AUTOMATIC
// and keys on the leading bytes/tokens of the request. So the preservation contract on
// this wire is not "forward the raw body verbatim" (that is the Anthropic passthrough,
// TestHTTPPlannerRawBodyPassthrough — whose openai_ignores_raw_body arm also proves raw
// Anthropic bytes never leak onto this wire); it is that the adapter's re-marshal of
// the canonical transcript is DETERMINISTIC, ORDER-PRESERVING, and APPEND-ONLY across
// turns: turn N's serialized model+messages head is a byte-prefix of turn N+1's body,
// so the provider's automatic prefix cache keeps hitting as the conversation grows.
//
// Each transformation the guard/serve path applies on this wire gets an arm:
//
//   - per-turn sampling re-injection (max_tokens / temperature / stream differ turn to
//     turn on the served path) must ride BEHIND the model+messages head, never inside it;
//   - the pre-send quarantine redaction (QuarantineOutboundMessages — the safety layer
//     on this wire) must be deterministic and ordinal-stable, so a redacted turn
//     re-serializes to the SAME stub bytes on every later turn. A safety rewrite whose
//     stub bytes changed per turn would bust the provider cache on the whole suffix
//     every turn — exactly the failure mode cache-default[22] names;
//   - the opt-in OpenAIToolMessagesAsText lowering (Qwen text blocks for servers that
//     reject OpenAI tool-continuation fields) rewrites messages per-message and in
//     order, so the lowered prefix must be byte-stable too.
//
// xAI rides the identical openAIAdapter (only the provider tag differs), so one
// provider arm covers both.

// openAIWireEcho decodes the prefix-relevant carriers of a marshaled chat-completions
// body. json.RawMessage keeps each element's input bytes verbatim, so the comparisons
// below are BYTE comparisons of the wire, not semantic ones.
type openAIWireEcho struct {
	Model    string            `json:"model"`
	Messages []json.RawMessage `json:"messages"`
	Tools    json.RawMessage   `json:"tools"`
}

// assertStableOpenAIPrefix asserts the cross-turn stable-prefix-bytes contract between
// the marshaled body of turn N (turnA, sharedLen messages) and turn N+1 (turnB, a
// strict extension): every shared message element is byte-identical, the tools block is
// byte-identical, and the literal leading bytes of BOTH bodies — `{"model":…,
// "messages":[…` through the last shared element — are the same run, so the provider's
// automatic prefix cache sees an unbroken head. Anything that only differs behind that
// head (sampling, stream flags, appended turns) cannot defeat reuse.
func assertStableOpenAIPrefix(t *testing.T, turnA, turnB []byte, sharedLen int) {
	t.Helper()
	var a, b openAIWireEcho
	if err := json.Unmarshal(turnA, &a); err != nil {
		t.Fatalf("turn A body: %v", err)
	}
	if err := json.Unmarshal(turnB, &b); err != nil {
		t.Fatalf("turn B body: %v", err)
	}
	if len(a.Messages) != sharedLen {
		t.Fatalf("turn A carries %d messages, want %d", len(a.Messages), sharedLen)
	}
	if len(b.Messages) <= sharedLen {
		t.Fatalf("turn B carries %d messages, want an extension beyond the %d shared", len(b.Messages), sharedLen)
	}
	for i := 0; i < sharedLen; i++ {
		if string(a.Messages[i]) != string(b.Messages[i]) {
			t.Errorf("message[%d] not byte-stable across turns (prefix cache would miss from here):\n turn A %s\n turn B %s", i, a.Messages[i], b.Messages[i])
		}
	}
	if string(a.Tools) != string(b.Tools) {
		t.Errorf("tools block not byte-stable across turns:\n turn A %s\n turn B %s", a.Tools, b.Tools)
	}
	// The raw wire head: the struct marshal puts model then messages first (no
	// ExtraBody in these arms, so field order is the struct order), which makes this
	// reconstruction the literal leading run of bytes the provider tokenizes.
	modelJSON, err := json.Marshal(a.Model)
	if err != nil {
		t.Fatal(err)
	}
	elems := make([]string, sharedLen)
	for i := 0; i < sharedLen; i++ {
		elems[i] = string(a.Messages[i])
	}
	head := `{"model":` + string(modelJSON) + `,"messages":[` + strings.Join(elems, ",")
	if !strings.HasPrefix(string(turnA), head) {
		t.Errorf("turn A body does not start with the model+messages head:\nhead %s\nbody %s", head, turnA)
	}
	if !strings.HasPrefix(string(turnB), head) {
		t.Errorf("turn B body does not start with turn A's model+messages head (stable prefix bytes broken):\nhead %s\nbody %s", head, turnB)
	}
}

// prefixStabilityTurns builds the two-turn fixture: turn one is the shared transcript
// (system, user, assistant tool call, tool result, assistant answer), turn two extends
// it with a follow-up round including a SECOND tool exchange, so the extension crosses
// every message shape the wire carries.
func prefixStabilityTurns(toolResult, secondToolResult string) (turn1, turn2 []Message) {
	turn1 = append(adapterTestMessages(toolResult),
		Message{Role: RoleAssistant, Content: "Economy is refundable within 24h."})
	turn2 = append(append([]Message(nil), turn1...),
		Message{Role: RoleUser, Content: "And business class?"},
		Message{Role: RoleAssistant, Content: "Checking.", ToolCalls: []ToolCall{
			{ID: "call_2", Type: "function", Function: Func{Name: "lookup", Arguments: `{"city":"JFK"}`}},
		}},
		Message{Role: RoleTool, ToolCallID: "call_2", Name: "lookup", Content: secondToolResult},
	)
	return turn1, turn2
}

// TestOpenAIStablePrefixBytesAcrossTurns is the native-tool-wire arm: with per-turn
// sampling re-injection (different max_tokens/temperature, and turn two streamed), the
// model+messages head of turn N stays a byte-prefix of turn N+1's body.
func TestOpenAIStablePrefixBytesAcrossTurns(t *testing.T) {
	adapter := openAIAdapter{provider: ProviderOpenAI}
	turn1, turn2 := prefixStabilityTurns(`{"fare":"$420"}`, `{"fare":"$1980"}`)

	turnA, err := adapter.MarshalRequest(adapterRequest{
		Model: "m", Messages: turn1, Tools: adapterTestTools(),
		MaxTokens: 64, Temperature: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	// The served path re-injects sampling every turn and may flip to streaming; none
	// of that may reach forward of the messages array.
	turnB, err := adapter.MarshalRequest(adapterRequest{
		Model: "m", Messages: turn2, Tools: adapterTestTools(),
		MaxTokens: 512, Temperature: 0.7, Stream: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	assertStableOpenAIPrefix(t, turnA, turnB, len(turn1))
}

// TestOpenAIQuarantineRedactionIsPrefixStableAcrossTurns is the safety-layer arm: the
// pre-send quarantine rewrites a poisoned tool result into a stub, and that stub must
// be the SAME bytes when the transcript is re-serialized one turn later — including
// when a SECOND poisoned result is quarantined behind it (the ordinal of the first stub
// must not shift).
func TestOpenAIQuarantineRedactionIsPrefixStableAcrossTurns(t *testing.T) {
	adapter := openAIAdapter{provider: ProviderOpenAI}
	turn1, turn2 := prefixStabilityTurns(injectionDoc, injectionDoc)

	safe1, qs1 := QuarantineOutboundMessages(turn1)
	if len(qs1) != 1 {
		t.Fatalf("turn 1 quarantines = %d, want 1 (fixture no longer trips the admitter)", len(qs1))
	}
	if safe1[3].Content == turn1[3].Content {
		t.Fatal("turn 1 tool result not rewritten — the redaction arm would be vacuous")
	}
	safe2, qs2 := QuarantineOutboundMessages(turn2)
	if len(qs2) != 2 {
		t.Fatalf("turn 2 quarantines = %d, want 2 (both poisoned results)", len(qs2))
	}

	turnA, err := adapter.MarshalRequest(adapterRequest{
		Model: "m", Messages: safe1, Tools: adapterTestTools(), MaxTokens: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	turnB, err := adapter.MarshalRequest(adapterRequest{
		Model: "m", Messages: safe2, Tools: adapterTestTools(), MaxTokens: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	assertStableOpenAIPrefix(t, turnA, turnB, len(safe1))
}

// TestOpenAIToolMessagesAsTextLoweringIsPrefixStable is the compat-lowering arm: with
// OpenAIToolMessagesAsText on, prior tool exchanges are lowered to Qwen text blocks —
// per-message and in order, so the lowered head must stay byte-stable as turns append.
func TestOpenAIToolMessagesAsTextLoweringIsPrefixStable(t *testing.T) {
	adapter := openAIAdapter{provider: ProviderOpenAI}
	turn1, turn2 := prefixStabilityTurns(`{"fare":"$420"}`, `{"fare":"$1980"}`)

	marshal := func(messages []Message) []byte {
		t.Helper()
		body, err := adapter.MarshalRequest(adapterRequest{
			Model: "m", Messages: messages, Tools: adapterTestTools(),
			MaxTokens: 64, OpenAIToolMessagesAsText: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		return body
	}
	turnA := marshal(turn1)
	turnB := marshal(turn2)
	if strings.Contains(string(turnB), `"role":"tool"`) {
		t.Fatal("lowering did not fire — role=tool survived, the arm would test the native wire twice")
	}
	assertStableOpenAIPrefix(t, turnA, turnB, len(turn1))
}
