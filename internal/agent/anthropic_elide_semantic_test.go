package agent

// anthropic_elide_semantic_test.go — the semantic-well-formedness witness for the elision
// post-splice guard (spliceVerdictMalformedResult / ElideReasonMalformedResult).
//
// The bug these tests pin: verifySplicedBody's ONLY structural gate was a re-decode through
// DecodeAnthropicMessagesRequest — fak's OWN permissive decoder, which accumulates text with a
// strings.Builder and silently DROPS empty blocks. So a spliced body carrying an empty `text`
// value, an empty message `content` array, or a `tool_result` with empty content re-decodes
// CLEANLY here yet is rejected by the real Anthropic Messages API with
// `400 … request … malformed`. Elision (ON by default) would then ship it and 400 the session.
// The fix scans the spliced result for those shapes and returns identity with a labeled reason
// instead of shipping a body the provider will reject.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// TestBodyHasEmptyBlockSemanticsCatchesEachShape is the direct unit witness for the scanner: it
// must flag each of the three Anthropic-invalid empties (empty text value, empty message content
// array, empty tool_result content) and must NOT flag a well-formed body — including a body whose
// tool_result content is a normal shrunk head+tail (the happy-path elision output).
func TestBodyHasEmptyBlockSemanticsCatchesEachShape(t *testing.T) {
	cases := []struct {
		name string
		body string
		want bool
	}{
		{
			name: "empty text value",
			body: `{"messages":[{"role":"user","content":[{"type":"text","text":""}]}]}`,
			want: true,
		},
		{
			name: "empty message content array",
			body: `{"messages":[{"role":"assistant","content":[]}]}`,
			want: true,
		},
		{
			name: "tool_result with empty string content",
			body: `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":""}]}]}`,
			want: true,
		},
		{
			name: "tool_result with empty content array",
			body: `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[]}]}]}`,
			want: true,
		},
		{
			name: "tool_result whose only block is empty text",
			body: `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":""}]}]}]}`,
			want: true,
		},
		{
			name: "well-formed non-empty text",
			body: `{"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`,
			want: false,
		},
		{
			name: "well-formed tool_result with text",
			body: `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"ok"}]}]}]}`,
			want: false,
		},
		{
			name: "bare-string message content is out of scope (not flagged)",
			body: `{"messages":[{"role":"user","content":"just a prompt"}]}`,
			want: false,
		},
		{
			name: "shrunk head+tail marker value is non-empty and NOT flagged",
			body: fmt.Sprintf(`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":%q}]}]}]}`,
				"HEAD"+elideMarkerf(1000)+"TAIL"),
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := bodyHasEmptyBlockSemantics([]byte(tc.body)); got != tc.want {
				t.Fatalf("bodyHasEmptyBlockSemantics = %v, want %v", got, tc.want)
			}
		})
	}
}

// elideWireBodyWithEmptyText builds the elision witness body but plants an empty `text` block in
// message [1] (an assistant turn OUTSIDE the eligible band — it is neither a tool_result nor
// oversized, so elision never touches it). msg[2] still carries the oversized eligible tool_result
// that elision SHOULD fire on. This is the shape that produces the real-world 400: the empty-text
// assistant turn (Claude Code emits these alongside a tool_use) rides along in a body that fak
// ACTIVELY transforms and asserts well-formed by shipping. bigA is the oversized eligible payload.
func elideWireBodyWithEmptyText(t *testing.T, bigA string) []byte {
	t.Helper()
	type obj = map[string]any
	cc := obj{"type": "ephemeral"}
	raw, err := json.Marshal(obj{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"system":     []obj{{"type": "text", "text": "policy header", "cache_control": cc}},
		"messages": []obj{
			{"role": "user", "content": []obj{{"type": "text", "text": "cached head context", "cache_control": cc}}}, // 0 — first breakpoint
			{"role": "assistant", "content": []obj{{"type": "text", "text": ""}}},                                    // 1 — EMPTY text (Anthropic 400 shape); untouched by elision
			{"role": "user", "content": []obj{{"type": "tool_result", "tool_use_id": "t2",
				"content": []obj{{"type": "text", "text": bigA}}}}}, // 2 — oversized eligible → elision fires here
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a3"}}}, // 3
			{"role": "user", "content": []obj{{"type": "text", "text": "u4"}}},      // 4
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a5"}}}, // 5
			{"role": "user", "content": []obj{{"type": "text", "text": "u6"}}},      // 6 — recent window filler
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a7"}}}, // 7
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// TestElideReturnsIdentityOnSemanticallyMalformedBody is the end-to-end witness for the fix: given
// a body that carries an empty-text block (an Anthropic 400 shape) AND a separate oversized
// eligible tool_result, the OLD behavior would splice the oversized result and SHIP the transformed
// body — which the API rejects with 400, since it still carries the empty text. The NEW behavior
// runs the semantic scan on the spliced result, sees the empty text, and returns the input
// UNCHANGED with ElideReasonMalformedResult, so fak never asserts-and-ships a body the provider
// will reject.
func TestElideReturnsIdentityOnSemanticallyMalformedBody(t *testing.T) {
	const threshold = 1024
	bigA := strings.Repeat("A", 4000) // oversized — WOULD be elided if the body were well-formed
	raw := elideWireBodyWithEmptyText(t, bigA)
	orig := append([]byte(nil), raw...)

	// Sanity: the input really does contain the empty-text 400 shape, so the test is non-vacuous.
	if !bodyHasEmptyBlockSemantics(orig) {
		t.Fatal("test setup wrong: input does not carry the empty-text shape the fix must catch")
	}
	// Sanity: the OLD code path (splice + re-decode only) WOULD have produced a body that still
	// contains the malformed empty block — prove the splice itself succeeds and shrinks bigA, so the
	// only thing standing between fak and shipping a 400 body is the new semantic gate.
	if _, err := DecodeAnthropicMessagesRequest(orig); err != nil {
		t.Fatalf("input must re-decode cleanly through fak's permissive decoder (that IS the bug): %v", err)
	}

	out, outcome := ElideAnthropicResultsWithOutcome(raw, threshold)

	if outcome.Reason != ElideReasonMalformedResult {
		t.Fatalf("expected identity with reason %q, got reason=%q (elided=%d)", ElideReasonMalformedResult, outcome.Reason, outcome.Elided)
	}
	if outcome.Elided != 0 || outcome.ShedBytes != 0 {
		t.Fatalf("a malformed-result bail must report no work: got Elided=%d ShedBytes=%d", outcome.Elided, outcome.ShedBytes)
	}
	// Identity: the returned body is byte-for-byte the input (elision shipped nothing).
	if !bytes.Equal(out, orig) {
		t.Error("expected identity (raw unchanged) on a would-be-malformed splice")
	}
	// The oversized payload is STILL present (it was NOT shrunk) — proves the bail happened
	// before shipping, not that the splice merely no-op'd.
	if !bytes.Contains(out, []byte(bigA)) {
		t.Error("oversized payload should be intact on the identity return")
	}
	// Input slice header not mutated in place.
	if !bytes.Equal(raw, orig) {
		t.Error("input raw was mutated in place")
	}
}

// TestElideStillFiresOnWellFormedBodyNoRegression is the happy-path no-regression guard: the SAME
// oversized eligible tool_result, in a body WITHOUT the empty-text block, must still be shrunk and
// the rewritten body must be both re-decodable AND semantically well-formed (the new scan passes).
// This proves the semantic gate is a targeted reject, not a blanket disable of elision.
func TestElideStillFiresOnWellFormedBodyNoRegression(t *testing.T) {
	const threshold = 1024
	bigA := strings.Repeat("A", 4000)
	bigB := strings.Repeat("B", 4000) // cache_control — must survive
	bigC := strings.Repeat("C", 4000) // recent — must survive
	raw := elideWireBody(t, bigA, bigB, bigC)

	out, outcome := ElideAnthropicResultsWithOutcome(raw, threshold)

	if outcome.Reason != ElideReasonNone {
		t.Fatalf("expected FIRED (ElideReasonNone) on a well-formed body, got %q", outcome.Reason)
	}
	if outcome.Elided != 1 || outcome.ShedBytes <= 0 {
		t.Fatalf("expected exactly 1 elided result with positive shed, got Elided=%d ShedBytes=%d", outcome.Elided, outcome.ShedBytes)
	}
	if bytes.Contains(out, []byte(bigA)) {
		t.Error("eligible oversized result A was NOT shrunk (elision wrongly suppressed)")
	}
	// The rewritten body re-decodes AND is semantically well-formed — the new gate does not flag it.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Errorf("rewritten body failed to re-decode: %v", err)
	}
	if bodyHasEmptyBlockSemantics(out) {
		t.Error("happy-path elision output was wrongly flagged as semantically malformed")
	}
}

// TestVerifySplicedBodyMalformedVerdict pins the shared verdict layer directly: a spliced body
// that decodes but carries an empty text block must yield spliceVerdictMalformedResult, not
// spliceVerdictOK — so BOTH the elision and compaction callers get the labeled bail. spans/pfxEnd
// are set so the prefix and tail checks pass, isolating the semantic verdict.
func TestVerifySplicedBodyMalformedVerdict(t *testing.T) {
	// A minimal well-formed input. spans covers the single messages[] element; pfxEnd=-1 (no
	// breakpoint), so the prefix check is just the array open and the tail check the trailing `]}`
	// — both survive a middle-only value edit, isolating the semantic verdict under test.
	raw := []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"hello world"}]}]}`)
	var o map[string]json.RawMessage
	if err := json.Unmarshal(raw, &o); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	_, spans, ok := decodeArrayElements(raw, o["messages"])
	if !ok || len(spans) != 1 {
		t.Fatalf("decodeArrayElements: ok=%v spans=%d", ok, len(spans))
	}
	// An out with the SAME leading/trailing bytes as raw but an emptied middle text value must
	// yield the malformed verdict (this is exactly what a bad splice would land).
	emptiedOut := bytes.Replace(raw, []byte(`"hello world"`), []byte(`""`), 1)
	if v := verifySplicedBody(raw, emptiedOut, spans, -1); v != spliceVerdictMalformedResult {
		t.Fatalf("expected spliceVerdictMalformedResult on emptied text, got verdict %d", v)
	}
	// Control: a genuinely well-formed same-tail out must return OK.
	wellFormed := bytes.Replace(raw, []byte(`"hello world"`), []byte(`"hi"`), 1)
	if v := verifySplicedBody(raw, wellFormed, spans, -1); v != spliceVerdictOK {
		t.Fatalf("expected spliceVerdictOK on well-formed splice, got verdict %d", v)
	}
}
