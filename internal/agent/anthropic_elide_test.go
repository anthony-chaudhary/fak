package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// elideWireBody builds a realistic /v1/messages body for the elision witness. Layout (len=10):
//
//	[0] user      — cached HEAD, carries the first cache_control breakpoint (the protected prefix)
//	[1] assistant — small
//	[2] user      — tool_result with an oversized text payload, NO cache_control  → ELIGIBLE (shrinks)
//	[3] assistant — small
//	[4] user      — tool_result with an oversized payload AND cache_control       → PROTECTED (cache)
//	[5] assistant — small
//	[6] user      — tool_result with an oversized payload, but inside the recent  → PROTECTED (recency)
//	[7] assistant · [8] user · [9] assistant — recent working-set filler
//
// big{A,B,C} are distinct long runs so the test can assert by content which results survived.
func elideWireBody(t *testing.T, bigA, bigB, bigC string) []byte {
	t.Helper()
	type obj = map[string]any
	cc := obj{"type": "ephemeral"}
	toolResult := func(id, text string, cached bool) obj {
		blk := obj{"type": "tool_result", "tool_use_id": id, "content": []obj{{"type": "text", "text": text}}}
		if cached {
			blk["cache_control"] = cc
		}
		return obj{"role": "user", "content": []obj{blk}}
	}
	text := func(role, s string) obj {
		return obj{"role": role, "content": []obj{{"type": "text", "text": s}}}
	}
	msgs := []obj{
		{"role": "user", "content": []obj{{"type": "text", "text": "cached head context", "cache_control": cc}}}, // 0 — first breakpoint
		text("assistant", "a1"),       // 1
		toolResult("t2", bigA, false), // 2 — ELIGIBLE
		text("assistant", "a3"),       // 3
		toolResult("t4", bigB, true),  // 4 — cache_control → PROTECTED
		text("assistant", "a5"),       // 5
		toolResult("t6", bigC, false), // 6 — recent (len-4) → PROTECTED
		text("assistant", "a7"),       // 7
		text("user", "u8"),            // 8
		text("assistant", "a9"),       // 9
	}
	raw, err := json.Marshal(obj{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"system":     []obj{{"type": "text", "text": "policy header", "cache_control": cc}},
		"messages":   msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// protectedPrefixEnd returns the absolute byte offset just past the first cache_control
// breakpoint message — the prefix the elision must preserve byte-for-byte.
func protectedPrefixEnd(t *testing.T, raw []byte) int {
	t.Helper()
	var o map[string]json.RawMessage
	if err := json.Unmarshal(raw, &o); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	elems, spans, ok := decodeArrayElements(raw, o["messages"])
	if !ok {
		t.Fatal("decodeArrayElements failed")
	}
	pfx := firstBreakpointMessage(elems)
	if pfx < 0 {
		return spans[0].start
	}
	return spans[pfx].end
}

// TestElideShrinksOldOversizedResultKeepsPrefixAndWorkingSet is the core witness: an oversized
// tool_result in the un-cached, non-recent middle is shrunk to head+tail, while the cached
// prefix, a cache_control-bearing result, and a recent result are ALL left byte-for-byte intact,
// and the rewritten body still decodes as a valid Anthropic request.
func TestElideShrinksOldOversizedResultKeepsPrefixAndWorkingSet(t *testing.T) {
	const threshold = 1024
	bigA := strings.Repeat("A", 4000) // eligible — should shrink
	bigB := strings.Repeat("B", 4000) // cache_control — must survive
	bigC := strings.Repeat("C", 4000) // recent — must survive
	raw := elideWireBody(t, bigA, bigB, bigC)
	orig := append([]byte(nil), raw...)
	prefixEnd := protectedPrefixEnd(t, orig)

	out, outcome := ElideAnthropicResultsWithOutcome(raw, threshold)

	if outcome.Reason != ElideReasonNone {
		t.Fatalf("expected FIRED (ElideReasonNone), got %q", outcome.Reason)
	}
	if outcome.Elided != 1 {
		t.Fatalf("expected exactly 1 elided result, got %d", outcome.Elided)
	}
	if outcome.ShedBytes <= 0 {
		t.Fatalf("expected positive ShedBytes, got %d", outcome.ShedBytes)
	}
	if len(out) >= len(orig) {
		t.Fatalf("expected a shorter body, got %d >= %d", len(out), len(orig))
	}
	// (a) The eligible oversized result was shrunk: its full run is gone, the marker is present.
	if bytes.Contains(out, []byte(bigA)) {
		t.Error("eligible oversized result A was NOT shrunk (full run still present)")
	}
	if !bytes.Contains(out, []byte("characters of older tool_result output elided")) {
		t.Error("elision marker missing from output")
	}
	// (b) The cache_control-bearing result and (c) the recent result both survive verbatim.
	if !bytes.Contains(out, []byte(bigB)) {
		t.Error("cache_control-bearing result B was wrongly shrunk (cache burst)")
	}
	if !bytes.Contains(out, []byte(bigC)) {
		t.Error("recent working-set result C was wrongly shrunk")
	}
	// (d) The protected prefix bytes are byte-identical.
	if prefixEnd > len(out) || !bytes.Equal(orig[:prefixEnd], out[:prefixEnd]) {
		t.Error("protected prefix bytes changed — cache hit would be lost")
	}
	// (e) The rewritten body still decodes as a valid Anthropic request.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Errorf("rewritten body failed to re-decode: %v", err)
	}
	// The input slice header is not mutated in place when a rewrite happens.
	if !bytes.Equal(raw, orig) {
		t.Error("input raw was mutated in place")
	}
}

// TestElideStringContentResult covers the bare-string tool_result content shape (some harnesses
// send content as a JSON string rather than an array of text blocks).
func TestElideStringContentResult(t *testing.T) {
	const threshold = 1024
	big := strings.Repeat("Z", 4000)
	type obj = map[string]any
	cc := obj{"type": "ephemeral"}
	raw, err := json.Marshal(obj{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"system":     []obj{{"type": "text", "text": "sys", "cache_control": cc}},
		"messages": []obj{
			{"role": "user", "content": []obj{{"type": "text", "text": "head", "cache_control": cc}}},       // 0 breakpoint
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a"}}},                          // 1
			{"role": "user", "content": []obj{{"type": "tool_result", "tool_use_id": "t", "content": big}}}, // 2 string content → eligible
			{"role": "assistant", "content": []obj{{"type": "text", "text": "b"}}},                          // 3
			{"role": "user", "content": []obj{{"type": "text", "text": "c"}}},                               // 4
			{"role": "assistant", "content": []obj{{"type": "text", "text": "d"}}},                          // 5
			{"role": "user", "content": []obj{{"type": "text", "text": "e"}}},                               // 6
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	out, outcome := ElideAnthropicResultsWithOutcome(raw, threshold)
	if outcome.Reason != ElideReasonNone || outcome.Elided != 1 {
		t.Fatalf("expected 1 elided string-content result, got reason=%q elided=%d", outcome.Reason, outcome.Elided)
	}
	if bytes.Contains(out, []byte(big)) {
		t.Error("string-content oversized result was not shrunk")
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Errorf("rewritten body failed to re-decode: %v", err)
	}
}

// TestElideNestedCacheControlIsProtected is the regression for the cache-prefix-burst /
// working-set-loss bugs the adversarial review found: a cache_control on a text block NESTED
// inside a tool_result.content array is a valid Anthropic breakpoint the shrinker can reach, but
// the SHALLOW detector missed it — so it could anchor PAST it (msg[0]) or shrink it (msg[2]),
// bursting the cache. The deep detector (messageHasCacheControlForElide) must protect both, while
// a clean oversized result still elides so the test is not vacuously identity.
func TestElideNestedCacheControlIsProtected(t *testing.T) {
	const threshold = 1024
	type obj = map[string]any
	cc := obj{"type": "ephemeral"}
	big0 := strings.Repeat("0", 4000)  // msg[0] nested-cc anchor — must survive
	big2 := strings.Repeat("2", 4000)  // msg[2] nested-cc — must survive
	bigC := strings.Repeat("C", 4000)  // msg[4] clean — must shrink
	nestedCC := func(id, text string) obj {
		return obj{"role": "user", "content": []obj{{"type": "tool_result", "tool_use_id": id,
			"content": []obj{{"type": "text", "text": text, "cache_control": cc}}}}}
	}
	cleanTR := func(id, text string) obj {
		return obj{"role": "user", "content": []obj{{"type": "tool_result", "tool_use_id": id,
			"content": []obj{{"type": "text", "text": text}}}}}
	}
	txt := func(role, s string) obj { return obj{"role": role, "content": []obj{{"type": "text", "text": s}}} }
	raw, err := json.Marshal(obj{
		"model": "claude-sonnet-4-6", "max_tokens": 1024,
		"system": []obj{{"type": "text", "text": "sys", "cache_control": cc}},
		"messages": []obj{
			nestedCC("t0", big0),    // 0 — nested-cc → the deep anchor (pfxEnd=0), protected
			txt("assistant", "a1"),  // 1
			nestedCC("t2", big2),    // 2 — nested-cc in the eligible band → must be SKIPPED
			txt("assistant", "a3"),  // 3
			cleanTR("t4", bigC),     // 4 — clean oversized, eligible → must SHRINK
			txt("assistant", "a5"),  // 5
			txt("user", "u6"),       // 6
			txt("assistant", "a7"),  // 7
			txt("user", "u8"),       // 8
			txt("assistant", "a9"),  // 9
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	orig := append([]byte(nil), raw...)
	out, outcome := ElideAnthropicResultsWithOutcome(raw, threshold)
	if outcome.Reason != ElideReasonNone || outcome.Elided != 1 {
		t.Fatalf("expected exactly the clean result to elide, got reason=%q elided=%d", outcome.Reason, outcome.Elided)
	}
	if !bytes.Contains(out, []byte(big0)) {
		t.Error("BUG: msg[0] nested-cc anchor was shrunk — head cache burst")
	}
	if !bytes.Contains(out, []byte(big2)) {
		t.Error("BUG: msg[2] nested-cc result was shrunk — cache-control message edited")
	}
	if bytes.Contains(out, []byte(bigC)) {
		t.Error("clean oversized result was NOT shrunk (test vacuous)")
	}
	// Head prefix (through the nested-cc anchor msg[0]) must be byte-identical.
	prefixEnd := protectedPrefixEnd(t, orig)
	if !bytes.Equal(orig[:prefixEnd], out[:prefixEnd]) {
		t.Error("head prefix bytes changed")
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Errorf("re-decode: %v", err)
	}
}

// TestElideDuplicateToolUseIDNotCorrupted is the regression for the json-corruption bug: a
// tool_result whose tool_use_id value is byte-identical to its string content, with tool_use_id
// ordered BEFORE content on the wire (so a bytes.Index over the whole block would mis-splice the
// id). objectValueSpan locates the content value by KEY, so the id stays intact and the content
// shrinks. Hand-authored bytes (Go's json.Marshal sorts keys, hiding the ordering).
func TestElideDuplicateToolUseIDNotCorrupted(t *testing.T) {
	const threshold = 100
	big := strings.Repeat("X", 300)
	raw := []byte(fmt.Sprintf(`{"model":"claude-sonnet-4-6","max_tokens":1024,`+
		`"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}],`+
		`"messages":[`+
		`{"role":"user","content":[{"type":"text","text":"head","cache_control":{"type":"ephemeral"}}]},`+
		`{"role":"assistant","content":[{"type":"text","text":"a"}]},`+
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"%s","content":"%s"}]},`+
		`{"role":"assistant","content":[{"type":"text","text":"b"}]},`+
		`{"role":"user","content":[{"type":"text","text":"c"}]},`+
		`{"role":"assistant","content":[{"type":"text","text":"d"}]},`+
		`{"role":"user","content":[{"type":"text","text":"e"}]}`+
		`]}`, big, big))
	out, outcome := ElideAnthropicResultsWithOutcome(raw, threshold)
	if outcome.Reason != ElideReasonNone || outcome.Elided != 1 {
		t.Fatalf("expected 1 elided result, got reason=%q elided=%d", outcome.Reason, outcome.Elided)
	}
	if !strings.Contains(string(out), `"tool_use_id":"`+big+`"`) {
		t.Error("BUG: tool_use_id was corrupted — the splice mis-located the content value")
	}
	if strings.Contains(string(out), `"content":"`+big+`"`) {
		t.Error("content value was NOT shrunk (the oversized payload survived)")
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Errorf("re-decode: %v", err)
	}
}

// TestObjectValueSpan pins the key-based value locator: it must return the EXACT span of each
// key's value even when a sibling holds identical bytes (the property that fixes the corruption).
func TestObjectValueSpan(t *testing.T) {
	obj := []byte(`{"a":"XYZ","b":"XYZ","c":123}`)
	aS, aE, ok := objectValueSpan(obj, "a")
	if !ok || string(obj[aS:aE]) != `"XYZ"` {
		t.Fatalf("a: got ok=%v span=%q", ok, obj[aS:aE])
	}
	bS, bE, ok := objectValueSpan(obj, "b")
	if !ok || string(obj[bS:bE]) != `"XYZ"` {
		t.Fatalf("b: got ok=%v span=%q", ok, obj[bS:bE])
	}
	if aS == bS {
		t.Error("a and b resolved to the SAME span despite distinct positions (locator is value-keyed, not key-keyed)")
	}
	cS, cE, ok := objectValueSpan(obj, "c")
	if !ok || string(obj[cS:cE]) != `123` {
		t.Fatalf("c: got ok=%v span=%q", ok, obj[cS:cE])
	}
	if _, _, ok := objectValueSpan(obj, "missing"); ok {
		t.Error("absent key must return ok=false")
	}
}

// TestElideIdentityCases pins the fail-safe identity returns: disabled, nothing oversized, and
// no cache anchor must all leave the body byte-for-byte unchanged.
func TestElideIdentityCases(t *testing.T) {
	big := strings.Repeat("A", 4000)
	withAnchor := elideWireBody(t, big, big, big)

	// 1. Disabled (threshold 0) is identity.
	if out, oc := ElideAnthropicResultsWithOutcome(withAnchor, 0); !bytes.Equal(out, withAnchor) || oc.Reason != ElideReasonOff {
		t.Errorf("threshold 0 must be identity/off, got reason=%q changed=%v", oc.Reason, !bytes.Equal(out, withAnchor))
	}
	// 2. Nothing oversized (huge threshold) is identity.
	if out, oc := ElideAnthropicResultsWithOutcome(withAnchor, 1<<20); !bytes.Equal(out, withAnchor) || oc.Reason != ElideReasonUnderThreshold {
		t.Errorf("nothing-oversized must be identity/under_threshold, got reason=%q changed=%v", oc.Reason, !bytes.Equal(out, withAnchor))
	}
	// 3. No cache anchor (no cache_control anywhere) is identity — we cannot know the cache boundary.
	type obj = map[string]any
	noAnchor, err := json.Marshal(obj{
		"model": "claude-sonnet-4-6", "max_tokens": 1024,
		"messages": []obj{
			{"role": "user", "content": []obj{{"type": "text", "text": "h"}}},
			{"role": "assistant", "content": []obj{{"type": "text", "text": "a"}}},
			{"role": "user", "content": []obj{{"type": "tool_result", "tool_use_id": "t", "content": []obj{{"type": "text", "text": big}}}}},
			{"role": "assistant", "content": []obj{{"type": "text", "text": "b"}}},
			{"role": "user", "content": []obj{{"type": "text", "text": "c"}}},
			{"role": "assistant", "content": []obj{{"type": "text", "text": "d"}}},
			{"role": "user", "content": []obj{{"type": "text", "text": "e"}}},
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if out, oc := ElideAnthropicResultsWithOutcome(noAnchor, 1024); !bytes.Equal(out, noAnchor) || oc.Reason != ElideReasonNoBreakpoint {
		t.Errorf("no cache anchor must be identity/no_breakpoint, got reason=%q changed=%v", oc.Reason, !bytes.Equal(out, noAnchor))
	}
	// 4. Non-JSON body is identity.
	if out, oc := ElideAnthropicResultsWithOutcome([]byte("not json"), 1024); !bytes.Equal(out, []byte("not json")) || oc.Reason != ElideReasonNonJSON {
		t.Errorf("non-JSON must be identity/non_json, got reason=%q", oc.Reason)
	}
}
