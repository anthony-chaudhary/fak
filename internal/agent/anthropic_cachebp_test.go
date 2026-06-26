package agent

import (
	"bytes"
	"strings"
	"testing"
)

// TestPlaceOnLastSystemBlock is the core offensive case: a body with a system head and NO
// cache_control gets a breakpoint spliced onto its LAST system block (caching tools+system),
// the result re-decodes, exactly one breakpoint is added, and every byte before it is verbatim.
func TestPlaceOnLastSystemBlock(t *testing.T) {
	raw := []byte(`{"model":"claude-x","max_tokens":100,` +
		`"system":[{"type":"text","text":"head A"},{"type":"text","text":"head B"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)

	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone {
		t.Fatalf("reason = %q, want placed (none)", oc.Reason)
	}
	if oc.Target != "system" {
		t.Fatalf("target = %q, want system", oc.Target)
	}
	if n := bytes.Count(out, []byte("cache_control")); n != 1 {
		t.Fatalf("cache_control count = %d, want exactly 1", n)
	}
	// It must land on the LAST system block (head B), not the first (head A).
	if !bytes.Contains(out, []byte(`"text":"head B","cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("breakpoint not on the last system block:\n%s", out)
	}
	if bytes.Contains(out, []byte(`"text":"head A","cache_control"`)) {
		t.Fatalf("breakpoint wrongly placed on the FIRST system block:\n%s", out)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("placed body failed to re-decode: %v", err)
	}
	// The prefix up to the rewritten block is byte-identical: the breakpoint is the only change,
	// and it sits inside the last system block, so everything before "head B" is untouched.
	idx := bytes.Index(raw, []byte(`{"type":"text","text":"head B"}`))
	if idx < 0 {
		t.Fatal("fixture sanity: last system block not found")
	}
	if !bytes.Equal(raw[:idx], out[:idx]) {
		t.Fatalf("bytes before the breakpoint changed:\nraw=%s\nout=%s", raw[:idx], out[:idx])
	}
}

// TestRespectsExistingBreakpointInSystem: a cache_control already on the head ⇒ identity.
func TestRespectsExistingBreakpointInSystem(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"h","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"x"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonAlreadySet {
		t.Fatalf("reason = %q, want already_set", oc.Reason)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("body must be returned unchanged when a breakpoint already exists")
	}
}

// TestRespectsExistingBreakpointInMessages: a cache_control anywhere (here on a recent turn, the
// Claude Code shape) ⇒ identity, so we never fight a layout the client already chose.
func TestRespectsExistingBreakpointInMessages(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"h"}],` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonAlreadySet {
		t.Fatalf("reason = %q, want already_set", oc.Reason)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("body must be unchanged when a breakpoint exists in messages")
	}
}

// TestPlaceOnLastToolWhenNoSystem: no system array ⇒ fall back to the last tools[] entry.
func TestPlaceOnLastToolWhenNoSystem(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"tools":[{"name":"a","input_schema":{"type":"object"}},{"name":"b","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone || oc.Target != "tools" {
		t.Fatalf("got reason=%q target=%q, want none/tools", oc.Reason, oc.Target)
	}
	if !bytes.Contains(out, []byte(`"name":"b","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("breakpoint not on the last tool:\n%s", out)
	}
	if bytes.Contains(out, []byte(`"name":"a","input_schema":{"type":"object"},"cache_control"`)) {
		t.Fatalf("breakpoint wrongly placed on a non-last tool:\n%s", out)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("placed body failed to re-decode: %v", err)
	}
}

// TestSystemStringFallsToTools: a bare-string system has no block to anchor on ⇒ use tools.
func TestSystemStringFallsToTools(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,"system":"you are helpful",` +
		`"tools":[{"name":"a","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	_, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone || oc.Target != "tools" {
		t.Fatalf("got reason=%q target=%q, want none/tools", oc.Reason, oc.Target)
	}
}

// TestNoStableHead: no system array and no tools ⇒ nothing safe to cache without touching the
// volatile message tail, so leave the body unchanged.
func TestNoStableHead(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,"system":"hi","messages":[{"role":"user","content":"x"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNoStableHead {
		t.Fatalf("reason = %q, want no_stable_head", oc.Reason)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("body must be unchanged when there is no stable head")
	}
}

// TestNonJSONAndEmptyAreIdentity: garbage and an empty body are returned unchanged.
func TestNonJSONAndEmptyAreIdentity(t *testing.T) {
	for _, raw := range [][]byte{[]byte("not json"), []byte(""), nil, []byte("[1,2,3]")} {
		out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
		if oc.Reason != BreakpointReasonNonJSON {
			t.Fatalf("reason for %q = %q, want non_json", raw, oc.Reason)
		}
		if !bytes.Equal(out, raw) {
			t.Fatalf("non-JSON body %q must be unchanged", raw)
		}
	}
}

// TestEmptyObjectBlock exercises the comma-free splice branch: an empty `{}` block gets the lone
// cache_control key with no leading comma, and the result is still valid JSON.
func TestEmptyObjectBlock(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,"system":[{"type":"text","text":"a"},{}],` +
		`"messages":[{"role":"user","content":"x"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone {
		t.Fatalf("reason = %q, want placed", oc.Reason)
	}
	if !bytes.Contains(out, []byte(`{"cache_control":{"type":"ephemeral"}}`)) {
		t.Fatalf("empty-object splice malformed:\n%s", out)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("placed body failed to re-decode: %v", err)
	}
}

// TestIdempotent: a second placement is a no-op, because the first one added a cache_control that
// the already_set guard then respects.
func TestIdempotent(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,"system":[{"type":"text","text":"h"}],` +
		`"messages":[{"role":"user","content":"x"}]}`)
	once := PlaceAnthropicCacheBreakpoint(raw)
	if bytes.Equal(once, raw) {
		t.Fatal("first placement should have changed the body")
	}
	twice, oc := PlaceAnthropicCacheBreakpointWithOutcome(once)
	if oc.Reason != BreakpointReasonAlreadySet {
		t.Fatalf("second placement reason = %q, want already_set", oc.Reason)
	}
	if !bytes.Equal(twice, once) {
		t.Fatal("placement must be idempotent")
	}
}

// TestPlacementEnablesCompaction is the synergy with the DEFENSIVE half: a body with no
// breakpoint cannot be compacted (CompactReasonNoBreakpoint — nothing to anchor on); after the
// offensive placer adds a breakpoint on the stable head, the SAME compaction fires and drops the
// un-cacheable middle. The offensive half thus turns on the defensive half for callers that never
// set cache_control themselves.
func TestPlacementEnablesCompaction(t *testing.T) {
	long := strings.Repeat("x", 400) // ~100 tokens at 4 chars/token
	var msgs strings.Builder
	for i := 0; i < 8; i++ {
		if i > 0 {
			msgs.WriteByte(',')
		}
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs.WriteString(`{"role":"` + role + `","content":"` + long + `"}`)
	}
	raw := []byte(`{"model":"m","max_tokens":100,"system":[{"type":"text","text":"stable head"}],` +
		`"messages":[` + msgs.String() + `]}`)

	const budget = 120
	if _, oc0 := CompactAnthropicHistoryWithOutcome(raw, budget); oc0.Reason != CompactReasonNoBreakpoint {
		t.Fatalf("pre-placement compaction reason = %q, want no_breakpoint", oc0.Reason)
	}

	placed, pl := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if pl.Reason != BreakpointReasonNone || pl.Target != "system" {
		t.Fatalf("placement reason=%q target=%q, want none/system", pl.Reason, pl.Target)
	}

	out, oc1 := CompactAnthropicHistoryWithOutcome(placed, budget)
	if oc1.Reason != CompactReasonNone {
		t.Fatalf("post-placement compaction reason = %q, want it to FIRE (none)", oc1.Reason)
	}
	if oc1.Dropped == 0 {
		t.Fatal("post-placement compaction fired but dropped nothing")
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("compacted body failed to re-decode: %v", err)
	}
}
