package agent

import (
	"bytes"
	"encoding/json"
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

func TestUpgradeStableCacheTTL1hOnSystemBreakpoint(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"stable head","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"recent","cache_control":{"type":"ephemeral"}}]}]}`)
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonNone || oc.Target != "system" {
		t.Fatalf("outcome = %+v, want system upgrade", oc)
	}
	if !bytes.Contains(out, []byte(`"cache_control":{"type":"ephemeral","ttl":"1h"}`)) {
		t.Fatalf("system cache_control was not upgraded to 1h:\n%s", out)
	}
	if !bytes.Contains(out, []byte(`"text":"recent","cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("message-tail breakpoint must stay 5m:\n%s", out)
	}
	cc := bytes.Index(raw, []byte(`"cache_control"`))
	if cc < 0 {
		t.Fatal("fixture sanity: missing cache_control")
	}
	if !bytes.Equal(raw[:cc], out[:cc]) {
		t.Fatalf("bytes before cache_control object changed:\nraw=%s\nout=%s", raw[:cc], out[:cc])
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("upgraded body failed to re-decode: %v", err)
	}
}

func TestUpgradeStableCacheTTL1hOnToolsBreakpoint(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"tools":[{"name":"search","description":"stable","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonNone || oc.Target != "tools" {
		t.Fatalf("outcome = %+v, want tools upgrade", oc)
	}
	if !bytes.Contains(out, []byte(`"cache_control":{"type":"ephemeral","ttl":"1h"}`)) {
		t.Fatalf("tools cache_control was not upgraded to 1h:\n%s", out)
	}
	cc := bytes.Index(raw, []byte(`"cache_control"`))
	if cc < 0 {
		t.Fatal("fixture sanity: missing cache_control")
	}
	if !bytes.Equal(raw[:cc], out[:cc]) {
		t.Fatalf("bytes before cache_control object changed:\nraw=%s\nout=%s", raw[:cc], out[:cc])
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("upgraded body failed to re-decode: %v", err)
	}
}

func TestUpgradeStableCacheTTL1hIgnoresMessageOnlyBreakpoint(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"stable but unmarked"}],` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"recent","cache_control":{"type":"ephemeral"}}]}]}`)
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonNoStableBreakpoint {
		t.Fatalf("reason=%q, want no stable breakpoint", oc.Reason)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("message-only breakpoint must be left unchanged")
	}
}

func TestUpgradeStableCacheTTL1hRespectsExistingTTL(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"stable","cache_control":{"type":"ephemeral","ttl":"1h"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonAlready1h || oc.Target != "system" {
		t.Fatalf("outcome=%+v, want already_1h on system", oc)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("already-1h breakpoint must be identity")
	}
}

func TestUpgradeStableCacheTTL1hRefusesVolatileHead(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"trace 550e8400-e29b-41d4-a716-446655440000"},` +
		`{"type":"text","text":"stable","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonVolatileHead || oc.Target != "system" {
		t.Fatalf("outcome=%+v, want volatile system refusal", oc)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("volatile head must be identity")
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

func TestM2StarAnchorHoistsVolatileSystemBlock(t *testing.T) {
	rawA := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"trace 11111111-2222-3333-4444-555555555555"},{"type":"text","text":"stable policy"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	rawB := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"trace aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},{"type":"text","text":"stable policy"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)

	outA, ocA := PlaceAnthropicCacheBreakpointWithOutcome(rawA)
	if ocA.Reason != BreakpointReasonNone || ocA.Target != "system" || !ocA.Rewritten {
		t.Fatalf("A outcome = %+v, want rewritten system placement", ocA)
	}
	if ocA.MovedVolatile != 1 || ocA.PredictedUplift <= 0 {
		t.Fatalf("A recommendation = %+v, want one moved volatile block with positive uplift", ocA)
	}
	outB, ocB := PlaceAnthropicCacheBreakpointWithOutcome(rawB)
	if ocB.Reason != BreakpointReasonNone || ocB.Target != "system" || !ocB.Rewritten {
		t.Fatalf("B outcome = %+v, want rewritten system placement", ocB)
	}
	if !bytes.Contains(outA, []byte(`"text":"stable policy","cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("stable block did not receive the breakpoint after hoist:\n%s", outA)
	}
	if bytes.Contains(outA, []byte(`555555555555","cache_control"`)) {
		t.Fatalf("volatile UUID block was incorrectly cached:\n%s", outA)
	}
	if !bytes.Equal(systemCachePrefix(t, outA), systemCachePrefix(t, outB)) {
		t.Fatalf("M2 hoist did not stabilize the cache prefix:\nA=%s\nB=%s", systemCachePrefix(t, outA), systemCachePrefix(t, outB))
	}
	if _, err := DecodeAnthropicMessagesRequest(outA); err != nil {
		t.Fatalf("rewritten body A failed to re-decode: %v", err)
	}
	if _, err := DecodeAnthropicMessagesRequest(outB); err != nil {
		t.Fatalf("rewritten body B failed to re-decode: %v", err)
	}
}

func TestM2StarAnchorPlacesBeforeVolatileSystemTail(t *testing.T) {
	rawA := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"stable policy"},{"type":"text","text":"trace 11111111-2222-3333-4444-555555555555"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	rawB := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"stable policy"},{"type":"text","text":"trace aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)

	outA, ocA := PlaceAnthropicCacheBreakpointWithOutcome(rawA)
	if ocA.Reason != BreakpointReasonNone || ocA.Target != "system" || ocA.Rewritten {
		t.Fatalf("A outcome = %+v, want non-rewrite system placement", ocA)
	}
	if ocA.MovedVolatile != 1 || ocA.PredictedUplift != 0 {
		t.Fatalf("A recommendation = %+v, want volatile already at tail with zero uplift", ocA)
	}
	outB, ocB := PlaceAnthropicCacheBreakpointWithOutcome(rawB)
	if ocB.Reason != BreakpointReasonNone || ocB.Target != "system" || ocB.Rewritten {
		t.Fatalf("B outcome = %+v, want non-rewrite system placement", ocB)
	}
	if !bytes.Contains(outA, []byte(`"text":"stable policy","cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("stable block did not receive the breakpoint before volatile tail:\n%s", outA)
	}
	if bytes.Contains(outA, []byte(`555555555555","cache_control"`)) {
		t.Fatalf("volatile UUID tail was incorrectly cached:\n%s", outA)
	}
	if !bytes.Equal(systemCachePrefix(t, outA), systemCachePrefix(t, outB)) {
		t.Fatalf("tail-volatile anchor did not stabilize the cache prefix:\nA=%s\nB=%s", systemCachePrefix(t, outA), systemCachePrefix(t, outB))
	}
}

// TestVolatileSystemStepsDownToTools is the core #806-bullet-2 case: the maximal head (tools+system)
// is NOT byte-stable because the system block carries a per-request UUID, so anchoring there would
// pay the cache-write premium for a prefix doomed to miss. The placer steps DOWN to caching just the
// stable tools head, leaving the volatile system untouched.
func TestVolatileSystemStepsDownToTools(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"session 550e8400-e29b-41d4-a716-446655440000"}],` +
		`"tools":[{"name":"a","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone || oc.Target != "tools" {
		t.Fatalf("got reason=%q target=%q, want none/tools (step down from volatile system)", oc.Reason, oc.Target)
	}
	// The breakpoint lands on the stable tools head, NOT on the volatile system block.
	if !bytes.Contains(out, []byte(`"input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("breakpoint not on the stable tools head:\n%s", out)
	}
	if bytes.Contains(out, []byte(`446655440000","cache_control"`)) {
		t.Fatalf("breakpoint wrongly anchored on the volatile system block:\n%s", out)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("placed body failed to re-decode: %v", err)
	}
}

// TestVolatileTimestampSystemStepsDown: a sub-day ISO timestamp in the system head is volatile (it
// changes faster than the 5-minute ephemeral TTL), so the same step-down to tools applies.
func TestVolatileTimestampSystemStepsDown(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"as of 2026-06-26T14:23:01Z, you are helpful"}],` +
		`"tools":[{"name":"a","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	_, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone || oc.Target != "tools" {
		t.Fatalf("got reason=%q target=%q, want none/tools", oc.Reason, oc.Target)
	}
}

// TestVolatileHeadNoStableFallbackBails: the only head is a volatile system block and there is no
// tools head to step down to, so there is no byte-stable span to anchor — leave the body unchanged.
func TestVolatileHeadNoStableFallbackBails(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"req 550e8400-e29b-41d4-a716-446655440000"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonVolatileHead {
		t.Fatalf("reason = %q, want volatile_head", oc.Reason)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("body must be unchanged when the only head span is volatile")
	}
}

// TestVolatileToolsHeadBails: a per-request nonce inside a tool description makes the tools head
// volatile; with no system head to fall back to, there is nothing byte-stable to cache.
func TestVolatileToolsHeadBails(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"tools":[{"name":"a","description":"trace 550e8400-e29b-41d4-a716-446655440000","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonVolatileHead {
		t.Fatalf("reason = %q, want volatile_head", oc.Reason)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("body must be unchanged when the tools head is volatile and there is no fallback")
	}
}

// TestDateOnlyHeadStillCaches is the false-positive guard: a date-ONLY token (the common "Today's
// date is ..." system shape) is byte-stable across a session's turns within the cache TTL, so it must
// NOT be flagged volatile — the maximal tools+system head is still cached on the last system block.
func TestDateOnlyHeadStillCaches(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"Today's date is 2026-06-26. You are helpful."}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	_, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone || oc.Target != "system" {
		t.Fatalf("got reason=%q target=%q, want none/system (date-only is stable, must still cache)", oc.Reason, oc.Target)
	}
}

func systemCachePrefix(t *testing.T, raw []byte) []byte {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal rewritten body: %v", err)
	}
	elems, spans, ok := decodeArrayElements(raw, obj["system"])
	if !ok {
		t.Fatal("decode system elements failed")
	}
	for i, el := range elems {
		if bytes.Contains(el, []byte("cache_control")) {
			return raw[:spans[i].end]
		}
	}
	t.Fatal("no system cache_control block found")
	return nil
}

// TestHeadValueIsVolatileBoundaries nails the detector's stable/volatile boundary directly: only an
// adjacent ISO date-TIME or a UUID is volatile; a date-only token, a bare time, and plain prose are
// stable (a false positive only skips a cache, a false negative caches a busting span).
func TestHeadValueIsVolatileBoundaries(t *testing.T) {
	volatile := []string{
		`2026-06-26T14:23:01Z`,                               // ISO datetime, T separator
		`2026-06-26 14:23`,                                   // ISO datetime, space separator
		`550e8400-e29b-41d4-a716-446655440000`,               // UUID
		`prefix 11111111-2222-3333-4444-555555555555 suffix`, // embedded UUID
	}
	stable := []string{
		`2026-06-26`,                    // date only — stable within the day
		`meeting at 14:30`,              // bare time, no date
		`you are a helpful assistant`,   // plain prose
		``,                              // empty
		`version 1.2.3-4567 of the cli`, // not a UUID shape
		`on 2026-06-26 the release ...`, // date not adjacent to a time
	}
	for _, s := range volatile {
		if !headValueIsVolatile(json.RawMessage(s)) {
			t.Errorf("headValueIsVolatile(%q) = false, want true", s)
		}
	}
	for _, s := range stable {
		if headValueIsVolatile(json.RawMessage(s)) {
			t.Errorf("headValueIsVolatile(%q) = true, want false", s)
		}
	}
}
