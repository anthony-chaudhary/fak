package agent

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
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

// TestUpgradeStableCacheTTL1hUpgradesEveryHeadBreakpoint pins the #5363 ordering fix:
// Claude Code 2.1.x marks BOTH the last tool and multiple system blocks, and Anthropic
// rejects a 1h breakpoint that comes after a 5m one in tools→system→messages order (the
// witnessed per-turn HTTP 400 "system.2.cache_control.ttl: a ttl='1h' cache_control block
// must not come after a ttl='5m' cache_control block"). Every head breakpoint must be
// upgraded together; the message-tail breakpoint stays 5m (it follows the 1h head, which
// is the legal descending order).
func TestUpgradeStableCacheTTL1hUpgradesEveryHeadBreakpoint(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"tools":[{"name":"a","input_schema":{"type":"object"}},{"name":"b","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],` +
		`"system":[{"type":"text","text":"identity","cache_control":{"type":"ephemeral"}},` +
		`{"type":"text","text":"project context","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"recent","cache_control":{"type":"ephemeral"}}]}]}`)
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonNone || oc.Target != "system" {
		t.Fatalf("outcome = %+v, want a full head upgrade labeled system", oc)
	}
	if n := bytes.Count(out, []byte(`"cache_control":{"type":"ephemeral","ttl":"1h"}`)); n != 3 {
		t.Fatalf("want ALL 3 head breakpoints (1 tools + 2 system) on the 1h tier, got %d:\n%s", n, out)
	}
	if !bytes.Contains(out, []byte(`"text":"recent","cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("message-tail breakpoint must stay 5m:\n%s", out)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("upgraded body failed to re-decode: %v", err)
	}

	// Idempotence: a second pass over the fully-upgraded body is already_1h identity.
	out2, oc2 := UpgradeAnthropicStableCacheTTL1h(out)
	if oc2.Reason != TTLUpgradeReasonAlready1h || !bytes.Equal(out2, out) {
		t.Fatalf("second pass = %+v, want already_1h identity", oc2)
	}
}

// TestUpgradeStableCacheTTL1hRefusesMixedExplicitTTL: an explicit caller ttl on ANY head
// breakpoint refuses the WHOLE upgrade — a partial edit around it could invert the
// provider's required non-increasing TTL order.
func TestUpgradeStableCacheTTL1hRefusesMixedExplicitTTL(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"tools":[{"name":"b","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral","ttl":"5m"}}],` +
		`"system":[{"type":"text","text":"stable","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonTTLAlreadySet {
		t.Fatalf("outcome = %+v, want ttl_already_set refusal", oc)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("a mixed explicit-ttl head must be identity")
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

// TestVolatileHeadClassify wires the NAMED classifier (#3341) end-to-end from the same head
// bytes the bool check scans: a head with a UUID, a JWT, and a SHA-256 hash yields the
// per-class diagnosis and one operator warning naming those classes. A stable head yields
// an empty warning.
func TestVolatileHeadClassify(t *testing.T) {
	head := json.RawMessage(`{"system":"agent session=550e8400-e29b-41d4-a716-446655440000 ` +
		`token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTYifQ.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9 ` +
		`build=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}`)
	rep := ClassifyVolatileHead(head)
	if rep.Total() != 3 {
		t.Fatalf("Total() = %d, want 3 (counts=%v)", rep.Total(), rep.Counts)
	}
	const wantWarn = "cache prefix unstable: uuid=1, jwt=1, hex_hash=1 — move dynamic values out of the system prompt"
	if got := rep.Warning(); got != wantWarn {
		t.Fatalf("Warning() =\n  %q\nwant\n  %q", got, wantWarn)
	}
	if w := ClassifyVolatileHead(json.RawMessage(`{"system":"today is 2026-06-26, be helpful"}`)).Warning(); w != "" {
		t.Fatalf("stable head Warning() = %q, want empty", w)
	}
}

// TestRespectsUnicodeEscapedBreakpoint is the #3774 regression: a client whose cache_control key
// is spelled with a JSON \uXXXX escape decodes to the SAME cache_control key. The byte-literal
// already-set scan missed it, so placement used to splice a SECOND breakpoint over the client's
// chosen layout. The guard now detects the escaped form and returns the body unchanged
// (already_set), exactly as it does for a literal key. Keys use a raw string so `_` reaches
// the fixture as the 6-byte JSON escape (backslash-u-0-0-5-f), not a decoded underscore.
func TestRespectsUnicodeEscapedBreakpoint(t *testing.T) {
	cases := []struct {
		name string
		raw  string
	}{
		{
			// underscore escaped: cache_control (split so the backslash-u stays literal).
			name: "underscore_escaped",
			raw: `{"model":"m","max_tokens":1,` +
				`"system":[{"type":"text","text":"stable head","cache\` + `u005fcontrol":{"type":"ephemeral"}}],` +
				`"messages":[{"role":"user","content":"x"}]}`,
		},
		{
			// leading letter escaped: cache_control (the c as c).
			name: "letter_escaped",
			raw: `{"model":"m","max_tokens":1,` +
				`"system":[{"type":"text","text":"stable head","\` + `u0063ache_control":{"type":"ephemeral"}}],` +
				`"messages":[{"role":"user","content":"x"}]}`,
		},
		{
			// escaped key on a hoist-shaped body (volatile block ahead of a stable one): the guard
			// must fire BEFORE the star-anchor hoist, so no breakpoint is moved or added.
			name: "escaped_on_hoist_shape",
			raw: `{"model":"m","max_tokens":1,` +
				`"system":[{"type":"text","text":"run 123e4567-e89b-12d3-a456-426614174000"},` +
				`{"type":"text","text":"a","cache\` + `u005fcontrol":{"type":"ephemeral"}},` +
				`{"type":"text","text":"b"}],"messages":[{"role":"user","content":"x"}]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(tc.raw)
			// Fixture sanity: the literal is genuinely NOT present, but the key decodes to it.
			if bytes.Contains(raw, []byte("cache_control")) {
				t.Fatalf("fixture sanity: raw carries the literal, not an escaped key:\n%s", raw)
			}
			var probe map[string]json.RawMessage
			if err := json.Unmarshal(raw, &probe); err != nil {
				t.Fatalf("fixture sanity: body is not valid JSON: %v", err)
			}
			out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
			if oc.Reason != BreakpointReasonAlreadySet {
				t.Fatalf("reason = %q, want already_set (escaped key must be respected)\nout: %s", oc.Reason, out)
			}
			if oc.Rewritten {
				t.Fatalf("guard fired but body was rewritten — a breakpoint was moved:\n%s", out)
			}
			if !bytes.Equal(out, raw) {
				t.Fatalf("escaped-key body must be returned byte-for-byte unchanged:\nraw: %s\nout: %s", raw, out)
			}
		})
	}
}

// TestBodyHasCacheControlKey pins the detector directly: the literal and every escaped spelling
// that decodes to cache_control are caught, while a body with an unrelated \u escape (or none) is
// not a false positive.
func TestBodyHasCacheControlKey(t *testing.T) {
	hits := []string{
		`{"a":{"cache_control":{"type":"ephemeral"}}}`,                     // literal
		`{"a":{"cache\` + `u005fcontrol":{"type":"ephemeral"}}}`,           // underscore escaped
		`{"a":{"\` + `u0063ache_control":{"type":"ephemeral"}}}`,           // leading letter escaped
		`{"a":{"\` + `u0063ache\` + `u005fcontrol":{"type":"ephemeral"}}}`, // both escaped
	}
	for _, s := range hits {
		if !bodyHasCacheControlKey([]byte(s)) {
			t.Fatalf("bodyHasCacheControlKey(%s) = false, want true", s)
		}
	}
	misses := []string{
		`{"system":[{"type":"text","text":"plain"}]}`,                    // no cache_control at all
		`{"system":[{"type":"text","text":"\` + `u00e9\` + `u00e8 x"}]}`, // \u escapes, but not the key
		`{"note":"we mention cache control with a space"}`,               // words, not the key
	}
	for _, s := range misses {
		if bodyHasCacheControlKey([]byte(s)) {
			t.Fatalf("bodyHasCacheControlKey(%s) = true, want false (false positive)", s)
		}
	}
}

func TestUpgradeAnthropicStableCacheTTL1hExtendsOrderedMessagePrefixes(t *testing.T) {
	raw := []byte(`{"model":"claude-sonnet-4","system":[{"type":"text","text":"stable head","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"first stable turn","cache_control":{"type":"ephemeral"}}]},{"role":"assistant","content":[{"type":"text","text":"answer"}]},{"role":"user","content":[{"type":"text","text":"second stable turn","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64}`)

	out, oc := UpgradeAnthropicStableCacheTTL1hWithMessagePrefixes(raw)
	if oc.Reason != TTLUpgradeReasonNone || oc.Target != "messages" {
		t.Fatalf("outcome = %+v, want message-prefix upgrade", oc)
	}
	if oc.UpgradedHeadBreakpoints != 1 || oc.UpgradedMessageBreakpoints != 2 {
		t.Fatalf("upgrade split = head %d message %d, want 1/2", oc.UpgradedHeadBreakpoints, oc.UpgradedMessageBreakpoints)
	}
	if got := bytes.Count(out, []byte(`"ttl":"1h"`)); got != 3 {
		t.Fatalf("1h breakpoint count = %d, want 3:\n%s", got, out)
	}
	// The rewrite must preserve the caller's bytes apart from the three ttl fields.
	stripped := bytes.ReplaceAll(out, []byte(`,"ttl":"1h"`), nil)
	if !bytes.Equal(stripped, raw) {
		t.Fatalf("rewrite changed non-ttl bytes:\n got %s\nwant %s", stripped, raw)
	}
}

func TestUpgradeAnthropicStableCacheTTL1hRefusesVolatileMessagePrefix(t *testing.T) {
	raw := []byte(`{"model":"claude-sonnet-4","system":[{"type":"text","text":"stable head","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"request 550e8400-e29b-41d4-a716-446655440000","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64}`)

	out, oc := UpgradeAnthropicStableCacheTTL1hWithMessagePrefixes(raw)
	if oc.Reason != TTLUpgradeReasonVolatileMessage || oc.Target != "messages" {
		t.Fatalf("outcome = %+v, want volatile-message refusal", oc)
	}
	if !bytes.Equal(out, raw) {
		t.Fatalf("refusal changed bytes:\n got %s\nwant %s", out, raw)
	}
}

func TestUpgradeAnthropicStableCacheTTL1hRefusesMixedTTLOrdering(t *testing.T) {
	// A caller-selected 5m head may not be followed by a fak-authored 1h message
	// breakpoint: Anthropic requires longer TTLs to precede shorter TTLs.
	raw := []byte(`{"model":"claude-sonnet-4","system":[{"type":"text","text":"stable head","cache_control":{"type":"ephemeral","ttl":"5m"}}],"messages":[{"role":"user","content":[{"type":"text","text":"stable turn","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64}`)
	out, oc := UpgradeAnthropicStableCacheTTL1hWithMessagePrefixes(raw)
	if oc.Reason != TTLUpgradeReasonTTLAlreadySet {
		t.Fatalf("outcome = %+v, want ttl_already_set", oc)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("ordering refusal changed bytes")
	}
}

func TestUpgradeAnthropicStableCacheTTL1hHeadOnlyIsDistinctAblation(t *testing.T) {
	raw := []byte(`{"model":"claude-sonnet-4","system":[{"type":"text","text":"head","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"history","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64}`)
	out, oc := UpgradeAnthropicStableCacheTTL1hHeadOnly(raw)
	if oc.Reason != TTLUpgradeReasonNone || oc.UpgradedHeadBreakpoints != 1 || oc.UpgradedMessageBreakpoints != 0 {
		t.Fatalf("head-only outcome = %+v", oc)
	}
	if got := bytes.Count(out, []byte(`"ttl":"1h"`)); got != 1 {
		t.Fatalf("head-only 1h count = %d, want 1: %s", got, out)
	}
	if !bytes.Contains(out, []byte(`"text":"history","cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("head-only arm changed message breakpoint: %s", out)
	}
}

// TestCachePrefixTransformsByteSafetyProperty drives the star-anchor and 1h
// upgrade together over a large deterministic corpus. Removing only the two
// documented insertions must recover the input byte-for-byte; this catches
// re-marshalling, whitespace normalization, key reordering, and escaped-string
// corruption anywhere in the request, not merely in the cached head.
func TestCachePrefixTransformsByteSafetyProperty(t *testing.T) {
	for i := 0; i < 1024; i++ {
		seed := sha256.Sum256([]byte(fmt.Sprintf("cache-prefix-shape-%d", i)))
		assertCachePrefixTransformsByteSafe(t, seed[:])
	}
}

// FuzzCachePrefixTransformsByteSafety keeps an explicit seed corpus of the
// message/tool/system shapes that have historically been fragile. Go's fuzzing
// engine then mutates the strings and shape selector while the property checks
// the real byte transforms rather than a decoded surrogate.
func FuzzCachePrefixTransformsByteSafety(f *testing.F) {
	for _, seed := range [][]byte{
		{0, 0},
		{1, '"', '\\', 0, 0xff},
		{2, '<', 't', 'o', 'o', 'l', '_', 'u', 's', 'e', '>'},
		{3, '\n', '\r', '\t'},
		{4, 0xe2, 0x80, 0xa8, 0xf0, 0x9f, 0x92, 0xa5},
		{5, '{', '}', '[', ']', ',', ':'},
		{6, 'c', 'a', 'c', 'h', 'e', '_', 'c', 'o', 'n', 't', 'r', 'o', 'l'},
		{7, 0, 1, 2, 3, 4, 5},
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, seed []byte) {
		assertCachePrefixTransformsByteSafe(t, seed)
	})
}

func assertCachePrefixTransformsByteSafe(t testing.TB, seed []byte) {
	t.Helper()
	raw := generatedStableAnthropicShape(seed)

	anchored, anchorOutcome := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if anchorOutcome.Reason != BreakpointReasonNone {
		t.Fatalf("star-anchor refused generated stable shape: reason=%q body=%s", anchorOutcome.Reason, raw)
	}
	const breakpoint = `,"cache_control":{"type":"ephemeral"}`
	if got := bytes.Count(anchored, []byte(breakpoint)); got != 1 {
		t.Fatalf("star-anchor insertion count=%d, want 1: %s", got, anchored)
	}
	if recovered := bytes.Replace(anchored, []byte(breakpoint), nil, 1); !bytes.Equal(recovered, raw) {
		t.Fatalf("star-anchor changed bytes outside its breakpoint\nraw: %s\nout: %s", raw, anchored)
	}

	upgraded, upgradeOutcome := UpgradeAnthropicStableCacheTTL1h(anchored)
	if upgradeOutcome.Reason != TTLUpgradeReasonNone {
		t.Fatalf("1h upgrade refused anchored stable shape: reason=%q body=%s", upgradeOutcome.Reason, anchored)
	}
	const ttl = `,"ttl":"1h"`
	if got := bytes.Count(upgraded, []byte(ttl)); got != 1 {
		t.Fatalf("1h insertion count=%d, want 1: %s", got, upgraded)
	}
	withoutTTL := bytes.Replace(upgraded, []byte(ttl), nil, 1)
	if !bytes.Equal(withoutTTL, anchored) {
		t.Fatalf("1h upgrade changed bytes outside ttl insertion\nanchored: %s\nupgraded: %s", anchored, upgraded)
	}
	if recovered := bytes.Replace(withoutTTL, []byte(breakpoint), nil, 1); !bytes.Equal(recovered, raw) {
		t.Fatalf("combined transforms changed cached-prefix bytes\nraw: %s\nout: %s", raw, upgraded)
	}

	anchoredAgain, secondAnchor := PlaceAnthropicCacheBreakpointWithOutcome(upgraded)
	if secondAnchor.Reason != BreakpointReasonAlreadySet || !bytes.Equal(anchoredAgain, upgraded) {
		t.Fatalf("star-anchor is not idempotent: reason=%q\nfirst: %s\nsecond: %s", secondAnchor.Reason, upgraded, anchoredAgain)
	}
	upgradedAgain, secondUpgrade := UpgradeAnthropicStableCacheTTL1h(upgraded)
	if secondUpgrade.Reason != TTLUpgradeReasonAlready1h || !bytes.Equal(upgradedAgain, upgraded) {
		t.Fatalf("1h upgrade is not idempotent: reason=%q\nfirst: %s\nsecond: %s", secondUpgrade.Reason, upgraded, upgradedAgain)
	}
}

func generatedStableAnthropicShape(seed []byte) []byte {
	if len(seed) == 0 {
		seed = []byte{0}
	}
	if len(seed) > 256 {
		seed = seed[:256]
	}
	// Hex deliberately cannot accidentally synthesize the UUID/timestamp/session
	// markers that define a volatile head. JSON quoting still exercises varied
	// lengths, empty values, nested objects, and field/whitespace layouts.
	text, _ := json.Marshal(hex.EncodeToString(seed))
	alt, _ := json.Marshal(hex.EncodeToString(append([]byte{0xa5}, seed...)))
	messages := fmt.Sprintf(`[{"role":"user","content":[{"type":"text","text":%s},{"type":"tool_result","tool_use_id":%s,"content":%s}]},{"role":"assistant","content":[{"type":"tool_use","id":%s,"name":"lookup","input":{"q":%s}}]}]`, text, alt, text, alt, text)

	switch seed[0] % 4 {
	case 0:
		return []byte(fmt.Sprintf("{\n  \"model\":\"m\", \"system\":[{\"type\":\"text\",\"text\":%s},{\"text\":%s,\"type\":\"text\"}],\n  \"messages\":%s, \"max_tokens\":64\n}", text, alt, messages))
	case 1:
		return []byte(fmt.Sprintf(`{"model":"m","tools":[{"name":"lookup","description":%s,"input_schema":{"properties":{"q":{"type":"string"}},"type":"object"}}],"messages":%s,"max_tokens":64}`, text, messages))
	case 2:
		return []byte(fmt.Sprintf(`{"messages":%s,"tools":[{"input_schema":{"type":"object"},"name":"lookup"}],"system":[{"text":%s,"type":"text"}],"model":"m","max_tokens":64}`, messages, text))
	default:
		return []byte(fmt.Sprintf("{ \"model\" : \"m\" , \"system\" : [ { \"type\" : \"text\" , \"meta\" : {\"nested\":[1,true,null]} , \"text\" : %s } ] , \"messages\" : %s , \"max_tokens\" : 64 }", alt, messages))
	}
}

func TestCachePrefixTransformsVolatileHeadsAreIdentity(t *testing.T) {
	volatile := []string{
		`request 550e8400-e29b-41d4-a716-446655440000`,
		`generated at 2026-08-08T12:34:56Z`,
	}
	for _, head := range volatile {
		headJSON, _ := json.Marshal(head)
		raw := []byte(fmt.Sprintf(`{"model":"m","system":[{"type":"text","text":%s}],"messages":[{"role":"user","content":"hi"}]}`, headJSON))
		anchored, anchorOutcome := PlaceAnthropicCacheBreakpointWithOutcome(raw)
		if anchorOutcome.Reason != BreakpointReasonVolatileHead || !bytes.Equal(anchored, raw) {
			t.Errorf("star-anchor volatile identity failed for %q: reason=%q out=%s", head, anchorOutcome.Reason, anchored)
		}

		withBreakpoint := bytes.Replace(raw, []byte(`}`), []byte(`,"cache_control":{"type":"ephemeral"}}`), 1)
		upgraded, upgradeOutcome := UpgradeAnthropicStableCacheTTL1h(withBreakpoint)
		if upgradeOutcome.Reason != TTLUpgradeReasonVolatileHead || !bytes.Equal(upgraded, withBreakpoint) {
			t.Errorf("1h volatile identity failed for %q: reason=%q out=%s", head, upgradeOutcome.Reason, upgraded)
		}
	}
}
