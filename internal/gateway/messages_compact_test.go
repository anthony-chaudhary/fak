package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compactcohere"
)

// #555 req.Raw step: the gateway compacts the OUTBOUND Anthropic passthrough body to a
// resident-token budget while keeping the cached prefix byte-identical. These tests
// exercise the GATING (the cache-safety of the rewrite itself is proven in
// internal/agent/anthropic_compact_test.go):
//   - OFF (budget 0) is identity.
//   - non-passthrough wire is identity even with a budget (the body is rebuilt downstream).
//   - ON + Anthropic passthrough compacts an oversized body but keeps the prefix verbatim.

// compactWireBody is a realistic /v1/messages body: a system array with a trailing
// cache_control breakpoint, plus nMsgs alternating turns whose 1st carries a per-message
// breakpoint — enough that a tight budget forces compaction.
func compactWireBody(t *testing.T, nMsgs int) []byte {
	t.Helper()
	type block map[string]any
	msgs := make([]map[string]any, 0, nMsgs)
	msgs = append(msgs, map[string]any{
		"role": "user",
		"content": []block{
			{"type": "text", "text": strings.Repeat("cached early context. ", 20), "cache_control": map[string]any{"type": "ephemeral"}},
		},
	})
	for i := 1; i < nMsgs; i++ {
		role := "user"
		if i%2 == 0 {
			role = "assistant"
		}
		msgs = append(msgs, map[string]any{
			"role":    role,
			"content": []block{{"type": "text", "text": strings.Repeat("conversation turn body. ", 15)}},
		})
	}
	raw, err := json.Marshal(map[string]any{
		"model": "claude-sonnet-4-6", "max_tokens": 1024, "stream": true,
		"system": []block{
			{"type": "text", "text": strings.Repeat("policy. ", 30), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

func anthropicPassthroughServer(budget int) *Server {
	return &Server{
		planner:              &agent.HTTPPlanner{Provider: agent.ProviderAnthropic},
		compactHistoryBudget: budget,
		logf:                 func(string, ...any) {},
	}
}

func TestMaybeUpgradeCacheTTL1HGate(t *testing.T) {
	raw := []byte(`{"model":"claude","max_tokens":1024,` +
		`"system":[{"type":"text","text":"stable policy","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	reqOff, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode off: %v", err)
	}
	if anthropicPassthroughServer(1200).maybeUpgradeAnthropicCacheTTL1H(reqOff) {
		t.Fatal("TTL upgrade must be gated off by default")
	}
	if !bytes.Equal(reqOff.Raw, raw) {
		t.Fatal("gated-off TTL upgrade must leave req.Raw unchanged")
	}

	reqOn, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode on: %v", err)
	}
	s := anthropicPassthroughServer(1200)
	s.cacheTTL1H = true
	if !s.maybeUpgradeAnthropicCacheTTL1H(reqOn) {
		t.Fatal("TTL upgrade gate should fire on a stable system breakpoint")
	}
	if !bytes.Contains(reqOn.Raw, []byte(`"cache_control":{"type":"ephemeral","ttl":"1h"}`)) {
		t.Fatalf("stable system breakpoint was not upgraded to 1h:\n%s", reqOn.Raw)
	}
	cc := bytes.Index(raw, []byte(`"cache_control"`))
	if cc < 0 {
		t.Fatal("fixture sanity: missing cache_control")
	}
	if !bytes.Equal(raw[:cc], reqOn.Raw[:cc]) {
		t.Fatalf("bytes before cache_control changed:\nraw=%s\nout=%s", raw[:cc], reqOn.Raw[:cc])
	}
}

// TestPrepareServedRequestUnionsExtendedCacheTTLBeta is the regression witness for the
// subscription-OAuth instant-400 (managed cache — ACTIVE, forced by --managed-cache on): when
// the managed-cache 1h TTL upgrade fires, the served request MUST union the extended-cache-ttl
// beta into the forwarded anthropic-beta set. The upgrade sets cache_control ttl:"1h", which
// Anthropic accepts only with that beta negotiated; the wrapped claude CLI defaults to the 5m
// tier and never sends it, so a body with ttl:"1h" but no beta is 400'd upstream as malformed.
// The bug was that only deferColdTools unioned its beta (toolSearchBeta) — the TTL upgrade had
// no analogous union, so the forced 1h posture shipped a malformed body.
func TestPrepareServedRequestUnionsExtendedCacheTTLBeta(t *testing.T) {
	raw := []byte(`{"model":"claude","max_tokens":1024,` +
		`"system":[{"type":"text","text":"stable policy","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	newReq := func(t *testing.T) *agent.AnthropicMessagesRequest {
		t.Helper()
		req, err := agent.DecodeAnthropicMessagesRequest(raw)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		return req
	}
	// The inbound betas the wrapped claude CLI negotiates — NOT the extended-cache-ttl one.
	const inboundBeta = "claude-code-20250219,fine-grained-tool-streaming-2025-05-14"
	inboundReq := func() *http.Request {
		r := httptest.NewRequest("POST", "/v1/messages", nil)
		r.Header.Set("anthropic-beta", inboundBeta)
		return r
	}

	// (a) managed cache ACTIVE → the upgrade fires; the extended-cache-ttl beta must be unioned
	// in AND the inbound betas must survive (union, not replace).
	t.Run("upgrade_fires_unions_beta", func(t *testing.T) {
		s := anthropicPassthroughServer(1200)
		s.cacheTTL1H = true
		s.metrics = newGatewayMetrics(time.Now())

		req := newReq(t)
		prep := s.prepareServedAnthropicRequest(context.Background(), inboundReq(), req, "", servedSessionTurn{})

		if !bytes.Contains(req.Raw, []byte(`"ttl":"1h"`)) {
			t.Fatalf("fixture sanity: TTL upgrade did not fire on a stable breakpoint:\n%s", req.Raw)
		}
		if !strings.Contains(prep.upstreamBeta, extendedCacheTTLBeta) {
			t.Fatalf("upstreamBeta = %q, want it to contain the extended-cache-ttl beta %q (ttl:1h without it is 400'd upstream)", prep.upstreamBeta, extendedCacheTTLBeta)
		}
		for _, want := range []string{"claude-code-20250219", "fine-grained-tool-streaming-2025-05-14"} {
			if !strings.Contains(prep.upstreamBeta, want) {
				t.Errorf("upstreamBeta = %q dropped inbound beta %q — must UNION, not replace", prep.upstreamBeta, want)
			}
		}
	})

	// (b) managed cache OFF → no upgrade, so the extended-cache-ttl beta must NOT be added (no
	// ttl:1h in the body means the beta would be gratuitous).
	t.Run("no_upgrade_no_beta", func(t *testing.T) {
		s := anthropicPassthroughServer(1200) // cacheTTL1H stays false
		s.metrics = newGatewayMetrics(time.Now())

		req := newReq(t)
		prep := s.prepareServedAnthropicRequest(context.Background(), inboundReq(), req, "", servedSessionTurn{})

		if bytes.Contains(req.Raw, []byte(`"ttl":"1h"`)) {
			t.Fatalf("fixture sanity: TTL upgrade must be gated off when cacheTTL1H is false:\n%s", req.Raw)
		}
		if strings.Contains(prep.upstreamBeta, extendedCacheTTLBeta) {
			t.Fatalf("upstreamBeta = %q must NOT carry the extended-cache-ttl beta when no upgrade fired", prep.upstreamBeta)
		}
	})
}

// TestMaybeUpgradeCacheTTL1HPlacesThenUpgrades (#2175): a caller that sends ZERO cache_control
// used to hit no_stable_breakpoint forever — upgrade only edits an EXISTING breakpoint, and the
// sibling that places one (agent.PlaceAnthropicCacheBreakpointWithOutcome) only ran behind the
// compaction gate (compactHistoryBudget>0), not the managed-cache posture. With --managed-cache
// ACTIVE and compaction OFF, the flagship lever must still place-then-upgrade as one transform.
func TestMaybeUpgradeCacheTTL1HPlacesThenUpgrades(t *testing.T) {
	raw := []byte(`{"model":"claude","max_tokens":1024,` +
		`"system":[{"type":"text","text":"stable policy, no caching hint at all"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	s := anthropicPassthroughServer(0) // compaction OFF: placement must not depend on it
	s.cacheTTL1H = true
	s.metrics = newGatewayMetrics(time.Now())

	if !s.maybeUpgradeAnthropicCacheTTL1H(req) {
		t.Fatal("managed-cache ACTIVE must place-then-upgrade a zero-cache_control stable head")
	}
	if !bytes.Contains(req.Raw, []byte(`"cache_control":{"type":"ephemeral","ttl":"1h"}`)) {
		t.Fatalf("system head was not placed+upgraded to 1h:\n%s", req.Raw)
	}
	if n := s.metrics.ttlUpgrades[agent.TTLUpgradeReasonNoStableBreakpoint]; n != 0 {
		t.Fatalf("no_stable_breakpoint must stay flat on the composed path, got %d", n)
	}
	if n := s.metrics.ttlUpgrades[cacheTTLUpgradePlacedAndUpgraded]; n != 1 {
		t.Fatalf("placed_and_upgraded must count the composed attempt, got %d", n)
	}
	if n := s.metrics.ttlUpgrades["upgraded"]; n != 0 {
		t.Fatalf("the composed row must not double-count into upgrade-only, got %d", n)
	}
}

// TestMaybeUpgradeCacheTTL1HVolatileHeadStaysByteSafe (#2185 criterion 2): the composed
// place-then-upgrade path (#2175) must stay byte-safe / identity-on-ambiguity — a caller that
// sends ZERO cache_control AND whose only cacheable head is VOLATILE (a per-request nonce in the
// tools block sits ahead of every system anchor, so no rewrite can make the prefix byte-stable)
// gets NO breakpoint placed: placement refuses with volatile_head, the outbound body is returned
// byte-for-byte unchanged, and the composed upgrade does NOT fire. The refusal is WITNESSED (the
// placement counter records volatile_head; the no_stable_breakpoint bail stays visible) so an
// ACTIVE-but-ineligible session is legible on /metrics instead of silently mutating the wire. This
// is the refute guard for the composed path: it proves the transform, not a default that always
// places. The success path is pinned by TestMaybeUpgradeCacheTTL1HPlacesThenUpgrades.
func TestMaybeUpgradeCacheTTL1HVolatileHeadStaysByteSafe(t *testing.T) {
	raw := []byte(`{"model":"claude","max_tokens":1024,` +
		`"tools":[{"name":"run","description":"per-request nonce 11111111-2222-3333-4444-555555555555","input_schema":{"type":"object"}}],` +
		`"system":[{"type":"text","text":"stable policy, no caching hint at all"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	s := anthropicPassthroughServer(0) // compaction OFF: the composed path must stand on its own
	s.cacheTTL1H = true                // --managed-cache ACTIVE
	s.metrics = newGatewayMetrics(time.Now())

	if s.maybeUpgradeAnthropicCacheTTL1H(req) {
		t.Fatal("a volatile-only head must NOT upgrade: no byte-stable prefix to secure")
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("volatile-head body must be forwarded byte-for-byte unchanged:\n%s", req.Raw)
	}
	if n := s.metrics.placementAttempts[agent.BreakpointReasonVolatileHead]; n != 1 {
		t.Fatalf("the composed placement must witness the volatile_head refusal, got %d", n)
	}
	if n := s.metrics.ttlUpgrades[cacheTTLUpgradePlacedAndUpgraded]; n != 0 {
		t.Fatalf("placed_and_upgraded must stay flat when placement refuses, got %d", n)
	}
	if n := s.metrics.ttlUpgrades[agent.TTLUpgradeReasonNoStableBreakpoint]; n != 1 {
		t.Fatalf("no_stable_breakpoint bail must stay witnessed on the refused path, got %d", n)
	}
}

// TestMaybeCompactOffIsIdentity: budget 0 forwards the body byte-for-byte unchanged.
func TestMaybeCompactOffIsIdentity(t *testing.T) {
	raw := compactWireBody(t, 16)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	anthropicPassthroughServer(0).maybeCompactAnthropicRaw(req)
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("budget 0 must leave req.Raw unchanged")
	}
}

// TestMaybeCompactNonPassthroughIsIdentity: a budget set but the upstream is NOT the
// Anthropic API (mock planner) → identity, because the body is rebuilt from req.Messages
// downstream and touching req.Raw would be pointless (and unsafe to claim cache-preserving).
func TestMaybeCompactNonPassthroughIsIdentity(t *testing.T) {
	raw := compactWireBody(t, 16)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	s := &Server{
		planner:              agent.NewMockPlanner("m"),
		compactHistoryBudget: 50,
		logf:                 func(string, ...any) {},
	}
	if s.anthropicPassthrough() {
		t.Fatal("mock planner must NOT be an anthropic passthrough")
	}
	s.maybeCompactAnthropicRaw(req)
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("non-passthrough wire must leave req.Raw unchanged")
	}
}

// TestMaybeCompactOnShortensKeepsPrefix: ON + Anthropic passthrough + an oversized history
// → the forwarded body is shorter, still decodes, and its cache prefix is byte-identical.
func TestMaybeCompactOnShortensKeepsPrefix(t *testing.T) {
	raw := compactWireBody(t, 20)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)

	// The prefix boundary: end of the last message bearing a cache_control breakpoint.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(orig, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	elems, spans, ok := decodeArrayElementsFromTest(t, orig, obj["messages"])
	if !ok {
		t.Fatal("decodeArrayElements failed")
	}
	split := spans[lastBreakpointMessageFromTest(elems)].end

	anthropicPassthroughServer(120).maybeCompactAnthropicRaw(req)

	if bytes.Equal(req.Raw, orig) {
		t.Fatalf("expected compaction with a 20-message body at budget=120, got identity")
	}
	if len(req.Raw) >= len(orig) {
		t.Fatalf("expected a shorter body, got %d >= %d", len(req.Raw), len(orig))
	}
	if split > len(req.Raw) || !bytes.Equal(orig[:split], req.Raw[:split]) {
		t.Fatalf("cache prefix bytes changed (split=%d)", split)
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
		t.Fatalf("compacted body failed to re-decode: %v", err)
	}
}

// sprawlWireBody is compactWireBody with each turn padded so the compactible suffix
// deterministically EXCEEDS a target resident-token budget — the "sprawl" the default-on
// trigger is meant to catch. tokensPerTurn is the ~4-chars/token estimate the compactor
// uses, so nMsgs*tokensPerTurn clears the budget with margin.
func sprawlWireBody(t *testing.T, nMsgs, charsPerTurn int) []byte {
	t.Helper()
	type block map[string]any
	msgs := make([]map[string]any, 0, nMsgs)
	msgs = append(msgs, map[string]any{
		"role": "user",
		"content": []block{
			{"type": "text", "text": strings.Repeat("cached early context. ", 20), "cache_control": map[string]any{"type": "ephemeral"}},
		},
	})
	body := strings.Repeat("x", charsPerTurn)
	for i := 1; i < nMsgs; i++ {
		role := "user"
		if i%2 == 0 {
			role = "assistant"
		}
		msgs = append(msgs, map[string]any{
			"role":    role,
			"content": []block{{"type": "text", "text": body}},
		})
	}
	raw, err := json.Marshal(map[string]any{
		"model": "claude-sonnet-4-6", "max_tokens": 1024, "stream": true,
		"system": []block{
			{"type": "text", "text": strings.Repeat("policy. ", 30), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// TestMaybeCompactDefaultBudgetTrigger is the default-on sprawl trigger: a server built at
// DefaultCompactHistoryBudget (what the CLI flag now defaults to) compacts a conversation
// whose suffix has sprawled past that budget, and keeps the cache_control prefix byte-
// identical. This is the live realization of "limit sprawl without net-charging more" — the
// cut only sheds the un-cacheable middle, never the cached prefix.
func TestMaybeCompactDefaultBudgetTrigger(t *testing.T) {
	// ~12 turns of ~24k chars each ≈ 6k tokens/turn ≈ 72k token suffix, well over the 48k
	// default — so the cut MUST fire at the default budget.
	raw := sprawlWireBody(t, 12, 24000)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(orig, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	elems, spans, ok := decodeArrayElementsFromTest(t, orig, obj["messages"])
	if !ok {
		t.Fatal("decodeArrayElements failed")
	}
	split := spans[lastBreakpointMessageFromTest(elems)].end

	// Built at the DEFAULT budget — no explicit operator value, exactly the CLI default path.
	anthropicPassthroughServer(DefaultCompactHistoryBudget).maybeCompactAnthropicRaw(req)

	if bytes.Equal(req.Raw, orig) {
		t.Fatalf("a body sprawled past the default budget must compact, got identity")
	}
	if len(req.Raw) >= len(orig) {
		t.Fatalf("expected a shorter body, got %d >= %d", len(req.Raw), len(orig))
	}
	if split > len(req.Raw) || !bytes.Equal(orig[:split], req.Raw[:split]) {
		t.Fatalf("cache prefix bytes changed (split=%d)", split)
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
		t.Fatalf("compacted body failed to re-decode: %v", err)
	}
}

// TestMaybeCompactDefaultBudgetLeavesShortSessionAlone: a short conversation whose suffix
// is well under the default budget is forwarded byte-for-byte even at the default-on budget
// — the trigger only fires on genuine sprawl, so a typical session is untouched.
func TestMaybeCompactDefaultBudgetLeavesShortSessionAlone(t *testing.T) {
	raw := compactWireBody(t, 8) // ~8 small turns, far under 48k tokens
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	anthropicPassthroughServer(DefaultCompactHistoryBudget).maybeCompactAnthropicRaw(req)
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("a short session under the default budget must be left byte-for-byte unchanged")
	}
}

func TestMaybeCompactAppliesM2SystemAnchorRewrite(t *testing.T) {
	rawA := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"trace 11111111-2222-3333-4444-555555555555"},{"type":"text","text":"stable policy"}],` +
		`"messages":[{"role":"user","content":"one"},{"role":"assistant","content":"two"},{"role":"user","content":"three"}]}`)
	rawB := []byte(`{"model":"m","max_tokens":1,` +
		`"system":[{"type":"text","text":"trace aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},{"type":"text","text":"stable policy"}],` +
		`"messages":[{"role":"user","content":"one"},{"role":"assistant","content":"two"},{"role":"user","content":"three"}]}`)
	reqA, err := agent.DecodeAnthropicMessagesRequest(rawA)
	if err != nil {
		t.Fatalf("decode A: %v", err)
	}
	reqB, err := agent.DecodeAnthropicMessagesRequest(rawB)
	if err != nil {
		t.Fatalf("decode B: %v", err)
	}

	anthropicPassthroughServer(DefaultCompactHistoryBudget).maybeCompactAnthropicRaw(reqA)
	anthropicPassthroughServer(DefaultCompactHistoryBudget).maybeCompactAnthropicRaw(reqB)

	if bytes.Equal(reqA.Raw, rawA) {
		t.Fatal("gateway preflight left volatile-before-stable system anchor unchanged")
	}
	if !bytes.Contains(reqA.Raw, []byte(`"text":"stable policy","cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("gateway preflight did not place the breakpoint on the stable system block:\n%s", reqA.Raw)
	}
	if bytes.Contains(reqA.Raw, []byte(`555555555555","cache_control"`)) {
		t.Fatalf("gateway preflight cached the volatile UUID block:\n%s", reqA.Raw)
	}
	if !bytes.Equal(systemCachePrefixFromTest(t, reqA.Raw), systemCachePrefixFromTest(t, reqB.Raw)) {
		t.Fatalf("gateway M2 rewrite did not make the forwarded cache prefix stable:\nA=%s\nB=%s", systemCachePrefixFromTest(t, reqA.Raw), systemCachePrefixFromTest(t, reqB.Raw))
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(reqA.Raw); err != nil {
		t.Fatalf("rewritten gateway body A failed to decode: %v", err)
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(reqB.Raw); err != nil {
		t.Fatalf("rewritten gateway body B failed to decode: %v", err)
	}
}

// The two helpers below let the gateway test reach the agent package's unexported span
// locators indirectly: we re-derive the boundary with the same public primitive the
// gateway relies on (DecodeAnthropicMessagesRequest round-trips), then compute the split
// by parsing here. They keep the test self-contained without exporting agent internals.
func decodeArrayElementsFromTest(t *testing.T, raw []byte, msgs json.RawMessage) ([]json.RawMessage, []elementSpanT, bool) {
	t.Helper()
	base := bytes.Index(raw, msgs)
	if base < 0 {
		return nil, nil, false
	}
	dec := json.NewDecoder(bytes.NewReader(msgs))
	if tok, err := dec.Token(); err != nil {
		return nil, nil, false
	} else if d, ok := tok.(json.Delim); !ok || d != '[' {
		return nil, nil, false
	}
	var elems []json.RawMessage
	var spans []elementSpanT
	for dec.More() {
		start := int(dec.InputOffset())
		for start < len(msgs) && (msgs[start] == ' ' || msgs[start] == ',' || msgs[start] == '\n' || msgs[start] == '\t' || msgs[start] == '\r') {
			start++
		}
		var el json.RawMessage
		if err := dec.Decode(&el); err != nil {
			return nil, nil, false
		}
		elems = append(elems, el)
		spans = append(spans, elementSpanT{start: base + start, end: base + int(dec.InputOffset())})
	}
	return elems, spans, true
}

type elementSpanT struct{ start, end int }

func lastBreakpointMessageFromTest(elems []json.RawMessage) int {
	last := -1
	for i, el := range elems {
		var m struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(el, &m) != nil {
			continue
		}
		var blocks []map[string]json.RawMessage
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		for _, b := range blocks {
			if _, ok := b["cache_control"]; ok {
				last = i
			}
		}
	}
	return last
}

func systemCachePrefixFromTest(t *testing.T, raw []byte) []byte {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal rewritten body: %v", err)
	}
	elems, spans, ok := decodeArrayElementsFromTest(t, raw, obj["system"])
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

// TestSpliceMaxTokensPreservesPrefix is the F13 regression: capping max_tokens must NOT
// re-marshal the whole body (which would sort the top-level keys and bust the cached prefix
// on a paced turn). The splice replaces only the integer and leaves every other byte — and so
// the cache_control prefix — byte-identical.
func TestSpliceMaxTokensPreservesPrefix(t *testing.T) {
	raw := []byte(`{"model":"claude","max_tokens":1024,"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]}`)
	out, ok := spliceMaxTokens(raw, 64)
	if !ok {
		t.Fatalf("spliceMaxTokens returned ok=false on a valid body")
	}
	if !bytes.Contains(out, []byte(`"max_tokens":64`)) {
		t.Fatalf("max_tokens not capped to 64: %s", out)
	}
	// Everything BEFORE max_tokens (the model key) and the whole system/messages tail must be
	// byte-identical — only the number changed, nothing reordered.
	if !bytes.HasPrefix(out, []byte(`{"model":"claude","max_tokens":`)) {
		t.Fatalf("top-level key order changed (cache prefix would bust): %s", out)
	}
	if !bytes.Contains(out, []byte(`"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}]`)) {
		t.Fatalf("system/cache_control bytes changed: %s", out)
	}
	// A body with no max_tokens, or a non-integer value, leaves the body untouched (ok=false).
	if _, ok := spliceMaxTokens([]byte(`{"model":"c","messages":[]}`), 64); ok {
		t.Fatalf("expected ok=false when max_tokens is absent")
	}
}

// headOrderedWireBody models real Claude Code traffic for the head anchor (#1407/#1408): a
// stable `system` cache_control breakpoint serialized BEFORE messages[] (struct field order is
// JSON key order), with the ONLY messages[] breakpoint on a RECENT turn (recentBpBack from the
// end). This is the dormant #1407 shape: the default first-breakpoint anchor protects almost the
// whole conversation, so compaction cannot fire no matter the budget. Mirrors headOrderedBody in
// internal/agent/anthropic_compact_test.go (gateway can't reach that unexported test helper).
func headOrderedWireBody(t *testing.T, nMsgs, recentBpBack int) []byte {
	t.Helper()
	type block map[string]any
	recentBpIdx := nMsgs - 1 - recentBpBack
	msgs := make([]map[string]any, 0, nMsgs)
	for i := 0; i < nMsgs; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		blk := block{"type": "text", "text": strings.Repeat("conversation turn body words. ", 12)}
		if i == recentBpIdx {
			blk["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		msgs = append(msgs, map[string]any{"role": role, "content": []block{blk}})
	}
	ordered := struct {
		Model     string           `json:"model"`
		MaxTokens int              `json:"max_tokens"`
		System    []block          `json:"system"`
		Messages  []map[string]any `json:"messages"`
	}{
		Model:     "claude-sonnet-4-6",
		MaxTokens: 1024,
		System: []block{
			{"type": "text", "text": "You are a coding agent."},
			{"type": "text", "text": strings.Repeat("policy text. ", 40), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		Messages: msgs,
	}
	raw, err := json.Marshal(ordered)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// TestFakCompactionShedNetSavingOnClaudeCodePath is the economic + attribution witness for the
// ONE cache-value mechanism that applies to the flagship `fak guard -- claude` path: history
// compaction-shed. It is the compaction analog of TestFakPlacementUnlocksProviderCacheSavings
// (docs/notes/FAK-OFFENSIVE-CACHE-PLACEMENT-SAVINGS-WITNESS-2026-07-01.md), which prices the
// OTHER mechanism (breakpoint placement) — but placement is identity on Claude Code (which marks
// its own head), so that witness claims none of the flagship-path value. This one does.
//
// The gap it closes (epic #1844 / #1407): the unit tests above prove head-anchored compaction
// FIRES on the real Claude-Code-shaped body (head-marked system + a recent message breakpoint) via
// the observed-cold long-session path. But none priced the net fak-authored saving or attributed
// it, so docs/notes/FAK-GUARD-CACHE-VALUE-SHARE-2026-07-01.md could only record fak_share = 0 on
// the guard path (anchor-starvation, no captured fire). This witness prices a real fire end-to-end
// through the LIVE gateway metrics and the SHIPPED CachePricing model, and proves the counterfactual
// (firstBP default on the identical body sheds nothing → $0 fak saving). Deterministic: no key, no
// network, no GPU.
//
// PROVENANCE (per docs/standards/net-true-value.md):
//   - The shed is WITNESSED: it comes from the real Server.compactAnthropicRawWithReason transform
//     on the real body, read back off the live gatewayMetrics adjudication summary (the same counter
//     /metrics and the guard exit line render), never a hand-computed number.
//   - The counterfactual is WITNESSED: the identical body under the firstBP default (compactAnchorHead
//     off) is forwarded unchanged → CompactionShedTokens == 0 → $0 fak saving.
//   - The attribution is STRUCTURAL: compaction-shed is fak-authored — the provider never sheds the
//     conversation middle, it re-bills it every turn. MechanismSavings splits it under owner=fak.
//   - The magnitude is MODELED: the shipped gateway.CachePricing model at the Opus-4.8 example base
//     input rate ($5/MTok), the same rate and model the placement witness uses, so the two artifacts
//     are directly comparable.
func TestFakCompactionShedNetSavingOnClaudeCodePath(t *testing.T) {
	// A real Claude-Code-shaped body: a system head that marks its OWN cache_control breakpoint
	// (the stable cached head the provider reuses) plus a RECENT message breakpoint. On the
	// warm-cache-safe firstBP default this anchors near the end and the lever stays dormant
	// (anchor-starved, #1407); only head re-anchoring makes the sprawled middle compactible.
	body := headOrderedWireBody(t, 120, 2)

	// --- Live fire on the flagship path (head-anchored, observed-cold long session) ---
	fireSrv := anthropicPassthroughServer(1200)
	fireSrv.compactAnchorHead = true
	fireSrv.metrics = newGatewayMetrics(time.Now())
	// This trace last served two provider-cache TTLs ago: its message-span suffix is provably
	// expired, so the head-anchored burst carries zero marginal penalty and fires horizon-free —
	// the un-budgeted plain-`fak guard -- claude` long-session case (#1407's cold path).
	now := time.Now()
	fireSrv.metrics.observeHarnessCoherence("long-session", now.Add(-2*compactcohere.DefaultProviderCacheTTL), "", false, "", false, false, 0, 0, 0)

	reqFire, err := agent.DecodeAnthropicMessagesRequest(body)
	if err != nil {
		t.Fatalf("decode fire body: %v", err)
	}
	origLen := len(reqFire.Raw)
	fired, reason := fireSrv.compactAnthropicRawWithReason(reqFire, 0, "long-session")
	if !fired || reason != "" {
		t.Fatalf("head-anchored compaction must fire on the observed-cold Claude Code body, got fired=%v reason=%q", fired, reason)
	}
	if len(reqFire.Raw) >= origLen {
		t.Fatalf("a fire must shrink the forwarded body, got %d (in %d)", len(reqFire.Raw), origLen)
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(reqFire.Raw); err != nil {
		t.Fatalf("compacted body failed to re-decode: %v", err)
	}

	// Read the WITNESSED shed off the live summary — the exact counter /metrics and the guard exit
	// line render — then split it by owner and price it with the shipped model.
	sum := fireSrv.metrics.adjudicationSummary()
	if sum.CompactionShedTokens == 0 {
		t.Fatal("a fired compaction must record a nonzero WITNESSED shed on the live summary")
	}
	mech := sum.MechanismSavings()
	if mech.FakCompactionShedTokens != sum.CompactionShedTokens {
		t.Fatalf("compaction shed must attribute to owner=fak: got %d, summary %d", mech.FakCompactionShedTokens, sum.CompactionShedTokens)
	}
	pricing := CachePricing{InputPerMTokUSD: ClaudeOpus48InputPerMTokUSD, OutputPerMTokUSD: ClaudeOpus48OutputPerMTokUSD}
	// The shed is input tokens fak did NOT forward: each would otherwise re-bill at the full input
	// price (the conversation middle is not cache_control-marked, so the provider never caches it —
	// it is uncached remainder every turn). The saving is thus shed × base input price, 100% fak.
	fakTokenEquiv := mech.FakTokenEquiv()
	fakSavingUSD := fakTokenEquiv * perToken(pricing.InputPerMTokUSD)
	if fakSavingUSD <= 0 {
		t.Fatalf("fak-authored compaction saving must be positive, got $%.6f", fakSavingUSD)
	}

	// --- Counterfactual: the SAME body on the firstBP default sheds nothing → $0 fak saving ---
	baseSrv := anthropicPassthroughServer(1200) // compactAnchorHead stays false (the default)
	baseSrv.metrics = newGatewayMetrics(time.Now())
	baseSrv.metrics.observeHarnessCoherence("long-session", now.Add(-2*compactcohere.DefaultProviderCacheTTL), "", false, "", false, false, 0, 0, 0)
	reqBase, err := agent.DecodeAnthropicMessagesRequest(body)
	if err != nil {
		t.Fatalf("decode base body: %v", err)
	}
	origBase := append([]byte(nil), reqBase.Raw...)
	baseFired, _ := baseSrv.compactAnthropicRawWithReason(reqBase, 0, "long-session")
	if baseFired {
		t.Fatal("the firstBP default must stay dormant (anchor-starved) on this Claude Code body — the #1407 baseline")
	}
	if !bytes.Equal(reqBase.Raw, origBase) {
		t.Fatal("a dormant firstBP default must forward the body byte-for-byte")
	}
	baseMech := baseSrv.metrics.adjudicationSummary().MechanismSavings()
	baseSavingUSD := baseMech.FakTokenEquiv() * perToken(pricing.InputPerMTokUSD)
	if baseSavingUSD != 0 {
		t.Fatalf("the firstBP-default counterfactual must save $0 of fak-authored value, got $%.6f", baseSavingUSD)
	}

	// The witness block (printed under -v), mirroring the placement note so the two mechanisms'
	// artifacts read the same way.
	t.Logf("FAK-AUTHORED COMPACTION-SHED SAVINGS WITNESS (#1407 head-anchored, flagship `fak guard -- claude` path)")
	t.Logf("  scope         : real Claude-Code-shaped body (system head marks its OWN cache_control + a recent message breakpoint); 120-turn session; observed-cold long-session trace; Opus-4.8 example input rate {$%.0f/MTok}", pricing.InputPerMTokUSD)
	t.Logf("  fire          : WITNESSED — real Server.compactAnthropicRawWithReason head-anchored fire, forwarded body %d→%d bytes, re-decodes clean, cached prefix byte-identical", origLen, len(reqFire.Raw))
	t.Logf("  counterfactual: WITNESSED — same body on the firstBP default (compactAnchorHead off) stays dormant (anchor-starved, #1407) → forwarded byte-for-byte → $0 fak saving")
	t.Logf("  attribution   : STRUCTURAL — compaction-shed is owner=fak (the provider re-bills the un-cached conversation middle every turn; only fak sheds it)")
	t.Logf("  magnitude     : MODELED   — shipped gateway.CachePricing at the base input rate, the same model+rate the offensive-placement witness uses")
	t.Logf("  ---- economics ----")
	t.Logf("  fak shed          : %d tok (WITNESSED, owner=fak, off the live adjudication summary)", sum.CompactionShedTokens)
	t.Logf("  fak saving        : $%.6f  (head-anchored long session)", fakSavingUSD)
	t.Logf("  firstBP baseline  : $%.6f  (the #1407 anchor-starved dormancy this fixes)", baseSavingUSD)
	t.Logf("  net fak-authored  : $%.6f  = 100%% fak-attributed (the provider never sheds the middle)", fakSavingUSD-baseSavingUSD)
}

// TestMaybeCompactAnchorHeadDormantWithoutTurnsLeft: --compact-anchor-head is opt-in but the
// request boundary carries no session-turns horizon (turnsLeft=0, e.g. DecideSession unwired) —
// the #1407/#1408 burst-economics gate stays conservative and does NOT fire, same as the default
// anchor. Proves the flag alone does not start bursting caches; it needs a real horizon too.
func TestMaybeCompactAnchorHeadDormantWithoutTurnsLeft(t *testing.T) {
	raw := headOrderedWireBody(t, 120, 2)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)

	s := anthropicPassthroughServer(1200)
	s.compactAnchorHead = true
	fired, reason := s.compactAnthropicRawWithReason(req, 0, "")
	if fired {
		t.Fatalf("head anchor with no turns-left horizon must NOT fire, got fired=true reason=%q", reason)
	}
	if reason != agent.CompactReasonBurstUnprofitable {
		t.Fatalf("reason=%q, want %q (bail must name the economics gate, not just stay silent)", reason, agent.CompactReasonBurstUnprofitable)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("a burst_unprofitable bail must leave req.Raw byte-identical")
	}
}

func TestMaybeCompactAnchorHeadBreakEvenKeepsWarmSpanWhenUnprofitable(t *testing.T) {
	raw := headOrderedWireBody(t, 120, 2)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)

	s := anthropicPassthroughServer(2400)
	s.compactAnchorHead = true
	fired, reason := s.compactAnthropicRawWithReason(req, 1, "")
	if fired {
		t.Fatalf("head anchor with a too-short break-even horizon must NOT fire, got fired=true reason=%q", reason)
	}
	if reason != agent.CompactReasonBurstUnprofitable {
		t.Fatalf("reason=%q, want %q", reason, agent.CompactReasonBurstUnprofitable)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatal("unprofitable cache-burst gate must retain the warm span byte-for-byte")
	}
}

// TestMaybeCompactAnchorHeadFiresWithTurnsLeft is the #1407/#1408 end-to-end witness on the LIVE
// gateway wiring: the default first-breakpoint anchor stays dormant on the real-traffic shape (a
// recent-only message breakpoint), but --compact-anchor-head + a wired DecideSession session with
// turns left to repay the one-time burst actually FIRES — sheds the middle and keeps the stable
// system head byte-identical, so the dominant cache read survives.
func TestMaybeCompactAnchorHeadFiresWithTurnsLeft(t *testing.T) {
	raw := headOrderedWireBody(t, 120, 2)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(orig, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	_, spans, ok := decodeArrayElementsFromTest(t, orig, obj["messages"])
	if !ok {
		t.Fatal("decodeArrayElements failed")
	}
	headEnd := spans[0].start // verbatim head region a head-anchored fire must preserve

	// 1. Confirm the default anchor is genuinely dormant on this shape first (the #1407 bug).
	def := anthropicPassthroughServer(1200)
	if fired, reason := def.compactAnthropicRawWithReason(req, 1000, ""); fired || reason != agent.CompactReasonUnderBudget {
		t.Fatalf("default anchor must stay dormant (under_budget) even with turnsLeft supplied (it only gates head mode); got fired=%v reason=%q", fired, reason)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("default-anchor dormant bail must leave req.Raw unchanged")
	}

	// 2. Head anchor + a generous turns-left horizon (the wired DecideSession case): FIRES.
	head := anthropicPassthroughServer(1200)
	head.compactAnchorHead = true
	fired, reason := head.compactAnthropicRawWithReason(req, 1000, "")
	if !fired || reason != "" {
		t.Fatalf("head anchor with a paying turns-left horizon must FIRE, got fired=%v reason=%q", fired, reason)
	}
	if bytes.Equal(req.Raw, orig) || len(req.Raw) >= len(orig) {
		t.Fatalf("a head-anchored fire must shrink the body, got %d (in %d)", len(req.Raw), len(orig))
	}
	if headEnd > len(req.Raw) || !bytes.Equal(orig[:headEnd], req.Raw[:headEnd]) {
		t.Fatalf("head-anchored fire changed the stable head bytes [0,%d) — would burst the dominant cache", headEnd)
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
		t.Fatalf("head-anchored body failed to decode: %v", err)
	}
}

// TestMaybeCompactAnchorHeadFiresOnObservedColdTrace is the un-budgeted long-session witness —
// the first value throughpath for a PLAIN `fak guard -- claude` run with no DecideSession turn
// budget wired. A trace that idled past the message-breakpoint cache TTL since its last served
// turn (the harness-coherence per-trace wall clock — OBSERVED, never guessed) has a provably
// expired message-span suffix, so the head-anchored burst carries zero marginal penalty and the
// economics gate fires horizon-free. A warm trace on the same server stays conservatively idle.
func TestMaybeCompactAnchorHeadFiresOnObservedColdTrace(t *testing.T) {
	s := anthropicPassthroughServer(1200)
	s.compactAnchorHead = true
	s.metrics = newGatewayMetrics(time.Now())

	// Prime the per-trace wall clocks: trace-cold last served TWO TTLs ago (provably expired
	// message spans); trace-warm just now (its cached suffix may still be warm).
	now := time.Now()
	s.metrics.observeHarnessCoherence("trace-cold", now.Add(-2*compactcohere.DefaultProviderCacheTTL), "", false, "", false, false, 0, 0, 0)
	s.metrics.observeHarnessCoherence("trace-warm", now, "", false, "", false, false, 0, 0, 0)

	// Warm trace, no horizon: the gate must stay conservative (identity, burst_unprofitable).
	reqWarm, err := agent.DecodeAnthropicMessagesRequest(headOrderedWireBody(t, 120, 2))
	if err != nil {
		t.Fatalf("decode warm: %v", err)
	}
	origWarm := append([]byte(nil), reqWarm.Raw...)
	if fired, reason := s.compactAnthropicRawWithReason(reqWarm, 0, "trace-warm"); fired || reason != agent.CompactReasonBurstUnprofitable {
		t.Fatalf("warm trace with no horizon must bail burst_unprofitable, got fired=%v reason=%q", fired, reason)
	}
	if !bytes.Equal(reqWarm.Raw, origWarm) {
		t.Fatal("warm-trace bail must leave req.Raw byte-identical")
	}

	// Cold trace, same server, still no horizon: FIRES.
	reqCold, err := agent.DecodeAnthropicMessagesRequest(headOrderedWireBody(t, 120, 2))
	if err != nil {
		t.Fatalf("decode cold: %v", err)
	}
	origCold := append([]byte(nil), reqCold.Raw...)
	fired, reason := s.compactAnthropicRawWithReason(reqCold, 0, "trace-cold")
	if !fired || reason != "" {
		t.Fatalf("provably-cold trace must fire horizon-free, got fired=%v reason=%q", fired, reason)
	}
	if bytes.Equal(reqCold.Raw, origCold) || len(reqCold.Raw) >= len(origCold) {
		t.Fatalf("a cold fire must shrink the body, got %d (in %d)", len(reqCold.Raw), len(origCold))
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(reqCold.Raw); err != nil {
		t.Fatalf("cold-fired body failed to decode: %v", err)
	}

	// An unknown trace (first turn — no lastTurn clock yet) is never cold.
	reqNew, err := agent.DecodeAnthropicMessagesRequest(headOrderedWireBody(t, 120, 2))
	if err != nil {
		t.Fatalf("decode new: %v", err)
	}
	if fired, reason := s.compactAnthropicRawWithReason(reqNew, 0, "trace-never-seen"); fired || reason != agent.CompactReasonBurstUnprofitable {
		t.Fatalf("an unseen trace must never read as cold, got fired=%v reason=%q", fired, reason)
	}
}

// TestMaybeCompactAnchorHeadFiresOnAssumedSessionPrior is the head of the goal: a WARM, un-budgeted
// session (no wired turn horizon, not idle past the TTL) that today refuses now FIRES early because
// the assumed-session-length prior presumes it will run long. This is the path that raises fak's own
// cache-value share on continuously-active long sessions.
func TestMaybeCompactAnchorHeadFiresOnAssumedSessionPrior(t *testing.T) {
	s := anthropicPassthroughServer(1200)
	s.compactAnchorHead = true
	// Pin the presumed horizon at a fixed 100: this test exercises the MECHANISM (a fresh trace,
	// maximally early, fires) at a known horizon, decoupled from the tuned DefaultAssumedSessionTurns
	// so a later recalibration of that default cannot silently flip this fire/no-fire assertion.
	s.assumeSessionTurns = 100
	s.metrics = newGatewayMetrics(time.Now())
	// A fresh trace (served-turn depth 0 ⇒ CurrentTurn 1): maximally early in a presumed-100-turn
	// session, so the burst has ~99 repaying turns ahead.
	req, err := agent.DecodeAnthropicMessagesRequest(headOrderedWireBody(t, 120, 2))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	fired, reason := s.compactAnthropicRawWithReason(req, 0, "trace-fresh")
	if !fired || reason != "" {
		t.Fatalf("warm un-budgeted session must fire on the assumed-session prior, got fired=%v reason=%q", fired, reason)
	}
	if bytes.Equal(req.Raw, orig) || len(req.Raw) >= len(orig) {
		t.Fatalf("a prior-driven fire must shrink the body, got %d (in %d)", len(req.Raw), len(orig))
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
		t.Fatalf("prior-fired body failed to decode: %v", err)
	}
}

// TestMaybeCompactAnchorHeadPriorRefusesLateInAssumedSession proves the prior is not a blunt "always
// fire": deep in a presumed-100-turn session (few repaying turns left) the SAME break-even economics
// keep the warm cache — the burst can no longer repay before the presumed end.
func TestMaybeCompactAnchorHeadPriorRefusesLateInAssumedSession(t *testing.T) {
	s := anthropicPassthroughServer(1200)
	s.compactAnchorHead = true
	// Fixed 100-turn horizon (mechanism test, decoupled from the tuned DefaultAssumedSessionTurns):
	// the 99-fold priming below targets CurrentTurn 100 == the presumed end, where the burst cannot
	// repay. Pinning keeps the "refuses at the end" assertion meaningful under any default retune.
	s.assumeSessionTurns = 100
	s.metrics = newGatewayMetrics(time.Now())
	// Drive the trace's served-turn depth close to the presumed end (99 folds ⇒ CurrentTurn 100),
	// so remainingTurns is ~0 and a non-trivial invalidated suffix cannot repay. Prime each fold at
	// "now" (warm — never idle-cold), so the refusal comes from the horizon, not the cold path.
	now := time.Now()
	for i := 0; i < 99; i++ {
		s.metrics.observeHarnessCoherence("trace-deep", now, "", false, "", false, false, 0, 0, 0)
	}
	req, err := agent.DecodeAnthropicMessagesRequest(headOrderedWireBody(t, 120, 2))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	if fired, reason := s.compactAnthropicRawWithReason(req, 0, "trace-deep"); fired || reason != agent.CompactReasonBurstUnprofitable {
		t.Fatalf("deep in a presumed session the burst must not repay, got fired=%v reason=%q", fired, reason)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatal("a late-session bail must leave req.Raw byte-identical")
	}
}

// TestMaybeCompactAnchorHeadPriorDisabledPreservesConservativeBail is the escape-hatch / byte-identity
// lock: with the prior disabled (assumeSessionTurns 0) a warm un-budgeted session bails exactly as it
// did before the prior existed — the default-off behavior is byte-for-byte unchanged.
func TestMaybeCompactAnchorHeadPriorDisabledPreservesConservativeBail(t *testing.T) {
	s := anthropicPassthroughServer(1200)
	s.compactAnchorHead = true
	s.assumeSessionTurns = 0 // prior OFF
	s.metrics = newGatewayMetrics(time.Now())
	req, err := agent.DecodeAnthropicMessagesRequest(headOrderedWireBody(t, 120, 2))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	if fired, reason := s.compactAnthropicRawWithReason(req, 0, "trace-fresh"); fired || reason != agent.CompactReasonBurstUnprofitable {
		t.Fatalf("with the prior disabled a warm session must bail burst_unprofitable, got fired=%v reason=%q", fired, reason)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatal("prior-disabled bail must leave req.Raw byte-identical")
	}
}

// TestMaybeCompactAnchorHeadBoundedTurnsLeftBeatsPrior proves precedence: a genuine wired turn
// horizon always wins over the prior. A bounded turnsLeft fires off the real remaining-turn budget
// even when the trace's served depth is primed deep (which would refuse under the prior alone).
func TestMaybeCompactAnchorHeadBoundedTurnsLeftBeatsPrior(t *testing.T) {
	s := anthropicPassthroughServer(1200)
	s.compactAnchorHead = true
	s.assumeSessionTurns = DefaultAssumedSessionTurns
	s.metrics = newGatewayMetrics(time.Now())
	now := time.Now()
	for i := 0; i < 99; i++ { // deep enough that the prior alone would refuse
		s.metrics.observeHarnessCoherence("trace-deep", now, "", false, "", false, false, 0, 0, 0)
	}
	req, err := agent.DecodeAnthropicMessagesRequest(headOrderedWireBody(t, 120, 2))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	fired, reason := s.compactAnthropicRawWithReason(req, 1000, "trace-deep")
	if !fired || reason != "" {
		t.Fatalf("a wired bounded horizon must win over the prior, got fired=%v reason=%q", fired, reason)
	}
	if bytes.Equal(req.Raw, orig) || len(req.Raw) >= len(orig) {
		t.Fatalf("a bounded-horizon fire must shrink the body, got %d (in %d)", len(req.Raw), len(orig))
	}
}

// TestServedTurnCountIncrementsPerFold locks the per-trace depth counter the prior reads as
// CurrentTurn: it advances one per served fold, and both an unseen trace and a nil-metrics server
// report 0.
func TestServedTurnCountIncrementsPerFold(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	now := time.Now()
	const k = 7
	for i := 0; i < k; i++ {
		m.observeHarnessCoherence("t", now, "", false, "", false, false, 0, 0, 0)
	}
	if got := m.servedTurnCount("t"); got != k {
		t.Fatalf("servedTurnCount=%d, want %d", got, k)
	}
	if got := m.servedTurnCount("unseen"); got != 0 {
		t.Fatalf("unseen trace servedTurnCount=%d, want 0", got)
	}
	var nilM *gatewayMetrics
	if got := nilM.servedTurnCount("t"); got != 0 {
		t.Fatalf("nil-metrics servedTurnCount=%d, want 0", got)
	}
}

func TestManagedCacheMessagePrefixUpgradeAndCreationAttribution(t *testing.T) {
	s := anthropicPassthroughServer(0)
	s.cacheTTL1H = true
	s.metrics = newGatewayMetrics(time.Now())
	req, err := agent.DecodeAnthropicMessagesRequest([]byte(`{"model":"claude-test","system":[{"type":"text","text":"head","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"stable history","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64}`))
	if err != nil {
		t.Fatal(err)
	}
	upgraded, messagePrefix := s.maybeUpgradeAnthropicCacheTTL1HScoped(req)
	if !upgraded || !messagePrefix {
		t.Fatalf("upgrade = %v, message prefix = %v; want true/true", upgraded, messagePrefix)
	}
	s.noteCtxValueTTL1hScope("trace-message", messagePrefix)
	s.metrics.recordCacheCreationTierSplit(1200, s.ttl1hActiveFor("trace-message"), s.ttl1hMessagePrefixFor("trace-message"))

	// The original head-only arm remains separately sweepable and attributable.
	s.noteCtxValueTTL1hScope("trace-head", false)
	s.metrics.recordCacheCreationTierSplit(300, s.ttl1hActiveFor("trace-head"), s.ttl1hMessagePrefixFor("trace-head"))
	summary := s.metrics.adjudicationSummary()
	if summary.CacheCreationTokensMessagePrefix != 1200 || summary.CacheCreationTokensHeadOnly != 300 {
		t.Fatalf("creation split = message %d head %d, want 1200/300", summary.CacheCreationTokensMessagePrefix, summary.CacheCreationTokensHeadOnly)
	}
	if summary.CacheCreationTokensUpgraded != 1500 {
		t.Fatalf("upgraded total = %d, want 1500", summary.CacheCreationTokensUpgraded)
	}
}

func TestCacheCreationSpanLabel(t *testing.T) {
	for _, tc := range []struct {
		tokens  int
		up, msg bool
		want    string
	}{
		{0, true, true, "none"},
		{10, false, false, "head_5m"},
		{10, true, false, "head_1h"},
		{10, true, true, "message_prefix_1h"},
	} {
		if got := cacheCreationSpanLabel(tc.tokens, tc.up, tc.msg); got != tc.want {
			t.Errorf("label(%d,%v,%v) = %q, want %q", tc.tokens, tc.up, tc.msg, got, tc.want)
		}
	}
}

func TestManagedCacheHeadOnlyEnvironmentAblation(t *testing.T) {
	t.Setenv("FAK_ABLATE_TTL_1H_HEAD_ONLY", "1")
	s := anthropicPassthroughServer(0)
	s.cacheTTL1H = true
	req, err := agent.DecodeAnthropicMessagesRequest([]byte(`{"model":"claude-test","system":[{"type":"text","text":"head","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"history","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64}`))
	if err != nil {
		t.Fatal(err)
	}
	upgraded, messagePrefix := s.maybeUpgradeAnthropicCacheTTL1HScoped(req)
	if !upgraded || messagePrefix {
		t.Fatalf("head-only ablation = upgraded %v message %v, want true/false", upgraded, messagePrefix)
	}
	if got := bytes.Count(req.Raw, []byte(`"ttl":"1h"`)); got != 1 {
		t.Fatalf("1h count = %d, want head only: %s", got, req.Raw)
	}
}
