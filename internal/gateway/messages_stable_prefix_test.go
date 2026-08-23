package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// messages_stable_prefix_test.go — issue #1540 (cache-frontier Next-50 item 22,
// epic #1490): the single CONSOLIDATED witness that the guard/serve request-side
// transform chain does not defeat provider reuse by rewriting the stable prefix, on
// BOTH provider wires.
//
// The default/evidence target of the item is: "the gateway's safety layer does not
// defeat provider reuse by rewriting stable prefixes." The Anthropic half was already
// witnessed per-transform (messages_compact_test.go, messages_elide_test.go,
// internal/agent/anthropic_cachebp_test.go), but two gaps stood open:
//
//  1. No witness ran the transforms IN THE SERVE ORDER handleAnthropicMessages applies
//     them and asserted the cache_control-protected prefix survives the whole chain.
//  2. The OpenAI-compatible wire — the stable system+tools head an OpenAI-shaped caller
//     sends — had NO byte-identity witness at all, so a future safety-layer transform
//     could silently rewrite it and bust provider reuse with no failing test.
//
// runServeRequestChain below is the SAME chain both wires pass through, so the two
// tests prove one property on both wires: the safety layer keeps the provider-cacheable
// prefix byte-stable.
//
// Resolved open question (the issue's Assumptions named it — "does the OpenAI-compatible
// serve path run any request-side body transform, or is it a byte passthrough?"): the
// prefix-rewriting req.Raw transforms (managed-cache TTL upgrade / ctx-view plan /
// history compaction / oversized-result elision / inbound tools[] splice / inbound
// system prune) are EVERY ONE gated on s.anthropicPassthroughFor and are identity off
// the Anthropic wire — and handleChatCompletions never invokes them at all (it rebuilds
// the outbound request from the decoded ChatRequest). So the OpenAI-compatible stable
// prefix is already carried through unchanged; the leaf is the witness alone, no new
// guard is needed. The gate (anthropicPassthroughFor) IS the identity-on-OpenAI-wire
// guard, and TestServeRequestChainKeepsOpenAIStablePrefix now pins it.
//
// spliceMaxTokens (the fourth transform the item names) is deliberately NOT in the
// shared chain: it edits only the max_tokens integer — a sampling parameter OUTSIDE the
// provider-cached content, and Anthropic-wire-only (applySessionPaceToAnthropicRequest;
// handleChatCompletions caps via a WithMaxTokens option, never req.Raw). Its
// content-prefix preservation is witnessed by TestSpliceMaxTokensPreservesPrefix.

// runServeRequestChain applies the gateway's request-side req.Raw transforms in the
// exact order handleAnthropicMessages runs them (internal/gateway/messages.go), so one
// call exercises the whole serve-path chain a request body passes through before it is
// forwarded upstream. It mutates req.Raw in place, exactly as the serve path does.
func runServeRequestChain(s *Server, req *agent.AnthropicMessagesRequest) {
	s.sanitizeAnthropicToolReferences(req) // correctness, every wire
	s.maybeUpgradeAnthropicCacheTTL1H(req) // managed-cache 1h TTL
	s.maybePlanAnthropicRaw(context.Background(), "stable-prefix-witness", req)
	s.compactAnthropicRawWithReason(req, 1000, "stable-prefix-witness") // #555 history compaction
	s.maybeElideAnthropicRaw(req)                                       // oversized tool_result elision
	s.maybeCompactInboundTools(req)                                     // #555 twin: prune floor-denied tool defs
	s.logInboundSystemPrune(s.maybeCompactInboundSystem(req))           // system-block prune + its witness
}

// TestServeRequestChainKeepsAnthropicCacheControlPrefix drives a real /v1/messages body
// through the whole serve-order transform chain on an Anthropic-passthrough server and
// asserts the cache_control-protected prefix is byte-for-byte identical afterward — for
// two bodies, each of which forces a DIFFERENT transform to fire (elision, then
// compaction), so the chain demonstrably preserves the prefix regardless of which member
// of the safety layer actually rewrites the body.
func TestServeRequestChainKeepsAnthropicCacheControlPrefix(t *testing.T) {
	// (a) An oversized old tool_result → elision fires; compaction stays under-budget.
	t.Run("elision_fires", func(t *testing.T) {
		req, err := agent.DecodeAnthropicMessagesRequest(elideWireBody(t))
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		orig := append([]byte(nil), req.Raw...)
		prefixEnd := firstBreakpointMessageEnd(t, orig) // protected prefix = through message[0]'s breakpoint

		s := anthropicPassthroughServer(DefaultCompactHistoryBudget) // large budget → compaction is a no-op here
		s.elideResultBytes = 2048
		s.metrics = newGatewayMetrics(time.Now())

		runServeRequestChain(s, req)

		if bytes.Equal(req.Raw, orig) {
			t.Fatal("expected the chain to shrink the oversized tool_result, got identity")
		}
		if len(req.Raw) >= len(orig) {
			t.Fatalf("expected a shorter body, got %d >= %d", len(req.Raw), len(orig))
		}
		if prefixEnd > len(req.Raw) || !bytes.Equal(orig[:prefixEnd], req.Raw[:prefixEnd]) {
			t.Fatalf("cache_control-protected prefix bytes changed (prefixEnd=%d) — provider reuse would be lost", prefixEnd)
		}
		if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
			t.Fatalf("body failed to re-decode after the chain: %v", err)
		}
	})

	// (b) A sprawled history → compaction fires; there is no oversized result to elide.
	t.Run("compaction_fires", func(t *testing.T) {
		req, err := agent.DecodeAnthropicMessagesRequest(compactWireBody(t, 20))
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		orig := append([]byte(nil), req.Raw...)
		split := lastBreakpointMessageEnd(t, orig) // protected prefix = through the last message breakpoint

		s := anthropicPassthroughServer(120) // tight budget → compaction fires
		s.elideResultBytes = 2048            // enabled, but this body carries no tool_result to shrink
		s.metrics = newGatewayMetrics(time.Now())

		runServeRequestChain(s, req)

		if bytes.Equal(req.Raw, orig) {
			t.Fatal("expected the chain to compact a 20-message body at budget=120, got identity")
		}
		if len(req.Raw) >= len(orig) {
			t.Fatalf("expected a shorter body, got %d >= %d", len(req.Raw), len(orig))
		}
		if split > len(req.Raw) || !bytes.Equal(orig[:split], req.Raw[:split]) {
			t.Fatalf("cache_control-protected prefix bytes changed (split=%d) — provider reuse would be lost", split)
		}
		if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
			t.Fatalf("body failed to re-decode after the chain: %v", err)
		}
	})
}

// TestServeRequestChainKeepsOpenAIStablePrefix is the item's new coverage: an
// OpenAI-compatible body with a stable system+tools head, run through the SAME serve-path
// transform chain on a server whose upstream is NOT the Anthropic API (a mock planner
// stands in for an OpenAI-compatible / vLLM / SGLang backend) — with EVERY prefix-rewriting
// lever turned ON. The stable prefix must be forwarded byte-for-byte unchanged, because
// every one of those transforms is gated on s.anthropicPassthroughFor and no-ops off the
// Anthropic wire. If any future edit made one of them fire on this wire and rewrite the
// head, this witness reddens. The assertion is expressed both as raw byte-identity and, in
// provider-cache terms, via the internal/cachemeta §A3 prefix-stability check (Diverge /
// StableTokens): the whole system+tools head stays cacheable across the transform.
func TestServeRequestChainKeepsOpenAIStablePrefix(t *testing.T) {
	// A minimal but realistic OpenAI /v1/chat/completions body: a stable system message +
	// a stable tools[] head (what an OpenAI-compatible provider prompt-caches positionally),
	// then the volatile user turn.
	body := []byte(`{"model":"gpt-4o",` +
		`"messages":[` +
		`{"role":"system","content":"You are a coding agent. These are the stable standing instructions the provider caches positionally, byte-for-byte, every turn."},` +
		`{"role":"user","content":"first question of the session"}` +
		`],` +
		`"tools":[{"type":"function","function":{"name":"read_file","description":"read a file from the repo","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],` +
		`"max_tokens":1024}`)

	// A non-Anthropic upstream with EVERY prefix-rewriting lever enabled — the strongest
	// form of the guard: even fully armed, the safety layer must not touch this wire.
	s := &Server{
		planner:              agent.NewMockPlanner("gpt-4o"),
		compactHistoryBudget: 50,
		compactAnchorHead:    true,
		assumeSessionTurns:   DefaultAssumedSessionTurns,
		elideResultBytes:     512,

		toolFloorDenies: func(string) bool { return true }, // "deny every tool" — would prune the whole tools[] on the Anthropic wire
		systemBlockDrop: func(string, string) bool { return true },
		logf:            func(string, ...any) {},
		metrics:         newGatewayMetrics(time.Now()),
	}
	if s.anthropicPassthrough() {
		t.Fatal("mock planner must NOT be an anthropic passthrough — the OpenAI-compatible wire under test")
	}

	req := &agent.AnthropicMessagesRequest{Model: "gpt-4o", Raw: append([]byte(nil), body...)}
	orig := append([]byte(nil), req.Raw...)

	runServeRequestChain(s, req)

	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("the safety layer rewrote the OpenAI-compatible body off the Anthropic wire — provider reuse would be lost:\nwant %s\ngot  %s", orig, req.Raw)
	}

	// Provider-cache framing (reuse the §A3 linter as the assertion helper, per the item):
	// model the stable system+tools head as prompt segments before and after the chain and
	// assert it stays a byte-identical cacheable prefix — no divergence introduced.
	before := openAIStableHeadSegments(t, orig)
	after := openAIStableHeadSegments(t, req.Raw)
	d := cachemeta.Diverge(before, after)
	if !d.Identical {
		t.Fatalf("OpenAI stable system+tools head diverged after the transform chain: %+v", d)
	}
	var wantStable int64
	for _, seg := range before {
		wantStable += seg.Tokens
	}
	if d.StableTokens != wantStable {
		t.Fatalf("stable head not fully provider-cacheable after the chain: StableTokens=%d want %d", d.StableTokens, wantStable)
	}
}

// firstBreakpointMessageEnd is the byte offset of the end of messages[0] — the protected
// prefix boundary for the elide fixture, whose first (and only message-level) cache_control
// breakpoint sits on message[0]. Mirrors the boundary math in messages_elide_test.go.
func firstBreakpointMessageEnd(t *testing.T, raw []byte) int {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	_, spans, ok := decodeArrayElementsFromTest(t, raw, obj["messages"])
	if !ok || len(spans) == 0 {
		t.Fatal("decodeArrayElements failed")
	}
	return spans[0].end
}

// lastBreakpointMessageEnd is the byte offset of the end of the LAST message carrying a
// cache_control breakpoint — the protected prefix boundary for the compaction fixture.
// Mirrors the boundary math in messages_compact_test.go.
func lastBreakpointMessageEnd(t *testing.T, raw []byte) int {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	elems, spans, ok := decodeArrayElementsFromTest(t, raw, obj["messages"])
	if !ok {
		t.Fatal("decodeArrayElements failed")
	}
	i := lastBreakpointMessageFromTest(elems)
	if i < 0 {
		t.Fatal("fixture sanity: no cache_control breakpoint in messages")
	}
	return spans[i].end
}

// openAIStableHeadSegments models the stable, provider-cacheable head of an OpenAI-compatible
// chat body as ordered cachemeta prompt segments: the leading system message content (SegStable)
// followed by the tools[] schema (SegToolSchema). Content is the exact serialized bytes, so a
// single rewritten byte in either segment makes cachemeta.Diverge report a break. Token counts are
// a coarse length-based estimate — only their equality before/after matters here.
func openAIStableHeadSegments(t *testing.T, body []byte) []cachemeta.PromptSegment {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		t.Fatalf("unmarshal OpenAI body: %v", err)
	}
	var msgs []json.RawMessage
	if err := json.Unmarshal(obj["messages"], &msgs); err != nil || len(msgs) == 0 {
		t.Fatalf("unmarshal messages: %v", err)
	}
	var sys struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(msgs[0], &sys); err != nil || sys.Role != "system" {
		t.Fatalf("expected messages[0] to be the stable system head, got %s", msgs[0])
	}
	segs := []cachemeta.PromptSegment{{
		Kind:    cachemeta.SegStable,
		Content: append([]byte(nil), sys.Content...),
		Tokens:  int64(len(sys.Content)/4 + 1),
	}}
	if tools := obj["tools"]; len(bytes.TrimSpace(tools)) > 0 {
		segs = append(segs, cachemeta.PromptSegment{
			Kind:    cachemeta.SegToolSchema,
			Content: append([]byte(nil), tools...),
			Tokens:  int64(len(tools)/4 + 1),
		})
	}
	return segs
}
