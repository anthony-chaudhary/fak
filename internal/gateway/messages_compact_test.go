package gateway

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
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
