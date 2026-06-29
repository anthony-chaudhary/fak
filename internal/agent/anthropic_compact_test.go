package agent

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// realisticBody builds a Claude-Code-shaped /v1/messages body: a system array with a
// cache_control breakpoint on its last block, a tools array, and nMsgs messages — the
// 2nd of which carries a cache_control breakpoint (an early-history incremental cache),
// plus one assistant tool_use / user tool_result pair near the middle. stream + max_tokens
// are top-level keys outside the cached prefix.
func realisticBody(t *testing.T, nMsgs int) []byte {
	t.Helper()
	type block map[string]any
	msgs := make([]map[string]any, 0, nMsgs)
	// msg 0: a long early user turn carrying the LAST cache_control breakpoint.
	msgs = append(msgs, map[string]any{
		"role": "user",
		"content": []block{
			{"type": "text", "text": strings.Repeat("early context that is cached. ", 20), "cache_control": map[string]any{"type": "ephemeral"}},
		},
	})
	// Fill the middle with alternating user/assistant turns, with one tool pair.
	for i := 1; i < nMsgs; i++ {
		if i == 3 {
			msgs = append(msgs, map[string]any{
				"role": "assistant",
				"content": []block{
					{"type": "tool_use", "id": "tu_1", "name": "Read", "input": map[string]any{"path": "x.go"}},
				},
			})
			continue
		}
		if i == 4 {
			msgs = append(msgs, map[string]any{
				"role": "user",
				"content": []block{
					{"type": "tool_result", "tool_use_id": "tu_1", "content": strings.Repeat("file body line. ", 30)},
				},
			})
			continue
		}
		role := "user"
		if i%2 == 0 {
			role = "assistant"
		}
		msgs = append(msgs, map[string]any{
			"role":    role,
			"content": []block{{"type": "text", "text": strings.Repeat("turn ", 20) + itoa(i)}},
		})
	}
	body := map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"stream":     true,
		"system": []block{
			{"type": "text", "text": "You are a coding agent."},
			{"type": "text", "text": strings.Repeat("policy text. ", 40), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"tools":    []block{{"name": "Read", "description": "read a file", "input_schema": map[string]any{"type": "object"}}},
		"messages": msgs,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	return raw
}

// lastBreakpointEnd returns the byte offset just past the last messages[] element that
// carries a cache_control breakpoint — the boundary the cached prefix must not cross.
func lastBreakpointEnd(t *testing.T, raw []byte) int {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	elems, spans, ok := decodeArrayElements(raw, obj["messages"])
	if !ok {
		t.Fatalf("decodeArrayElements failed")
	}
	idx := lastBreakpointMessage(elems)
	if idx < 0 {
		return arrayContentStart(spans)
	}
	return spans[idx].end
}

// TestCompactPreservesCachePrefix is the core invariant: a non-identity compaction copies
// the protected prefix (through the last breakpoint message) byte-for-byte, so the upstream
// cache hit survives.
func TestCompactPreservesCachePrefix(t *testing.T) {
	raw := realisticBody(t, 16)
	split := lastBreakpointEnd(t, raw)

	out := CompactAnthropicHistory(raw, 200) // small budget → must compact
	if bytes.Equal(out, raw) {
		t.Fatalf("expected compaction with a 16-message body at budget=200, got identity")
	}
	if len(out) >= len(raw) {
		t.Fatalf("expected a shorter body, got %d >= %d", len(out), len(raw))
	}
	if split > len(out) || !bytes.Equal(raw[:split], out[:split]) {
		t.Fatalf("cache prefix bytes changed: prefix end=%d, lenOut=%d", split, len(out))
	}
}

// TestCompactStillDecodesAndPairs proves the rewritten body is a valid Messages request
// with strict role alternation preserved and no orphaned tool_result.
func TestCompactStillDecodesAndPairs(t *testing.T) {
	raw := realisticBody(t, 16)
	out := CompactAnthropicHistory(raw, 200)
	if bytes.Equal(out, raw) {
		t.Fatalf("expected compaction, got identity")
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("compacted body failed to decode: %v", err)
	}
	// Re-parse the wire messages and check no tool_result lacks a preceding tool_use.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(out, &obj); err != nil {
		t.Fatalf("unmarshal out: %v", err)
	}
	var msgs []anthropicInboundMessage
	if err := json.Unmarshal(obj["messages"], &msgs); err != nil {
		t.Fatalf("unmarshal messages: %v", err)
	}
	seenToolUse := map[string]bool{}
	for _, m := range msgs {
		var blocks []anthropicInboundBlock
		_ = json.Unmarshal(m.Content, &blocks)
		for _, b := range blocks {
			if b.Type == "tool_use" {
				seenToolUse[b.ID] = true
			}
			if b.Type == "tool_result" && b.ToolUseID != "" && !seenToolUse[b.ToolUseID] {
				t.Fatalf("orphaned tool_result for %q — its tool_use was dropped", b.ToolUseID)
			}
		}
	}
	// The stub must be present and name a positive count.
	if !strings.Contains(string(out), compactStubPrefix) {
		t.Fatalf("expected a %q stub in the compacted body", compactStubPrefix)
	}
}

// TestCompactIdentityCases asserts the fail-safe: any condition under which the rewrite
// cannot be proven cache-safe returns the input unchanged.
func TestCompactIdentityCases(t *testing.T) {
	full := realisticBody(t, 16)
	cases := []struct {
		name   string
		raw    []byte
		budget int
	}{
		{"budget zero", full, 0},
		{"budget negative", full, -5},
		{"non-json", []byte("not json at all"), 1000},
		{"empty", nil, 1000},
		{"no messages key", []byte(`{"model":"x","max_tokens":1}`), 1000},
		{"too few messages", realisticBody(t, 2), 10},
		{"suffix already under budget", full, 1 << 20},
		{"no breakpoint anywhere", noBreakpointBody(t), 50},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := CompactAnthropicHistory(c.raw, c.budget)
			if !bytes.Equal(out, c.raw) {
				t.Fatalf("expected identity for %q, body changed (%d -> %d bytes)", c.name, len(c.raw), len(out))
			}
		})
	}
}

// noBreakpointBody is a valid multi-message body with NO cache_control anywhere — the
// gateway cannot know the cache boundary, so compaction must be identity.
func noBreakpointBody(t *testing.T) []byte {
	t.Helper()
	msgs := make([]map[string]any, 0, 8)
	for i := 0; i < 8; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs = append(msgs, map[string]any{"role": role, "content": strings.Repeat("plain turn text ", 30)})
	}
	raw, err := json.Marshal(map[string]any{
		"model": "claude", "max_tokens": 512, "messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// TestCompactToolPairNotOrphaned forces the kept-window boundary to land exactly on the
// user tool_result turn and asserts the preceding assistant tool_use turn is pulled in, so
// the wire never carries a tool_result whose tool_use was dropped.
func TestCompactToolPairNotOrphaned(t *testing.T) {
	// Build a body whose only compactible bulk forces the window to start near the tool
	// pair (msgs 3=tool_use, 4=tool_result in realisticBody).
	raw := realisticBody(t, 16)
	// A budget that keeps roughly the tail but lands the boundary in the tool region.
	for _, budget := range []int{120, 180, 260, 340} {
		out := CompactAnthropicHistory(raw, budget)
		if bytes.Equal(out, raw) {
			continue // identity is always safe
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(out, &obj); err != nil {
			t.Fatalf("budget=%d: unmarshal: %v", budget, err)
		}
		var msgs []anthropicInboundMessage
		if err := json.Unmarshal(obj["messages"], &msgs); err != nil {
			t.Fatalf("budget=%d: unmarshal messages: %v", budget, err)
		}
		seen := map[string]bool{}
		for _, m := range msgs {
			var blocks []anthropicInboundBlock
			_ = json.Unmarshal(m.Content, &blocks)
			for _, b := range blocks {
				if b.Type == "tool_use" {
					seen[b.ID] = true
				}
				if b.Type == "tool_result" && b.ToolUseID != "" && !seen[b.ToolUseID] {
					t.Fatalf("budget=%d: orphaned tool_result %q", budget, b.ToolUseID)
				}
			}
		}
	}
}

// TestCompactBreakpointOnLastMessage: when the last breakpoint is the final message, there
// is nothing after it to compact → identity.
func TestCompactBreakpointOnLastMessage(t *testing.T) {
	msgs := make([]map[string]any, 0, 6)
	for i := 0; i < 6; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		blk := map[string]any{"type": "text", "text": strings.Repeat("x ", 50)}
		if i == 5 {
			blk["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		msgs = append(msgs, map[string]any{"role": role, "content": []map[string]any{blk}})
	}
	raw, err := json.Marshal(map[string]any{"model": "c", "max_tokens": 1, "messages": msgs})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if out := CompactAnthropicHistory(raw, 10); !bytes.Equal(out, raw) {
		t.Fatalf("expected identity when the breakpoint is on the last message")
	}
}

// claudeCodeShapedBody models the REAL Claude Code growing-conversation cache layout
// (Anthropic prompt-caching docs: up to 4 breakpoints — one on the static system/tools head,
// and one or more on RECENT turns to incrementally cache the growing prefix). The decisive
// difference from realisticBody: there is a breakpoint on msg 0 (the early cached head) AND a
// breakpoint on a RECENT message (near the tail), so the LAST breakpoint is near the end.
// nMsgs total; the breakpoint sits on the message recentBpBack from the end.
func claudeCodeShapedBody(t *testing.T, nMsgs, recentBpBack int) []byte {
	t.Helper()
	type block map[string]any
	recentBpIdx := nMsgs - 1 - recentBpBack
	msgs := make([]map[string]any, 0, nMsgs)
	for i := 0; i < nMsgs; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		blk := block{"type": "text", "text": strings.Repeat("conversation turn body words. ", 12) + itoa(i)}
		// Breakpoint on msg 0 (early cached head) and on a RECENT message (growing-prefix cache).
		if i == 0 || i == recentBpIdx {
			blk["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		msgs = append(msgs, map[string]any{"role": role, "content": []block{blk}})
	}
	raw, err := json.Marshal(map[string]any{
		"model": "claude-sonnet-4-6", "max_tokens": 1024,
		"system": []block{
			{"type": "text", "text": "You are a coding agent."},
			{"type": "text", "text": strings.Repeat("policy text. ", 40), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// fullyCacheControlledBody models the dogfood shape where Claude Code has made every
// message part of a provider-warm prompt cache span. A smaller prompt is not automatically
// cheaper here: deleting the middle would deliberately burst the cached suffix.
func fullyCacheControlledBody(t *testing.T, nMsgs int) []byte {
	t.Helper()
	type block map[string]any
	msgs := make([]map[string]any, 0, nMsgs)
	for i := 0; i < nMsgs; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs = append(msgs, map[string]any{
			"role": role,
			"content": []block{{
				"type":          "text",
				"text":          strings.Repeat("cache-hot conversation body. ", 20) + itoa(i),
				"cache_control": map[string]any{"type": "ephemeral"},
			}},
		})
	}
	raw, err := json.Marshal(map[string]any{
		"model": "claude-sonnet-4-6", "max_tokens": 1024,
		"messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// firstMessageBreakpointEnd returns the byte offset just past the FIRST messages[] element
// carrying a cache_control breakpoint — the stable cached HEAD the provider reuses every turn.
func firstMessageBreakpointEnd(t *testing.T, raw []byte) int {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	elems, spans, ok := decodeArrayElements(raw, obj["messages"])
	if !ok {
		t.Fatalf("decodeArrayElements failed")
	}
	for i, el := range elems {
		var m struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(el, &m) == nil && rawHasCacheControl(m.Content) {
			return spans[i].end
		}
	}
	return arrayContentStart(spans)
}

// TestCompactFiresOnClaudeCodeMultiBreakpointShape is the regression for the silent
// no-op-on-real-traffic bug: with a breakpoint on the static head AND on a recent turn (the
// real Claude Code layout), a large session MUST still shed the un-cacheable MIDDLE turns —
// the span between the cached head and the recent kept window, which the provider re-bills
// every turn anyway (it is beyond the recent breakpoint's 20-block backward lookback). The
// kept HEAD (through the first message breakpoint) must stay byte-identical so the dominant
// cache read survives.
//
// Before the anchor fix, CompactAnthropicHistory anchored the protected prefix on the LAST
// breakpoint (the recent turn), so the whole conversation was "protected" and it shed 0%.
func TestCompactFiresOnClaudeCodeMultiBreakpointShape(t *testing.T) {
	raw := claudeCodeShapedBody(t, 120, 3) // 120 turns, recent breakpoint kept in the tail
	head := firstMessageBreakpointEnd(t, raw)

	out := CompactAnthropicHistory(raw, 1200) // budget far below the 120-turn middle, but keeps the recent breakpoint
	if bytes.Equal(out, raw) {
		t.Fatalf("REGRESSION: compaction is a no-op on the real Claude Code multi-breakpoint shape — "+
			"it must shed the un-cacheable middle turns (inbound %d bytes, budget 1200)", len(raw))
	}
	if len(out) >= len(raw) {
		t.Fatalf("expected a shorter body, got %d >= %d", len(out), len(raw))
	}
	// The stable cached HEAD (through the first message breakpoint) must be byte-identical.
	if head > len(out) || !bytes.Equal(raw[:head], out[:head]) {
		t.Fatalf("cached head prefix changed: head end=%d, lenOut=%d", head, len(out))
	}
	// The result must still be a valid, well-paired Messages request.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("compacted body failed to decode: %v", err)
	}
}

func TestCompactRefusesToBurstFullyCachedHistory(t *testing.T) {
	raw := fullyCacheControlledBody(t, 80)

	out, outcome := CompactAnthropicHistoryWithOutcome(raw, 400)
	if !bytes.Equal(out, raw) {
		t.Fatalf("fully cache_control-marked history must stay identity, changed %d -> %d bytes", len(raw), len(out))
	}
	if outcome.Reason != CompactReasonCachedSpan {
		t.Fatalf("Reason=%q, want %q", outcome.Reason, CompactReasonCachedSpan)
	}
	if outcome.Dropped != 0 || outcome.ShedTokens != 0 {
		t.Fatalf("cached-span bail must not claim shed work: %+v", outcome)
	}
}

// recentOnlyBreakpointBody models the ACTUAL Claude Code layout the 2026-06-29 ablation note
// observed: the ONLY messages[] cache_control sits on a RECENT turn (recentBpBack from the end),
// with NO breakpoint on msg 0. firstBreakpointMessage therefore anchors near the end, the
// protected prefix swallows ~the whole conversation, and the compactible suffix is structurally
// tiny — the dormant-on-real-traffic shape (#1407). Contrast claudeCodeShapedBody, which ALSO
// marks msg 0 and so (artificially) lets compaction fire.
func recentOnlyBreakpointBody(t *testing.T, nMsgs, recentBpBack int) []byte {
	t.Helper()
	type block map[string]any
	recentBpIdx := nMsgs - 1 - recentBpBack
	msgs := make([]map[string]any, 0, nMsgs)
	for i := 0; i < nMsgs; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		blk := block{"type": "text", "text": strings.Repeat("conversation turn body words. ", 12) + itoa(i)}
		if i == recentBpIdx {
			blk["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		msgs = append(msgs, map[string]any{"role": role, "content": []block{blk}})
	}
	raw, err := json.Marshal(map[string]any{
		"model": "claude-sonnet-4-6", "max_tokens": 1024,
		"system":   []block{{"type": "text", "text": "You are a coding agent.", "cache_control": map[string]any{"type": "ephemeral"}}},
		"messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// TestCompactAnchorStarvedDiagnostic is the witness for #1409: on the real Claude Code layout (a
// recent-only message breakpoint), compaction bails under_budget AND flags AnchorStarved, because
// the protected prefix already exceeds the budget — the signal that distinguishes this structural
// dormancy from a benign short-session idle. A short session and a system-only breakpoint are NOT
// starved.
func TestCompactAnchorStarvedDiagnostic(t *testing.T) {
	// 1. Recent-only breakpoint, long conversation: anchor near the end → prefix >> budget.
	raw := recentOnlyBreakpointBody(t, 120, 2)
	out, outcome := CompactAnthropicHistoryWithOutcome(raw, 1200)
	if !bytes.Equal(out, raw) {
		t.Fatalf("expected identity (under_budget) on a recent-only-breakpoint body, body changed")
	}
	if outcome.Reason != CompactReasonUnderBudget {
		t.Fatalf("Reason=%q, want %q", outcome.Reason, CompactReasonUnderBudget)
	}
	if !outcome.AnchorStarved {
		t.Fatalf("expected AnchorStarved=true (prefix %d tok > budget 1200, suffix %d tok) — the anchor swallowed the conversation",
			outcome.ProtectedPrefixTokens, outcome.SuffixTokens)
	}
	if outcome.ProtectedPrefixTokens <= 1200 {
		t.Fatalf("the protected prefix must exceed the budget to be starved, got %d", outcome.ProtectedPrefixTokens)
	}

	// 2. Genuinely short session (prefix small, well under a huge budget): benign idle, NOT starved.
	short := claudeCodeShapedBody(t, 6, 1)
	if _, shortOut := CompactAnthropicHistoryWithOutcome(short, 1<<20); shortOut.AnchorStarved {
		t.Fatalf("a genuinely short session must NOT be anchor-starved (prefix %d, huge budget)", shortOut.ProtectedPrefixTokens)
	}

	// 3. System-only breakpoint (pfxEnd = -1): the whole array is compactible, so an under_budget
	//    there is genuinely benign — prefix tokens are 0, never starved.
	sysMsgs := make([]map[string]any, 0, 6)
	for i := 0; i < 6; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		sysMsgs = append(sysMsgs, map[string]any{"role": role, "content": []map[string]any{{"type": "text", "text": strings.Repeat("body ", 25)}}})
	}
	sysOnly, err := json.Marshal(map[string]any{
		"model": "claude", "max_tokens": 512,
		"system":   []map[string]any{{"type": "text", "text": "sys", "cache_control": map[string]any{"type": "ephemeral"}}},
		"messages": sysMsgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if _, sysOut := CompactAnthropicHistoryWithOutcome(sysOnly, 1<<20); sysOut.AnchorStarved {
		t.Fatalf("a system-only breakpoint (pfxEnd=-1) must never be anchor-starved")
	}
}

func TestCacheBurstBreakEvenTurns(t *testing.T) {
	// Dropping 20k cache-hot tokens saves only their discounted read cost each future
	// turn. If the edit invalidates 40k warm suffix tokens, Anthropic-like 1h write/read
	// multipliers (1.25 vs 0.1) need ceil((1.25-0.1)*40000 / (0.1*20000)) = 23 turns.
	if got := CacheBurstBreakEvenTurns(20000, 40000, 0.1, 1.25); got != 23 {
		t.Fatalf("break-even = %d, want 23", got)
	}
	// Segment-level vCache surgery changes the same economics by shrinking the invalidated
	// suffix. A 5k-token segment penalty repays in three future turns under the same drop.
	if got := CacheBurstBreakEvenTurns(20000, 5000, 0.1, 1.25); got != 3 {
		t.Fatalf("sub-vcache break-even = %d, want 3", got)
	}
	if got := CacheBurstBreakEvenTurns(0, 5000, 0.1, 1.25); got != int(^uint(0)>>1) {
		t.Fatalf("no-saving break-even = %d, want MaxInt", got)
	}
	if got := CacheBurstBreakEvenTurns(20000, 0, 0.1, 1.25); got != 0 {
		t.Fatalf("no-penalty break-even = %d, want 0", got)
	}
}

func TestCacheBurstPaysBackOnKnownSessionHorizon(t *testing.T) {
	// Same economics as TestCacheBurstBreakEvenTurns: full-suffix burst needs 23 future
	// turns; sub-vCache-style surgery needs only three.
	cases := []struct {
		name                    string
		totalTurns, currentTurn int
		invalidatedSuffixTokens int
		want                    bool
	}{
		{"turn 20 of 50 pays back full suffix", 50, 20, 40000, true},
		{"turn 40 of 50 is too late for full suffix", 50, 40, 40000, false},
		{"turn 47 of 50 pays back sub-vcache surgery", 50, 47, 5000, true},
		{"turn 48 of 50 is too late even for sub-vcache surgery", 50, 48, 5000, false},
		{"unknown horizon does not burst on a guess", 0, 20, 5000, false},
		{"zero penalty always pays back", 50, 49, 0, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := CacheBurstPaysBack(tc.totalTurns, tc.currentTurn, 20000, tc.invalidatedSuffixTokens, 0.1, 1.25)
			if got != tc.want {
				t.Fatalf("CacheBurstPaysBack = %v, want %v", got, tc.want)
			}
		})
	}
}

// assertAlternation fails if any two consecutive messages share a role — Anthropic rejects
// that with a 400, and the splice's synthetic stub must not introduce it (F7).
func assertAlternation(t *testing.T, out []byte) {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(out, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	var ms []struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(obj["messages"], &ms); err != nil {
		t.Fatalf("unmarshal messages: %v", err)
	}
	prev := ""
	for i, m := range ms {
		if m.Role == prev {
			t.Fatalf("adjacent %q turns at index %d — alternation broken, Anthropic would 400", m.Role, i)
		}
		prev = m.Role
	}
}

// TestCompactStubNeverBreaksAlternation is the F7 regression: with the first-breakpoint anchor
// the protected prefix routinely ends on a USER turn (msg 0), so a hardcoded user-role stub
// would produce two consecutive user turns. The stub role must alternate with both neighbors.
// Swept across breakpoint placements, budgets, and message counts.
func TestCompactStubNeverBreaksAlternation(t *testing.T) {
	for _, n := range []int{20, 41, 80, 121} {
		for _, recentBack := range []int{1, 2, 5} {
			for _, budget := range []int{100, 200, 400, 800} {
				raw := claudeCodeShapedBody(t, n, recentBack)
				out := CompactAnthropicHistory(raw, budget)
				if bytes.Equal(out, raw) {
					continue // identity is always safe
				}
				assertAlternation(t, out)
				if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
					t.Fatalf("n=%d recent=%d budget=%d: compacted body failed to decode: %v", n, recentBack, budget, err)
				}
			}
		}
	}
}

// TestCompactSystemOnlyBreakpoint covers the case where ONLY the system block carries the
// cache_control breakpoint (no per-message breakpoint): every message is compactible, the
// system+array-head prefix is preserved, and the result still decodes.
func TestCompactSystemOnlyBreakpoint(t *testing.T) {
	msgs := make([]map[string]any, 0, 10)
	for i := 0; i < 10; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs = append(msgs, map[string]any{
			"role":    role,
			"content": []map[string]any{{"type": "text", "text": strings.Repeat("body ", 25)}},
		})
	}
	raw, err := json.Marshal(map[string]any{
		"model": "claude", "max_tokens": 512,
		"system": []map[string]any{
			{"type": "text", "text": "sys", "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	out := CompactAnthropicHistory(raw, 100)
	if bytes.Equal(out, raw) {
		t.Fatalf("expected compaction with a system-only breakpoint, got identity")
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("system-only compacted body failed to decode: %v", err)
	}
	// The whole array head (system + the messages `[`) must be byte-identical.
	var obj map[string]json.RawMessage
	_ = json.Unmarshal(raw, &obj)
	_, spans, _ := decodeArrayElements(raw, obj["messages"])
	head := arrayContentStart(spans)
	if !bytes.Equal(raw[:head], out[:head]) {
		t.Fatalf("array-head prefix changed under a system-only breakpoint")
	}
}

// TestCompactToViewStubsElidedMiddleTurn is the byte-level witness for the ctxplan-view
// req.Raw transform (#927): the planner's resident set drops the off-topic middle turn,
// and CompactAnthropicHistoryToView replaces it in place with a same-role stub while the
// cached prefix, the resident turns, AND the message count / role alternation are all
// preserved byte-for-byte. Fail-safe identity when nothing is elided.
func TestCompactToViewStubsElidedMiddleTurn(t *testing.T) {
	raw := []byte(`{"model":"claude","max_tokens":4096,` +
		`"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[` +
		`{"role":"user","content":"rotate the auth token and check the refund policy"},` +
		`{"role":"assistant","content":"weather sunny 22C light wind from the west"},` +
		`{"role":"user","content":"what is the auth token rotation and refund window"}]}`)

	// The planner's resident view: system + first user + last user (the weather span elided).
	planned := []Message{
		{Role: RoleSystem, Content: "You are a coding agent."},
		{Role: RoleUser, Content: "rotate the auth token and check the refund policy"},
		{Role: RoleUser, Content: "what is the auth token rotation and refund window"},
	}

	out, outcome := CompactAnthropicHistoryToView(raw, planned)
	if outcome.Reason != CompactReasonNone {
		t.Fatalf("expected a fire (Reason==None), got %q (Dropped=%d)", outcome.Reason, outcome.Dropped)
	}
	if outcome.Dropped != 1 {
		t.Fatalf("expected 1 stubbed message, got %d", outcome.Dropped)
	}
	s := string(out)

	// The off-topic span is gone (stubbed); the resident turns are verbatim.
	if strings.Contains(s, "weather sunny 22C") {
		t.Errorf("the elided weather span must not survive in the rewritten body")
	}
	if !strings.Contains(s, "[fak] ctxview-elided") {
		t.Errorf("the elided turn must be replaced by a [fak] ctxview-elided stub")
	}
	if !strings.Contains(s, "rotate the auth token") || !strings.Contains(s, "what is the auth token rotation") {
		t.Errorf("the resident turns must survive verbatim")
	}

	// The cached system prefix is byte-identical.
	wantSys := `"system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}}]`
	if !strings.Contains(s, wantSys) {
		t.Errorf("the cached system prefix must be byte-identical")
	}

	// The message COUNT and role alternation survive (same-role stub in place of the original).
	if c := strings.Count(s, `"role":`); c != 3 {
		t.Errorf("the rewritten body must keep all 3 messages (one stubbed in place), got %d role keys", c)
	}
	// The stub carries the assistant role (same as the original middle turn) and the ctxview
	// marker. json.Marshal sorts keys, so check each independently.
	if !strings.Contains(s, `"role":"assistant"`) || !strings.Contains(s, `"[fak] ctxview-elided`) {
		t.Errorf("the stub must carry the assistant role + the ctxview-elided marker (json key order is unspecified)")
	}

	// The result re-decodes as a valid request.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("rewritten body failed to decode: %v", err)
	}
}

// TestCompactToViewIdentityWhenAllResident asserts the fail-safe no-op: when the planner
// kept everything (every message is resident), the body is returned byte-for-byte unchanged.
func TestCompactToViewIdentityWhenAllResident(t *testing.T) {
	raw := []byte(`{"model":"claude","max_tokens":4096,` +
		`"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[` +
		`{"role":"user","content":"hello"},` +
		`{"role":"assistant","content":"hi there"}]}`)

	// Every message is resident — nothing to elide.
	planned := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "hello"},
		{Role: RoleAssistant, Content: "hi there"},
	}
	out, outcome := CompactAnthropicHistoryToView(raw, planned)
	if outcome.Reason == CompactReasonNone {
		t.Fatalf("expected a bail (nothing to elide), got a fire")
	}
	if !bytes.Equal(out, raw) {
		t.Errorf("all-resident planned view must return the body byte-for-byte unchanged")
	}
}

// TestCompactToViewKeepsToolBlocks asserts that a message carrying tool_use / tool_result
// blocks is NEVER stubbed (elementTextContent bails on non-text blocks), so tool_use ↔
// tool_result pairings stay intact.
func TestCompactToViewKeepsToolBlocks(t *testing.T) {
	raw := []byte(`{"model":"claude","max_tokens":4096,` +
		`"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[` +
		`{"role":"user","content":"rotate the auth token"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"Read","input":{"path":"x.go"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"file contents here"}]},` +
		`{"role":"user","content":"what is the auth token rotation"}]}`)

	// The planner elided the tool pair (kept only the two text turns) — but the transform
	// must KEEP the tool blocks verbatim because it cannot confidently match them.
	planned := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "rotate the auth token"},
		{Role: RoleUser, Content: "what is the auth token rotation"},
	}
	out, outcome := CompactAnthropicHistoryToView(raw, planned)
	s := string(out)
	// The tool_use and tool_result survive unchanged.
	if !strings.Contains(s, `"tool_use"`) || !strings.Contains(s, `"tool_result"`) {
		t.Errorf("tool_use/tool_result blocks must be kept verbatim (never stubbed):\n%s", s)
	}
	if outcome.Reason == CompactReasonNone && outcome.Dropped > 0 {
		// It may have stubbed the text-only middle turn if any existed; the tool blocks are safe regardless.
	}
	// The result re-decodes.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("rewritten body failed to decode: %v", err)
	}
}
