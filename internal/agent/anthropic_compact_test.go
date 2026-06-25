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
	raw := claudeCodeShapedBody(t, 120, 2) // 120 turns, recent breakpoint 2 from the end
	head := firstMessageBreakpointEnd(t, raw)

	out := CompactAnthropicHistory(raw, 400) // budget far below the 120-turn middle
	if bytes.Equal(out, raw) {
		t.Fatalf("REGRESSION: compaction is a no-op on the real Claude Code multi-breakpoint shape — "+
			"it must shed the un-cacheable middle turns (inbound %d bytes, budget 400)", len(raw))
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
