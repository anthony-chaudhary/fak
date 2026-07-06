package agent

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// anthropic_compact_goalpin_test.go — the byte-level passthrough half of the goal pin (#845). The
// decoded planner pins a RoleGoal span as the context heap's GC root (ctxplan_session.go); this
// surface — the flagship `fak guard -- claude` Anthropic passthrough, which never sees the decoded
// role — marks the goal in the message TEXT (compactGoalMarker) and HOISTS it out of the
// compaction drop range so a session's standing goal survives a compaction it would otherwise fall
// into. These witnesses prove the goal survives, the drop still sheds the rest of the middle,
// alternation / tool-pairing / cache-prefix are all preserved, and — the load-bearing invariant —
// a body with NO marker compacts byte-for-byte as before.

// goalMarkedBody builds a Claude-Code-shaped /v1/messages body: a system array with a
// cache_control breakpoint on its last block, one tool, and nMsgs alternating messages whose
// FIRST message carries the single message-level cache_control breakpoint (so the protected prefix
// is msg 0, a user turn). The message at goalIdx carries the goal marker; every other middle turn
// carries a unique, droppable text. withToolPair injects an assistant tool_use / user tool_result
// pair at indices 3/4 to prove pairing survives the hoist.
func goalMarkedBody(t *testing.T, nMsgs, goalIdx int, withToolPair bool) []byte {
	t.Helper()
	type block map[string]any
	if goalIdx <= 0 || goalIdx >= nMsgs {
		t.Fatalf("goalIdx %d must be a compactible middle index in (0, %d)", goalIdx, nMsgs)
	}
	msgs := make([]map[string]any, 0, nMsgs)
	// msg 0: the cached head — a user turn carrying the only message-level breakpoint.
	msgs = append(msgs, map[string]any{
		"role": "user",
		"content": []block{
			{"type": "text", "text": strings.Repeat("early cached context. ", 20), "cache_control": map[string]any{"type": "ephemeral"}},
		},
	})
	for i := 1; i < nMsgs; i++ {
		naturalRole := "user"
		if i%2 == 1 {
			naturalRole = "assistant"
		}
		if withToolPair && i == 3 {
			msgs = append(msgs, map[string]any{
				"role":    "assistant",
				"content": []block{{"type": "tool_use", "id": "tu_1", "name": "Read", "input": map[string]any{"path": "x.go"}}},
			})
			continue
		}
		if withToolPair && i == 4 {
			msgs = append(msgs, map[string]any{
				"role":    "user",
				"content": []block{{"type": "tool_result", "tool_use_id": "tu_1", "content": strings.Repeat("file body line. ", 30)}},
			})
			continue
		}
		text := strings.Repeat("droppable middle turn ", 20) + itoa(i)
		if i == goalIdx {
			// The goal turn keeps its natural alternating role so the SOURCE body is valid, and
			// leads with the marker so isCompactGoalText pins it.
			text = compactGoalMarker + " keep the auth-token rotation runbook as the standing goal (turn " + itoa(i) + ")"
		}
		msgs = append(msgs, map[string]any{
			"role":    naturalRole,
			"content": []block{{"type": "text", "text": text}},
		})
	}
	raw, err := json.Marshal(map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"stream":     true,
		"system": []block{
			{"type": "text", "text": "You are a coding agent."},
			{"type": "text", "text": strings.Repeat("policy text. ", 40), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"tools":    []block{{"name": "Read", "description": "read a file", "input_schema": map[string]any{"type": "object"}}},
		"messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// goalText is the distinctive content the goal turn at goalIdx carries in goalMarkedBody.
func goalText(goalIdx int) string {
	return compactGoalMarker + " keep the auth-token rotation runbook as the standing goal (turn " + itoa(goalIdx) + ")"
}

// firstBreakpointEnd returns the byte offset just past the FIRST message-level breakpoint — the
// protected prefix the firstBP anchor must copy verbatim.
func firstBreakpointEnd(t *testing.T, raw []byte) int {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	elems, spans, ok := decodeArrayElements(raw, obj["messages"])
	if !ok {
		t.Fatalf("decodeArrayElements failed")
	}
	idx := firstBreakpointMessage(elems)
	if idx < 0 {
		return arrayContentStart(spans)
	}
	return spans[idx].end
}

// TestCompactHoistsGoalOutOfDropRange is the headline: a goal-marked turn sitting in the
// compactible middle SURVIVES a tight-budget compaction verbatim, while the rest of the middle is
// still shed to a stub — the whole point of the pin.
func TestCompactHoistsGoalOutOfDropRange(t *testing.T) {
	const nMsgs, goalIdx = 16, 6
	raw := goalMarkedBody(t, nMsgs, goalIdx, false)

	out, outcome := CompactAnthropicHistoryWithOutcome(raw, 200) // tight budget → must compact
	if outcome.Reason != CompactReasonNone {
		t.Fatalf("expected a fire (Reason==None), got %q", outcome.Reason)
	}
	if bytes.Equal(out, raw) {
		t.Fatalf("expected a rewrite, got identity")
	}
	if len(out) >= len(raw) {
		t.Fatalf("expected a shorter body, got %d >= %d", len(out), len(raw))
	}
	s := string(out)

	// The goal survives verbatim...
	if !strings.Contains(s, goalText(goalIdx)) {
		t.Fatalf("the goal turn must survive compaction verbatim; it was laundered away")
	}
	// ...while a DIFFERENT droppable middle turn is gone (real shed happened).
	if strings.Contains(s, "droppable middle turn "+itoa(2)) {
		t.Fatalf("a non-goal middle turn should have been dropped; nothing was shed")
	}
	// The compaction stub stands in for the dropped middle.
	if !strings.Contains(s, compactStubPrefix) {
		t.Fatalf("expected a %q stub for the dropped (non-goal) middle", compactStubPrefix)
	}

	// Structural guarantees: decodes, strict alternation, cache prefix intact.
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("goal-pinned body failed to decode: %v", err)
	}
	assertAlternation(t, out)
	split := firstBreakpointEnd(t, raw)
	if split > len(out) || !bytes.Equal(raw[:split], out[:split]) {
		t.Fatalf("cache prefix bytes changed under the goal hoist (split=%d, lenOut=%d)", split, len(out))
	}
}

// TestCompactWithoutGoalMarkerUnchanged is the load-bearing invariant: the SAME body shape with NO
// marker takes the ordinary contiguous path (the goal turn's position is just another dropped
// turn), so the marker is the ONLY trigger for the new behavior.
func TestCompactWithoutGoalMarkerUnchanged(t *testing.T) {
	// Build a body identical in shape but where the "goal" index carries plain droppable text.
	raw := goalMarkedBody(t, 16, 6, false)
	plain := bytes.Replace(raw, []byte(goalText(6)), []byte(strings.Repeat("droppable middle turn ", 20)+itoa(6)), 1)
	if bytes.Equal(plain, raw) {
		t.Fatalf("failed to strip the marker for the control body")
	}

	// Detection must see no goal in the plain body.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(plain, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	elems, _, ok := decodeArrayElements(plain, obj["messages"])
	if !ok {
		t.Fatalf("decodeArrayElements failed")
	}
	if g := lastGoalPinInRange(elems, 1, len(elems)); g >= 0 {
		t.Fatalf("no message carries the marker, but lastGoalPinInRange found a goal at %d", g)
	}

	// It still takes the ordinary contiguous path: across budgets it fires at least once, and no
	// fire ever conjures a marker or breaks alternation (the goal branch is unreachable here).
	fired := false
	for _, budget := range []int{100, 200, 400, 800} {
		out, outcome := CompactAnthropicHistoryWithOutcome(plain, budget)
		if outcome.Reason != CompactReasonNone {
			continue
		}
		fired = true
		if strings.Contains(string(out), compactGoalMarker) {
			t.Fatalf("budget=%d: no marker should survive when none was present", budget)
		}
		assertAlternation(t, out)
	}
	if !fired {
		t.Fatalf("the marker-free control body never compacted across the swept budgets")
	}
}

// TestCompactGoalPinToolPairingSurvives puts the goal AND a tool_use/tool_result pair in the
// compactible middle: the goal hoists out, the tool pair either drops together or is kept together,
// never orphaned.
func TestCompactGoalPinToolPairingSurvives(t *testing.T) {
	const nMsgs, goalIdx = 18, 8
	raw := goalMarkedBody(t, nMsgs, goalIdx, true)
	for _, budget := range []int{120, 200, 300, 450} {
		out, outcome := CompactAnthropicHistoryWithOutcome(raw, budget)
		if outcome.Reason != CompactReasonNone {
			continue // identity/bail is always safe
		}
		if !strings.Contains(string(out), goalText(goalIdx)) {
			t.Fatalf("budget=%d: the goal must survive the hoist even with a tool pair present", budget)
		}
		assertAlternation(t, out)
		assertNoOrphanToolResult(t, out, budget)
		if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
			t.Fatalf("budget=%d: decode failed: %v", budget, err)
		}
	}
}

// assertNoOrphanToolResult fails if any tool_result's tool_use was dropped.
func assertNoOrphanToolResult(t *testing.T, out []byte, budget int) {
	t.Helper()
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

// TestCompactGoalPinAlternationSweep is the adversarial witness for the two-boundary hoist logic:
// across message counts, budgets, and goal positions (both user- and assistant-role goals, which
// flip the [stub,goal] vs [goal,stub] ordering), every FIRE must keep the goal and never break
// alternation or the re-decode proof.
func TestCompactGoalPinAlternationSweep(t *testing.T) {
	for _, n := range []int{12, 17, 24, 41} {
		for _, goalIdx := range []int{2, 3, 5, 8} {
			if goalIdx >= n-1 {
				continue
			}
			raw := goalMarkedBody(t, n, goalIdx, false)
			for _, budget := range []int{80, 150, 300, 600} {
				out, outcome := CompactAnthropicHistoryWithOutcome(raw, budget)
				if outcome.Reason != CompactReasonNone {
					continue // identity/bail is always safe
				}
				assertAlternation(t, out)
				assertFirstMessageUser(t, out)
				if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
					t.Fatalf("n=%d goal=%d budget=%d: decode failed: %v", n, goalIdx, budget, err)
				}
				if !strings.Contains(string(out), goalText(goalIdx)) {
					t.Fatalf("n=%d goal=%d budget=%d: a FIRE dropped the pinned goal", n, goalIdx, budget)
				}
			}
		}
	}
}

// assertFirstMessageUser fails if the first messages[] element is not role "user" — Anthropic
// requires the leading turn to be a user turn (a leading assistant turn 400s), and neither
// DecodeAnthropicMessagesRequest nor assertAlternation catches that.
func assertFirstMessageUser(t *testing.T, out []byte) {
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
	if len(ms) == 0 || ms[0].Role != "user" {
		role := "<none>"
		if len(ms) > 0 {
			role = ms[0].Role
		}
		t.Fatalf("first message role = %q, want \"user\" — Anthropic 400s on a leading non-user turn", role)
	}
}

// systemOnlyGoalBody builds a system-only-cache body (no message-level breakpoint ⇒ pfxEnd<0, the
// whole array compactible) with a goal marker at goalIdx (whose natural alternating role is
// user for even / assistant for odd indices).
func systemOnlyGoalBody(t *testing.T, nMsgs, goalIdx int) []byte {
	t.Helper()
	type block map[string]any
	msgs := make([]map[string]any, 0, nMsgs)
	for i := 0; i < nMsgs; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		text := strings.Repeat("body turn ", 20) + itoa(i)
		if i == goalIdx {
			text = compactGoalMarker + " standing instruction: never touch the vendored tree (turn " + itoa(i) + ")"
		}
		msgs = append(msgs, map[string]any{"role": role, "content": []block{{"type": "text", "text": text}}})
	}
	raw, err := json.Marshal(map[string]any{
		"model": "claude", "max_tokens": 512,
		"system":   []block{{"type": "text", "text": "sys", "cache_control": map[string]any{"type": "ephemeral"}}},
		"messages": msgs,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// TestCompactGoalPinSystemOnlyAnchor covers the pfxEnd<0 path: only the system block is cached, so
// the whole message array is compactible and the hoisted pair LEADS the array (no protected message
// precedes it). Swept over a USER-role goal (index 4) and an ASSISTANT-role goal (index 3): the
// goal survives, the array head stays byte-identical, alternation holds, AND the leading turn is a
// user turn (the leading-role invariant a naive [stub,goal] order would violate).
func TestCompactGoalPinSystemOnlyAnchor(t *testing.T) {
	for _, goalIdx := range []int{4 /* user */, 3 /* assistant */} {
		raw := systemOnlyGoalBody(t, 14, goalIdx)
		fired := false
		for _, budget := range []int{150, 250, 400, 600} {
			out, outcome := CompactAnthropicHistoryWithOutcome(raw, budget)
			if outcome.Reason != CompactReasonNone {
				continue // a tight budget that can't seat the kept-window role bails identity — fail safe
			}
			fired = true
			if !strings.Contains(string(out), compactGoalMarker) {
				t.Fatalf("goalIdx=%d budget=%d: the goal must survive under the system-only (pfxEnd<0) anchor", goalIdx, budget)
			}
			assertAlternation(t, out)
			assertFirstMessageUser(t, out) // the leading-role invariant a naive [stub,goal] order would break
			if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
				t.Fatalf("goalIdx=%d budget=%d: system-only goal-pinned body failed to decode: %v", goalIdx, budget, err)
			}
			// The array head (system + `[`) is byte-identical.
			var obj map[string]json.RawMessage
			_ = json.Unmarshal(raw, &obj)
			_, spans, _ := decodeArrayElements(raw, obj["messages"])
			head := arrayContentStart(spans)
			if !bytes.Equal(raw[:head], out[:head]) {
				t.Fatalf("goalIdx=%d budget=%d: array-head prefix changed under a system-only goal hoist", goalIdx, budget)
			}
		}
		if !fired {
			t.Fatalf("goalIdx=%d: never fired across the swept budgets under a system-only anchor", goalIdx)
		}
	}
}

// TestCompactGoalPinHeadAnchor covers the flagship head-anchored path (CompactAnchorHead ⇒ pfxEnd<0,
// the whole array compactible). With ColdCache the burst is zero-penalty so the gate fires. The goal
// survives and — the leading-role invariant — the first turn is a user turn regardless of the
// goal's own role.
func TestCompactGoalPinHeadAnchor(t *testing.T) {
	for _, goalIdx := range []int{4 /* user */, 3 /* assistant */} {
		raw := systemOnlyGoalBody(t, 14, goalIdx)
		fired := false
		for _, budget := range []int{150, 250, 400, 600} {
			out, outcome := CompactAnthropicHistoryWithOptions(raw, CompactOptions{
				Budget: budget, Anchor: CompactAnchorHead, ColdCache: true,
			})
			if outcome.Reason != CompactReasonNone {
				continue
			}
			fired = true
			if !strings.Contains(string(out), compactGoalMarker) {
				t.Fatalf("goalIdx=%d budget=%d: the goal must survive the head-anchored hoist", goalIdx, budget)
			}
			assertAlternation(t, out)
			assertFirstMessageUser(t, out)
			if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
				t.Fatalf("goalIdx=%d budget=%d: head-anchored goal-pinned body failed to decode: %v", goalIdx, budget, err)
			}
		}
		if !fired {
			t.Fatalf("goalIdx=%d: never fired across the swept budgets under the head anchor", goalIdx)
		}
	}
}

// TestCompactGoalWithCacheControlBailsIdentity: a goal turn that itself carries a cache_control
// breakpoint is warm cached history — dropping/moving it would burst the suffix, so the
// conservative action is identity (CompactReasonCachedSpan), exactly as a normal cached-span drop.
func TestCompactGoalWithCacheControlBailsIdentity(t *testing.T) {
	type block map[string]any
	msgs := []map[string]any{
		{"role": "user", "content": []block{{"type": "text", "text": strings.Repeat("head ", 30), "cache_control": map[string]any{"type": "ephemeral"}}}},
		{"role": "assistant", "content": []block{{"type": "text", "text": strings.Repeat("a ", 30)}}},
		// A goal turn carrying its OWN breakpoint (cached history).
		{"role": "user", "content": []block{{"type": "text", "text": compactGoalMarker + " cached goal", "cache_control": map[string]any{"type": "ephemeral"}}}},
		{"role": "assistant", "content": []block{{"type": "text", "text": strings.Repeat("b ", 30)}}},
		{"role": "user", "content": []block{{"type": "text", "text": strings.Repeat("c ", 30)}}},
		{"role": "assistant", "content": []block{{"type": "text", "text": strings.Repeat("d ", 30)}}},
	}
	raw, err := json.Marshal(map[string]any{"model": "claude", "max_tokens": 256, "messages": msgs})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	out, outcome := CompactAnthropicHistoryWithOutcome(raw, 40)
	if outcome.Reason != CompactReasonCachedSpan {
		t.Fatalf("expected CachedSpan identity for a cache_control-bearing goal, got %q", outcome.Reason)
	}
	if !bytes.Equal(out, raw) {
		t.Fatalf("a cached-goal bail must return the body unchanged")
	}
}

// TestGoalMarkerDetection unit-tests the marker predicate: leading marker (with/without leading
// whitespace) and a line-start marker after a preamble are goals; a mid-sentence mention is not; a
// tool-block message is never pinnable even if its text would match.
func TestGoalMarkerDetection(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{compactGoalMarker + " do the thing", true},
		{"   " + compactGoalMarker + " leading whitespace", true},
		{"a preamble line\n" + compactGoalMarker + " marker on its own line", true},
		{"please remember the " + compactGoalMarker + " later", false},
		{"no marker here at all", false},
		{"", false},
	}
	for _, c := range cases {
		if got := isCompactGoalText(c.text); got != c.want {
			t.Errorf("isCompactGoalText(%q) = %v, want %v", c.text, got, c.want)
		}
	}

	// A tool_use message whose text-equivalent would match must NOT be pinnable (elementTextContent
	// returns ok=false for tool blocks), so hoisting can never orphan a tool pair.
	toolMsg := json.RawMessage(`{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"R","input":{}}]}`)
	if isGoalPinnedMessage(toolMsg) {
		t.Errorf("a tool_use message must never be a pinnable goal")
	}
	textGoal := json.RawMessage(`{"role":"user","content":[{"type":"text","text":"` + compactGoalMarker + ` x"}]}`)
	if !isGoalPinnedMessage(textGoal) {
		t.Errorf("a plain-text user turn leading with the marker must be a pinnable goal")
	}
}
