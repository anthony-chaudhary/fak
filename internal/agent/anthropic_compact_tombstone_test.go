package agent

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// anthropic_compact_tombstone_test.go — witnesses for the originating-task tombstone. When
// compaction drops the session's FIRST user turn (the de-facto task) and no [fak:goal] pin covers
// it, the synthetic stub carries a bounded, one-line excerpt of that turn instead of laundering it
// into a bare "[fak] compacted N earlier turn(s) ... detail is omitted" count. This is the
// automatic, lossy counterpart to the verbatim goal pin, and the fix for the model-switch symptom
// where a resuming model finds the stub and no longer knows what it was asked to do.

// tombstoneTaskBody builds a system-only-cache body (no message-level breakpoint ⇒ the whole array
// is compactible, so index 0 — the originating task — is droppable) whose FIRST user turn carries
// `task` and every other turn carries unique droppable filler.
func tombstoneTaskBody(t *testing.T, nMsgs int, task string) []byte {
	t.Helper()
	type block map[string]any
	msgs := make([]map[string]any, 0, nMsgs)
	for i := 0; i < nMsgs; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		text := strings.Repeat("droppable filler turn ", 20) + itoa(i)
		if i == 0 {
			text = task // the originating task, a leading user turn
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

// stubContentText returns the decoded text content of the compaction stub in a rewritten body, or
// fails if none is present. Decoding first sidesteps the JSON/strconv escaping of the excerpt.
func stubContentText(t *testing.T, out []byte) string {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(out, &obj); err != nil {
		t.Fatalf("unmarshal out: %v", err)
	}
	elems, _, ok := decodeArrayElements(out, obj["messages"])
	if !ok {
		t.Fatalf("decodeArrayElements failed")
	}
	for _, el := range elems {
		if txt, ok := elementTextContent(el); ok && strings.Contains(txt, compactStubPrefix) {
			return txt
		}
	}
	t.Fatalf("no stub message found in the compacted body")
	return ""
}

// TestCompactTombstonesOriginatingTask is the headline: across both firing anchors (the system-only
// firstBP path and the head-anchored path), a dropped originating task leaves a bounded excerpt in
// the stub, whitespace-collapsed, while alternation / decode / cache-head are all preserved.
func TestCompactTombstonesOriginatingTask(t *testing.T) {
	const task = "Please  rotate the auth tokens\nand update the runbook — this is the ORIGINATING task."
	const wantExcerpt = "rotate the auth tokens and update the runbook" // whitespace-collapsed, no newline
	raw := tombstoneTaskBody(t, 14, task)

	// The array head (system + `[`) that must survive byte-for-byte.
	var obj map[string]json.RawMessage
	_ = json.Unmarshal(raw, &obj)
	_, spans, _ := decodeArrayElements(raw, obj["messages"])
	head := arrayContentStart(spans)

	check := func(name string, out []byte) {
		content := stubContentText(t, out)
		if !strings.Contains(content, compactTombstonePrefix) {
			t.Fatalf("%s: stub carries no tombstone line: %q", name, content)
		}
		if !strings.Contains(content, wantExcerpt) {
			t.Fatalf("%s: tombstone missing the whitespace-collapsed excerpt; got %q", name, content)
		}
		if strings.Contains(content, "\nand update") {
			t.Fatalf("%s: excerpt kept a raw newline instead of collapsing it: %q", name, content)
		}
		assertAlternation(t, out)
		assertFirstMessageUser(t, out)
		if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
			t.Fatalf("%s: compacted body failed to decode: %v", name, err)
		}
		if !bytes.Equal(raw[:head], out[:head]) {
			t.Fatalf("%s: array-head prefix changed — the tombstone must not touch cached bytes", name)
		}
	}

	firedSystem, firedHead := false, false
	for _, budget := range []int{120, 200, 350, 550} {
		if out, oc := CompactAnthropicHistoryWithOutcome(raw, budget); oc.Reason == CompactReasonNone {
			firedSystem = true
			check("system-only", out)
		}
		if out, oc := CompactAnthropicHistoryWithOptions(raw, CompactOptions{Budget: budget, Anchor: CompactAnchorHead, ColdCache: true}); oc.Reason == CompactReasonNone {
			firedHead = true
			check("head-anchor", out)
		}
	}
	if !firedSystem {
		t.Fatalf("system-only path never fired across the swept budgets")
	}
	if !firedHead {
		t.Fatalf("head-anchored path never fired across the swept budgets")
	}
}

// TestCompactNoTombstoneWhenTaskProtected proves the "no change when the task survives" invariant:
// realisticBody protects msg 0 (it carries the message breakpoint), so the originating task is in
// the verbatim prefix and no tombstone is emitted — the stub is the bare count, byte-identical to
// the pre-tombstone behavior.
func TestCompactNoTombstoneWhenTaskProtected(t *testing.T) {
	raw := realisticBody(t, 16)
	fired := false
	for _, budget := range []int{150, 250, 500} {
		out, oc := CompactAnthropicHistoryWithOutcome(raw, budget)
		if oc.Reason != CompactReasonNone {
			continue
		}
		fired = true
		if strings.Contains(string(out), compactTombstonePrefix) {
			t.Fatalf("budget=%d: the task is in the protected prefix — no tombstone should be emitted", budget)
		}
	}
	if !fired {
		t.Fatalf("realisticBody never compacted across the swept budgets")
	}
}

// TestCompactGoalPathNoTombstone proves the goal path stays the bare-count stub: a marked goal is
// hoisted verbatim (the fidelity path), so the stub carries no redundant excerpt.
func TestCompactGoalPathNoTombstone(t *testing.T) {
	raw := systemOnlyGoalBody(t, 14, 4)
	fired := false
	for _, budget := range []int{150, 250, 400} {
		out, oc := CompactAnthropicHistoryWithOutcome(raw, budget)
		if oc.Reason != CompactReasonNone {
			continue
		}
		fired = true
		if !strings.Contains(string(out), compactGoalMarker) {
			t.Fatalf("budget=%d: the goal must survive", budget)
		}
		if strings.Contains(string(out), compactTombstonePrefix) {
			t.Fatalf("budget=%d: the goal path preserves the task verbatim — it must not also emit a tombstone", budget)
		}
	}
	if !fired {
		t.Fatalf("systemOnlyGoalBody never compacted across the swept budgets")
	}
}

// TestOriginatingTaskDigest unit-tests the range/collapse logic directly.
func TestOriginatingTaskDigest(t *testing.T) {
	mk := func(role, text string) json.RawMessage {
		b, _ := json.Marshal(map[string]any{"role": role, "content": text})
		return b
	}
	elems := []json.RawMessage{
		mk("user", "  do the   thing\nacross\tlines  "),
		mk("assistant", "ok"),
		mk("user", "follow up"),
	}
	if got := originatingTaskDigest(elems, 0, 3); got != "do the thing across lines" {
		t.Fatalf("collapse/excerpt = %q", got)
	}
	// Task protected (index 0 sits before the drop range) ⇒ empty, so the stub is unchanged.
	if got := originatingTaskDigest(elems, 1, 3); got != "" {
		t.Fatalf("protected task should yield empty, got %q", got)
	}
	// Empty range ⇒ empty.
	if got := originatingTaskDigest(elems, 0, 0); got != "" {
		t.Fatalf("empty range should yield empty, got %q", got)
	}
	// A leading [fak:goal] marker is stripped (defensive — the marked case is hoisted, not dropped).
	marked := []json.RawMessage{mk("user", compactGoalMarker+"  the standing task")}
	if got := originatingTaskDigest(marked, 0, 1); got != "the standing task" {
		t.Fatalf("marker strip = %q", got)
	}
}

// TestOriginatingTaskDigestTruncates: a long task is bounded to compactTombstoneCap runes + ellipsis.
func TestOriginatingTaskDigestTruncates(t *testing.T) {
	long := strings.Repeat("x", compactTombstoneCap+50)
	b, _ := json.Marshal(map[string]any{"role": "user", "content": long})
	got := originatingTaskDigest([]json.RawMessage{b}, 0, 1)
	if r := []rune(got); len(r) != compactTombstoneCap+1 {
		t.Fatalf("truncated length = %d runes, want %d (cap + ellipsis)", len(r), compactTombstoneCap+1)
	}
	if !strings.HasSuffix(got, "…") {
		t.Fatalf("truncated excerpt must end with an ellipsis: %q", got)
	}
}

// TestOriginatingTaskDigestSkipsToolResult: a tool_result-only leading turn is never an originating
// task, so the digest declines it (leaving the bare stub) rather than excerpting tool blocks.
func TestOriginatingTaskDigestSkipsToolResult(t *testing.T) {
	b, _ := json.Marshal(map[string]any{
		"role":    "user",
		"content": []map[string]any{{"type": "tool_result", "tool_use_id": "t1", "content": "x"}},
	})
	if got := originatingTaskDigest([]json.RawMessage{b}, 0, 1); got != "" {
		t.Fatalf("tool_result turn must not be excerpted, got %q", got)
	}
}

// TestTruncateRunes covers the rune-boundary helper.
func TestTruncateRunes(t *testing.T) {
	if got := truncateRunes("héllo", 10); got != "héllo" {
		t.Fatalf("no-trim = %q", got)
	}
	if got := truncateRunes("héllo", 3); got != "hél…" {
		t.Fatalf("trim = %q", got)
	}
}
