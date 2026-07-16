package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// breakpointBody builds a /v1/messages body with nMsgs alternating turns, a cache_control
// breakpoint on each message index in markMsgs, and optional top-level system/tools head
// marks. It is the inbound-shape fixture for the #2786 layout recorder.
func breakpointBody(t *testing.T, nMsgs int, markMsgs []int, markSystem, markTools bool) []byte {
	t.Helper()
	marked := map[int]bool{}
	for _, i := range markMsgs {
		marked[i] = true
	}
	msgs := make([]map[string]any, 0, nMsgs)
	for i := 0; i < nMsgs; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		blk := map[string]any{"type": "text", "text": strings.Repeat("turn body. ", 5)}
		if marked[i] {
			blk["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		msgs = append(msgs, map[string]any{"role": role, "content": []map[string]any{blk}})
	}
	body := map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"messages":   msgs,
	}
	sysBlk := map[string]any{"type": "text", "text": "You are a coding agent."}
	if markSystem {
		sysBlk["cache_control"] = map[string]any{"type": "ephemeral"}
	}
	body["system"] = []map[string]any{sysBlk}
	toolBlk := map[string]any{"name": "Read", "description": "read a file"}
	if markTools {
		toolBlk["cache_control"] = map[string]any{"type": "ephemeral"}
	}
	body["tools"] = []map[string]any{toolBlk}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	return raw
}

// TestRecordInboundBreakpointsReflectsMarks is the #2786 acceptance-gate witness: the recorded
// breakpoint_positions must correctly reflect the inbound body's cache-control marks — the
// exact indices and roles that were marked, and nothing else.
func TestRecordInboundBreakpointsReflectsMarks(t *testing.T) {
	// The real Claude Code shape the issue names: a static head AND a recent turn.
	raw := breakpointBody(t, 8, []int{0, 6}, true, false)
	got, ok := RecordInboundBreakpoints(raw)
	if !ok {
		t.Fatalf("RecordInboundBreakpoints: ok=false on a well-formed body")
	}
	if got.Count != 2 {
		t.Fatalf("Count = %d, want 2 (marked indices 0 and 6)", got.Count)
	}
	if got.Messages != 8 {
		t.Errorf("Messages = %d, want 8", got.Messages)
	}
	if !got.SystemMarked {
		t.Errorf("SystemMarked = false, want true (the static head is marked)")
	}
	if got.ToolsMarked {
		t.Errorf("ToolsMarked = true, want false (tools was left unmarked)")
	}
	want := []BreakpointPosition{{Index: 0, Role: "user"}, {Index: 6, Role: "user"}}
	for i, w := range want {
		if got.Positions[i] != w {
			t.Errorf("Positions[%d] = %+v, want %+v", i, got.Positions[i], w)
		}
	}
	// An UNMARKED body must record zero positions — the recorder reports the marks that are
	// there, never a mark that is not.
	bare, ok := RecordInboundBreakpoints(breakpointBody(t, 5, nil, false, false))
	if !ok {
		t.Fatalf("RecordInboundBreakpoints: ok=false on an unmarked body")
	}
	if bare.Count != 0 || len(bare.Positions) != 0 {
		t.Errorf("unmarked body recorded %d position(s), want 0: %+v", bare.Count, bare.Positions)
	}
}

// TestRecordInboundBreakpointsRolesAndOrder pins that a breakpoint's ROLE travels with its
// index and that Positions stays ascending — the layout is useless for a range comparison if
// the order is not stable.
func TestRecordInboundBreakpointsRolesAndOrder(t *testing.T) {
	raw := breakpointBody(t, 6, []int{5, 1, 3}, false, true)
	got, ok := RecordInboundBreakpoints(raw)
	if !ok {
		t.Fatalf("RecordInboundBreakpoints: ok=false")
	}
	if !got.ToolsMarked {
		t.Errorf("ToolsMarked = false, want true")
	}
	want := []BreakpointPosition{
		{Index: 1, Role: "assistant"},
		{Index: 3, Role: "assistant"},
		{Index: 5, Role: "assistant"},
	}
	if got.Count != len(want) {
		t.Fatalf("Count = %d, want %d", got.Count, len(want))
	}
	for i, w := range want {
		if got.Positions[i] != w {
			t.Errorf("Positions[%d] = %+v, want %+v", i, got.Positions[i], w)
		}
	}
}

// readerCachedThrough is the DOWNSTREAM reader's derivation, written here exactly as a ledger
// consumer would write it: the last messages[] index covered by a live breakpoint's cached
// prefix, computed from the raw recorded layout alone (a breakpoint at i caches messages[0..i]).
// It returns -1 when no MESSAGE breakpoint exists — the honest answer even when the stable
// system/tools head is marked, since that head covers no message index. Keeping it in the test
// (not the recorder) is the #2786 scope fence: the record states the marks, the reader rules.
func readerCachedThrough(b InboundBreakpoints) int {
	cachedThrough := -1
	for _, p := range b.Positions {
		if p.Index > cachedThrough {
			cachedThrough = p.Index
		}
	}
	return cachedThrough
}

// TestInboundBreakpointsAnswersShedSpanCached is the #2786 Done-condition witness: a reader can
// determine, per session, whether a shed span sat inside a LIVE cached prefix — using only the
// recorded layout plus the drop range from the same fire, with no help from this package.
//
// A drop of [dropStart, keepStart) was inside a live cached prefix iff the highest recorded
// breakpoint Index >= keepStart-1 (the last dropped index).
func TestInboundBreakpointsAnswersShedSpanCached(t *testing.T) {
	shedWasCached := func(b InboundBreakpoints, keepStart int) bool {
		return readerCachedThrough(b) >= keepStart-1
	}

	// Breakpoint at index 6 caches messages[0..6]: a drop of [1,5) is wholly inside it.
	warm, ok := RecordInboundBreakpoints(breakpointBody(t, 8, []int{0, 6}, true, false))
	if !ok {
		t.Fatalf("record warm body: ok=false")
	}
	if got := readerCachedThrough(warm); got != 6 {
		t.Fatalf("reader derived cached-through = %d, want 6", got)
	}
	if !shedWasCached(warm, 5) {
		t.Errorf("drop [1,5) under a breakpoint at 6: reader concluded COLD, want cached")
	}

	// Only an EARLY breakpoint at index 1: a drop of [2,6) sits past the cached prefix.
	cold, ok := RecordInboundBreakpoints(breakpointBody(t, 8, []int{1}, true, false))
	if !ok {
		t.Fatalf("record cold body: ok=false")
	}
	if got := readerCachedThrough(cold); got != 1 {
		t.Fatalf("reader derived cached-through = %d, want 1", got)
	}
	if shedWasCached(cold, 6) {
		t.Errorf("drop [2,6) past a breakpoint at 1: reader concluded CACHED, want cold")
	}

	// A HEAD-only body caches no message index, so no shed span is inside a message
	// breakpoint's prefix — the record must not let a reader infer coverage from a head mark.
	head, ok := RecordInboundBreakpoints(breakpointBody(t, 6, nil, true, true))
	if !ok {
		t.Fatalf("record head-only body: ok=false")
	}
	if !head.SystemMarked || !head.ToolsMarked {
		t.Fatalf("head marks not recorded: %+v", head)
	}
	if got := readerCachedThrough(head); got != -1 {
		t.Errorf("head-only reader derived cached-through = %d, want -1 (no message is covered)", got)
	}
	if shedWasCached(head, 1) {
		t.Errorf("head-only body: reader concluded a shed span was CACHED, want cold")
	}
}

// TestRecordInboundBreakpointsFailSafe pins the unreadable-body posture: ok=false yields NO
// claim, so an unparseable body can never be persisted as a confident zero-breakpoint session.
func TestRecordInboundBreakpointsFailSafe(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  []byte
	}{
		{"not json", []byte("this is not json")},
		{"json array not object", []byte(`[{"role":"user"}]`)},
		{"no messages key", []byte(`{"model":"claude-sonnet-4-6"}`)},
		{"messages not an array", []byte(`{"messages":"nope"}`)},
		{"empty", nil},
	} {
		if got, ok := RecordInboundBreakpoints(tc.raw); ok {
			t.Errorf("%s: ok=true, want false (got %+v)", tc.name, got)
		}
	}
	// A well-formed body with an EMPTY messages array is readable: "we looked, there were
	// none" (ok=true, Count 0) must not collapse into the unreadable case.
	if got, ok := RecordInboundBreakpoints([]byte(`{"messages":[]}`)); !ok || got.Count != 0 || got.Messages != 0 {
		t.Errorf("empty messages array: ok=%v got=%+v, want ok=true with a zero-count record", ok, got)
	}
}

// TestInboundBreakpointsJSONFieldNames pins the durable field names the cachevaluereport lane
// persists and reads: `breakpoint_positions` (with per-position index/role) is the name #2786
// specifies, so a ledger writer and reader cannot drift from the contract.
func TestInboundBreakpointsJSONFieldNames(t *testing.T) {
	rec, ok := RecordInboundBreakpoints(breakpointBody(t, 4, []int{2}, true, false))
	if !ok {
		t.Fatalf("RecordInboundBreakpoints: ok=false")
	}
	blob, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal record: %v", err)
	}
	for _, needle := range []string{
		`"breakpoint_positions"`,
		`"breakpoint_count":1`,
		`"index":2`,
		`"role":"user"`,
		`"messages":4`,
		`"system_marked":true`,
	} {
		if !strings.Contains(string(blob), needle) {
			t.Errorf("marshalled record missing %s\n got: %s", needle, blob)
		}
	}
	var back InboundBreakpoints
	if err := json.Unmarshal(blob, &back); err != nil {
		t.Fatalf("unmarshal record: %v", err)
	}
	// The persisted row must be as answerable as the in-memory record: a reader restoring it
	// from the ledger derives the same cached extent, or the durable shape lost the fact.
	if readerCachedThrough(back) != readerCachedThrough(rec) {
		t.Errorf("round-trip reader cached-through = %d, want %d", readerCachedThrough(back), readerCachedThrough(rec))
	}
}

// TestRecordInboundBreakpointsDoesNotPerturbCompaction pins the distinct-data-path fence: the
// recorder is read-only, so compaction's byte-level result is identical whether or not the
// layout was recorded. If this ever fails, recording has developed a side effect on the fire
// path and the cache guarantee is no longer a bytes.Equal.
func TestRecordInboundBreakpointsDoesNotPerturbCompaction(t *testing.T) {
	raw := realisticBody(t, 40)
	before, outcomeBefore := CompactAnthropicHistoryWithOutcome(raw, 200)
	if _, ok := RecordInboundBreakpoints(raw); !ok {
		t.Fatalf("RecordInboundBreakpoints: ok=false on the realistic body")
	}
	after, outcomeAfter := CompactAnthropicHistoryWithOutcome(raw, 200)
	if string(before) != string(after) {
		t.Errorf("compaction output changed after recording the layout (recorder is not read-only)")
	}
	// CompactOutcome carries []byte fields, so compare the scalar verdict axes rather than
	// the struct itself.
	if outcomeBefore.Reason != outcomeAfter.Reason ||
		outcomeBefore.Dropped != outcomeAfter.Dropped ||
		outcomeBefore.ShedTokens != outcomeAfter.ShedTokens {
		t.Errorf("compaction outcome changed after recording: %+v vs %+v", outcomeBefore, outcomeAfter)
	}
}
