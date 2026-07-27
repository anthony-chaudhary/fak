package sessionaudit

import "testing"

// withToolID is withTool plus the block id the API assigns to every tool call — the field
// the split-record accounting keys on.
func withToolID(name, id string) func(map[string]any) {
	return func(rec map[string]any) {
		msg := rec["message"].(map[string]any)
		blocks, _ := msg["content"].([]any)
		msg["content"] = append(blocks, map[string]any{
			"type": "tool_use", "name": name, "id": id, "input": map[string]any{},
		})
	}
}

// THE REGRESSION THIS FILE EXISTS FOR. Claude Code writes one assistant response as
// several records sharing a message id, and the later ones are where the tool calls
// actually live. The message-id dedup guard must still suppress the turn and its tokens —
// those really are repeated — without taking the tool calls down with them.
//
// Census over 712 transcripts before the fix: 16,289 of 18,776 tool_use blocks (86.8%)
// were on a record the dedup discarded.
func TestSplitAssistantRecordsKeepTheirToolCalls(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("m1", 500, 0, 0),
		assistantRecord("m1", 500, 0, 0, withToolID("Bash", "t1")),
		assistantRecord("m1", 500, 0, 0, withToolID("Agent", "t2")),
	}))
	if s.Error != "" {
		t.Fatal(s.Error)
	}
	// The dedup still holds where it should.
	if s.AssistantTurns != 1 {
		t.Fatalf("assistant turns = %d, want 1 — the split records are ONE turn", s.AssistantTurns)
	}
	if s.Tokens.Output != 500 {
		t.Fatalf("output = %d, want 500 — usage is repeated on every split record and must count once", s.Tokens.Output)
	}
	if s.DupAssistantLines != 2 {
		t.Fatalf("dup assistant lines = %d, want 2", s.DupAssistantLines)
	}
	// ...and the tool calls survive it.
	if s.Tools["Bash"] != 1 || s.Tools["Agent"] != 1 {
		t.Fatalf("tools = %v, want one Bash and one Agent: a call on a split record is a real call", s.Tools)
	}
	if s.NToolUse != 2 {
		t.Fatalf("tool_use count = %d, want 2", s.NToolUse)
	}
}

// The block id is what makes counting them safe: 41 of 18,790 blocks in the same census
// really were repeated across records, so the same call must not be counted twice.
func TestARepeatedToolUseBlockIsCountedOnce(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("m1", 10, 0, 0, withToolID("Bash", "t1")),
		assistantRecord("m1", 10, 0, 0, withToolID("Bash", "t1")),
	}))
	if s.Tools["Bash"] != 1 {
		t.Fatalf("Bash calls = %d, want 1 — the same block id is the same call", s.Tools["Bash"])
	}
	if s.NToolUse != 1 {
		t.Fatalf("tool_use count = %d, want 1", s.NToolUse)
	}
}

// The rescue is fail-closed. On a split record an id-less block is skipped, because
// nothing distinguishes it from a repeat of a block already counted and inventing tool
// calls is the worse error. The census found no id-less blocks at all, so this costs
// nothing real — and it is what keeps a genuinely repeated record from inflating the tool
// mix (see TestDuplicateBilledTurnCountedOnce, which asserts exactly that).
func TestIDLessToolCallsOnASplitRecordAreNotGuessedAt(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("m1", 10, 0, 0, withTool("Read")),
		assistantRecord("m1", 10, 0, 0, withTool("Read")),
	}))
	if s.Tools["Read"] != 1 {
		t.Fatalf("Read calls = %d, want 1 — only the first record's call is provably a call", s.Tools["Read"])
	}
	if s.NToolUse != 1 {
		t.Fatalf("tool_use count = %d, want 1", s.NToolUse)
	}
}

// WHY THIS MATTERS BEYOND THE TOOL-MIX TABLE. The delegation fold cross-examines the
// `isSidechain` marker with spawn-tool calls, so an uninstrumented corpus reports UNKNOWN
// instead of a 0% it did not earn. Spawn calls ride on split records like every other tool
// call, so while they were being dropped that safeguard was reading zero spawns and
// handing back exactly the unearned zero it exists to refuse.
func TestASpawnCallOnASplitRecordStillEarnsTheUnknown(t *testing.T) {
	s := Analyze(writeTranscript(t, []map[string]any{
		assistantRecord("m1", 900, 0, 0),
		assistantRecord("m1", 900, 0, 0, withToolID("Agent", "t1")),
	}))
	agg := AggregateSessions([]Session{s})
	d := FoldDelegationShare(agg.PerTrack, agg.ToolMix)

	if d.SpawnCalls != 1 {
		t.Fatalf("spawn calls = %d, want 1 — the Agent call is on the split record", d.SpawnCalls)
	}
	if !d.UnderInstrumented {
		t.Fatal("a corpus that spawned work and marked no turn is under-instrumented")
	}
	if d.OutputShare != nil {
		t.Fatalf("delegated share = %v, want UNKNOWN", *d.OutputShare)
	}
}
