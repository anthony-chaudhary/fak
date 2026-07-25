package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A compact Claude Code transcript fixture. Each line is one JSONL record. The load-
// bearing shape being exercised: one assistant response is split across several records
// stamped with the SAME message id, so the parser must collapse them into one turn.
const batchFixture = `
{"type":"assistant","message":{"id":"m1","content":[{"type":"text","text":"looking"}]}}
{"type":"assistant","message":{"id":"m1","content":[{"type":"tool_use","name":"Read"},{"type":"tool_use","name":"Read"}]}}
{"type":"user","message":{"content":[{"type":"tool_result","content":"ok"},{"type":"tool_result","content":"ok"}]}}
{"type":"assistant","message":{"id":"m2","content":[{"type":"tool_use","name":"Bash"}]}}
{"type":"user","message":{"content":[{"type":"tool_result","content":"done"}]}}
{"type":"assistant","message":{"id":"m3","content":[{"type":"text","text":"all set"}]}}
{"type":"assistant","isSidechain":true,"message":{"id":"s1","content":[{"type":"tool_use","name":"Grep"},{"type":"tool_use","name":"Grep"}]}}
`

func TestParseTranscriptTurns_CollapsesSplitsAndSkipsSidechain(t *testing.T) {
	rows, err := ParseTranscriptTurns(strings.NewReader(batchFixture))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	// Three logical turns: m1 (split text + 2 tool_use → 2 calls), m2 (1 call), m3
	// (text only → 0). The sidechain record contributes nothing.
	want := []int{2, 1, 0}
	if len(rows) != len(want) {
		t.Fatalf("turns = %d, want %d (rows=%v)", len(rows), len(want), rows)
	}
	for i, w := range want {
		if rows[i].ToolCalls != w {
			t.Errorf("turn %d tool calls = %d, want %d", i, rows[i].ToolCalls, w)
		}
	}
}

func TestFoldTurnBatch_KPI(t *testing.T) {
	rows, err := ParseTranscriptTurns(strings.NewReader(batchFixture))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	s := FoldTurnBatch(rows)
	if s.Turns != 3 || s.ToolCalls != 3 || s.ToolTurns != 2 || s.BatchedTurns != 1 {
		t.Fatalf("counts = %+v", s)
	}
	// 3 calls / 3 turns = 1.0 per assistant turn; 1 of 2 tool turns batched = 0.5.
	if s.ToolCallsPerAssistantTurn != 1.0 {
		t.Errorf("tool_calls_per_assistant_turn = %v, want 1.0", s.ToolCallsPerAssistantTurn)
	}
	if s.BatchedTurnRate != 0.5 {
		t.Errorf("batched_turn_rate = %v, want 0.5", s.BatchedTurnRate)
	}
}

func TestFoldTurnBatch_ZeroDenominatorsAreSafe(t *testing.T) {
	// Empty transcript: no turns, no NaN.
	if s := FoldTurnBatch(nil); s.Turns != 0 || s.ToolCallsPerAssistantTurn != 0 || s.BatchedTurnRate != 0 {
		t.Fatalf("empty fold = %+v, want all zero", s)
	}
	// A session that called no tool: per-turn mean is 0 and the batched rate stays 0
	// (no tool turns to divide by), never a divide-by-zero.
	s := FoldTurnBatch([]TurnBatchRow{{ToolCalls: 0}, {ToolCalls: 0}})
	if s.Turns != 2 || s.ToolTurns != 0 || s.BatchedTurnRate != 0 || s.ToolCallsPerAssistantTurn != 0 {
		t.Fatalf("no-tool fold = %+v", s)
	}
}

func TestParseTranscriptTurns_EmptyIDSplitsStaySeparate(t *testing.T) {
	// Two consecutive assistant records with no id (and no requestId) must NOT merge:
	// each is its own turn, matching the reference script's `mid is not None` guard.
	fixture := `
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read"}]}}
`
	rows, err := ParseTranscriptTurns(strings.NewReader(fixture))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("turns = %d, want 2 (empty ids must not merge): %v", len(rows), rows)
	}
}

func TestParseTranscriptTurns_RequestIDFallbackAndTornLine(t *testing.T) {
	// The two split records share a requestId (message.id absent) → one turn, 2 calls.
	// The middle line is torn JSON and must be skipped, not fatal.
	fixture := `
{"type":"assistant","requestId":"r1","message":{"content":[{"type":"tool_use","name":"Read"}]}}
{not valid json
{"type":"assistant","requestId":"r1","message":{"content":[{"type":"tool_use","name":"Edit"}]}}
`
	rows, err := ParseTranscriptTurns(strings.NewReader(fixture))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(rows) != 1 || rows[0].ToolCalls != 2 {
		t.Fatalf("rows = %v, want one turn with 2 calls", rows)
	}
}

func TestScanTranscriptBatch_File(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "session.jsonl")
	if err := os.WriteFile(path, []byte(batchFixture), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	s, err := ScanTranscriptBatch(path)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if s.Turns != 3 || s.ToolCalls != 3 || s.BatchedTurns != 1 {
		t.Fatalf("scan stats = %+v", s)
	}
	if _, err := ScanTranscriptBatch(filepath.Join(dir, "missing.jsonl")); err == nil {
		t.Fatal("expected an error opening a missing transcript")
	}
}

func TestTurnBatchStats_JSONFieldNames(t *testing.T) {
	// The KPI names the issue (#5019) requires must survive marshaling verbatim so a
	// dispatch/witness metrics consumer keys on them.
	b, err := json.Marshal(FoldTurnBatch([]TurnBatchRow{{ToolCalls: 2}}))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, key := range []string{"tool_calls_per_assistant_turn", "batched_turn_rate", "batched_turns", "tool_turns"} {
		if !strings.Contains(string(b), key) {
			t.Errorf("marshaled KPI missing %q: %s", key, b)
		}
	}
	if TurnBatchSchema != "fak.turnbatch.v1" {
		t.Errorf("schema = %q", TurnBatchSchema)
	}
}
