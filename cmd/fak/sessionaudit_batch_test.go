package main

// Tests for `fak session-audit batch` (#5799) — the first tracked, non-test consumer of
// the internal/agent turn-batching KPI. #5019 shipped the fold and closed while nothing
// ever called it, so these tests pin the two properties that keep it from going orphan
// again: a session must PERSIST a fak.turnbatch.v1 reading through the verb (not a bespoke
// scan), and the headline rate must stay numerically comparable to the codex-side
// tool_calls_per_turn rather than drifting into a second bespoke number.

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codexlifecycle"
)

// batchTranscript is one Claude Code transcript whose shape makes the fold's load-bearing
// turn-collapsing observable: m1 is ONE assistant response split across two JSONL records
// (text block, then a two-tool_use block) under the same message id. A parser that failed
// to merge the split would report 4 turns and dilute every rate, so the expected numbers
// below are only reachable with the collapse.
//
// It also carries the two records the durability contract must survive: a sidechain
// (subagent) record whose three calls belong to another track, and a torn non-JSON line.
const batchTranscript = `{"type":"assistant","message":{"id":"m1","content":[{"type":"text","text":"planning"}]}}
{"type":"assistant","message":{"id":"m1","content":[{"type":"tool_use","name":"Read"},{"type":"tool_use","name":"Grep"}]}}
{"type":"user","message":{"content":[{"type":"tool_result"}]}}
{"type":"assistant","message":{"id":"m2","content":[{"type":"tool_use","name":"Bash"}]}}
{"type":"user","message":{"content":[{"type":"tool_result"}]}}
{"type":"assistant","message":{"id":"m3","content":[{"type":"text","text":"done"}]}}
{"type":"assistant","isSidechain":true,"message":{"id":"s1","content":[{"type":"tool_use"},{"type":"tool_use"},{"type":"tool_use"}]}}
{"type":"assistant","message":{"id":"m4","conte`

// wantBatch is what batchTranscript folds to: turns m1=[2 calls], m2=[1], m3=[0 text-only].
// tool_calls_per_assistant_turn = 3/3 = 1.0 (all turns), tool_calls_per_tool_turn = 3/2 =
// 1.5 (tool-calling turns only), batched_turn_rate = 1/2 = 0.5.
var wantBatch = agent.TurnBatchStats{
	Turns: 3, ToolCalls: 3, ToolTurns: 2, BatchedTurns: 1,
	ToolCallsPerAssistantTurn: 1, BatchedTurnRate: .5,
}

func writeBatchTranscript(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(batchTranscript+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func readLedger(t *testing.T, path string) []turnBatchRow {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var rows []turnBatchRow
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var row turnBatchRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("ledger line is not a row: %v (%s)", err, line)
		}
		rows = append(rows, row)
	}
	return rows
}

// TestSessionAuditBatchPersistsReading is the #5799 acceptance witness: one session, one
// verb invocation, one durable fak.turnbatch.v1 row — no bespoke scan. The second run
// proves the ledger APPENDS rather than truncating, which is what makes the series usable
// for the before/after cadence comparison the epic needs.
func TestSessionAuditBatchPersistsReading(t *testing.T) {
	path := writeBatchTranscript(t, "sess-abc.jsonl")
	ledger := filepath.Join(t.TempDir(), "turnbatch.jsonl")
	var stdout, stderr bytes.Buffer

	for i := 0; i < 2; i++ {
		if code := runSessionAuditBatch(&stdout, &stderr, []string{"--ledger", ledger, path}); code != 0 {
			t.Fatalf("run %d: exit=%d stderr=%s", i, code, stderr.String())
		}
	}

	rows := readLedger(t, ledger)
	if len(rows) != 2 {
		t.Fatalf("ledger has %d rows, want 2 appended", len(rows))
	}
	row := rows[0]
	if row.Schema != agent.TurnBatchSchema {
		t.Fatalf("schema=%q want %q", row.Schema, agent.TurnBatchSchema)
	}
	if row.TurnBatchStats != wantBatch {
		t.Fatalf("stats=%+v want %+v", row.TurnBatchStats, wantBatch)
	}
	if row.ToolCallsPerToolTurn != 1.5 {
		t.Fatalf("tool_calls_per_tool_turn=%v want 1.5", row.ToolCallsPerToolTurn)
	}
	// Evidence-minimal: the session id is the bare base name, and no transcript text
	// (prompt, tool name, tool result) may reach the durable ledger.
	if row.Session != "sess-abc" {
		t.Fatalf("session=%q want %q", row.Session, "sess-abc")
	}
	raw, _ := json.Marshal(row)
	for _, leaked := range []string{"planning", "done", "Grep", "tool_result", path} {
		if bytes.Contains(raw, []byte(leaked)) {
			t.Fatalf("row leaked transcript text %q: %s", leaked, raw)
		}
	}
	if _, err := time.Parse(time.RFC3339, row.RecordedAt); err != nil {
		t.Fatalf("recorded_at=%q: %v", row.RecordedAt, err)
	}
	if !strings.Contains(stdout.String(), "per_tool_turn=1.5") {
		t.Fatalf("stdout does not state the per-tool-turn rate: %s", stdout.String())
	}
}

// TestSessionAuditBatchHeadlineMatchesCodex holds the cross-engine comparability the issue
// asks for: tool_calls_per_assistant_turn must equal what internal/codexlifecycle computes
// for the SAME (turns, tool_calls) pair, so the Claude and codex readings sit on one axis.
// It calls the real codex fold rather than mirroring its formula — a drift in either
// rounding or denominator reds this test instead of silently splitting the metric in two.
func TestSessionAuditBatchHeadlineMatchesCodex(t *testing.T) {
	for _, tc := range []struct{ turns, calls int }{{3, 3}, {3, 5}, {7, 2}, {4, 9}} {
		row := foldTurnBatchRow("s", agent.FoldTurnBatch(spreadCalls(tc.turns, tc.calls)), time.Unix(0, 0))
		codex := codexlifecycle.RollUp([]codexlifecycle.SessionStats{{
			Session: "s", Turns: tc.turns, ToolCalls: tc.calls, ZeroTool: map[string]int{},
		}}, 0)
		if row.ToolCallsPerAssistantTurn != codex.Totals.ToolCallsPerTurn {
			t.Fatalf("turns=%d calls=%d: turnbatch=%v codex=%v — denominator or rounding drifted apart",
				tc.turns, tc.calls, row.ToolCallsPerAssistantTurn, codex.Totals.ToolCallsPerTurn)
		}
	}
}

// spreadCalls builds `turns` rows carrying `calls` tool calls in total, so a fold over them
// has exactly the totals the codex side is handed.
func spreadCalls(turns, calls int) []agent.TurnBatchRow {
	rows := make([]agent.TurnBatchRow, turns)
	for i := 0; i < calls; i++ {
		rows[i%turns].ToolCalls++
	}
	return rows
}

// TestFoldTurnBatchRowGuardsZeroDenominator pins the derived rate this verb adds on top of
// the pure fold. A session that never called a tool has a zero denominator: reporting NaN
// would not merely look wrong, it does not survive json.Marshal, so it would CORRUPT the
// ledger line rather than degrade it.
func TestFoldTurnBatchRowGuardsZeroDenominator(t *testing.T) {
	for _, tc := range []struct {
		name       string
		stats      agent.TurnBatchStats
		wantPerool float64
	}{
		{"text-only session", agent.TurnBatchStats{Turns: 4}, 0},
		{"empty transcript", agent.TurnBatchStats{}, 0},
		{"rounds to one decimal", agent.TurnBatchStats{Turns: 3, ToolCalls: 5, ToolTurns: 3}, 1.7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			row := foldTurnBatchRow("s", tc.stats, time.Unix(0, 0))
			if math.IsNaN(row.ToolCallsPerToolTurn) || row.ToolCallsPerToolTurn != tc.wantPerool {
				t.Fatalf("tool_calls_per_tool_turn=%v want %v", row.ToolCallsPerToolTurn, tc.wantPerool)
			}
			if _, err := json.Marshal(row); err != nil {
				t.Fatalf("row does not survive a JSON round trip: %v", err)
			}
		})
	}
}

// TestSessionAuditBatchDryRunAndMissingTranscript pins the two paths that must NOT touch
// the ledger: an explicit dry run, and a transcript that cannot be read.
func TestSessionAuditBatchDryRunAndMissingTranscript(t *testing.T) {
	dir := t.TempDir()
	for _, tc := range []struct {
		name     string
		argv     []string
		wantCode int
	}{
		{"dry run", []string{"--dry-run", writeBatchTranscript(t, "s.jsonl")}, 0},
		{"missing transcript", []string{filepath.Join(dir, "absent.jsonl")}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ledger := filepath.Join(dir, tc.name+".jsonl")
			var stdout, stderr bytes.Buffer
			argv := append([]string{"--ledger", ledger}, tc.argv...)
			if code := runSessionAuditBatch(&stdout, &stderr, argv); code != tc.wantCode {
				t.Fatalf("exit=%d want %d stderr=%s", code, tc.wantCode, stderr.String())
			}
			if _, err := os.Stat(ledger); !os.IsNotExist(err) {
				t.Fatalf("ledger was written anyway (err=%v)", err)
			}
		})
	}
}
