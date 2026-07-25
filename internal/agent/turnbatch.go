package agent

// turnbatch.go — the tool-calls-per-assistant-turn batching KPI (#5019, part of the
// mode-debt census #4397). Opus tends to issue tool calls one-per-turn — narrating
// between single calls — where codex batches independent shell calls in a single cell.
// That cadence gap is measurable but, until now, only via a bespoke scan: no typed
// per-session KPI captured how often a turn batched independent tool calls, so a
// launch-time "batch independent calls" nudge had no metric to be measured against.
//
// This is the Go measurement primitive for that KPI: a pure fold over ONE Claude Code
// session transcript that reports tool_calls_per_assistant_turn and a batched-turn
// rate. It is the sibling of internal/codexlifecycle's ToolCallsPerTurn (which folds
// codex rollouts) and the typed peer of the tool_calls_per_turn number
// tools/transcript_workload.py already derives from the same transcript shape.
//
// It copies NO prompt, tool argument, tool result, diff, or model text into its
// report: only structural per-turn counts and derived rates survive — the same
// evidence-minimal contract the codex health fold holds.

import (
	"bufio"
	"encoding/json"
	"io"
	"math"
	"os"
)

// TurnBatchSchema identifies the report shape. /1: the first typed batching KPI for
// Claude/Opus session transcripts.
const TurnBatchSchema = "fak.turnbatch.v1"

// BatchedTurnMinCalls is the number of tool calls in ONE assistant turn that makes it a
// "batched" turn — a turn that issued two or more (presumptively independent) tool
// calls in a single model response rather than serializing them across turns. A turn
// with a single tool call is not batched; a text-only turn cannot be.
const BatchedTurnMinCalls = 2

// TurnBatchRow is one logical assistant turn distilled from a transcript: the number of
// tool_use blocks that single model response issued. Claude Code writes one assistant
// API response as SEVERAL JSONL records — one per content block — so the parser collapses
// those splits (keyed by message id) into one row before folding; see ParseTranscriptTurns.
type TurnBatchRow struct {
	ToolCalls int `json:"tool_calls"`
}

// TurnBatchStats is the per-session batching KPI folded from a transcript's turns.
// ToolCallsPerAssistantTurn is the headline the issue names; BatchedTurnRate is the
// "batched-turn rate" — the fraction of tool-CALLING turns that issued two or more
// calls at once (a text-only turn is excluded from the denominator: it had nothing to
// batch). Raw counts are retained so a consumer can recompute either rate over a
// different denominator without re-reading the transcript.
type TurnBatchStats struct {
	Turns                     int     `json:"turns"`
	ToolCalls                 int     `json:"tool_calls"`
	ToolTurns                 int     `json:"tool_turns"`
	BatchedTurns              int     `json:"batched_turns"`
	ToolCallsPerAssistantTurn float64 `json:"tool_calls_per_assistant_turn"`
	BatchedTurnRate           float64 `json:"batched_turn_rate"`
}

// FoldTurnBatch folds per-turn rows into the session KPI. Pure: no IO, deterministic.
// The two derived rates are guarded against a zero denominator (an empty transcript, or
// a session that never called a tool) and reported at 0 rather than NaN. Rounding
// matches the codex health fold: one decimal for the per-turn mean, three for the rate.
func FoldTurnBatch(rows []TurnBatchRow) TurnBatchStats {
	s := TurnBatchStats{Turns: len(rows)}
	for _, r := range rows {
		s.ToolCalls += r.ToolCalls
		if r.ToolCalls >= 1 {
			s.ToolTurns++
		}
		if r.ToolCalls >= BatchedTurnMinCalls {
			s.BatchedTurns++
		}
	}
	if s.Turns > 0 {
		s.ToolCallsPerAssistantTurn = math.Round(float64(s.ToolCalls)/float64(s.Turns)*10) / 10
	}
	if s.ToolTurns > 0 {
		s.BatchedTurnRate = math.Round(float64(s.BatchedTurns)/float64(s.ToolTurns)*1000) / 1000
	}
	return s
}

// ParseTranscriptTurns reads ONE Claude Code session transcript (JSONL) into per-turn
// rows in file order. It reproduces the load-bearing turn-collapsing that
// tools/transcript_workload.py documents: Claude Code emits one assistant API response
// as several JSONL records (one per thinking/text/tool_use content block) stamped with
// the SAME message id, so a turn MUST be keyed by that id and the split records merged —
// otherwise turns over-count ~3x and the tool-call fraction is diluted by the text-only
// splits. Sidechain (subagent/workflow) records live in their own track and are skipped.
//
// Durability contract, matching the codex rollout parser: a torn or non-JSON line is
// skipped, never fatal; only a reader error is returned.
func ParseTranscriptTurns(r io.Reader) ([]TurnBatchRow, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 256*1024), 64*1024*1024)

	var rows []TurnBatchRow
	var curID string
	curTools := 0
	haveCur := false
	flush := func() {
		if haveCur {
			rows = append(rows, TurnBatchRow{ToolCalls: curTools})
			haveCur, curTools, curID = false, 0, ""
		}
	}

	for sc.Scan() {
		var rec struct {
			Type        string `json:"type"`
			IsSidechain bool   `json:"isSidechain"`
			RequestID   string `json:"requestId"`
			Message     struct {
				ID      string          `json:"id"`
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		}
		if json.Unmarshal(sc.Bytes(), &rec) != nil {
			continue
		}
		if rec.IsSidechain {
			continue
		}
		switch rec.Type {
		case "assistant":
			id := rec.Message.ID
			if id == "" {
				id = rec.RequestID
			}
			n := countToolUse(rec.Message.Content)
			// Merge only a NON-empty id that matches the open turn (an empty id never
			// merges, so each usage-less split stays its own turn — matching the script).
			if haveCur && id != "" && id == curID {
				curTools += n
				continue
			}
			flush()
			haveCur, curID, curTools = true, id, n
		case "user":
			// A user message (a tool_result or a human turn) closes the open assistant
			// turn: any later assistant record opens a fresh turn.
			flush()
		}
	}
	flush()
	if err := sc.Err(); err != nil {
		return rows, err
	}
	return rows, nil
}

// countToolUse counts the tool_use blocks in an assistant message's content. A content
// value that is not a block array (e.g. a bare string) carries no tool call and counts 0.
func countToolUse(raw json.RawMessage) int {
	if len(raw) == 0 {
		return 0
	}
	var blocks []struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &blocks) != nil {
		return 0
	}
	n := 0
	for _, b := range blocks {
		if b.Type == "tool_use" {
			n++
		}
	}
	return n
}

// ScanTranscriptBatch opens a transcript file, parses its turns, and folds the batching
// KPI in one call — the file-path convenience a session-audit / dispatch-metrics caller
// uses. The parser's durability contract is preserved: a torn line inside the file is
// skipped, so only an open or read error surfaces.
func ScanTranscriptBatch(path string) (TurnBatchStats, error) {
	fh, err := os.Open(path)
	if err != nil {
		return TurnBatchStats{}, err
	}
	defer fh.Close()
	rows, err := ParseTranscriptTurns(fh)
	if err != nil {
		return TurnBatchStats{}, err
	}
	return FoldTurnBatch(rows), nil
}
