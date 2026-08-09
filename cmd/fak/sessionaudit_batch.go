package main

// sessionaudit_batch.go — `fak session-audit batch`, the first real consumer of the
// turn-batching KPI in internal/agent/turnbatch.go (#5799, the unmet remainder of #5019).
//
// #5019 shipped the measurement primitive and closed, but the KPI was an ORPHAN: nothing
// in tracked code ever called ParseTranscriptTurns / FoldTurnBatch / ScanTranscriptBatch,
// so the number the package exists to produce was never read. A metric with no consumer
// cannot grade a launch-time "batch independent calls" nudge, cannot be ratcheted by a
// monitoring lane, and cannot be compared across engines — which is why wiring one
// consumer is what unblocks the rest of the tool-call-width epic.
//
// WHY THE CONSUMER LIVES HERE AND NOT IN internal/sessionaudit. sessionaudit is the
// natural-looking home — it already discovers and folds these very transcripts, and its
// delegated.go even names ParseTranscriptTurns in prose. It cannot be the caller: the
// architest layering gate (TestNoUpwardImports) puts sessionaudit at tier 1 and agent at
// tier 4, so sessionaudit importing agent is an upward import and reds ARCH_LAYER_VIOLATION.
// cmd/fak sits above both and already imports internal/agent, so the composition seam is
// the CLI — which is also the shape #5799 proposes ("a fak verb that folds a session
// transcript and emits the fak.turnbatch.v1 payload").
//
// THE DENOMINATOR DECISION, MADE EXPLICITLY RATHER THAN INHERITED. #5799 flags that the
// KPI's two rates answer different questions and that a readout showing them side by side
// without saying so will mislead:
//
//   - tool_calls_per_assistant_turn = tool_calls / turns — divides by ALL assistant turns,
//     text-only ones included. This is byte-for-byte the convention (and the one-decimal
//     rounding) internal/codexlifecycle's HealthTotals.ToolCallsPerTurn uses for codex
//     rollouts, so the two engines are directly comparable rather than two bespoke numbers.
//   - batched_turn_rate = batched_turns / tool_turns — divides by TOOL-CALLING turns only,
//     because a text-only turn had nothing to batch and would dilute the rate.
//
// So this row carries a `denominators` block that states each one in words, and adds the
// third reading #5799 asks for — tool_calls_per_tool_turn — which is the same numerator
// over the batched-rate denominator. That derived rate is computed HERE, from the raw
// counts the fold already retains for exactly this purpose ("a consumer can recompute
// either rate over a different denominator without re-reading the transcript"), so the
// fold in internal/agent stays pure and its rounding and zero-denominator behaviour are
// untouched.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// turnBatchLedgerRel is the durable per-session batching ledger, a sibling of the other
// docs/nightrun ledgers (session-audit.jsonl, memory-value.jsonl). One appended line per
// folded session is what makes the reading PERSIST instead of being recomputed by hand.
const turnBatchLedgerRel = "docs/nightrun/turnbatch.jsonl"

// turnBatchDenominators states, in words, what each rate divides by. It ships inside the
// row rather than in a doc so a dashboard reading the ledger cannot put the two rates on
// one axis without the difference being right there in the payload.
type turnBatchDenominators struct {
	ToolCallsPerAssistantTurn string `json:"tool_calls_per_assistant_turn"`
	ToolCallsPerToolTurn      string `json:"tool_calls_per_tool_turn"`
	BatchedTurnRate           string `json:"batched_turn_rate"`
	CodexComparable           string `json:"codex_comparable"`
}

// turnBatchRow is one persisted fak.turnbatch.v1 reading for ONE session. The embedded
// TurnBatchStats inlines the KPI's own field names, so the ledger's wire shape is the
// schema's shape plus provenance and the explicit denominator gloss.
//
// Evidence-minimal, matching the contract turnbatch.go holds: only the session's BASE
// name and structural counts are recorded — no prompt, tool argument, tool result, diff,
// or model text, and not even the absolute path (which can carry a namespace or a
// worktree name).
type turnBatchRow struct {
	Schema     string `json:"schema"`
	RecordedAt string `json:"recorded_at"`
	Session    string `json:"session"`

	agent.TurnBatchStats

	ToolCallsPerToolTurn float64               `json:"tool_calls_per_tool_turn"`
	Denominators         turnBatchDenominators `json:"denominators"`
}

// foldTurnBatchRow builds the persisted row from a folded session. Pure and deterministic
// given `now`: no IO, so the row shape is testable without a transcript on disk.
//
// The derived per-tool-turn rate is guarded against a zero denominator (a session that
// never called a tool) and reported as 0 rather than NaN — the same guard, and the same
// one-decimal rounding, the fold in internal/agent applies to its own mean. A NaN here
// would not merely look wrong: it does not survive a JSON round trip, so it would corrupt
// the ledger line rather than degrade it.
func foldTurnBatchRow(session string, st agent.TurnBatchStats, now time.Time) turnBatchRow {
	row := turnBatchRow{
		Schema:         agent.TurnBatchSchema,
		RecordedAt:     now.UTC().Format(time.RFC3339),
		Session:        session,
		TurnBatchStats: st,
		Denominators: turnBatchDenominators{
			ToolCallsPerAssistantTurn: "tool_calls / turns (ALL assistant turns, text-only included)",
			ToolCallsPerToolTurn:      "tool_calls / tool_turns (turns that called at least one tool)",
			BatchedTurnRate:           "batched_turns / tool_turns (a text-only turn had nothing to batch)",
			CodexComparable:           "tool_calls_per_assistant_turn matches internal/codexlifecycle HealthTotals.tool_calls_per_turn: same denominator (all turns) and same one-decimal rounding",
		},
	}
	if st.ToolTurns > 0 {
		row.ToolCallsPerToolTurn = math.Round(float64(st.ToolCalls)/float64(st.ToolTurns)*10) / 10
	}
	return row
}

// appendTurnBatchRow appends row as one JSON line to the ledger at path (created if
// absent). json.Encoder.Encode writes the marshaled row plus its '\n' terminator in a
// single Write, so the O_APPEND write is atomic — the single-line-append convention the
// sibling docs/nightrun ledgers use, under which concurrent fleet writers never interleave
// a partial row. HTML escaping is off so the ledger stays byte-faithful and diffable.
func appendTurnBatchRow(path string, row turnBatchRow) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	return enc.Encode(row)
}

// sessionNameForBatch reduces a transcript path to the evidence-minimal session id: the
// base name with its .jsonl suffix removed. An empty result falls back to the base name so
// a row is never anonymous.
func sessionNameForBatch(path string) string {
	base := filepath.Base(path)
	if trimmed := strings.TrimSuffix(base, ".jsonl"); trimmed != "" {
		return trimmed
	}
	return base
}

func runSessionAuditBatch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session-audit batch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "also print the row as JSON to stdout")
	ledger := fs.String("ledger", turnBatchLedgerRel, "durable ledger to append one row to")
	dryRun := fs.Bool("dry-run", false, "compute and print the row but do NOT append")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: fak session-audit batch [--json] [--ledger PATH] [--dry-run] <session.jsonl>")
		return 2
	}
	path := fs.Arg(0)
	st, err := agent.ScanTranscriptBatch(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak session-audit batch: %v\n", err)
		return 1
	}
	row := foldTurnBatchRow(sessionNameForBatch(path), st, time.Now())
	if !*dryRun {
		if err := appendTurnBatchRow(*ledger, row); err != nil {
			fmt.Fprintf(stderr, "fak session-audit batch: append %s: %v\n", *ledger, err)
			return 1
		}
		fmt.Fprintf(stderr, "appended 1 row to %s\n", *ledger)
	}
	if *asJSON || *dryRun {
		return writeJSON(stdout, row)
	}
	fmt.Fprintf(stdout, "session-audit batch: %s turns=%d tool_calls=%d per_assistant_turn=%.1f per_tool_turn=%.1f batched=%d/%d rate=%.3f\n",
		row.Session, row.Turns, row.ToolCalls, row.ToolCallsPerAssistantTurn,
		row.ToolCallsPerToolTurn, row.BatchedTurns, row.ToolTurns, row.BatchedTurnRate)
	return 0
}
