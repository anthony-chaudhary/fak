package query

// adapter.go — the bridge from the real transcript parser (internal/resume/transcript)
// into the Turn projection the query engine runs over. The engine is deliberately
// parser-agnostic (it runs over []Turn, whatever fills them), so a durable projection
// (C3) can feed it the same shape; this adapter is the filesystem-transcript feeder that
// exists today, the one #4193 names as the current-only way to read transcript content.
//
// The adapter maps what the parser cleanly exposes: Role (turn-open/turn-close), the
// tool-terminal shape (a tool_use paired with its tool_result), and the turn text. It does
// NOT fabricate fields the parser does not attest — ToolFailed (an error verdict) and Files
// (paths from tool args/results) are left zero here, to be populated by a producer that
// actually adjudicates them (the owned-turn producer, #2388) rather than guessed. Bytes is
// left nil: the adapter is the redacted-text feeder, and the verbatim-bytes projection is
// supplied by the durable span store (C3), not re-synthesized from parsed text.

import (
	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
)

// TurnsFromRecords projects parsed transcript records into query Turns, in record order.
// Synthetic (harness-injected) records are skipped — they are not session turns a user
// asked about. Each Turn's Index is its position in the returned slice.
func TurnsFromRecords(recs []transcript.Record) []Turn {
	out := make([]Turn, 0, len(recs))
	for _, r := range recs {
		if r.IsSynthetic() {
			continue
		}
		role := r.Role()
		if role == "" {
			continue
		}
		tool := r.LastToolUseName()
		out = append(out, Turn{
			Index:    len(out),
			Role:     role,
			Tool:     tool,
			ToolTerm: tool != "" && r.HasToolResult(),
			Text:     r.Text(),
		})
	}
	return out
}
