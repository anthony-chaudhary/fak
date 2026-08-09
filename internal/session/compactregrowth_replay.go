package session

// compactregrowth_replay.go — COUNTERFACTUAL replay of a candidate repeated-tool-result
// dedup mechanism over the SAME post-fire windows the regrowth audit already measures
// (#5254, child of #4768).
//
// Why this exists. #5254's DoD item 3 asks for a `compact-audit` re-run showing the
// REPEATED_TOOL_RESULT window share and tool_result/* dup_bytes "materially reduced on new
// sessions". That measurement is unreachable in one pass by construction: it is defined over
// sessions that do not exist yet, and the shipped mechanism (a gateway-side fold,
// `agent.ElideMessages`) rewrites the copy fak forwards upstream while the Codex rollout row
// the audit mines is written before fak sees the turn and stays byte-identical. See
// docs/notes/COMPACTION-REPEATED-TOOL-RESULT-DEDUP-2026-07-19.md.
//
// What IS measurable today, on the corpus that already exists: replay the candidate fold over
// every post-fire window and count how much of the duplication the audit already found the fold
// would actually have collapsed. That is a real, checkable bound with a stated denominator, and
// it is falsifiable — unlike a "materially reduced" read off a corpus that is ~96% traffic fak
// never routed.
//
// Two properties this file is built to expose rather than hide:
//
//  1. REACH. A gateway-side fold only ever sees ONE wire. A body whose earliest occurrence
//     precedes this window's compaction fire was compacted OUT of that wire, so the fold has
//     nothing left to match it against — even though the audit correctly types it a duplicate
//     (the audit's `seen` table is session-wide, spanning fires). Those bytes are counted
//     separately as CrossFire and are NOT claimable by a within-wire mechanism. This is the
//     issue's own "the same output re-entering the window it was just compacted out of", and it
//     is precisely the part a wire-local fold cannot fix.
//
//  2. LOSS. A dedup that destroys content is a correctness bug, not a saving. The property is
//     lossless-by-RELOCATION, checked at SPAN granularity because that is the granularity the
//     mechanism works at: every line the fold removes must still be reachable verbatim in a
//     strictly-earlier body of the same window. Checking it at BODY granularity instead ("a body
//     that is not a whole-body duplicate must survive byte-identical") is the wrong test and
//     manufactures a false-positive rate out of nothing — two different tool outputs routinely
//     share a long identical run, and folding it is the mechanism working. The false-positive
//     rate is reported beside the win instead of being assumed to be zero.
//
// Tier note: internal/session is tier 1 (stdlib only) and cannot import the elider
// (internal/agent, tier 4). The mechanism is therefore INJECTED at this seam as a plain
// func(bodies []string) []string by a caller that may import both (cmd/fak). This file knows
// nothing about how the fold works, which is what lets it score any candidate mechanism.
//
// Privacy: bodies are held in memory for the lifetime of ONE window and are never written to a
// receipt, a report field, or disk. Every exported field below is a count or a byte total.

import (
	"encoding/json"
	"strings"
)

// RegrowthReplayFolder is the candidate dedup mechanism under counterfactual test. It receives
// the decoded tool-result bodies of ONE post-fire window in wire order and must return a slice
// of the SAME length, where element i is the (possibly folded) rendering of body i. Returning a
// slice of any other length is treated as a replay error and the window is skipped, because a
// mechanism that drops a span outright cannot be scored on the same axis as one that shortens it.
type RegrowthReplayFolder func(bodies []string) []string

// regrowthReplayDefaultBudget bounds the tool-result bytes held for one open window. The measured
// corpus averages ~683 KB of tool_result bytes per post-fire window, so this is ~100x headroom;
// a window that exceeds it is skipped and counted rather than silently truncated, so an outlier
// can never quietly inflate or deflate the replayed bound.
const regrowthReplayDefaultBudget = 64 << 20

// RegrowthReplayOptions arms the replay. A nil Fold leaves the whole pass off and costs nothing.
type RegrowthReplayOptions struct {
	// Fold is the mechanism under test.
	Fold RegrowthReplayFolder
	// WindowByteBudget caps the tool-result bytes retained for one window; <= 0 uses the default.
	WindowByteBudget int64
	// MinDupLines is the candidate's own minimum-duplicate-run floor, used ONLY for the reach
	// diagnostic DupRowsUnderLineFloor (a body with fewer lines than the floor can never fold, no
	// matter how often it repeats). 0 disables that diagnostic. The replay never enforces it.
	MinDupLines int
}

// RegrowthReplayStat is the scored result of a replay. Every field is a count or a byte total;
// no body ever reaches this struct.
type RegrowthReplayStat struct {
	// Denominator.
	Rollouts        int   `json:"rollouts"`
	Windows         int   `json:"windows"`           // post-fire windows carrying >= 1 tool-result row
	ToolResultRows  int   `json:"tool_result_rows"`  // rows scored
	ToolResultBytes int64 `json:"tool_result_bytes"` // their rollout row bytes

	// BEFORE — the audit's own session-wide duplicate verdict, re-totalled per window so the
	// anomaly rule (RegrowthDupToolMinRows / RegrowthDupMinBytes) can be re-applied after the fold.
	AnomalyWindowsBefore int   `json:"anomaly_windows_before"`
	DupRowsBefore        int   `json:"dup_rows_before"`
	DupBytesBefore       int64 `json:"dup_bytes_before"`

	// REACH — how much of that duplication a WITHIN-WIRE fold can even see.
	InWindowDupRows   int   `json:"in_window_dup_rows"` // earliest occurrence is inside this window: foldable
	InWindowDupBytes  int64 `json:"in_window_dup_bytes"`
	CrossFireDupRows  int   `json:"cross_fire_dup_rows"` // earliest occurrence precedes the fire: NOT foldable
	CrossFireDupBytes int64 `json:"cross_fire_dup_bytes"`
	// DupRowsUnderLineFloor counts in-window duplicate rows whose body has fewer lines than
	// MinDupLines — foldable in principle, unreachable by a line-run matcher.
	DupRowsUnderLineFloor int `json:"dup_rows_under_line_floor,omitempty"`

	// AFTER — the same accounting recomputed on the folded bodies.
	AnomalyWindowsAfter int   `json:"anomaly_windows_after"`
	DupRowsAfter        int   `json:"dup_rows_after"`
	DupBytesAfter       int64 `json:"dup_bytes_after"`
	WindowsCollapsed    int   `json:"windows_collapsed"` // fired before, does not fire after
	FoldedRows          int   `json:"folded_rows"`
	ShedBytes           int64 `json:"shed_bytes"`

	// LOSS — the counterfactual's correctness side, measured at SPAN granularity because that is
	// the granularity the mechanism works at. A body-level check ("a body that is not a whole-body
	// duplicate must survive byte-identical") is the wrong test and reads as a catastrophic
	// false-positive rate: two genuinely different tool outputs routinely share a long identical
	// line run (two `git status` runs, two builds of the same target), and folding that shared run
	// is the mechanism working, not failing.
	//
	// The real property is lossless-by-RELOCATION: every line the fold removes must still be
	// reachable, verbatim, in an EARLIER body of the same window. RemovedLinesLost counts lines
	// removed that appear nowhere earlier — content actually destroyed — and is the number that
	// must be zero.
	RemovedLinesRelocated int64 `json:"removed_lines_relocated"`
	RemovedLinesLost      int64 `json:"removed_lines_lost"`
	// Diagnostics, not defects: how much of the fold's work is invisible to the audit's body-level
	// duplicate accounting.
	WholeBodyDupRows int `json:"whole_body_dup_rows"` // byte-identical to an earlier body
	PartialFoldRows  int `json:"partial_fold_rows"`   // folded, but NOT a whole-body duplicate

	// Fidelity caveats, reported so the bound above is read with them.
	TruncatedRows        int `json:"truncated_rows"`         // clipped by the scanner's 128 KB head bound
	BudgetSkippedWindows int `json:"budget_skipped_windows"` // over WindowByteBudget, excluded entirely
	ShapeErrorWindows    int `json:"shape_error_windows"`    // folder returned a wrong-length slice
}

// Add folds another rollout's replay stat into r, so a corpus sweep can total them.
func (r *RegrowthReplayStat) Add(o RegrowthReplayStat) {
	r.Rollouts += o.Rollouts
	r.Windows += o.Windows
	r.ToolResultRows += o.ToolResultRows
	r.ToolResultBytes += o.ToolResultBytes
	r.AnomalyWindowsBefore += o.AnomalyWindowsBefore
	r.DupRowsBefore += o.DupRowsBefore
	r.DupBytesBefore += o.DupBytesBefore
	r.InWindowDupRows += o.InWindowDupRows
	r.InWindowDupBytes += o.InWindowDupBytes
	r.CrossFireDupRows += o.CrossFireDupRows
	r.CrossFireDupBytes += o.CrossFireDupBytes
	r.DupRowsUnderLineFloor += o.DupRowsUnderLineFloor
	r.AnomalyWindowsAfter += o.AnomalyWindowsAfter
	r.DupRowsAfter += o.DupRowsAfter
	r.DupBytesAfter += o.DupBytesAfter
	r.WindowsCollapsed += o.WindowsCollapsed
	r.FoldedRows += o.FoldedRows
	r.ShedBytes += o.ShedBytes
	r.RemovedLinesRelocated += o.RemovedLinesRelocated
	r.RemovedLinesLost += o.RemovedLinesLost
	r.WholeBodyDupRows += o.WholeBodyDupRows
	r.PartialFoldRows += o.PartialFoldRows
	r.TruncatedRows += o.TruncatedRows
	r.BudgetSkippedWindows += o.BudgetSkippedWindows
	r.ShapeErrorWindows += o.ShapeErrorWindows
}

// replayRow is one captured tool-result row of the open window.
type replayRow struct {
	body      string
	rowLen    int64
	dupAudit  bool // the audit's session-wide duplicate verdict for this row
	truncated bool
}

// regrowthReplay accumulates the open window's tool-result bodies and scores them at close.
type regrowthReplay struct {
	opt    RegrowthReplayOptions
	budget int64

	rows  []replayRow
	bytes int64
	over  bool

	stat RegrowthReplayStat
}

func newRegrowthReplay(opt RegrowthReplayOptions) *regrowthReplay {
	if opt.Fold == nil {
		return nil
	}
	b := opt.WindowByteBudget
	if b <= 0 {
		b = regrowthReplayDefaultBudget
	}
	return &regrowthReplay{opt: opt, budget: b}
}

// observe captures one tool-result row of the open window. raw is the row's `output` JSON slot
// exactly as the audit hashed it; dup is the audit's session-wide duplicate verdict.
func (rp *regrowthReplay) observe(raw []byte, rowLen int64, dup, truncated bool) {
	if rp.over {
		return
	}
	if rp.bytes+rowLen > rp.budget {
		rp.over = true
		rp.rows = nil
		rp.bytes = 0
		return
	}
	rp.bytes += rowLen
	rp.rows = append(rp.rows, replayRow{
		body:      decodeToolResultBody(raw),
		rowLen:    rowLen,
		dupAudit:  dup,
		truncated: truncated,
	})
}

// decodeToolResultBody renders the `output` slot the way the DECODED wire carries it — a Codex
// function_call_output.output is a JSON string, and agent.ElideMessages sees its unescaped text
// (real newlines), not the escaped JSON literal. Folding the escaped form would see a single line
// and understate reach to zero, so this unwrap is load-bearing for fidelity, not cosmetic. A
// non-string slot (the list form, ~0.3% of rows) is scored as its raw text.
func decodeToolResultBody(raw []byte) string {
	if len(raw) > 0 && raw[0] == '"' {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return s
		}
	}
	return string(raw)
}

// lineSet is the DISTINCT lines of a body. Set semantics (not a multiset) are deliberate: the
// question a loss check answers is "are these bytes still reachable somewhere earlier", which is a
// membership question, not a count.
func lineSet(s string) map[string]struct{} {
	m := make(map[string]struct{})
	for _, ln := range strings.Split(s, "\n") {
		m[ln] = struct{}{}
	}
	return m
}

// closeWindow scores the captured window and resets for the next fire. Called from the tracker's
// window close so the replay's window boundaries are exactly the audit's.
func (rp *regrowthReplay) closeWindow() {
	rows := rp.rows
	over := rp.over
	rp.rows = nil
	rp.bytes = 0
	rp.over = false

	if over {
		rp.stat.BudgetSkippedWindows++
		return
	}
	if len(rows) == 0 {
		return
	}
	rp.stat.Windows++
	rp.stat.ToolResultRows += len(rows)

	bodies := make([]string, len(rows))
	// firstAt maps a body's content to the index of its EARLIEST occurrence in this window, and
	// count how many times it occurs. Identity is content, never call id — the issue's spans arrive
	// under different call ids, so a call-id key would find nothing.
	firstAt := make(map[string]int, len(rows))
	occurs := make(map[string]int, len(rows))
	for i, r := range rows {
		bodies[i] = r.body
		rp.stat.ToolResultBytes += r.rowLen
		if r.truncated {
			rp.stat.TruncatedRows++
		}
		if _, ok := firstAt[r.body]; !ok {
			firstAt[r.body] = i
		}
		occurs[r.body]++
	}

	// BEFORE, plus the reach split.
	dupRowsBefore, dupBytesBefore := 0, int64(0)
	for i, r := range rows {
		if !r.dupAudit {
			continue
		}
		dupRowsBefore++
		dupBytesBefore += r.rowLen
		if firstAt[r.body] < i {
			// Its earliest occurrence is inside this same post-fire window: a within-wire fold can
			// see both copies and match them.
			rp.stat.InWindowDupRows++
			rp.stat.InWindowDupBytes += r.rowLen
			if n := rp.opt.MinDupLines; n > 0 && strings.Count(r.body, "\n")+1 < n {
				rp.stat.DupRowsUnderLineFloor++
			}
		} else {
			// The audit typed it a duplicate against something it saw BEFORE this fire. That copy was
			// compacted out of the wire, so no within-wire mechanism can collapse this row.
			rp.stat.CrossFireDupRows++
			rp.stat.CrossFireDupBytes += r.rowLen
		}
	}
	rp.stat.DupRowsBefore += dupRowsBefore
	rp.stat.DupBytesBefore += dupBytesBefore
	anomalyBefore := dupRowsBefore >= RegrowthDupToolMinRows && dupBytesBefore >= RegrowthDupMinBytes
	if anomalyBefore {
		rp.stat.AnomalyWindowsBefore++
	}

	folded := rp.opt.Fold(bodies)
	if len(folded) != len(bodies) {
		// A mechanism that changes the span COUNT is not comparable on this axis; refuse to score it
		// rather than silently crediting a deletion as a fold.
		rp.stat.ShapeErrorWindows++
		return
	}

	// LOSS, at span granularity: every line the fold removes from body i must still be reachable
	// verbatim in a STRICTLY-EARLIER body of this window. A removed line seen nowhere earlier is
	// content destroyed — the one failure that would make a "bytes saved" headline a lie.
	seenLines := make(map[string]struct{})
	for i, r := range rows {
		orig := lineSet(r.body)
		if folded[i] != r.body {
			kept := lineSet(folded[i])
			for ln := range orig {
				if _, ok := kept[ln]; ok {
					continue
				}
				if _, ok := seenLines[ln]; ok {
					rp.stat.RemovedLinesRelocated++
				} else {
					rp.stat.RemovedLinesLost++
				}
			}
			if occurs[r.body] > 1 && firstAt[r.body] < i {
				rp.stat.WholeBodyDupRows++
			} else {
				rp.stat.PartialFoldRows++
			}
		}
		for ln := range orig {
			seenLines[ln] = struct{}{}
		}
	}

	// AFTER. The rollout row length minus whatever the fold shed from its body, re-scored under the
	// audit's own duplicate rule so before/after are the same measurement.
	afterLen := make([]int64, len(rows))
	for i, r := range rows {
		shed := int64(len(r.body) - len(folded[i]))
		if shed < 0 {
			shed = 0
		}
		if shed > 0 {
			rp.stat.FoldedRows++
			rp.stat.ShedBytes += shed
		}
		afterLen[i] = r.rowLen - shed
	}
	firstAfter := make(map[string]int, len(rows))
	dupRowsAfter, dupBytesAfter := 0, int64(0)
	for i, r := range rows {
		f := folded[i]
		unchanged := f == r.body
		isDup := false
		switch {
		case r.dupAudit && unchanged:
			// Untouched by the fold and still byte-identical to the earlier copy the audit matched —
			// including every cross-fire row, whose partner is not in this window at all.
			isDup = true
		default:
			if j, ok := firstAfter[f]; ok && j < i && afterLen[i] >= RegrowthDupMinBytes {
				isDup = true
			}
		}
		if _, ok := firstAfter[f]; !ok {
			firstAfter[f] = i
		}
		if isDup {
			dupRowsAfter++
			dupBytesAfter += afterLen[i]
		}
	}
	rp.stat.DupRowsAfter += dupRowsAfter
	rp.stat.DupBytesAfter += dupBytesAfter
	anomalyAfter := dupRowsAfter >= RegrowthDupToolMinRows && dupBytesAfter >= RegrowthDupMinBytes
	if anomalyAfter {
		rp.stat.AnomalyWindowsAfter++
	}
	if anomalyBefore && !anomalyAfter {
		rp.stat.WindowsCollapsed++
	}
}
