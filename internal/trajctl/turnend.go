package trajctl

// turnend.go — issue #2539, spine step 6 of the trajectory-control epic (#2533):
// score-at-turn-end so the curve gains a point every turn end instead of only
// when a scorer is run by hand. `fak guard-stophook` fires at every Claude Code
// turn end; this is the pure fold it runs there, plus the PreCompact twin that
// marks a context reset. Keeping the sampling here — not in the hook shell —
// keeps it deterministic and tier-1: the hook injects the evidence window and
// session stamp, this folds them into rows, and the impurity (clock, git,
// transcript resolution, and the wall-clock deadline that bounds the hook) stays
// at the call site.

import (
	"fmt"
	"sort"
)

const (
	// CompactionBoundaryMethod is the reserved method id of the context-reset
	// marker the PreCompact twin emits. It is its own curve series, so a boundary
	// never perturbs a progress method's values; a curve reader treats a flat
	// stretch spanning a boundary differently from a flat stretch within one
	// context.
	CompactionBoundaryMethod = "compaction-boundary"
	// CompactionBoundaryVersion is this marker's version.
	CompactionBoundaryVersion = "1"
)

// SampleReason names why a turn-boundary sampling pass ran. It travels on the
// [TurnSample] so an observer can tell a turn-end score pass from a compaction
// marker without re-deriving it.
type SampleReason string

const (
	// ReasonTurnEnd is the stop-hook cadence: an assistant turn ended and the
	// cheap scorers ran for the session's open objectives.
	ReasonTurnEnd SampleReason = "turn-end"
	// ReasonCompaction is the PreCompact twin: the context was reset, and a
	// boundary marker was appended for each open objective.
	ReasonCompaction SampleReason = "compaction"
)

// Stamp is the session attribution the turn-boundary runner writes onto every row
// it produces. The scorers themselves are session-agnostic pure folds; the runner
// owns attribution because only the hook knows which session and run it fired in.
type Stamp struct {
	SessionID string
	RunID     string
}

// SampleFailure records one scorer invocation that errored or panicked during a
// sampling pass. It is captured, never raised: the fail-open contract is that a
// buggy scorer costs its own row, not the turn — the hook and the session run on.
type SampleFailure struct {
	ObjectiveID string
	Method      string
	Detail      string
}

// TurnSample is the deterministic result of one turn-boundary sampling pass: the
// rows to append and the per-scorer failures that were swallowed fail-open. The
// caller appends Rows (via [AppendSample], or [Append] with [ScoreRecord]) and
// may surface Failures for observability. Producing the sample mutates nothing
// and touches no ledger; the append is the caller's single side effect.
type TurnSample struct {
	Reason   SampleReason
	Rows     []ScoreRow
	Failures []SampleFailure
}

// Sample runs each cheap scorer over each OPEN objective and folds the results
// into the rows to append at a turn end. It is a pure, deterministic fold with
// one guarantee layered over the scorers: every Score call is wrapped so a panic
// becomes a recorded [SampleFailure] instead of unwinding the caller — the
// fail-open rung the stop-hook needs. Runtime is bounded by construction: at most
// len(open objectives) * len(scorers) already-bounded pure calls, in a fixed
// (lexical objective id, then caller's scorer order) order. A wall-clock deadline,
// if the host wants one, wraps this at the call site (the stop-hook runs it under
// a context); trajctl stays clock-free and tier-1.
//
// objectives is the folded objective set ([State.Objectives]); only open
// objectives (active or paused) are scored — a met or abandoned objective's curve
// is closed. scorers is the cheap set the host elected to run at turn cadence (W3
// commit, W2 stall); judge scorers are excluded here by the caller's choice, not
// by this fold. win carries the injected evidence and UnixMillis stamp; stamp
// attributes each produced row to the firing session and run.
//
// For an objective with a declared plan the W3 commit scorer yields a row every
// pass (its progress fraction, even 0), so a planned objective accumulates at
// least one point per turn end. A planless objective yields a point only when a
// scorer has a live signal: the runner never fabricates a witness to pad the
// curve.
func Sample(objectives map[string]Objective, scorers []Scorer, win EvidenceWindow, stamp Stamp) TurnSample {
	out := TurnSample{Reason: ReasonTurnEnd}
	for _, id := range openObjectiveIDs(objectives) {
		obj := objectives[id]
		for _, sc := range scorers {
			rows, failure := scoreGuarded(sc, obj, win)
			if failure != nil {
				out.Failures = append(out.Failures, *failure)
				continue
			}
			for _, r := range rows {
				out.Rows = append(out.Rows, stampRow(r, win, stamp))
			}
		}
	}
	return out
}

// CompactionBoundary returns one boundary-marker row per OPEN objective, stamping
// a context reset at win.UnixMillis. It is the PreCompact twin of [Sample]: a
// curve reader uses the markers to see where the context was reset, since a flat
// stretch across a reset is a different signal from a flat stretch within one
// context. Each marker is its own method series ([CompactionBoundaryMethod]) at
// witness W0 with Value 0 and no progress claim, so it never perturbs a scorer's
// curve.
func CompactionBoundary(objectives map[string]Objective, win EvidenceWindow, stamp Stamp) TurnSample {
	out := TurnSample{Reason: ReasonCompaction}
	for _, id := range openObjectiveIDs(objectives) {
		row := ScoreRow{
			ObjectiveID: id,
			Value:       0,
			Method:      CompactionBoundaryMethod,
			Version:     CompactionBoundaryVersion,
			Witness:     W0,
		}
		if stamp.SessionID != "" {
			row.Evidence = []EvidenceRef{{Kind: "compaction", Ref: stamp.SessionID}}
		}
		out.Rows = append(out.Rows, stampRow(row, win, stamp))
	}
	return out
}

// AppendSample writes every row of a [TurnSample] to the ledger at path, in
// order, and returns the count appended. It is the thin I/O wrapper over the pure
// [Sample]/[CompactionBoundary] folds so the stop-hook wiring is a one-liner. A
// row that fails validation stops the append and returns the count written so far
// with the error; at turn cadence the caller treats a short append as non-fatal
// (fail-open), but the error is surfaced so a poisoned row is never silent.
func AppendSample(path string, sample TurnSample) (int, error) {
	n := 0
	for _, r := range sample.Rows {
		if err := Append(path, ScoreRecord(r)); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}

// scoreGuarded calls sc.Score(obj, win) behind a recover shield so a panicking
// scorer yields a [SampleFailure] rather than unwinding the turn. A scorer that
// returns normally — even with zero rows — is not a failure.
func scoreGuarded(sc Scorer, obj Objective, win EvidenceWindow) (rows []ScoreRow, failure *SampleFailure) {
	defer func() {
		if r := recover(); r != nil {
			rows = nil
			failure = &SampleFailure{
				ObjectiveID: obj.ID,
				Method:      sc.Method(),
				Detail:      fmt.Sprintf("scorer panicked: %v", r),
			}
		}
	}()
	return sc.Score(obj, win), nil
}

// stampRow writes session attribution and, when unset, the window's timestamp
// onto a produced row. The runner owns attribution: scorers are session-agnostic,
// so SessionID/RunID are applied here, and UnixMillis is backfilled from the
// window if a scorer left it zero.
func stampRow(r ScoreRow, win EvidenceWindow, stamp Stamp) ScoreRow {
	if r.UnixMillis == 0 {
		r.UnixMillis = win.UnixMillis
	}
	r.SessionID = stamp.SessionID
	r.RunID = stamp.RunID
	return r
}

// openObjectiveIDs returns the ids of the open (active or paused) objectives in
// lexical order, so a sampling pass is deterministic regardless of map iteration
// order.
func openObjectiveIDs(objectives map[string]Objective) []string {
	ids := make([]string, 0, len(objectives))
	for id, obj := range objectives {
		if objectiveOpen(obj.Status) {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}
