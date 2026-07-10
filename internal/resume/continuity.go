package resume

// continuity.go — the RESUME-FIDELITY witness (#4145): the pure fold that turns
// internal/trajctl's already-shipped W3 verified-progress rows into the one bit
// FoldResumeState needs to tell a resume that TOOK (the objective's verified progress
// curve advanced) from one that merely produced a clean terminal turn without moving the
// objective (a `took_no_progress`).
//
// # Why the terminal-turn text is not enough
//
// Today `took` fires on `NewTurns > 0 && Outcome == OutcomeProgressed` (outcome.go). But
// OutcomeProgressed is read from the terminal turn's TEXT (it looks clean/live) and
// NewTurnsAfter is a pure COUNT — so together they prove "a clean turn landed," never "the
// objective advanced." A resume can burn turns re-reading, apologizing, or spinning and
// still present a clean terminal turn. This fold adds the missing witness: the W3
// commit-progress cursor (trajctl's deterministic, zero-model-call outcome scorer) compared
// across the last launch boundary.
//
// # Content-blind, clock-free, and it RE-READS — never a new sensor
//
// It reasons only over ScoreRows the shell already read from the trajctl ledger. It does not
// re-measure progress (trajctl owns that); it re-projects trajctl's W3 rows onto the resume
// boundary. Same rows in, same verdict out — no clock, no I/O. WITNESSED (this W3 signal)
// stays in its own field and is NEVER summed with the OBSERVED counts elsewhere in the
// package.

import "github.com/anthony-chaudhary/fak/internal/trajctl"

// w3Epsilon guards the float compare: W3 commit-progress values are exact k/n fractions, but
// a hair of representation error must never read as "advanced". A real phase advance moves
// the fraction by at least 1/len(plan), far above this floor.
const w3Epsilon = 1e-9

// ContinuityWitness is the per-resume verified-progress verdict: did the objective's W3
// curve actually move across the last launch, or did the resume only produce turns? It is a
// FOLD of trajctl W3 rows, carried into ResumeFacts so FoldResumeState can split `took` from
// `took_no_progress` without importing trajctl itself.
type ContinuityWitness struct {
	// Witnessed is true iff a W3 curve exists for the scored objective (at least one W3 row
	// was handed in). False means the session is un-instrumented for verified progress — the
	// fold has no witness, so FoldResumeState falls back to the legacy text-cleanliness
	// reading (`took`) rather than inventing a verdict from evidence that does not exist.
	Witnessed bool `json:"witnessed"`
	// Advanced is true iff a post-launch W3 row's value strictly exceeds the pre-launch
	// cursor — the objective's verified progress moved. Only meaningful when Witnessed.
	Advanced bool `json:"advanced"`
	// PreValue is the max W3 value at or before the launch boundary — the pre-crash cursor.
	PreValue float64 `json:"pre_value"`
	// PostValue is the max W3 value strictly after the launch boundary — 0 when no post-launch
	// W3 row landed (a productive-but-uncommitted stretch reads as a flat post cursor).
	PostValue float64 `json:"post_value"`
	// W3Rows is how many W3 rows fed the compare — the evidence count, for the readout.
	W3Rows int `json:"w3_rows"`
}

// FoldW3Continuity folds the objective's W3 ScoreRows into the continuity witness across a
// launch boundary. rows is the trajctl score history the shell already read; objectiveID
// scopes the compare to one objective ("" folds across every W3 row handed in, for a
// single-objective session); sinceUnixMillis is the last launch time in epoch-millis (the
// same boundary NewTurnsAfter uses, in millis to match ScoreRow.UnixMillis).
//
// A row is POST iff sinceUnixMillis > 0 AND its UnixMillis is strictly greater — so an
// unstamped W3 row (UnixMillis == 0) or one at/before the boundary counts toward the PRE
// cursor, never as post-resume progress. That is the fail-closed choice: unattributable
// verified work never credits a resume with progress it cannot prove happened after it.
// Total over any input; empty rows fold to the un-witnessed floor (Witnessed=false).
func FoldW3Continuity(rows []trajctl.ScoreRow, objectiveID string, sinceUnixMillis int64) ContinuityWitness {
	var w ContinuityWitness
	var havePost bool
	for _, r := range rows {
		if r.Witness != trajctl.W3 {
			continue
		}
		if objectiveID != "" && r.ObjectiveID != objectiveID {
			continue
		}
		w.Witnessed = true
		w.W3Rows++
		if sinceUnixMillis > 0 && r.UnixMillis > sinceUnixMillis {
			if !havePost || r.Value > w.PostValue {
				w.PostValue = r.Value
			}
			havePost = true
		} else if r.Value > w.PreValue {
			w.PreValue = r.Value
		}
	}
	w.Advanced = havePost && w.PostValue > w.PreValue+w3Epsilon
	return w
}
