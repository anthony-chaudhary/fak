// Package climb is the importable surface of fak's improvement-loop engine.
//
// An improvement loop proposes candidates, measures each one, and keeps only
// those that beat the running baseline. The engine lives in internal/rsiloop and
// its keep-or-revert rule in internal/shipgate; Go's internal/ rule seals both
// from any other module. This package re-exports the part a loop author needs so
// a new proposer, metric or search policy can be tried from another repository
// without forking this one.
//
// Every type here is an alias and every function a thin forwarder, so a loop
// built through this package runs the same engine and the same rule as the
// in-tree loops. The package adds no behavior and holds no state.
//
// What you supply is a Harness:
//
//   - BaselineMetric measures the starting point.
//   - Candidates proposes what to try; this is where a search policy lives.
//   - Measure applies one candidate and returns its Measurement.
//   - Classify, optionally, states which evidence class a candidate falls in.
//
// What you do not supply is the decision rule. Run folds every Measurement
// through Evaluate, which keeps a candidate only when the signals its evidence
// class requires are all present: a metric gain, a green suite and a clean
// tree for ClassFull, fewer for the other classes. A loop author cannot change
// that rule or set a verdict directly.
//
// The rule's inputs are yours, though. Measure and Classify are your code, and
// the engine does not re-derive what they report, so a keep is only as honest
// as the harness that measured it. The same holds for the journal: a Row is a
// plain record, and a file of rows proves nothing about who wrote it. Trust in
// a result comes from running a harness you can read, not from the keep-bit
// alone.
//
// One rule to know before naming candidates: a candidate whose Label contains
// "policy", "gateway", "compaction" or "subturn", or whose Payload names a path
// under those subsystems, is reverted unless its Measurement carries a
// GymReceipt that verifies. That gate exists for fak's own self-modifying
// surfaces; an outside loop that trips it by accident should rename the label.
//
// The engine-internal read side (journal replay, variant archives, cache-parity
// checks, worktree helpers) is deliberately not re-exported.
package climb

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/gym"
	"github.com/anthony-chaudhary/fak/internal/rsiloop"
	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

type (
	// Harness is the set of callbacks a loop author supplies.
	Harness = rsiloop.Harness
	// Candidate is one proposed change; Payload is opaque to the engine.
	Candidate = rsiloop.Candidate
	// Measurement is what Measure reports for one candidate.
	Measurement = rsiloop.Measurement
	// GymReceipt is the multi-turn execution receipt some candidates must carry.
	GymReceipt = gym.GymReceipt
	// Scorecard is optional structured telemetry; the rule never reads it.
	Scorecard = rsiloop.Scorecard
	// ScoreComponent is one named axis of a Scorecard.
	ScoreComponent = rsiloop.ScoreComponent
	// Journal is an append-only JSONL record of a run.
	Journal = rsiloop.Journal
	// Row is one journal line: a candidate, its measurement and its verdict.
	Row = rsiloop.Row
	// Result summarizes a run.
	Result = rsiloop.Result
	// Observer sees each Row after it is journaled; it cannot change a verdict.
	Observer = rsiloop.Observer

	// Decision is the verdict on one candidate.
	Decision = shipgate.Decision
	// EvidenceClass selects which signals a candidate must show to be kept.
	EvidenceClass = shipgate.EvidenceClass
	// Profile is the set of signals one EvidenceClass requires.
	Profile = shipgate.Profile
	// Witness is the evidence Evaluate judges.
	Witness = shipgate.Witness
)

const (
	REVERT   = shipgate.REVERT
	KEEP     = shipgate.KEEP
	ESCALATE = shipgate.ESCALATE

	// ClassFull requires a metric gain, a green suite and a clean tree.
	ClassFull = shipgate.ClassFull
	// ClassDocsOnly requires only a clean tree.
	ClassDocsOnly = shipgate.ClassDocsOnly
	// ClassProofCarrying requires a metric gain and a clean tree, no suite run.
	ClassProofCarrying = shipgate.ClassProofCarrying

	DefaultTransientMeasurementRecoveryLimit = rsiloop.DefaultTransientMeasurementRecoveryLimit
)

// Run drives the loop: it measures the baseline, then judges each candidate in
// order, raising the baseline on every keep. k is how many consecutive
// non-keeps escalate and stop the run (3 when k <= 0). maxCycles caps the
// candidates tried (all of them when maxCycles <= 0).
func Run(h Harness, j *Journal, k, maxCycles int) (Result, error) {
	return rsiloop.Run(h, j, k, maxCycles)
}

// RunObserved is Run with an Observer called once per journaled row.
func RunObserved(h Harness, j *Journal, k, maxCycles int, obs Observer) (Result, error) {
	return rsiloop.RunObserved(h, j, k, maxCycles, obs)
}

// NewJournal opens an appending JSONL journal at path. An empty path or "-"
// writes to standard output.
func NewJournal(path string) (*Journal, error) { return rsiloop.NewJournal(path) }

// Evaluate applies the keep-or-revert rule to one witness.
func Evaluate(w Witness) (Decision, Witness) { return shipgate.Evaluate(w) }

// ProfileFor reports which signals an EvidenceClass requires.
func ProfileFor(c EvidenceClass) Profile { return shipgate.ProfileFor(c) }

// ClassifyPaths derives an EvidenceClass from the paths a candidate touched.
func ClassifyPaths(paths []string, isDoc func(string) bool) EvidenceClass {
	return shipgate.ClassifyPaths(paths, isDoc)
}

// NewTransientMeasureError marks a Measure failure as infrastructure trouble to
// retry, not a verdict on the candidate.
func NewTransientMeasureError(err error) error { return rsiloop.NewTransientMeasureError(err) }

// IsTransientMeasureError reports whether err was marked transient.
func IsTransientMeasureError(err error) bool { return rsiloop.IsTransientMeasureError(err) }

// DefaultTransientRetryBackoff is the delay used between transient retries when
// a Harness supplies none.
func DefaultTransientRetryBackoff(attempt int) time.Duration {
	return rsiloop.DefaultTransientRetryBackoff(attempt)
}
