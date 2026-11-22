// Package climb is the importable surface of fak's improvement-loop engine.
//
// An improvement loop proposes candidates, measures each one, and keeps only
// those that beat the running baseline. The engine lives in internal/rsiloop and
// its keep-or-revert verdict in internal/shipgate; Go's internal/ rule seals both
// from any other module. This package re-exports the part a loop author needs so
// a new proposer, metric or search policy can be tried from another repository
// without forking this one.
//
// Every name here is a type alias or a var/const re-export, so a value built
// through this package is the same value the in-tree loops use, and it earns the
// same keep-bit. The package adds no behavior and holds no state.
//
// What you supply is a Harness:
//
//   - BaselineMetric measures the starting point.
//   - Candidates proposes what to try; this is where a search policy lives.
//   - Measure applies one candidate and returns its Measurement.
//   - Classify, optionally, proves which evidence a candidate needs.
//
// What you do not supply is the verdict. Run folds every Measurement through
// Evaluate, so a candidate is kept only when its metric improved, its suite is
// green and its tree is clean, as its evidence class requires. A loop author
// cannot mark their own candidate as kept.
//
// The engine-internal read side (journal replay, variant archives, cache-parity
// checks, worktree helpers) is deliberately not re-exported.
package climb

import (
	"github.com/anthony-chaudhary/fak/internal/rsiloop"
	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// The loop: what an author constructs and what the engine returns.
type (
	Harness        = rsiloop.Harness
	Candidate      = rsiloop.Candidate
	Measurement    = rsiloop.Measurement
	Scorecard      = rsiloop.Scorecard
	ScoreComponent = rsiloop.ScoreComponent
	Journal        = rsiloop.Journal
	Row            = rsiloop.Row
	Result         = rsiloop.Result
	Observer       = rsiloop.Observer
)

// The verdict: the keep-bit and the evidence it is computed from.
type (
	Decision      = shipgate.Decision
	EvidenceClass = shipgate.EvidenceClass
	Profile       = shipgate.Profile
	Witness       = shipgate.Witness
	Gate          = shipgate.Gate
)

const (
	REVERT   = shipgate.REVERT
	KEEP     = shipgate.KEEP
	ESCALATE = shipgate.ESCALATE

	ClassFull          = shipgate.ClassFull
	ClassDocsOnly      = shipgate.ClassDocsOnly
	ClassProofCarrying = shipgate.ClassProofCarrying
)

var (
	Run         = rsiloop.Run
	RunObserved = rsiloop.RunObserved
	NewJournal  = rsiloop.NewJournal

	Evaluate      = shipgate.Evaluate
	NewGate       = shipgate.NewGate
	ProfileFor    = shipgate.ProfileFor
	ClassifyPaths = shipgate.ClassifyPaths

	NewTransientMeasureError = rsiloop.NewTransientMeasureError
	IsTransientMeasureError  = rsiloop.IsTransientMeasureError
)
