package main

import (
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// The placement ladder's evidence half (epic #5416 track E).
//
// dispatch_rung_pin.go decides what to DO with a resolved placement; nothing resolved one,
// so every worker got nil and the ladder moved no traffic. This is the half that resolves
// it: the account roster plus the append-only turn journal, folded into a grade, handed to
// `Roster.Place` for the class of work an operator DECLARED this issue to be. Together the
// two halves are the automatic three-stratum placement, on the dispatch path, which is
// where a fleet's token volume actually is.
//
// Every refusal here is NAMED and reported in the tick payload. A ladder that silently
// does nothing is indistinguishable from a ladder that is broken, and the first question an
// operator asks about an automatic placer is always "why did it not fire" — so each way of
// having nothing to say gets its own reason rather than a shared shrug.
//
// Three deliberate asymmetries with the `fak route --place` operator surface:
//
//  1. No hand declarations. The CLI accepts --capability, where an operator asserts what a
//     model can do. This path grades ONLY from observed outcomes, because an assertion is
//     exactly how a fleet ends up running its work on a rung nobody measured, and there is
//     no human in this loop to sanity-check one. An operator who wants to see what a
//     declaration WOULD do still has the CLI, which prints the pool it placed from.
//  2. A freshness window by default. Capability is a property of a model AS DEPLOYED, and a
//     re-quantised local build or a re-pointed fleet endpoint keeps the id while changing
//     the thing. The window also drops rows nobody stamped — which is safe in one specific
//     direction: losing evidence can only fail the grade floor, an ungraded candidate can
//     only stay unmeasured, and dispatch_rung_pin.go refuses an unmeasured placement. So
//     every path out of a thinner corpus ends at "leave the worker alone".
//  3. A size cap on the journal read. This runs once per worker launch rather than once per
//     operator command, and an append-only file on a busy fleet has no upper bound. Past
//     the cap the seam reports itself off rather than stalling a launch.

const (
	// dispatchRungWindowEnv overrides the freshness window; dispatchRungWindowDefault is
	// what an operator who has not thought about it gets. Same grammar as --since.
	dispatchRungWindowEnv     = "FLEET_DISPATCH_RUNG_WINDOW"
	dispatchRungWindowDefault = "30d"

	// dispatchRungJournalCap bounds the per-launch read. ~40k turns at a typical row size.
	dispatchRungJournalCap = 8 << 20
)

// The closed vocabulary of reasons the ladder did not pin this worker. Reported under the
// payload key rung_pin_skipped, one string, no free text.
const (
	rungSkipOutranked   = "outranked"              // somebody already decided this worker's model
	rungSkipNoWorkClass = "no-work-class"          // no trusted tier label to grade against
	rungSkipBadWindow   = "bad-window"             // the operator's window spec does not parse
	rungSkipNoRoster    = "no-roster"              // no account roster, or it does not load
	rungSkipRosterEmpty = "roster-binds-no-models" // a roster that places nothing
	rungSkipNoJournal   = "no-journal"             // the fleet has produced no turn evidence
	rungSkipJournalBig  = "journal-too-large"      // past the per-launch read cap
	rungSkipJournalBad  = "journal-unreadable"     // present but it does not read
	rungSkipNoEvidence  = "no-graded-evidence"     // nothing survived the window
	rungSkipRefused     = "placement-refused"      // Place said no
	rungSkipUnmeasured  = "placement-unmeasured"   // the ladder walked to the top rung ungraded
	rungSkipNotApplied  = "placement-not-applied"  // the pure guard refused a resolved placement
)

// dispatchRungPlacementEnabled reports whether the opt-in placement-ladder seam
// (FLEET_DISPATCH_RUNG_PLACEMENT) is switched on. Default (unset / an off-ish value) is OFF.
// Mirrors dispatchPlacementEvidenceEnabled's truthy/falsy grammar.
//
// Off-by-default is not timidity here. A device- or fleet-rung model id is only launchable
// by a worker whose seat routes to that endpoint, and nothing on this path can verify that
// — the fleet trust boundary (#5421, track G) is what makes it verifiable. Until then,
// turning this on is an operator asserting their seats can reach their own hardware.
func dispatchRungPlacementEnabled() bool {
	raw, ok := os.LookupEnv("FLEET_DISPATCH_RUNG_PLACEMENT")
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "0", "off", "false", "no", "disable", "disabled":
		return false
	}
	return true
}

// applyRungPlacement resolves the automatic placement for this slot and applies it to a
// worker that nothing else pinned. It returns the (possibly unchanged) policy and the
// reason it did not pin, empty when it did.
//
// The seam being OFF returns an empty reason rather than a named one, so an unconfigured
// tick adds no payload key at all and stays byte-identical to before this existed. Every
// other silence is a decision, and says so.
func applyRungPlacement(root string, labels []string, p workerModelPolicy) (workerModelPolicy, string) {
	if !dispatchRungPlacementEnabled() {
		return p, ""
	}
	// Checked here as well as inside placeUnpinnedWorker: there it is the guarantee, here it
	// is what keeps a pinned worker from paying for a roster load and a journal read to be
	// told something it already knew.
	if p.Source != modelSourceSeatDefault {
		return p, rungSkipOutranked
	}
	class, _ := dispatchtick.WorkClassForIssue(labels)
	if class == "" {
		// PolicyFor("") floors an unknown class at T0, so placing an unclassified slot would
		// walk it straight to the vendor rung. Refusing is the whole point of that floor.
		return p, rungSkipNoWorkClass
	}
	rosterPath := dispatchAccountsRosterPath(root)
	if rosterPath == "" {
		return p, rungSkipNoRoster
	}
	roster, err := modelroute.LoadRoster(rosterPath)
	if err != nil {
		return p, rungSkipNoRoster
	}
	evidence, reason := dispatchRungEvidence(root)
	if reason != "" {
		return p, reason
	}
	rung, reason := resolveRungPlacement(roster, class, evidence)
	if reason != "" {
		return p, reason
	}
	out := placeUnpinnedWorker(p, rung)
	if out.Source != modelSourceRung {
		// Unreachable while the two halves agree, and reported rather than swallowed if they
		// ever stop agreeing: a resolved placement that vanished without a reason is the one
		// failure an operator could not diagnose from the payload.
		return p, rungSkipNotApplied
	}
	return out, ""
}

// dispatchRungEvidence folds the turn journal into per-model class evidence, or names why
// there is none to fold.
func dispatchRungEvidence(root string) (map[string][]modelroute.ClassEvidence, string) {
	window := strings.TrimSpace(os.Getenv(dispatchRungWindowEnv))
	if window == "" {
		window = dispatchRungWindowDefault
	}
	d, err := parseSinceWindow(window)
	if err != nil {
		return nil, rungSkipBadWindow
	}
	path := filepath.Join(root, dispatchtick.RunsDirName, dispatchTurnJournalName)
	st, err := os.Stat(path)
	if err != nil {
		return nil, rungSkipNoJournal
	}
	if st.Size() > dispatchRungJournalCap {
		return nil, rungSkipJournalBig
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, rungSkipJournalBad
	}
	defer f.Close()
	outcomes, jstats, err := modelroute.ReadTurnOutcomes(f)
	if err != nil {
		return nil, rungSkipJournalBad
	}
	if len(outcomes) == 0 && jstats.Malformed > 0 {
		// Rows are present and not one of them parsed. The reader is deliberately forgiving
		// — a torn final line must not discard a good corpus — but a journal that produced
		// NOTHING is a broken producer, and reporting it as "no evidence yet" would send an
		// operator looking for turns that are already there.
		return nil, rungSkipJournalBad
	}
	evidence, _ := modelroute.FoldTurnOutcomes(outcomes, modelroute.FoldOptions{Since: time.Now().Add(-d)})
	if len(evidence) == 0 {
		return nil, rungSkipNoEvidence
	}
	return evidence, ""
}

// resolveRungPlacement grades the roster's bound models against the folded evidence and
// places this class of work on the cheapest rung that can hold it.
//
// Pure: no clock, no filesystem, no environment. The candidate pool is built with NO hand
// declarations — this path admits measured evidence only, which is asymmetry (1) in the
// file header.
func resolveRungPlacement(roster modelroute.Roster, class modelroute.WorkClass, evidence map[string][]modelroute.ClassEvidence) (*modelroute.Placement, string) {
	base := placementCandidates(roster, nil)
	if len(base) == 0 {
		return nil, rungSkipRosterEmpty
	}
	pool, _, _ := gradedPool(base, evidence, modelroute.DefaultGradeFloor())
	rung, err := roster.Place(class, pool)
	if err != nil {
		return nil, rungSkipRefused
	}
	if !rung.Measured {
		// Not a placement about this model: it is the zero-value capability being admitted on
		// the strength of the top rung. Named separately from no-graded-evidence because the
		// cures differ — this one has evidence, just not enough of it, for this class.
		return nil, rungSkipUnmeasured
	}
	return &rung, ""
}
