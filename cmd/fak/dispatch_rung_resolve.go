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
// Four deliberate asymmetries with the `fak route --place` operator surface:
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
//  4. A launchability gate (rungReach). The CLI reports a rung; this path LAUNCHES onto one,
//     and the two are not the same claim. A cheapest rung the worker's backend cannot dial
//     is a walled slot, not a saving, so the operator declares which accounts are reachable
//     and an undeclared one places nothing.

const (
	// dispatchRungWindowEnv overrides the freshness window; dispatchRungWindowDefault is
	// what an operator who has not thought about it gets. Same grammar as --since.
	dispatchRungWindowEnv     = "FLEET_DISPATCH_RUNG_WINDOW"
	dispatchRungWindowDefault = "30d"

	// dispatchRungJournalCap bounds the per-launch read. ~40k turns at a typical row size.
	dispatchRungJournalCap = 8 << 20

	// dispatchRungAccountsEnv is the operator's declaration of which roster ACCOUNTS the
	// launching backend can actually dial — a comma-separated list of Account.ID, or "*".
	// There is no default: see rungReach.
	dispatchRungAccountsEnv = "FLEET_DISPATCH_RUNG_ACCOUNTS"
	// dispatchRungAccountsAll is the whole-roster assertion. It is spelled, never inferred.
	dispatchRungAccountsAll = "*"
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

	// The launchability half. Distinct reasons because the cures are different: the first is
	// an operator who has not said anything yet, the second is one who said something that
	// selects nothing (a typo'd account id looks exactly like an idle ladder otherwise).
	rungSkipNoReachDecl = "no-account-reach-declared" // nobody declared which accounts are dialable
	rungSkipUnreachable = "accounts-bind-no-models"   // the declared accounts bind nothing placeable
)

// dispatchRungPlacement is the ladder's DECLARATION: the config-surface setting
// `fak dispatch tick --rung-placement` writes, zero value OFF. Declared rather than read from
// the process environment because a placement posture is behavior, not a credential — the
// CONFIG_NOT_ENV rule internal/envconfiglint ratchets. A package seam rather than a parameter
// for the reason dispatchTickView is one: the switch is read from three call chains (the
// placement half here, the escalation half in dispatch_rung_escalate.go, and the operator
// ledger surface in dispatch_rung_ledger.go), none of which otherwise needs to carry it.
// evaluateDispatchTick publishes the parsed dispatchTickOptions into it, beside its siblings
// in dispatch_placement_evidence.go.
var dispatchRungPlacement bool

// dispatchRungPlacementEnabled reports whether the opt-in placement-ladder seam
// (`fak dispatch tick --rung-placement`) is switched on. Default (undeclared) is OFF.
//
// Off-by-default is not timidity here. A device- or fleet-rung model id is only launchable
// by a worker whose seat routes to that endpoint, and nothing on this path can PROBE that.
// What it can do is refuse to guess: rungReach makes the operator name the accounts their
// seats reach, so the claim is written down and enforced against every rung instead of
// being implied by this one boolean. That is an assertion made explicit, not a verified
// fact — the fleet trust boundary (#5421, track G) is what would make it verifiable.
func dispatchRungPlacementEnabled() bool { return dispatchRungPlacement }

// rungReach is the operator's declaration of which roster accounts THIS backend can dial.
//
// It exists because a placement and a launch are different claims. `Roster.Place` answers
// "which rung is cheapest for this class of work"; it says nothing about whether the worker
// about to be launched can open a socket to that rung. A device-rung id is only launchable
// by a seat whose backend is already pointed at that endpoint, and WorkerLaunch carries no
// base URL and no credential — the model id is the whole instruction. So a pin the backend
// cannot serve is not a cheaper rung, it is a slot that walls on model_unknown.
//
// Fail-closed, with NO default. Turning the ladder on and declaring nothing places nothing.
// That is deliberate and it is the difference between a boundary and a decoration: an
// allowlist that defaults to "everything" reads as protection while enforcing nothing, so an
// operator who mistypes the variable name would get unrestricted automatic placement while
// believing they had constrained it. The whole-roster case is available and must be SPELLED
// ("*"), which keeps it an assertion the operator made rather than one the code assumed.
//
// This is a LAUNCHABILITY boundary, not a residency one. It answers "can this worker reach
// that endpoint", not "may this payload leave the box" — the second question belongs to the
// residency floor in internal/engine (#5421) and is not weakened, widened, or restated here.
// Nothing in this file exempts a rung from that floor, including the vendor rung, which is
// filtered by exactly the same rule as the other two.
type rungReach struct {
	All      bool
	Accounts map[string]bool
}

// dispatchRungReach parses the declaration. ok is false when the operator declared nothing
// usable — an unset variable, or one that names no account at all — and the ladder then has
// nothing it is allowed to dial.
func dispatchRungReach() (rungReach, bool) {
	raw, ok := os.LookupEnv(dispatchRungAccountsEnv)
	if !ok {
		return rungReach{}, false
	}
	reach := rungReach{Accounts: map[string]bool{}}
	for _, f := range strings.Split(raw, ",") {
		f = strings.TrimSpace(f)
		switch f {
		case "":
			continue
		case dispatchRungAccountsAll:
			reach.All = true
		default:
			// Account ids are matched exactly, the same way Roster.account matches them. A
			// case-folded match here would admit an account the roster would then fail to
			// resolve, which is a worse answer than refusing the typo.
			reach.Accounts[f] = true
		}
	}
	if !reach.All && len(reach.Accounts) == 0 {
		return rungReach{}, false
	}
	return reach, true
}

// filter drops every candidate whose serving account the operator did not declare dialable.
//
// It asks Roster.Resolve — the SAME function Place will ask — rather than re-deriving the
// model→account mapping from Bindings, so the account this gate judges is by construction
// the account the placement would have used. (Resolve's fall back to the default account
// cannot fire here: every candidate came from a binding.)
//
// A candidate Resolve REJECTS is kept, not dropped. A dangling binding is a misconfigured
// roster, Place is already fail-loud about it, and swallowing it here would turn a broken
// roster into a quiet "nothing was reachable" — a different diagnosis with a different cure.
func (r rungReach) filter(roster modelroute.Roster, in []modelroute.Candidate) []modelroute.Candidate {
	if r.All {
		return in
	}
	out := make([]modelroute.Candidate, 0, len(in))
	for _, c := range in {
		t, err := roster.Resolve(c.Model)
		if err != nil || r.Accounts[t.Account] {
			out = append(out, c)
		}
	}
	return out
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
	// Read before the roster load and the journal read for the same reason the precedence
	// check is: an undeclared reach is knowable from the environment alone, and a launch
	// should not pay for two file reads to be told the ladder was never allowed to dial.
	reach, ok := dispatchRungReach()
	if !ok {
		return p, rungSkipNoReachDecl
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
	rung, reason := resolveRungPlacement(roster, class, evidence, reach)
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
//
// The reach filter runs BEFORE the walk rather than as a check on the winner, so an
// unreachable device rung falls through to a reachable fleet rung instead of abandoning the
// slot to the seat's vendor default — narrowing what the fleet can dial should cost it the
// unreachable rungs, not the ladder. Narrowing cannot promote a weaker model: Place fixes
// its top rung from the static zone ladder, not from the pool, so an unmeasured candidate
// stays barred by rule 2 even when every rung above it has been filtered away, and rule 1's
// tier floor is a property of the WORK and never moves at all.
func resolveRungPlacement(roster modelroute.Roster, class modelroute.WorkClass, evidence map[string][]modelroute.ClassEvidence, reach rungReach) (*modelroute.Placement, string) {
	base := placementCandidates(roster, nil)
	if len(base) == 0 {
		return nil, rungSkipRosterEmpty
	}
	if base = reach.filter(roster, base); len(base) == 0 {
		return nil, rungSkipUnreachable
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
