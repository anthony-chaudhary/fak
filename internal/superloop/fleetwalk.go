package superloop

// fleetwalk.go — the generic meta-walker's worst-first KEY (issue #4958): ONE product
// over the three per-loop dimensions the operator supervises a fleet by.
//
// The `tend` fold used to score a KindLoop member on liveness alone (dark/stale), so
// the worst-first SELECT could never surface a loop that is up-but-not-working
// (SPINNING, #4956) or emitting follow-on work nobody advances (ORPHANED, #4957).
// The tend-fleet meta-walk ranks every enumerated fleet loop on the PRODUCT of the
// three dimensions instead:
//
//	liveness   ×  progress   ×  follow-on
//	stale = 2     spinning = 2   orphaned = 2
//	live  = 1     else     = 1   else     = 1
//
// [FleetDebt] is that product minus one, so a clean live leaf folds to debt 0 and the
// existing worst-first sort (tier, then debt descending) ranks the fleet without a
// rival walker. A product — not a sum — because the dimensions COMPOUND: a stale loop
// that is also spinning is not "two small problems", it is one loop paying cadence
// cost for zero verified output (debt 3, ahead of a merely-stale peer at 1), and a
// spinning loop whose emissions are also orphaned pays on both grains at once
// (debt 3), ahead of either fault alone.
//
// Two deliberate honesty edges, both inherited rather than invented here:
//
//   - DARK stays on the [MemberStatus].Dark flag (worst band, tier 0), exactly as the
//     roster enumeration (#4955) and the hand-named KindLoop fold already carry it —
//     the product's liveness factor weighs the STALE slippage of a still-ticking
//     loop, never double-counts a dark one.
//   - an UNREAD axis multiplies by 1 (surface-only): an unmeasured progress read
//     (ProgressUnmeasured) or an unresolvable follow-on read (FollowonUnknown) never
//     fabricates debt — the same fail-closed asymmetry ClassifyProgress (#4956) and
//     ClassifyFollowon (#4957) keep at the verdict grain.
//
// The verdicts themselves come from the sibling leaves — [ClassifyProgress] (#4956)
// and [ClassifyFollowon] (#4957); this file only FOLDS them into the ranking key the
// tend-fleet intent walks worst-first.

// fleetLivenessFactor weighs the liveness dimension of a still-ticking loop: a stale
// loop (slipping past its cadence) doubles the product; a live one is neutral. A DARK
// loop's urgency is already carried by the Dark flag into tier 0 — the worst band —
// so it stays factor 1 here rather than being counted twice.
func fleetLivenessFactor(state string, dark bool) int {
	if !dark && state == "stale" {
		return 2
	}
	return 1
}

// fleetProgressFactor weighs the verified-progress dimension (#4956): SPINNING —
// ticking on cadence with zero advanced verified progress — doubles the product.
// Every other verdict (advancing, a proven idle park, an unproven idle, an
// unmeasured read, an unread axis) is neutral: only the ledger-witnessed
// live-but-producing-nothing state is debt.
func fleetProgressFactor(p MemberProgress) int {
	if p == ProgressSpinning {
		return 2
	}
	return 1
}

// fleetFollowOnFactor weighs the follow-on dimension (#4957): ORPHANED — emitted
// work nobody advances — doubles the product. Advancing, unknown, and unread all
// stay neutral (an unresolvable emission never fabricates debt).
func fleetFollowOnFactor(f MemberFollowon) int {
	if f == FollowonOrphaned {
		return 2
	}
	return 1
}

// FleetProduct folds one fleet loop's three dimension verdicts into the worst-first
// product (>= 1; 1 = clean). Inputs are the roster's folded liveness (state + dark),
// the #4956 progress verdict, and the #4957 follow-on verdict; an unread axis ("")
// multiplies by 1.
func FleetProduct(state string, dark bool, prog MemberProgress, followOn MemberFollowon) int {
	return fleetLivenessFactor(state, dark) * fleetProgressFactor(prog) * fleetFollowOnFactor(followOn)
}

// FleetDebt is [FleetProduct] shifted onto the walk's debt scale: product minus one,
// so a clean live loop folds to 0 and the ordinary worst-first sort (tier, then debt
// descending) ranks the fleet on the product with no rival walker. The single source
// of truth for a fleet-enumerated loop's debt: [LoopFleetStatuses] applies it over
// the liveness axis alone, and the shell RE-applies it once the progress/follow-on
// verdicts are read, so the two folds can never disagree.
func FleetDebt(state string, dark bool, prog MemberProgress, followOn MemberFollowon) int {
	return FleetProduct(state, dark, prog, followOn) - 1
}
