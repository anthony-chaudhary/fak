package main

// dispatch_prereq.go -- the dependency soft-hold seam. Like the known-bad hold (dispatch_knownbad.go)
// it is a runtime overlay the dispatch verbs apply to the built RouterPayload, NOT part of the
// (lease-held) routing fold. It reads the "depends-on:/blocked-by: #N" edges the router already
// parsed onto each IssueRoute.BlockedBy and, reusing the tested internal/dispatchorder engine, holds
// back any dispatchable leaf whose prerequisite is still an OPEN candidate this tick. A held leaf is
// removed from its lane (so PickTargetIssue cannot select it) and surfaced in the skipped set with
// reason BLOCKED_BY_OPEN_PREREQ -- legible, not silently dropped.
//
// The hold is single-tick and self-clearing: the tick refetches the full open backlog each iteration,
// so a prerequisite is present (and its dependent held) exactly until it CLOSES, at which point it
// leaves the candidate set and the engine fails open (the dependent dispatches). No ledger, no clock,
// no persisted state -- pure over the payload, so a peer can re-derive it.

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// reasonBlockedByOpenPrereq is the closed-vocabulary skip reason a SkippedIssue carries when it was
// held back because a prerequisite it named ("depends-on:/blocked-by: #N") is still an open candidate
// this tick. Registered in dos.toml [reasons.BLOCKED_BY_OPEN_PREREQ] so the skip is a structured,
// refusal verifiable with `dos man wedge BLOCKED_BY_OPEN_PREREQ --explain`, not free text.
const reasonBlockedByOpenPrereq = "BLOCKED_BY_OPEN_PREREQ"

// holdOpenPrereqForRoute is the pure dependency soft-hold fold: given a routed payload, it moves every
// DISPATCHABLE issue (in a lane) whose BlockedBy names a prerequisite still open this tick out of its
// lane and into the skipped set with reason BLOCKED_BY_OPEN_PREREQ, and leaves every other issue where
// it was. It is payload-only (no root/ledger/clock), so same payload in -> same payload out.
//
// The presence universe -- what counts as "still open" -- is payload.Issues ∪ payload.SkippedHumanBlocked,
// i.e. the full open set the router saw: an open-but-undispatchable prerequisite (human-blocked,
// known-bad-held, unrouted) still holds its dependent, because it is not CLOSED. Only a prerequisite
// absent from both (closed -> gone) fails open. The fail-open and 2-cycle-safe invariants come from the
// shared dispatchorder engine (BlockedByOpenPrereq); this wrapper only projects the payload into that
// engine's Candidate set and folds the verdict back.
func holdOpenPrereqForRoute(payload dispatchtick.RouterPayload) dispatchtick.RouterPayload {
	// Build the candidate set: every routed/unrouted issue carries its BlockedBy edges; every
	// already-skipped-but-still-open issue contributes PRESENCE only (SkippedIssue has no body/edges),
	// so a dependent of a skipped-open prerequisite is still held.
	cands := make([]dispatchorder.Candidate, 0, len(payload.Issues)+len(payload.SkippedHumanBlocked))
	for _, iss := range payload.Issues {
		cands = append(cands, dispatchorder.Candidate{ID: strconv.Itoa(iss.Number), BlockedBy: iss.BlockedBy})
	}
	for _, sk := range payload.SkippedHumanBlocked {
		cands = append(cands, dispatchorder.Candidate{ID: strconv.Itoa(sk.Number)})
	}
	blocked := dispatchorder.BlockedByOpenPrereq(cands)
	if len(blocked) == 0 {
		return payload
	}

	// Identify the held issues in payload.Issues order (deterministic). Only a DISPATCHABLE issue
	// (in a lane) is held -- an unrouted issue is not pickable anyway, so holding it would be noise.
	// The map value is the issue's still-open prerequisite ids, which is also this hold's evidence:
	// a number is present here only when `open` was non-empty, so presence == held.
	held := map[int][]string{}
	stepByNum := map[int]int{}
	var heldRoutes []dispatchtick.IssueRoute
	for _, iss := range payload.Issues {
		stepByNum[iss.Number] = routeIssueSteps(iss)
		if iss.Lane == "" {
			continue
		}
		open := blocked[strconv.Itoa(iss.Number)]
		if _, done := held[iss.Number]; len(open) == 0 || done {
			continue
		}
		held[iss.Number] = open
		heldRoutes = append(heldRoutes, iss)
	}
	if len(held) == 0 {
		return payload
	}
	// The rest -- lane rebuild, candidate drop, skipped rows, counts -- is the shared
	// dispatch-hold rewrite (dispatch_hold.go).
	return applyDispatchHold(payload, held, heldRoutes, stepByNum, reasonBlockedByOpenPrereq,
		func(iss dispatchtick.IssueRoute) string {
			return openPrereqNextAction(held[iss.Number])
		})
}

// openPrereqNextAction is the "what unblocks this" hint a held row carries: the open prerequisite(s)
// the issue named, and the self-clearing exit (they close -> this dispatches next tick).
func openPrereqNextAction(open []string) string {
	refs := make([]string, len(open))
	for i, n := range open {
		refs[i] = "#" + n
	}
	joined := strings.Join(refs, ", ")
	return fmt.Sprintf("held: prerequisite %s still open; dispatches once %s closes", joined, joined)
}

// openPrereqBlockedSkipped selects the BLOCKED_BY_OPEN_PREREQ rows out of the router's skipped set --
// the dependency holds, distinct from the static human-blocked and dynamic known-bad rows.
func openPrereqBlockedSkipped(router dispatchtick.RouterPayload) []dispatchtick.SkippedIssue {
	out := make([]dispatchtick.SkippedIssue, 0)
	for _, s := range router.SkippedHumanBlocked {
		if s.Reason == reasonBlockedByOpenPrereq {
			out = append(out, s)
		}
	}
	return out
}
