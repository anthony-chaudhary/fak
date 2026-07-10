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
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// reasonBlockedByOpenPrereq is the closed-vocabulary skip reason a SkippedIssue carries when it was
// held back because a prerequisite it named ("depends-on:/blocked-by: #N") is still an open candidate
// this tick. Registered in dos.toml [reasons.BLOCKED_BY_OPEN_PREREQ] so the skip is a structured,
// `dos check-reason`-verifiable refusal, not free text.
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
	held := map[int]bool{}
	openPrereqs := map[int][]string{}
	stepByNum := map[int]int{}
	var heldRoutes []dispatchtick.IssueRoute
	for _, iss := range payload.Issues {
		stepByNum[iss.Number] = routeIssueSteps(iss)
		if iss.Lane == "" {
			continue
		}
		open := blocked[strconv.Itoa(iss.Number)]
		if len(open) == 0 || held[iss.Number] {
			continue
		}
		held[iss.Number] = true
		openPrereqs[iss.Number] = open
		heldRoutes = append(heldRoutes, iss)
	}
	if len(held) == 0 {
		return payload
	}

	// Rebuild the lane map without the held issues, dropping any lane held to empty and re-deriving
	// its count/step budget. Stale per-issue maps for a held number are pruned so a consumer reading
	// them cannot resurrect the hold. This removal is what makes PickTargetIssue unable to select a
	// held leaf: pick.Numbers flows from the lane's Issues.
	newLanes := make(map[string]dispatchtick.RouterLaneGroup, len(payload.Lanes))
	routedSteps := 0
	for lane, grp := range payload.Lanes {
		kept := make([]int, 0, len(grp.Issues))
		steps := 0
		for _, n := range grp.Issues {
			if held[n] {
				continue
			}
			kept = append(kept, n)
			steps += stepByNum[n]
		}
		if len(kept) == 0 {
			continue
		}
		grp.Issues = kept
		grp.Count = len(kept)
		grp.StepBudget = steps
		grp.Priority = prunePrereqIntMap(grp.Priority, held)
		grp.Generation = prunePrereqStrMap(grp.Generation, held)
		grp.WorkUnits = prunePrereqStrMap(grp.WorkUnits, held)
		grp.IssueSteps = prunePrereqIntMap(grp.IssueSteps, held)
		routedSteps += steps
		newLanes[lane] = grp
	}

	// Drop held issues from the candidate list too, so `fak dispatch route --json` never offers a
	// held issue as a routable candidate.
	keptIssues := make([]dispatchtick.IssueRoute, 0, len(payload.Issues))
	for _, iss := range payload.Issues {
		if held[iss.Number] {
			continue
		}
		keptIssues = append(keptIssues, iss)
	}

	// Append the held issues to the skipped set as BLOCKED_BY_OPEN_PREREQ rows, then re-sort the whole
	// skipped slice highest-number-first to match the router's own ordering.
	skipped := append([]dispatchtick.SkippedIssue(nil), payload.SkippedHumanBlocked...)
	for _, iss := range heldRoutes {
		skipped = append(skipped, dispatchtick.SkippedIssue{
			Number:        iss.Number,
			Title:         iss.Title,
			Reason:        reasonBlockedByOpenPrereq,
			NextAction:    openPrereqNextAction(openPrereqs[iss.Number]),
			WorkUnit:      iss.WorkUnit,
			ExpectedSteps: iss.ExpectedSteps,
		})
	}
	sort.SliceStable(skipped, func(i, j int) bool { return skipped[i].Number > skipped[j].Number })

	byReason := map[string]int{}
	for k, v := range payload.Counts.SkippedByReason {
		byReason[k] = v
	}
	byReason[reasonBlockedByOpenPrereq] += len(held)

	payload.Lanes = newLanes
	payload.Issues = keptIssues
	payload.SkippedHumanBlocked = skipped
	payload.Counts.Routed -= len(held)
	if payload.Counts.Routed < 0 {
		payload.Counts.Routed = 0
	}
	payload.Counts.RoutedStepBudget = routedSteps
	payload.Counts.SkippedHumanBlocked = len(skipped)
	payload.Counts.SkippedByReason = byReason
	return payload
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

// prunePrereqIntMap / prunePrereqStrMap return m without the held keys (nil when nothing remains), so
// a rebuilt lane group carries no stale per-issue entry for a held number.
func prunePrereqIntMap(m map[int]int, held map[int]bool) map[int]int {
	if len(m) == 0 {
		return m
	}
	out := make(map[int]int, len(m))
	for k, v := range m {
		if !held[k] {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func prunePrereqStrMap(m map[int]string, held map[int]bool) map[int]string {
	if len(m) == 0 {
		return m
	}
	out := make(map[int]string, len(m))
	for k, v := range m {
		if !held[k] {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
