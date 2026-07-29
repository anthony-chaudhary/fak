package main

// dispatch_hold.go -- the one payload rewrite shared by every dispatch soft-hold.
//
// `fak dispatch` grew three independent holds, each a runtime overlay the dispatch verbs
// apply to the built RouterPayload: the operator pause (BLOCKED_BY_HUMAN,
// dispatch_guard_escalations.go), the live known-bad scope hold (BLOCKED_BY_KNOWN_BAD,
// dispatch_knownbad.go) and the open-prerequisite hold (BLOCKED_BY_OPEN_PREREQ,
// dispatch_prereq.go). Only the SELECTION differs between them -- who is held, on what
// evidence, and what hint unblocks them. Everything after the selection was the same
// rewrite copied three times, so it lives here once.
//
// What stays with each hold: its predicate, its closed-vocabulary skip reason, and its
// next-action hint. What moved here: rebuild the lanes without the held issues, drop them
// from the candidate list, append them to the skipped set as `reason` rows, re-sort
// highest-number-first to match the router's own ordering, and re-derive Counts. There is
// no clock and no I/O here, so same payload in -> same payload out and every hold stays a
// witness a peer can re-derive.

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// applyDispatchHold rewrites payload so that every issue named in `held` is removed from
// its lane and from the candidate list and instead appears in the skipped set carrying
// `reason` and the per-issue hint `nextAction` returns. `heldRoutes` is the held issues in
// payload.Issues order -- the caller builds it while selecting, so the emitted rows stay
// deterministic -- and `stepByNum` maps issue number to its step budget so the per-lane
// budgets can be re-derived after removal.
//
// H is the caller's own per-issue evidence type (a paused intent, a known-bad record, the
// open prerequisite ids): PRESENCE in `held` is what "held" means here, the value is only
// the caller's business and is read back through `nextAction`.
//
// A hold with nothing held must return the payload untouched before calling this, exactly
// as the three copies did -- this function assumes len(held) > 0.
func applyDispatchHold[H any](
	payload dispatchtick.RouterPayload,
	held map[int]H,
	heldRoutes []dispatchtick.IssueRoute,
	stepByNum map[int]int,
	reason string,
	nextAction func(iss dispatchtick.IssueRoute) string,
) dispatchtick.RouterPayload {
	// Rebuild the lane map without the held issues, dropping any lane held to empty and
	// re-deriving its count/step budget. Stale per-issue maps (priority/generation/etc.)
	// for a held number are pruned so a consumer reading them cannot resurrect the hold.
	// This removal is what makes PickTargetIssue unable to select a held leaf: pick.Numbers
	// flows from the lane's Issues.
	newLanes := make(map[string]dispatchtick.RouterLaneGroup, len(payload.Lanes))
	routedSteps := 0
	for lane, grp := range payload.Lanes {
		kept := make([]int, 0, len(grp.Issues))
		steps := 0
		for _, n := range grp.Issues {
			if _, isHeld := held[n]; isHeld {
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
		grp.Priority = pruneHeldEntries(grp.Priority, held)
		grp.Generation = pruneHeldEntries(grp.Generation, held)
		grp.WorkUnits = pruneHeldEntries(grp.WorkUnits, held)
		grp.IssueSteps = pruneHeldEntries(grp.IssueSteps, held)
		routedSteps += steps
		newLanes[lane] = grp
	}

	// Drop held issues from the candidate list too, so `fak dispatch route --json` never
	// offers a held issue as a routable candidate.
	keptIssues := make([]dispatchtick.IssueRoute, 0, len(payload.Issues))
	for _, iss := range payload.Issues {
		if _, isHeld := held[iss.Number]; isHeld {
			continue
		}
		keptIssues = append(keptIssues, iss)
	}

	// Append the held issues to the skipped set, then re-sort the whole skipped slice
	// highest-number-first to match the router's own ordering.
	skipped := append([]dispatchtick.SkippedIssue(nil), payload.SkippedHumanBlocked...)
	for _, iss := range heldRoutes {
		skipped = append(skipped, dispatchtick.SkippedIssue{
			Number:        iss.Number,
			Title:         iss.Title,
			Reason:        reason,
			NextAction:    nextAction(iss),
			WorkUnit:      iss.WorkUnit,
			ExpectedSteps: iss.ExpectedSteps,
		})
	}
	sort.SliceStable(skipped, func(i, j int) bool { return skipped[i].Number > skipped[j].Number })

	byReason := map[string]int{}
	for k, v := range payload.Counts.SkippedByReason {
		byReason[k] = v
	}
	byReason[reason] += len(held)

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

// pruneHeldEntries returns m without the held keys, or nil when nothing remains, so a
// rebuilt lane group carries no stale per-issue entry for a held number. An already-empty
// m is returned as-is (nil stays nil), matching the three per-hold pruners it replaces.
func pruneHeldEntries[V any, H any](m map[int]V, held map[int]H) map[int]V {
	if len(m) == 0 {
		return m
	}
	out := make(map[int]V, len(m))
	for k, v := range m {
		if _, isHeld := held[k]; !isHeld {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
