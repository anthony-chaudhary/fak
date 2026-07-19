package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

// humanBlockedGuardEscalations folds each session to its latest Stop row and projects
// only genuine HUMAN_RESIDUAL outcomes into the dispatcher's existing closed token.
// Session ids are opaque, so Number stays zero and Title carries the stable identity.
func humanBlockedGuardEscalations(ledgerPath string) []dispatchtick.SkippedIssue {
	ledgerPath = strings.TrimSpace(ledgerPath)
	if ledgerPath == "" {
		return nil
	}
	b, err := os.ReadFile(ledgerPath)
	if err != nil {
		return nil
	}
	rows := jsonlledger.Parse(string(b), func(r guardStopRecord) bool { return r.Schema == guardStopRecordSchema })
	latest := map[string]guardStopRecord{}
	order := []string{}
	for _, row := range rows {
		session := strings.TrimSpace(row.Session)
		if session == "" {
			continue
		}
		if _, seen := latest[session]; !seen {
			order = append(order, session)
		}
		latest[session] = row
	}
	out := make([]dispatchtick.SkippedIssue, 0, len(order))
	for _, session := range order {
		row := latest[session]
		if guardStopDisposition(row.Disposition) != stopDispOperatorDirectedEscalate {
			continue
		}
		out = append(out, dispatchtick.SkippedIssue{
			Title:      "guard session " + session,
			Reason:     reasonBlockedByHuman,
			NextAction: "operator must resolve the HUMAN_RESIDUAL escalation before dispatch resumes",
			WorkUnit:   "session",
		})
	}
	return out
}

func mergeHumanBlockedSkipped(routerRows, guardRows []dispatchtick.SkippedIssue) []dispatchtick.SkippedIssue {
	out := make([]dispatchtick.SkippedIssue, 0, len(routerRows)+len(guardRows))
	seen := map[string]bool{}
	add := func(row dispatchtick.SkippedIssue) {
		key := strings.ToLower(strings.TrimSpace(row.Title))
		if row.Number != 0 {
			key = fmt.Sprintf("issue:%d", row.Number)
		}
		if key == "" || seen[key] {
			return
		}
		seen[key] = true
		out = append(out, row)
	}
	for _, row := range routerRows {
		add(row)
	}
	for _, row := range guardRows {
		add(row)
	}
	return out
}

// holdSteerPausedForRoute is the operator-pause half of this seam (#5031): the
// impure wrapper the dispatch route calls every tick. It loads the overlay
// pause ledger (`fak steer pause` / `fak steer resume`, .fak/steer-pauses.jsonl)
// and moves each actively paused unit's BOUND issue out of its dispatch lane
// into the skipped set under the dispatcher's EXISTING BLOCKED_BY_HUMAN token —
// the same closed reason the guard escalations above project, so `fak dispatch
// skipped` and every other consumer see the hold with no new vocabulary. It
// FAILS OPEN: a missing or unreadable ledger holds nothing, and dispatch never
// stalls on the overlay's absence.
func holdSteerPausedForRoute(root string, payload dispatchtick.RouterPayload) dispatchtick.RouterPayload {
	active := steerpr.ActivePauses(steerpr.LoadPauses(steerpr.PauseLedgerPath(root)))
	if len(active) == 0 {
		return payload
	}
	return applySteerPauseHold(payload, active)
}

// applySteerPauseHold is the pure pause-hold fold (the shape of
// applyKnownBadHold, matched by bound issue NUMBER instead of path
// intersection): given a routed payload and the actively paused units, it
// moves every routed issue an operator paused out of its dispatch lane and
// into the skipped set with reason BLOCKED_BY_HUMAN, and leaves every other
// issue exactly where it was. Same inputs -> same payload out, and the input's
// shared maps are never mutated — the hold rewrites only the FUTURE routing
// view, which is exactly why a pause is not a kill: an in-flight worker's run
// state is not touched by anything reachable from here.
func applySteerPauseHold(payload dispatchtick.RouterPayload, active map[string]steerpr.Pause) dispatchtick.RouterPayload {
	if len(active) == 0 {
		return payload
	}
	byIssue := map[int]steerpr.Pause{}
	for _, p := range active {
		if n := p.IssueNumber(); n > 0 {
			byIssue[n] = p
		}
	}
	held := map[int]steerpr.Pause{}
	stepByNum := map[int]int{}
	var heldRoutes []dispatchtick.IssueRoute
	for _, iss := range payload.Issues {
		stepByNum[iss.Number] = routeIssueSteps(iss)
		if _, done := held[iss.Number]; done {
			continue
		}
		if p, isPaused := byIssue[iss.Number]; isPaused {
			held[iss.Number] = p
			heldRoutes = append(heldRoutes, iss)
		}
	}
	if len(held) == 0 {
		return payload
	}

	// Rebuild the lane map without the held issues, dropping any lane held to
	// empty and re-deriving its count/step budget; stale per-issue maps are
	// pruned so a consumer reading them cannot resurrect the hold.
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
		grp.Priority = steerPausePruneInt(grp.Priority, held)
		grp.Generation = steerPausePruneStr(grp.Generation, held)
		grp.WorkUnits = steerPausePruneStr(grp.WorkUnits, held)
		grp.IssueSteps = steerPausePruneInt(grp.IssueSteps, held)
		routedSteps += steps
		newLanes[lane] = grp
	}

	// Drop held issues from the candidate list too, so a held issue is never
	// offered as a routable candidate.
	keptIssues := make([]dispatchtick.IssueRoute, 0, len(payload.Issues))
	for _, iss := range payload.Issues {
		if _, isHeld := held[iss.Number]; isHeld {
			continue
		}
		keptIssues = append(keptIssues, iss)
	}

	// Append the held issues to the skipped set as BLOCKED_BY_HUMAN rows —
	// the existing token, so this is backpressure through the existing seam,
	// not a second mechanism — then re-sort highest-number-first to match the
	// router's own ordering.
	skipped := append([]dispatchtick.SkippedIssue(nil), payload.SkippedHumanBlocked...)
	for _, iss := range heldRoutes {
		skipped = append(skipped, dispatchtick.SkippedIssue{
			Number:        iss.Number,
			Title:         iss.Title,
			Reason:        reasonBlockedByHuman,
			NextAction:    steerPauseNextAction(held[iss.Number]),
			WorkUnit:      iss.WorkUnit,
			ExpectedSteps: iss.ExpectedSteps,
		})
	}
	sort.SliceStable(skipped, func(i, j int) bool { return skipped[i].Number > skipped[j].Number })

	byReason := map[string]int{}
	for k, v := range payload.Counts.SkippedByReason {
		byReason[k] = v
	}
	byReason[reasonBlockedByHuman] += len(held)

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

// steerPauseNextAction is the "what unblocks this" hint a paused row carries:
// who paused it, since when, the reason if one was recorded, and the release
// verb — pause is not a kill, so the hint also says an in-flight worker may
// still land.
func steerPauseNextAction(p steerpr.Pause) string {
	reason := ""
	if strings.TrimSpace(p.Note) != "" {
		reason = fmt.Sprintf(" (%s)", strings.TrimSpace(p.Note))
	}
	return fmt.Sprintf("operator %s paused the %s intent since %s%s — an in-flight worker may still land; `fak steer resume %s` releases the hold",
		p.By, p.Leaf, p.At, reason, p.Leaf)
}

// steerPausePruneInt / steerPausePruneStr return m without the held keys, or
// nil when nothing remains, so a rebuilt lane group carries no stale per-issue
// entry for a held number. (Local twins of the known-bad hold's pruners, which
// are typed to its record.)
func steerPausePruneInt(m map[int]int, held map[int]steerpr.Pause) map[int]int {
	if len(m) == 0 {
		return m
	}
	out := make(map[int]int, len(m))
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

func steerPausePruneStr(m map[int]string, held map[int]steerpr.Pause) map[int]string {
	if len(m) == 0 {
		return m
	}
	out := make(map[int]string, len(m))
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
