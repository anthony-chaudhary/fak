package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

type dispatchWaveCandidateCollector struct {
	root, explicitLane string
	router             dispatchtick.RouterPayload
	explicitIssues     bool
	requestedSet       map[int]bool
	seenRoutes         map[int]bool
	refuse             func(int, string, string, string)
	exclude            map[string]bool
	intentHolds        map[int]string
	liveIssues         map[int]bool
	cooled             map[int]bool
	skipIssues         map[int]bool
	newlyUnblocked     map[int]bool
	readyState         dispatchPrereqState
	snapshot           *runsSnapshot
	profile            string
	lanes              []string
	issueByLane        map[string]int
	metadata           map[string]dispatchWaveCandidate
	candidates         []dispatchorder.Candidate
	unscopedByLane     map[string][]int
	scopedByLane       map[string]bool
}

func (collector *dispatchWaveCandidateCollector) addScopedRoutes() {
	root, router := collector.root, collector.router
	explicitLane := collector.explicitLane
	explicitIssues, requestedSet, seenRoutes := collector.explicitIssues, collector.requestedSet, collector.seenRoutes
	refuse, exclude, intentHolds := collector.refuse, collector.exclude, collector.intentHolds
	liveIssues, cooled, skipIssues := collector.liveIssues, collector.cooled, collector.skipIssues
	newlyUnblocked, readyState, snap := collector.newlyUnblocked, collector.readyState, collector.snapshot
	profile := collector.profile
	meta, cands := collector.metadata, collector.candidates
	unscopedByLane, scopedByLane := collector.unscopedByLane, collector.scopedByLane
	for _, route := range router.Issues {
		if explicitIssues && !requestedSet[route.Number] {
			continue
		}
		seenRoutes[route.Number] = true
		lane := strings.TrimSpace(route.Lane)
		if lane == "" {
			refuse(route.Number, dispatchWaveIssueRefusalRouting, dispatchWaveReasonIssueUnroutable, route.UnroutedReason)
			continue
		}
		if explicitLane != "" && lane != explicitLane {
			refuse(route.Number, dispatchWaveIssueRefusalRouting, dispatchWaveReasonLaneMismatch, fmt.Sprintf("routed lane %q does not match pinned lane %q", lane, explicitLane))
			continue
		}
		if exclude[lane] {
			refuse(route.Number, dispatchWaveIssueRefusalEligibility, dispatchWaveReasonLaneUnavailable, fmt.Sprintf("routed lane %q is excluded or already leased", lane))
			continue
		}
		if detail := strings.TrimSpace(intentHolds[route.Number]); detail != "" {
			refuse(route.Number, dispatchWaveIssueRefusalIntent, leaseref.ReasonIntentCollision, detail)
			continue
		}
		if liveIssues[route.Number] {
			refuse(route.Number, dispatchWaveIssueRefusalEligibility, dispatchWaveReasonIssueInFlight, "a live dispatch worker already owns this issue")
			continue
		}
		if cooled[route.Number] {
			refuse(route.Number, dispatchWaveIssueRefusalEligibility, dispatchWaveReasonIssueCooldown, "the issue is inside the dispatch retry cooldown")
			continue
		}
		if skipIssues[route.Number] {
			refuse(route.Number, dispatchWaveIssueRefusalEligibility, dispatchWaveReasonIssueIneligible, "the issue is held by the caller or attempt budget")
			continue
		}
		paths := append([]string(nil), route.Paths...)
		if len(paths) == 0 {
			unscopedByLane[lane] = append(unscopedByLane[lane], route.Number)
			continue
		}
		scopedByLane[lane] = true
		id := waveCandidateID(lane, route.Number)
		leaseID := dispatchIssueLeaseID(lane, route.Number)
		stepBudget := dispatchWaveRouteStepBudget(route)
		priority := dispatchtick.PriorityWeightDefault
		if grp, ok := router.Lanes[lane]; ok {
			if w, ok := grp.Priority[route.Number]; ok {
				priority = w
			}
		}
		meta[id] = dispatchWaveCandidate{
			ID:         id,
			Lane:       lane,
			LeaseID:    leaseID,
			Issue:      route.Number,
			BaseWeight: priority,
			ReadySince: dispatchIssueReadySinceStamp(root, readyState, route.Number),
			StepBudget: stepBudget,
			Tree:       paths,
			Scoped:     true,
		}
		lastAttempt := int64(0)
		if attemptedAt, ok := snap.latest[route.Number]; ok {
			lastAttempt = attemptedAt.Unix()
		}
		cands = append(cands, dispatchorder.Candidate{
			ID:              id,
			Key:             id,
			Lane:            leaseID,
			Tree:            paths,
			Mode:            "exclusive",
			UpdatedUnix:     dispatchWaveReleaseStamp(dispatchWaveOrderStamp(profile, priority, stepBudget, dispatchtick.IsCoreSourceLaneTree(paths)), newlyUnblocked[route.Number]),
			CreatedUnix:     int64(route.Number),
			LastAttemptUnix: lastAttempt,
		})
	}

	collector.candidates = cands
}

func (collector *dispatchWaveCandidateCollector) addLaneFallbacks() {
	root, router, explicitIssues := collector.root, collector.router, collector.explicitIssues
	explicitLane, exclude, liveIssues := collector.explicitLane, collector.exclude, collector.liveIssues
	cooled, skipIssues := collector.cooled, collector.skipIssues
	newlyUnblocked, readyState, snap := collector.newlyUnblocked, collector.readyState, collector.snapshot
	profile, lanes := collector.profile, collector.lanes
	issueByLane, meta, cands := collector.issueByLane, collector.metadata, collector.candidates
	unscopedByLane, scopedByLane := collector.unscopedByLane, collector.scopedByLane
	for i, lane := range lanes {
		if explicitLane != "" && lane != explicitLane {
			continue
		}
		if exclude[lane] {
			continue
		}
		if scopedByLane[lane] {
			continue
		}
		grp := router.Lanes[lane]
		nums := append([]int(nil), unscopedByLane[lane]...)
		if len(router.Issues) == 0 && !explicitIssues {
			nums = append([]int(nil), grp.Issues...)
		}
		nums = dispatchWaveOrderLaneIssues(root, nums, grp.Priority, readyState)
		issue, ok := firstLaunchableIssue(nums, liveIssues, cooled, skipIssues)
		if !ok {
			continue
		}
		priority := dispatchtick.PriorityWeightDefault
		if w, ok := grp.Priority[issue]; ok {
			priority = w
		}
		id := waveCandidateID(lane, issue)
		if _, exists := meta[id]; exists {
			continue
		}
		leaseID := dispatchIssueLeaseID(lane, issue)
		stepBudget := dispatchWaveLaneStepBudget(grp)
		issueByLane[lane] = issue
		meta[id] = dispatchWaveCandidate{
			ID:         id,
			Lane:       lane,
			LeaseID:    leaseID,
			Issue:      issue,
			BaseWeight: priority,
			ReadySince: dispatchIssueReadySinceStamp(root, readyState, issue),
			StepBudget: stepBudget,
			Tree:       append([]string(nil), grp.Tree...),
		}
		lastAttempt := int64(0)
		if attemptedAt, ok := snap.latest[issue]; ok {
			lastAttempt = attemptedAt.Unix()
		}
		cands = append(cands, dispatchorder.Candidate{
			ID:              id,
			Key:             id,
			Lane:            leaseID,
			Tree:            grp.Tree,
			Mode:            "exclusive",
			UpdatedUnix:     dispatchWaveReleaseStamp(dispatchWaveOrderStamp(profile, priority, stepBudget, dispatchtick.IsCoreSourceLaneTree(grp.Tree)), newlyUnblocked[issue]),
			CreatedUnix:     int64(grp.Count*len(lanes) + (len(lanes) - i)),
			LastAttemptUnix: lastAttempt,
		})
	}

	collector.candidates = cands
}
