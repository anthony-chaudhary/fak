package dispatchtick

import (
	"fmt"
	"sort"
	"strings"
)

func ComputeRouterCoverage(issuesFetched, issueLimit int) RouterCoverage {
	if issueLimit <= 0 {
		issueLimit = 1000
	}
	truncated := issuesFetched >= issueLimit
	notes := []string{}
	if truncated {
		notes = append(notes, fmt.Sprintf("gh fetch returned %d open issue(s) = the --issue-limit cap; older open issues may be unrouted - raise --issue-limit", issuesFetched))
	}
	return RouterCoverage{
		Complete:      !truncated,
		Truncated:     truncated,
		IssuesFetched: issuesFetched,
		IssueLimit:    issueLimit,
		Notes:         notes,
	}
}

type RouterPayloadInput struct {
	Workspace string
	Routes    []IssueRoute
	Trees     map[string][]string
	// Priority maps an issue number to its dispatch-priority weight for issues
	// that carry a priority/* label (unlabeled issues are omitted). It drives the
	// priority-first ordering of each lane group's Issues (#1395).
	Priority         map[int]int
	MaxUnroutedFrac  float64
	FetchError       string
	Coverage         RouterCoverage
	SkippedBlocked   []Issue
	SkippedDuplicate []Issue
	BlockedLabelName string
}

func BuildRouterPayload(in RouterPayloadInput) RouterPayload {
	maxUnrouted := in.MaxUnroutedFrac
	if maxUnrouted == 0 {
		maxUnrouted = 0.25
	}
	byConf := map[string]int{}
	for conf := range ConfidenceRank {
		byConf[conf] = 0
	}
	lanes := map[string]RouterLaneGroup{}
	laneRoutes := map[string][]IssueRoute{}
	routedStepBudget := 0
	for _, r := range in.Routes {
		byConf[r.Confidence] = byConf[r.Confidence] + 1
		if r.Lane == "" {
			continue
		}
		laneRoutes[r.Lane] = append(laneRoutes[r.Lane], r)
		grp := lanes[r.Lane]
		grp.Tree = append([]string(nil), in.Trees[r.Lane]...)
		grp.Count++
		stepBudget := routeStepBudget(r)
		grp.StepBudget += stepBudget
		routedStepBudget += stepBudget
		grp.Issues = append(grp.Issues, r.Number)
		if r.WorkUnit != "" {
			if grp.WorkUnits == nil {
				grp.WorkUnits = map[int]string{}
			}
			grp.WorkUnits[r.Number] = r.WorkUnit
		}
		if r.ExpectedSteps > 0 {
			if grp.IssueSteps == nil {
				grp.IssueSteps = map[int]int{}
			}
			grp.IssueSteps[r.Number] = r.ExpectedSteps
		}
		if w := laneIssueWeight(in.Priority, r.Number); w != PriorityWeightDefault {
			if grp.Priority == nil {
				grp.Priority = map[int]int{}
			}
			grp.Priority[r.Number] = w
		}
		if r.Generation != "" {
			if grp.Generation == nil {
				grp.Generation = map[int]string{}
			}
			grp.Generation[r.Number] = r.Generation
		}
		lanes[r.Lane] = grp
	}
	// Order each lane's candidates priority-first (#1395): the heaviest priority/P*
	// label wins, oldest-first within a tier, so an old priority/P1 surfaces ahead
	// of newer unlabeled noise. The picker (cmd/fak) re-applies this with the
	// caller's recency tiebreak; ordering here keeps the payload itself honest for
	// any consumer that reads lanes[*].issues directly.
	for lane, grp := range lanes {
		cands := make([]LaneCandidate, len(grp.Issues))
		for i, n := range grp.Issues {
			cands[i] = LaneCandidate{Number: n, Weight: laneIssueWeight(grp.Priority, n)}
		}
		grp.Issues = OrderLaneCandidates(cands, false)
		grp.SubLanes = buildRouterSubLanes(laneRoutes[lane])
		lanes[lane] = grp
	}
	total := len(in.Routes)
	unrouted := byConf["none"]
	routed := total - unrouted
	frac := 0.0
	if total > 0 {
		frac = float64(unrouted) / float64(total)
		frac = float64(int(frac*10000+0.5)) / 10000
	}

	skipped := make([]SkippedIssue, 0, len(in.SkippedBlocked))
	skippedByReason := map[string]int{}
	for _, issue := range in.SkippedBlocked {
		sk := classifySkippedIssue(issue, in.BlockedLabelName)
		skipped = append(skipped, sk)
		if sk.Reason != "" {
			skippedByReason[sk.Reason]++
		}
	}
	for _, issue := range in.SkippedDuplicate {
		sk := classifyDuplicateRiskIssue(issue)
		skipped = append(skipped, sk)
		if sk.Reason != "" {
			skippedByReason[sk.Reason]++
		}
	}
	sort.Slice(skipped, func(i, j int) bool { return skipped[i].Number > skipped[j].Number })

	coverage := in.Coverage
	if coverage.IssueLimit == 0 && coverage.IssuesFetched == 0 && len(coverage.Notes) == 0 {
		coverage = RouterCoverage{Complete: true, Notes: []string{}}
	}

	ok, verdict, finding, reason, next := true, "OK", "routed", "", "dos-dispatch workers fold lanes[<their lane>].issues into the dispositions sidecar"
	switch {
	case strings.TrimSpace(in.FetchError) != "":
		ok, verdict, finding = false, "FETCH_ERROR", "fetch_error"
		reason = strings.TrimSpace(in.FetchError)
		next = "fix the gh/dos read-back error, then re-run the lane router"
	case !coverage.Complete:
		ok, verdict, finding = false, "ACTION", "incomplete_coverage"
		reason = fmt.Sprintf("routed %d/%d fetched, but the open-issue fetch was truncated, so some open issues were never routed - %s",
			routed, total, strings.Join(coverage.Notes, "; "))
		next = "re-run with a higher --issue-limit so every open issue is routed"
	case total > 0 && frac > maxUnrouted:
		ok, verdict, finding = false, "ACTION", "high_unrouted"
		reason = fmt.Sprintf("%d/%d open issues UNROUTED (frac=%.4g > %.4g)", unrouted, total, frac, maxUnrouted)
		next = "operator: add scopes/labels or extend SCOPE_ALIAS so workers can target these"
	default:
		skippedNote := ""
		if len(skipped) > 0 {
			skippedNote = fmt.Sprintf("; %d skipped", len(skipped))
		}
		reason = fmt.Sprintf("%d/%d open issues routed to %d lane(s); %d UNROUTED%s", routed, total, len(lanes), unrouted, skippedNote)
	}

	issues := append([]IssueRoute(nil), in.Routes...)
	sort.Slice(issues, func(i, j int) bool { return routeSortLess(issues[i], issues[j]) })
	repairQueues := routerRepairQueues(issues, skipped)
	unroutable := unroutableBacklogRows(issues, skipped)

	return RouterPayload{
		Schema:              RouterSchema,
		OK:                  ok,
		Verdict:             verdict,
		Finding:             finding,
		Reason:              reason,
		NextAction:          next,
		Workspace:           in.Workspace,
		Coverage:            coverage,
		Counts:              RouterCounts{Open: total, Routed: routed, Unrouted: unrouted, UnroutedFrac: frac, RoutedStepBudget: routedStepBudget, ByConfidence: byConf, SkippedHumanBlocked: len(skipped), SkippedByReason: skippedByReason},
		Lanes:               lanes,
		Issues:              issues,
		RepairQueues:        repairQueues,
		UnroutableBacklog:   unroutable,
		SkippedHumanBlocked: skipped,
	}
}

func buildRouterSubLanes(routes []IssueRoute) []RouterSubLane {
	if len(routes) < 2 {
		return nil
	}
	groups := map[string]*RouterSubLane{}
	for _, route := range routes {
		prefix := routeSubLanePrefix(route.Paths)
		if prefix == "" {
			continue
		}
		grp := groups[prefix]
		if grp == nil {
			grp = &RouterSubLane{Prefix: prefix}
			groups[prefix] = grp
		}
		grp.Count++
		grp.StepBudget += routeStepBudget(route)
		grp.Issues = append(grp.Issues, route.Number)
	}
	if len(groups) < 2 {
		return nil
	}
	out := make([]RouterSubLane, 0, len(groups))
	for _, grp := range groups {
		sort.Slice(grp.Issues, func(i, j int) bool { return grp.Issues[i] > grp.Issues[j] })
		out = append(out, *grp)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Prefix < out[j].Prefix
	})
	return out
}

func routeSubLanePrefix(paths []string) string {
	prefix := ""
	for _, path := range paths {
		next := pathOwnershipPrefix(path)
		if next == "" {
			continue
		}
		if prefix == "" {
			prefix = next
			continue
		}
		if prefix != next {
			return ""
		}
	}
	return prefix
}

func pathOwnershipPrefix(path string) string {
	parts := strings.Split(strings.Trim(strings.ReplaceAll(path, "\\", "/"), "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		return ""
	}
	switch parts[0] {
	case "internal", "cmd", "tools", "docs", "examples", "experiments", "visuals":
		if len(parts) >= 2 {
			return parts[0] + "/" + parts[1]
		}
		return parts[0]
	case ".github", ".claude":
		if len(parts) >= 2 {
			return parts[0] + "/" + parts[1]
		}
		return parts[0]
	default:
		if len(parts) >= 2 {
			return parts[0] + "/" + parts[1]
		}
		return parts[0]
	}
}

func routerRepairQueues(routes []IssueRoute, skipped []SkippedIssue) []RouterRepairQueue {
	queues := map[string]*RouterRepairQueue{}
	add := func(kind string, issue int, stepBudget int, childIssueBudget int, reason string) {
		if stepBudget <= 0 {
			stepBudget = 1
		}
		queue := queues[kind]
		if queue == nil {
			queue = &RouterRepairQueue{
				Kind:       kind,
				NextAction: routerRepairAction(kind),
				ByReason:   map[string]int{},
			}
			queues[kind] = queue
		}
		queue.Count++
		queue.StepBudget += stepBudget
		queue.ChildIssueBudget += childIssueBudget
		if reason != "" {
			queue.ByReason[reason]++
		}
		if issue > 0 && len(queue.Issues) < 12 {
			queue.Issues = append(queue.Issues, issue)
		}
	}
	for _, route := range routes {
		if route.Lane == "" {
			add("route", route.Number, routeStepBudget(route), 0, "ISSUE_UNROUTED")
			continue
		}
		add("dispatch", route.Number, routeStepBudget(route), 0, "")
	}
	for _, skippedIssue := range skipped {
		kind := routerRepairKind(skippedIssue.Reason)
		add(kind, skippedIssue.Number, skippedIssueStepBudget(skippedIssue), skippedIssueChildIssueBudget(skippedIssue, kind), skippedIssue.Reason)
	}
	out := make([]RouterRepairQueue, 0, len(queues))
	for _, queue := range queues {
		if len(queue.ByReason) == 0 {
			queue.ByReason = nil
		}
		out = append(out, *queue)
	}
	sort.Slice(out, func(i, j int) bool {
		ri, rj := routerRepairRank(out[i].Kind), routerRepairRank(out[j].Kind)
		if ri != rj {
			return ri < rj
		}
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Kind < out[j].Kind
	})
	return out
}

func unroutableBacklogRows(routes []IssueRoute, skipped []SkippedIssue) []UnroutableBacklogRow {
	var rows []UnroutableBacklogRow
	for _, route := range routes {
		bucket := routeUnroutableBucket(route)
		if bucket == "" {
			continue
		}
		rows = append(rows, UnroutableBacklogRow{
			Number:     route.Number,
			Title:      route.Title,
			Bucket:     bucket,
			Reason:     firstNonEmptyString(route.UnroutedReason, route.Signal),
			NextAction: unroutableBacklogAction(bucket),
		})
	}
	for _, issue := range skipped {
		bucket := skippedUnroutableBucket(issue)
		if bucket == "" {
			continue
		}
		rows = append(rows, UnroutableBacklogRow{
			Number:     issue.Number,
			Title:      issue.Title,
			Bucket:     bucket,
			Reason:     issue.Reason,
			NextAction: unroutableBacklogAction(bucket),
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		ri, rj := unroutableBacklogRank(rows[i].Bucket), unroutableBacklogRank(rows[j].Bucket)
		if ri != rj {
			return ri < rj
		}
		return rows[i].Number < rows[j].Number
	})
	return rows
}

func routeUnroutableBucket(route IssueRoute) string {
	if route.Lane == "" {
		if strings.HasPrefix(route.Signal, "exclusive-scope:") || strings.Contains(route.UnroutedReason, "exclusive-lane") {
			return "exclusive_lane"
		}
		return "no_lane"
	}
	if route.SignalConflict && strings.HasPrefix(route.Signal, "path-ambiguous:") {
		return "ambiguous_path"
	}
	return ""
}

func skippedUnroutableBucket(issue SkippedIssue) string {
	switch issue.Reason {
	case "ISSUE_SCOPE_INCOMPLETE", "ISSUE_TRIAGE_ONLY":
		return "missing_scope"
	default:
		return ""
	}
}

func unroutableBacklogAction(bucket string) string {
	switch bucket {
	case "no_lane":
		return "add a lane-bearing title scope, label, or concrete path hint"
	case "exclusive_lane":
		return "operator must pick a non-exclusive lane or launch the exclusive work manually"
	case "missing_scope":
		return "add worker-ready scope, done condition, witness, likely files, and work-unit metadata"
	case "ambiguous_path":
		return "narrow path hints to one lane or add matching scope/label context"
	default:
		return "inspect the route row before dispatch"
	}
}

func unroutableBacklogRank(bucket string) int {
	switch bucket {
	case "no_lane":
		return 0
	case "exclusive_lane":
		return 1
	case "missing_scope":
		return 2
	case "ambiguous_path":
		return 3
	default:
		return 9
	}
}

func skippedIssueStepBudget(issue SkippedIssue) int {
	if issue.ExpectedSteps > 0 {
		return issue.ExpectedSteps
	}
	return 1
}

func skippedIssueChildIssueBudget(issue SkippedIssue, kind string) int {
	if kind != "split" {
		return 0
	}
	if issue.ExpectedSteps <= 0 {
		return 1
	}
	return (issue.ExpectedSteps + MaxDispatchExpectedSteps - 1) / MaxDispatchExpectedSteps
}

func routerRepairKind(reason string) string {
	switch reason {
	case ReasonBlockedByHuman:
		return "human"
	case ReasonHumanBlockUnverified:
		return "decide"
	case ReasonDuplicateRisk:
		return "duplicate"
	case "ISSUE_NOT_DISPATCH_LEAF", "ISSUE_OVERSIZED_EXPECTED_STEPS":
		return "split"
	case "ISSUE_SCOPE_INCOMPLETE", "ISSUE_TRIAGE_ONLY":
		return "scope"
	case "ISSUE_UNROUTED":
		return "route"
	case "ISSUE_LIVE_UNARMORED", "ISSUE_NOISE_CONTROL_INCOMPLETE", "ISSUE_AGENT_CONTEXT_INCOMPLETE":
		return "noise"
	case "ISSUE_PRIVATE_BOUNDARY":
		return "private"
	default:
		return "other"
	}
}

func routerRepairRank(kind string) int {
	switch kind {
	case "dispatch":
		return 0
	case "duplicate":
		return 1
	case "split":
		return 2
	case "scope":
		return 3
	case "route":
		return 4
	case "decide":
		return 5
	case "noise":
		return 6
	case "private":
		return 7
	case "human":
		return 8
	default:
		return 9
	}
}

func routerRepairAction(kind string) string {
	switch kind {
	case "dispatch":
		return "launch scoped leaf issues through their routed lanes"
	case "duplicate":
		return "dedupe duplicate-risk rows before spawning a worker"
	case "split":
		return fmt.Sprintf("decompose non-leaves or oversized rows into child issues with <= %d expected steps", MaxDispatchExpectedSteps)
	case "scope":
		return "add worker-ready scope, done condition, witness, and agent context before dispatch"
	case "route":
		return "add lane/path hints or extend routing aliases so each issue maps to one lane"
	case "noise":
		return "add trigger, batch policy, agent context, and live dedupe/cap evidence"
	case "private":
		return "remove private/operator-only evidence or move the work to the private companion repo"
	case "decide":
		return "spawn a fresh-context decision agent (meta issue) to confirm a genuine external/human blocker or drop the blocked-by-human label so a worker can dispatch it"
	case "human":
		return "wait for the human blocker to clear before worker dispatch"
	default:
		return "inspect the skipped reason before dispatch"
	}
}
