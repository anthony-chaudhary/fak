package issueorchestrator

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/debtlane"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// PlanWaves partitions candidate issues into ordered, collision-free, concurrent-safe waves.
func PlanWaves(issues []Issue, opts WavePlanOptions) Plan {
	waveSizeCap := opts.WaveSize
	if waveSizeCap <= 0 {
		waveSizeCap = 4
	}

	if opts.Limit > 0 && len(issues) > opts.Limit {
		issues = issues[:opts.Limit]
	}

	workspace := opts.WorkspaceRoot
	if workspace == "" {
		workspace = "."
	}

	plan := Plan{
		Schema:         WavePlanSchema,
		Workspace:      workspace,
		TotalIssues:    len(issues),
		WaveSizeCap:    waveSizeCap,
		TargetIssues:   opts.TargetIssues,
		TargetPoints:   opts.TargetPoints,
		ExcludedIssues: opts.ExcludedIssues,
		ExcludedLanes:  opts.ExcludedLanes,
	}

	excludedIssuesMap := make(map[int]bool)
	for _, num := range opts.ExcludedIssues {
		if num > 0 {
			excludedIssuesMap[num] = true
		}
	}

	excludedLanesMap := make(map[string]bool)
	for _, l := range opts.ExcludedLanes {
		trimmed := strings.ToLower(strings.TrimSpace(l))
		if trimmed != "" {
			excludedLanesMap[trimmed] = true
		}
	}

	heldLanesMap := make(map[string]bool)
	if opts.AutoDetectHeld && workspace != "" {
		discovered, _ := debtlane.DiscoverHeldLanes(workspace)
		for _, l := range discovered {
			lower := strings.ToLower(strings.TrimSpace(l))
			if lower != "" {
				heldLanesMap[lower] = true
				plan.HeldLanes = append(plan.HeldLanes, l)
			}
		}
		sort.Strings(plan.HeldLanes)
	}

	graph := opts.Graph
	if graph == nil && workspace != "" {
		graph, _ = debtlane.BuildInternalImportGraph(workspace)
	}

	// Detect duplicates by Key
	keyCounts := make(map[string][]int)
	for _, iss := range issues {
		key := strings.TrimSpace(iss.Key)
		if key != "" {
			keyCounts[key] = append(keyCounts[key], iss.Number)
		}
	}
	for key, nums := range keyCounts {
		if len(nums) > 1 {
			plan.Duplicates = append(plan.Duplicates, DuplicateGroup{
				Key:          key,
				Count:        len(nums),
				IssueNumbers: nums,
			})
		}
	}
	sort.Slice(plan.Duplicates, func(i, j int) bool {
		return plan.Duplicates[i].Key < plan.Duplicates[j].Key
	})

	seenKeys := make(map[string]bool)
	var dispatchableLeaves []Issue

	for _, iss := range issues {
		if iss.Number > 0 && excludedIssuesMap[iss.Number] {
			continue
		}
		laneLower := strings.ToLower(strings.TrimSpace(iss.Lane))
		if laneLower != "" && excludedLanesMap[laneLower] {
			continue
		}
		if opts.LaneFilter != "" && !strings.EqualFold(iss.Lane, opts.LaneFilter) {
			continue
		}

		key := strings.TrimSpace(iss.Key)
		if key != "" {
			if seenKeys[key] {
				continue // Skip duplicate keys in wave planning
			}
			seenKeys[key] = true
		}

		// Check if held by an active lease
		if isHeld(iss, heldLanesMap) {
			plan.HeldIssues = append(plan.HeldIssues, iss.Number)
			continue
		}

		// Evaluate subdivide / triage / dispatchable
		if isSubdivideTarget(iss) {
			plan.Subdividable++
			budget := (iss.ExpectedSteps + 4) / 5
			if budget < 2 {
				budget = 2
			}
			plan.Subdivide = append(plan.Subdivide, SubdivideRow{
				Key:              key,
				IssueNumber:      iss.Number,
				Title:            iss.Title,
				Reasons:          []string{"epic/oversized issue exceeding leaf step budget or spanning multiple subsystems"},
				ExpectedSteps:    iss.ExpectedSteps,
				ChildIssueBudget: budget,
				Lane:             iss.Lane,
				Paths:            append([]string(nil), iss.Paths...),
			})
			continue
		}

		if isTriageTarget(iss) {
			plan.TriageOnly++
			plan.Triage = append(plan.Triage, TriageRow{
				Key:             key,
				IssueNumber:     iss.Number,
				Title:           iss.Title,
				Dispatchability: iss.Dispatchability,
				Reasons:         []string{"missing concrete paths, unclear scope, or unmet problem frame"},
			})
			continue
		}

		plan.Dispatchable++
		dispatchableLeaves = append(dispatchableLeaves, iss)
	}

	// Sort dispatchable leaves: urgent/critical first, then core centrality, then expected steps ascending
	sort.SliceStable(dispatchableLeaves, func(i, j int) bool {
		ui := isUrgent(dispatchableLeaves[i])
		uj := isUrgent(dispatchableLeaves[j])
		if ui != uj {
			return ui // urgent first
		}
		ci := dispatchableLeaves[i].Centrality == "core"
		cj := dispatchableLeaves[j].Centrality == "core"
		if ci != cj {
			return ci
		}
		if dispatchableLeaves[i].ExpectedSteps != dispatchableLeaves[j].ExpectedSteps {
			return dispatchableLeaves[i].ExpectedSteps < dispatchableLeaves[j].ExpectedSteps
		}
		return dispatchableLeaves[i].Number < dispatchableLeaves[j].Number
	})

	var serialWaves []Wave
	var parallelWaves []Wave

	for _, leaf := range dispatchableLeaves {
		if isSerialSingleton(leaf) {
			sw := Wave{
				Safety:       WaveSafetySerialSingleton,
				Issues:       []Issue{leaf},
				IssueNumbers: []int{leaf.Number},
				Lanes:        []string{leaf.Lane},
				Paths:        append([]string(nil), leaf.Paths...),
				WaveSize:     1,
				StepBudget:   leaf.ExpectedSteps,
			}
			serialWaves = append(serialWaves, sw)
			continue
		}

		// Find first open parallel wave that can safely accommodate leaf
		placed := false
		for wi := range parallelWaves {
			w := &parallelWaves[wi]
			if len(w.Issues) >= waveSizeCap {
				continue
			}
			conflict := false
			for _, existing := range w.Issues {
				if issuesCollide(leaf, existing, graph) {
					conflict = true
					break
				}
			}
			if !conflict {
				w.Issues = append(w.Issues, leaf)
				w.IssueNumbers = append(w.IssueNumbers, leaf.Number)
				if leaf.Lane != "" && !containsStr(w.Lanes, leaf.Lane) {
					w.Lanes = append(w.Lanes, leaf.Lane)
				}
				for _, p := range leaf.Paths {
					if !containsStr(w.Paths, p) {
						w.Paths = append(w.Paths, p)
					}
				}
				w.WaveSize = len(w.Issues)
				w.StepBudget += leaf.ExpectedSteps
				placed = true
				break
			}
		}

		if !placed {
			nw := Wave{
				Safety:       WaveSafetyDisjointLeaf,
				Issues:       []Issue{leaf},
				IssueNumbers: []int{leaf.Number},
				Lanes:        nil,
				Paths:        append([]string(nil), leaf.Paths...),
				WaveSize:     1,
				StepBudget:   leaf.ExpectedSteps,
			}
			if leaf.Lane != "" {
				nw.Lanes = []string{leaf.Lane}
			}
			parallelWaves = append(parallelWaves, nw)
		}
	}

	// Order: parallel disjoint waves first for velocity, then serial singletons
	allWaves := append(parallelWaves, serialWaves...)

	var selectedWaves []Wave
	var plannedIssuesCount, plannedStepsCount int

	for i := range allWaves {
		w := allWaves[i]
		w.Index = len(selectedWaves) + 1
		w.ID = fmt.Sprintf("wave-%d", w.Index)
		w.WaveSize = len(w.Issues)

		// Populate LeaseRegion and LeaseLanes
		var allPaths []string
		laneSet := make(map[string]bool)
		for _, iss := range w.Issues {
			if len(iss.Paths) > 0 {
				allPaths = append(allPaths, iss.Paths...)
			} else if iss.Lane != "" {
				laneSet[iss.Lane] = true
			}
		}
		w.LeaseRegion = minimalRoots(allPaths)
		for l := range laneSet {
			w.LeaseLanes = append(w.LeaseLanes, l)
		}
		sort.Strings(w.LeaseLanes)

		selectedWaves = append(selectedWaves, w)
		plannedIssuesCount += len(w.Issues)
		plannedStepsCount += w.StepBudget

		// Campaign target check
		if opts.TargetIssues > 0 && plannedIssuesCount >= opts.TargetIssues {
			break
		}
		if opts.TargetPoints > 0 && plannedStepsCount >= opts.TargetPoints {
			break
		}
		if opts.MaxWaves > 0 && len(selectedWaves) >= opts.MaxWaves {
			break
		}
	}

	plan.Waves = selectedWaves
	plan.TotalWaves = len(selectedWaves)
	plan.PlannedIssues = plannedIssuesCount
	plan.PlannedSteps = plannedStepsCount

	return plan
}

// PlanCandidates adapts []issuepolicy.Candidate to Plan.
func PlanCandidates(candidates []issuepolicy.Candidate, opts WavePlanOptions) Plan {
	var issues []Issue
	for _, c := range candidates {
		rev := issuepolicy.ReviewCandidate(c, opts.toPolicyOptions())
		iss := Issue{
			Number:          rev.IssueNumber,
			Key:             c.Key,
			Title:           c.Title,
			Lane:            rev.Lane,
			Paths:           append([]string(nil), rev.Paths...),
			ExpectedSteps:   rev.ExpectedSteps,
			Labels:          append([]string(nil), c.Labels...),
			Centrality:      string(c.ProblemFrame.Centrality),
			ProblemFrame:    c.ProblemFrame,
			Dispatchability: rev.Dispatchability,
		}
		if iss.Number == 0 {
			iss.Number = parseNumberFromKey(c.Key)
		}
		issues = append(issues, iss)
	}
	return PlanWaves(issues, opts)
}

// PlanIssueDrafts adapts []issuepolicy.IssueDraft to Plan.
func PlanIssueDrafts(drafts []issuepolicy.IssueDraft, opts WavePlanOptions) Plan {
	var issues []Issue
	for _, d := range drafts {
		cand := issuepolicy.CandidateFromIssueDraft(d)
		rev := issuepolicy.ReviewIssueDraft(d, opts.toPolicyOptions())
		var lbls []string
		for _, l := range d.Labels {
			if l.Name != "" {
				lbls = append(lbls, l.Name)
			}
		}
		iss := Issue{
			Number:          d.Number,
			Key:             cand.Key,
			Title:           d.Title,
			Lane:            rev.Lane,
			Paths:           append([]string(nil), rev.Paths...),
			ExpectedSteps:   rev.ExpectedSteps,
			Labels:          lbls,
			URL:             d.URL,
			Centrality:      string(cand.ProblemFrame.Centrality),
			ProblemFrame:    cand.ProblemFrame,
			Dispatchability: rev.Dispatchability,
		}
		issues = append(issues, iss)
	}
	return PlanWaves(issues, opts)
}

func (o WavePlanOptions) toPolicyOptions() issuepolicy.Options {
	return issuepolicy.Options{
		StrictProjectWork: o.StrictProjectWork,
	}
}

func isHeld(iss Issue, heldLanes map[string]bool) bool {
	if len(heldLanes) == 0 {
		return false
	}
	if iss.Lane != "" && heldLanes[strings.ToLower(iss.Lane)] {
		return true
	}
	for _, p := range iss.Paths {
		clean := filepath.ToSlash(filepath.Clean(p))
		for held := range heldLanes {
			target := "internal/" + held
			if clean == target || strings.HasPrefix(clean, target+"/") {
				return true
			}
		}
	}
	return false
}

func isSubdivideTarget(iss Issue) bool {
	if iss.ExpectedSteps > 15 {
		return true
	}
	// If issue spans 3 or more distinct internal packages, it is an epic that should split
	pkgs := make(map[string]bool)
	for _, p := range iss.Paths {
		norm := filepath.ToSlash(filepath.Clean(p))
		if strings.HasPrefix(norm, "internal/") {
			parts := strings.Split(norm, "/")
			if len(parts) >= 2 {
				pkgs[parts[1]] = true
			}
		}
	}
	return len(pkgs) >= 3
}

func isTriageTarget(iss Issue) bool {
	if strings.TrimSpace(iss.Title) == "" {
		return true
	}
	if iss.Dispatchability != "" && iss.Dispatchability != issuepolicy.Dispatchable {
		return true
	}
	return false
}

func isSerialSingleton(iss Issue) bool {
	lane := strings.ToLower(strings.TrimSpace(iss.Lane))
	switch lane {
	case "abi", "kernel", "adjudicator", "policy", "gateway", "vdso", "shipgate", "architest":
		return true
	}
	for _, p := range iss.Paths {
		norm := filepath.ToSlash(filepath.Clean(p))
		if norm == "go.mod" || norm == "go.sum" || norm == "dos.toml" {
			return true
		}
		for _, core := range []string{"internal/abi", "internal/kernel", "internal/adjudicator", "internal/policy", "internal/gateway", "internal/vdso"} {
			if norm == core || strings.HasPrefix(norm, core+"/") {
				return true
			}
		}
	}
	return false
}

func isUrgent(iss Issue) bool {
	for _, lbl := range iss.Labels {
		lower := strings.ToLower(strings.TrimSpace(lbl))
		if lower == "urgent" || lower == "critical" || lower == "p0" || strings.HasPrefix(lower, "priority/p0") {
			return true
		}
	}
	return false
}

func issuesCollide(a, b Issue, graph map[string]map[string]struct{}) bool {
	// 1. Same lane always collides (two workers cannot hold the same lane lease simultaneously)
	if a.Lane != "" && b.Lane != "" && strings.EqualFold(a.Lane, b.Lane) {
		return true
	}

	// 2. Path tree overlap
	if len(a.Paths) > 0 && len(b.Paths) > 0 {
		for _, pa := range a.Paths {
			for _, pb := range b.Paths {
				if debtlane.TreesOverlap(pa, pb) {
					return true
				}
			}
		}
	}

	// 2. Package import graph contention
	if a.Lane != "" && b.Lane != "" && graph != nil {
		if debtlane.ImportsContend(a.Lane, b.Lane, graph) {
			return true
		}
	}

	return false
}

func minimalRoots(paths []string) []string {
	if len(paths) == 0 {
		return nil
	}
	uniq := make(map[string]bool)
	for _, p := range paths {
		cleaned := filepath.ToSlash(filepath.Clean(p))
		if cleaned != "" && cleaned != "." {
			uniq[cleaned] = true
		}
	}
	sorted := make([]string, 0, len(uniq))
	for p := range uniq {
		sorted = append(sorted, p)
	}
	sort.Strings(sorted)

	var roots []string
	for _, p := range sorted {
		covered := false
		for _, r := range roots {
			if p == r || strings.HasPrefix(p, r+"/") {
				covered = true
				break
			}
		}
		if !covered {
			roots = append(roots, p)
		}
	}
	return roots
}

func containsStr(slice []string, val string) bool {
	for _, s := range slice {
		if s == val {
			return true
		}
	}
	return false
}

func parseNumberFromKey(key string) int {
	var n int
	for _, part := range strings.Split(key, "-") {
		var val int
		if _, err := fmt.Sscanf(part, "%d", &val); err == nil && val > 0 {
			n = val
		}
	}
	return n
}
