package issueorchestrator

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"
)

// StepBudgetAssessment captures the outcome of adaptive step budgeting evaluation.
type StepBudgetAssessment struct {
	ElasticAdmitted bool     `json:"elastic_admitted"`
	RequiresSplit   bool     `json:"requires_split"`
	AdvisoryLog     string   `json:"advisory_log,omitempty"`
	Packages        []string `json:"packages,omitempty"`
	NominalSteps    int      `json:"nominal_steps"`
}

// SubdividedIssue represents an architectural sub-task created from splitting an epic.
type SubdividedIssue struct {
	Number        int      `json:"number,omitempty"`
	ParentNumber  int      `json:"parent_number,omitempty"`
	Key           string   `json:"key"`
	Title         string   `json:"title"`
	Lane          string   `json:"lane"`
	Package       string   `json:"package,omitempty"`
	Paths         []string `json:"paths"`
	ExpectedSteps int      `json:"expected_steps"`
}

// SubdivideChild is an alias for SubdividedIssue.
type SubdivideChild = SubdividedIssue

// ToIssue converts a SubdividedIssue back to a regular Issue for wave planning.
func (s SubdividedIssue) ToIssue() Issue {
	return Issue{
		Number:        s.Number,
		Key:           s.Key,
		Title:         s.Title,
		Lane:          s.Lane,
		Paths:         append([]string(nil), s.Paths...),
		ExpectedSteps: s.ExpectedSteps,
	}
}

// AssessStepBudget evaluates an issue's expected steps against package scope.
// Single/cohesive-package tasks (<=2 packages) are admitted under elasticity up to maxSinglePkgSteps (default 25).
// Multi-package tasks (>2 packages) adhere to strict limits (maxMultiPkgSteps, default 8).
func AssessStepBudget(issue Issue, maxSinglePkgSteps int, maxMultiPkgSteps int) StepBudgetAssessment {
	if maxSinglePkgSteps <= 0 {
		maxSinglePkgSteps = 25
	}
	if maxMultiPkgSteps <= 0 {
		maxMultiPkgSteps = 8
	}

	pkgs := extractPackages(issue)
	assessment := StepBudgetAssessment{
		NominalSteps: issue.ExpectedSteps,
		Packages:     pkgs,
	}

	isSingleOrCohesive := len(pkgs) <= 2

	if isSingleOrCohesive {
		if issue.ExpectedSteps > maxSinglePkgSteps {
			assessment.RequiresSplit = true
			assessment.ElasticAdmitted = false
			assessment.AdvisoryLog = fmt.Sprintf("[split-required] issue #%d step budget (%d) exceeds maximum single-package elasticity limit (%d)", issue.Number, issue.ExpectedSteps, maxSinglePkgSteps)
		} else if issue.ExpectedSteps >= 9 {
			assessment.ElasticAdmitted = true
			assessment.RequiresSplit = false
			assessment.AdvisoryLog = fmt.Sprintf("[advisory] issue #%d step budget (%d) exceeds nominal leaf limit; admitted under step elasticity", issue.Number, issue.ExpectedSteps)
		} else {
			assessment.ElasticAdmitted = false
			assessment.RequiresSplit = false
		}
	} else {
		// Multi-package tasks (> 2 distinct packages)
		if issue.ExpectedSteps > maxMultiPkgSteps {
			assessment.RequiresSplit = true
			assessment.ElasticAdmitted = false
			assessment.AdvisoryLog = fmt.Sprintf("[split-required] issue #%d spans %d packages with %d steps exceeding multi-package limit (%d)", issue.Number, len(pkgs), issue.ExpectedSteps, maxMultiPkgSteps)
		} else {
			assessment.ElasticAdmitted = false
			assessment.RequiresSplit = false
		}
	}

	return assessment
}

// SplitEpicArchitectural decomposes an epic along package / sub-lane boundaries,
// apportioning steps proportionately across the resulting child issues.
func SplitEpicArchitectural(issue Issue) []SubdividedIssue {
	if len(issue.Paths) == 0 {
		numParts := (issue.ExpectedSteps + 14) / 15
		if numParts < 2 {
			numParts = 2
		}
		stepsPerPart := apportionSteps(issue.ExpectedSteps, numParts, nil)
		res := make([]SubdividedIssue, numParts)
		for i := 0; i < numParts; i++ {
			subKey := issue.Key
			if subKey != "" {
				subKey = fmt.Sprintf("%s-part-%d", subKey, i+1)
			} else {
				subKey = fmt.Sprintf("issue-%d-part-%d", issue.Number, i+1)
			}
			res[i] = SubdividedIssue{
				ParentNumber:  issue.Number,
				Key:           subKey,
				Title:         fmt.Sprintf("%s (part %d)", issue.Title, i+1),
				Lane:          issue.Lane,
				ExpectedSteps: stepsPerPart[i],
			}
		}
		return res
	}

	pkgGroups := make(map[string][]string)
	var orderedPkgs []string
	for _, p := range issue.Paths {
		pkg := pathPackage(p)
		if _, exists := pkgGroups[pkg]; !exists {
			orderedPkgs = append(orderedPkgs, pkg)
		}
		pkgGroups[pkg] = append(pkgGroups[pkg], p)
	}

	if len(orderedPkgs) > 1 {
		weights := make([]int, len(orderedPkgs))
		for i, pkg := range orderedPkgs {
			weights[i] = len(pkgGroups[pkg])
		}

		stepsPerPkg := apportionSteps(issue.ExpectedSteps, len(orderedPkgs), weights)
		res := make([]SubdividedIssue, len(orderedPkgs))
		for i, pkg := range orderedPkgs {
			subKey := issue.Key
			if subKey != "" {
				subKey = fmt.Sprintf("%s-%s", subKey, pkg)
			} else {
				subKey = fmt.Sprintf("issue-%d-%s", issue.Number, pkg)
			}
			res[i] = SubdividedIssue{
				ParentNumber:  issue.Number,
				Key:           subKey,
				Title:         fmt.Sprintf("%s [%s]", issue.Title, pkg),
				Lane:          pkg,
				Package:       pkg,
				Paths:         pkgGroups[pkg],
				ExpectedSteps: stepsPerPkg[i],
			}
		}
		return res
	}

	// Single package with paths exceeding budget
	pkg := orderedPkgs[0]
	numParts := (issue.ExpectedSteps + 14) / 15
	if numParts < 2 {
		numParts = 2
	}
	allPaths := pkgGroups[pkg]
	stepsPerPart := apportionSteps(issue.ExpectedSteps, numParts, nil)
	res := make([]SubdividedIssue, numParts)

	pathsPerPart := make([][]string, numParts)
	for i, p := range allPaths {
		idx := i % numParts
		pathsPerPart[idx] = append(pathsPerPart[idx], p)
	}

	for i := 0; i < numParts; i++ {
		subKey := issue.Key
		if subKey != "" {
			subKey = fmt.Sprintf("%s-part-%d", subKey, i+1)
		} else {
			subKey = fmt.Sprintf("issue-%d-part-%d", issue.Number, i+1)
		}
		res[i] = SubdividedIssue{
			ParentNumber:  issue.Number,
			Key:           subKey,
			Title:         fmt.Sprintf("%s (part %d)", issue.Title, i+1),
			Lane:          issue.Lane,
			Package:       pkg,
			Paths:         pathsPerPart[i],
			ExpectedSteps: stepsPerPart[i],
		}
	}
	return res
}

// AdjustIssueSteps dynamically updates an issue's expected steps within a wave plan,
// recomputes wave step budgets, and records an advisory log in plan diagnostics.
func AdjustIssueSteps(plan *Plan, issueNum int, newSteps int, reason string) error {
	if plan == nil {
		return errors.New("plan is nil")
	}
	if newSteps < 0 {
		return fmt.Errorf("newSteps must be non-negative, got %d", newSteps)
	}

	found := false
	var oldSteps int
	var location string

	for wi := range plan.Waves {
		w := &plan.Waves[wi]
		for ii := range w.Issues {
			if w.Issues[ii].Number == issueNum {
				found = true
				oldSteps = w.Issues[ii].ExpectedSteps
				w.Issues[ii].ExpectedSteps = newSteps
				location = w.ID
				break
			}
		}
		if found {
			waveBudget := 0
			for _, iss := range w.Issues {
				waveBudget += iss.ExpectedSteps
			}
			w.StepBudget = waveBudget
			break
		}
	}

	if !found {
		for si := range plan.Subdivide {
			if plan.Subdivide[si].IssueNumber == issueNum {
				found = true
				oldSteps = plan.Subdivide[si].ExpectedSteps
				plan.Subdivide[si].ExpectedSteps = newSteps
				childBudget := (newSteps + 4) / 5
				if childBudget < 2 {
					childBudget = 2
				}
				plan.Subdivide[si].ChildIssueBudget = childBudget
				location = "subdivide queue"
				break
			}
		}
	}

	if !found {
		return fmt.Errorf("issue #%d not found in plan", issueNum)
	}

	totalPlannedSteps := 0
	for _, w := range plan.Waves {
		totalPlannedSteps += w.StepBudget
	}
	plan.PlannedSteps = totalPlannedSteps

	if plan.Diagnostics == nil {
		plan.Diagnostics = &PlanDiagnostics{}
	}
	var logMsg string
	if reason != "" {
		logMsg = fmt.Sprintf("[step-adjust] issue #%d in %s adjusted from %d to %d steps: %s", issueNum, location, oldSteps, newSteps, reason)
	} else {
		logMsg = fmt.Sprintf("[step-adjust] issue #%d in %s adjusted from %d to %d steps", issueNum, location, oldSteps, newSteps)
	}
	plan.Diagnostics.AdvisoryWarnings = append(plan.Diagnostics.AdvisoryWarnings, logMsg)

	return nil
}

func pathPackage(p string) string {
	norm := filepath.ToSlash(filepath.Clean(p))
	if strings.HasPrefix(norm, "internal/") {
		parts := strings.Split(norm, "/")
		if len(parts) >= 2 && parts[1] != "" {
			return parts[1]
		}
	}
	if strings.HasPrefix(norm, "cmd/") {
		parts := strings.Split(norm, "/")
		if len(parts) >= 2 && parts[1] != "" {
			return parts[1]
		}
	}
	parts := strings.Split(norm, "/")
	if len(parts) > 1 && parts[0] != "" && parts[0] != "." {
		return parts[0]
	}
	return "general"
}

func extractPackages(issue Issue) []string {
	pkgSet := make(map[string]bool)
	for _, p := range issue.Paths {
		pkg := pathPackage(p)
		if pkg != "" && pkg != "general" {
			pkgSet[pkg] = true
		}
	}
	if len(pkgSet) == 0 && strings.TrimSpace(issue.Lane) != "" {
		pkgSet[strings.ToLower(strings.TrimSpace(issue.Lane))] = true
	}
	pkgs := make([]string, 0, len(pkgSet))
	for k := range pkgSet {
		pkgs = append(pkgs, k)
	}
	sort.Strings(pkgs)
	return pkgs
}

func apportionSteps(totalSteps int, n int, weights []int) []int {
	if n <= 0 {
		return nil
	}
	if totalSteps <= 0 {
		totalSteps = n
	}
	res := make([]int, n)
	if weights == nil {
		weights = make([]int, n)
		for i := range weights {
			weights[i] = 1
		}
	}
	totalWeight := 0
	for _, w := range weights {
		totalWeight += w
	}
	if totalWeight <= 0 {
		totalWeight = n
		for i := range weights {
			weights[i] = 1
		}
	}

	type remainder struct {
		index int
		rem   float64
	}
	remList := make([]remainder, n)
	allocated := 0

	for i := 0; i < n; i++ {
		exact := float64(totalSteps) * float64(weights[i]) / float64(totalWeight)
		floor := int(math.Floor(exact))
		if floor < 1 {
			floor = 1
		}
		res[i] = floor
		allocated += floor
		remList[i] = remainder{index: i, rem: exact - float64(floor)}
	}

	if allocated < totalSteps {
		sort.SliceStable(remList, func(i, j int) bool {
			return remList[i].rem > remList[j].rem
		})
		diff := totalSteps - allocated
		for i := 0; i < diff; i++ {
			idx := remList[i%n].index
			res[idx]++
		}
	} else if allocated > totalSteps {
		sort.SliceStable(remList, func(i, j int) bool {
			return remList[i].rem < remList[j].rem
		})
		diff := allocated - totalSteps
		for i := 0; i < diff; i++ {
			maxIdx := -1
			maxVal := 1
			for j := 0; j < n; j++ {
				if res[j] > maxVal {
					maxVal = res[j]
					maxIdx = j
				}
			}
			if maxIdx >= 0 {
				res[maxIdx]--
			}
		}
	}

	return res
}
