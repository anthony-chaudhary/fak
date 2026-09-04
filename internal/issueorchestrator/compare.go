package issueorchestrator

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// CompareResult details campaign progress against a prior baseline snapshot.
type CompareResult struct {
	Schema           string  `json:"schema"`
	StartingIssues   int     `json:"starting_issues"`
	CurrentIssues    int     `json:"current_issues"`
	ClosedIssues     int     `json:"closed_issues"`
	ClosedNumbers    []int   `json:"closed_numbers"`
	ClosedPercent    float64 `json:"closed_percent"`
	StartingWaves    int     `json:"starting_waves"`
	CurrentWaves     int     `json:"current_waves"`
	ClosedWaves      int     `json:"closed_waves"`
	StartingSteps    int     `json:"starting_steps"`
	CurrentSteps     int     `json:"current_steps"`
	RetiredSteps     int     `json:"retired_steps"`
	CampaignComplete bool    `json:"campaign_complete"`
}

// Compare measures burndown velocity and closed issues against a baseline plan.
func Compare(current, baseline Plan) CompareResult {
	currentNums := make(map[int]bool)
	for _, w := range current.Waves {
		for _, iss := range w.Issues {
			if iss.Number > 0 {
				currentNums[iss.Number] = true
			}
		}
	}

	var closedNumbers []int
	for _, w := range baseline.Waves {
		for _, iss := range w.Issues {
			if iss.Number > 0 && !currentNums[iss.Number] {
				closedNumbers = append(closedNumbers, iss.Number)
			}
		}
	}
	sort.Ints(closedNumbers)

	closedCount := len(closedNumbers)
	startingCount := baseline.PlannedIssues
	if startingCount == 0 {
		startingCount = baseline.TotalIssues
	}
	currentCount := current.PlannedIssues

	var closedPct float64
	if startingCount > 0 {
		closedPct = math.Round((float64(closedCount)/float64(startingCount))*1000) / 10
	}

	retiredSteps := baseline.PlannedSteps - current.PlannedSteps
	if retiredSteps < 0 {
		retiredSteps = 0
	}

	closedWaves := baseline.TotalWaves - current.TotalWaves
	if closedWaves < 0 {
		closedWaves = 0
	}

	complete := (currentCount == 0 && startingCount > 0)
	if baseline.TargetIssues > 0 && closedCount >= baseline.TargetIssues {
		complete = true
	}
	if baseline.TargetPoints > 0 && retiredSteps >= baseline.TargetPoints {
		complete = true
	}

	return CompareResult{
		Schema:           CompareSchema,
		StartingIssues:   startingCount,
		CurrentIssues:    currentCount,
		ClosedIssues:     closedCount,
		ClosedNumbers:    closedNumbers,
		ClosedPercent:    closedPct,
		StartingWaves:    baseline.TotalWaves,
		CurrentWaves:     current.TotalWaves,
		ClosedWaves:      closedWaves,
		StartingSteps:    baseline.PlannedSteps,
		CurrentSteps:     current.PlannedSteps,
		RetiredSteps:     retiredSteps,
		CampaignComplete: complete,
	}
}

// CompareReport formats CompareResult as aligned terminal text.
func CompareReport(current, baseline Plan) string {
	res := Compare(current, baseline)
	var b strings.Builder

	b.WriteString("=== FAK ISSUE ORCHESTRATOR: CAMPAIGN BURNDOWN COMPARISON ===\n")
	b.WriteString(fmt.Sprintf("Campaign Progress:   %d/%d issue(s) closed (%.1f%%) · %d step(s) retired\n",
		res.ClosedIssues, res.StartingIssues, res.ClosedPercent, res.RetiredSteps,
	))
	b.WriteString(fmt.Sprintf("Waves Burndown:      %d starting wave(s) → %d current wave(s) (%d wave(s) completed)\n",
		res.StartingWaves, res.CurrentWaves, res.ClosedWaves,
	))
	if len(res.ClosedNumbers) > 0 {
		numStrs := make([]string, 0, len(res.ClosedNumbers))
		for _, n := range res.ClosedNumbers {
			numStrs = append(numStrs, fmt.Sprintf("#%d", n))
		}
		b.WriteString(fmt.Sprintf("Closed Issues:       %s\n", strings.Join(numStrs, ", ")))
	}
	if res.CampaignComplete {
		b.WriteString("Campaign Status:     COMPLETED (campaign target achieved)\n")
	} else {
		b.WriteString(fmt.Sprintf("Campaign Status:     ACTIVE (%d issue(s) remaining across %d wave(s))\n",
			res.CurrentIssues, res.CurrentWaves,
		))
	}

	return b.String()
}
