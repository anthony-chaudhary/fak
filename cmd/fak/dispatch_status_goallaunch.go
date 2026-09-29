package main

// dispatch_status_goallaunch.go — the `fak dispatch status` goal-launch-rate block
// (gateshare/fak/lesson-3).
//
// # The defect this fixes
//
// `.goal-runs/*.receipt.json` distinguishes `plan_only` from `launched` per run, but
// nothing anywhere AGGREGATED it. On the corpus measured 2026-09-28, 24 of 25
// receipts were `plan_only` and exactly ONE had launched — a 4% launch rate that no
// card, scorecard, or status surface reported. Meanwhile every investment in the
// dispatch stack (arbiter, dedup, lane derivation, capacity planning) improves the
// QUALITY OF A QUEUE THAT MOSTLY NEVER DRAINS.
//
// # Why the ratio is the KPI
//
// A plan-only run is not a failure — it is correct behavior when the launch is
// deliberately gated. But an unmeasured plan_only rate is indistinguishable from a
// broken launcher: both read as "the loop is running." Publishing the ratio makes
// the two tellable apart, which is the only reason this block exists.
//
// # What it does NOT claim
//
// This counts RECEIPTS, not outcomes. A launched run that shipped nothing counts as
// launched here; landing is the throughput headline's job (dispatch_status_throughput.go).
// The two blocks are deliberately non-overlapping: one answers "did we start?", the
// other answers "did it land?".

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// dispatchGoalLaunchRate is the goal-launch census over the .goal-runs receipt corpus.
type dispatchGoalLaunchRate struct {
	Dir          string         `json:"dir"`
	Receipts     int            `json:"receipts"`
	Launched     int            `json:"launched"`
	PlanOnly     int            `json:"plan_only"`
	Other        int            `json:"other"`
	Unparsed     int            `json:"unparsed"`
	RatePermille int            `json:"launch_rate_permille"`
	Status       string         `json:"status"` // ok | low | unknown
	Outcomes     map[string]int `json:"outcomes,omitempty"`
	NextAction   string         `json:"next_action,omitempty"`
}

// Launch-rate statuses. Closed set.
const (
	dispatchLaunchOK      = "ok"
	dispatchLaunchLow     = "low"
	dispatchLaunchUnknown = "unknown"
)

// dispatchLaunchLowPermille is the boundary below which the launcher, not the
// selector, is the suspected fault. 800‰ (80%): below it, more than one in five
// planned runs never started, which is a launch bug to diagnose rather than a
// selection quality question to tune.
const dispatchLaunchLowPermille = 800

// dispatchGoalLaunchRateFold reads the .goal-runs receipt corpus and folds the
// launch ratio. A missing or empty directory is UNKNOWN, never 0% — the same
// honesty rule the throughput block follows: an absent measurement must not render
// as a failing one.
func dispatchGoalLaunchRateFold(dir string) *dispatchGoalLaunchRate {
	g := &dispatchGoalLaunchRate{Dir: dir, Status: dispatchLaunchUnknown, Outcomes: map[string]int{}}
	entries, err := os.ReadDir(dir)
	if err != nil {
		g.NextAction = "no goal-run receipts at " + dir + " — launch rate unmeasured, not zero"
		return g
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".receipt.json") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	for _, name := range names {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			g.Unparsed++
			continue
		}
		var payload struct {
			Outcome string `json:"outcome"`
		}
		if err := json.Unmarshal(raw, &payload); err != nil {
			g.Unparsed++
			continue
		}
		g.Receipts++
		outcome := strings.TrimSpace(payload.Outcome)
		if outcome == "" {
			outcome = "(empty)"
		}
		g.Outcomes[outcome]++
		switch outcome {
		case "launched":
			g.Launched++
		case "plan_only":
			g.PlanOnly++
		default:
			g.Other++
		}
	}
	if g.Receipts == 0 {
		g.NextAction = "no goal-run receipts at " + dir + " — launch rate unmeasured, not zero"
		return g
	}
	g.RatePermille = (g.Launched * 1000) / g.Receipts
	if g.RatePermille < dispatchLaunchLowPermille {
		g.Status = dispatchLaunchLow
		g.NextAction = fmt.Sprintf("%d of %d planned runs never launched — suspect the LAUNCHER (preflight, argv-vs-binary skew, or lease refusal), not the selector", g.PlanOnly, g.Receipts)
	} else {
		g.Status = dispatchLaunchOK
		g.NextAction = "launch rate healthy"
	}
	if g.Unparsed > 0 {
		g.NextAction += fmt.Sprintf(" (%d receipt(s) unparseable and EXCLUDED from the ratio)", g.Unparsed)
	}
	return g
}

// dispatchGoalLaunchLine renders the census as one operator line.
func dispatchGoalLaunchLine(g *dispatchGoalLaunchRate) string {
	if g == nil {
		return ""
	}
	if g.Receipts == 0 {
		return fmt.Sprintf("goal launches: unmeasured (%d receipt(s) found) — not a zero rate", g.Unparsed)
	}
	return fmt.Sprintf("goal launches: %d/%d launched (%.1f%%) [%s]; %s",
		g.Launched, g.Receipts, float64(g.Launched)*100/float64(g.Receipts), g.Status, g.NextAction)
}
