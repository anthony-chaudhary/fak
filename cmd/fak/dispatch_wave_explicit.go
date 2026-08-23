package main

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func parseDispatchWaveIssueNumbers(values []string) ([]int, error) {
	seen := map[int]bool{}
	out := []int{}
	for _, value := range values {
		for _, token := range strings.Split(value, ",") {
			raw := strings.TrimSpace(token)
			if raw == "" {
				return nil, errors.New("empty issue number")
			}
			norm := strings.ToLower(raw)
			for _, prefix := range []string{"issue", "gh", "bug"} {
				if strings.HasPrefix(norm, prefix) {
					norm = strings.TrimLeft(strings.TrimSpace(strings.TrimPrefix(norm, prefix)), "-# ")
					break
				}
			}
			norm = strings.TrimSpace(strings.TrimPrefix(norm, "#"))
			n, err := strconv.Atoi(norm)
			if err != nil || n <= 0 {
				return nil, fmt.Errorf("%q is not a positive issue number", raw)
			}
			if !seen[n] {
				seen[n] = true
				out = append(out, n)
			}
		}
	}
	return out, nil
}

func dispatchWaveSeedExplicitIssueReceipt(rec map[string]any, requested []int) {
	if len(requested) == 0 || rec == nil {
		return
	}
	rec["requested_issues"] = append([]int(nil), requested...)
	rec["selected_issues"] = []int{}
	rec["refused_issues"] = []dispatchWaveIssueRefusal{}
}

func dispatchWaveAttachExplicitIssueReceipt(rec map[string]any, price dispatchWavePrice) {
	if len(price.RequestedIssues) == 0 || rec == nil {
		return
	}
	rec["requested_issues"] = append([]int(nil), price.RequestedIssues...)
	rec["selected_issues"] = append([]int{}, price.SelectedIssues...)
	rec["refused_issues"] = append([]dispatchWaveIssueRefusal{}, price.RefusedIssues...)
}

func dispatchWaveRefuseAllExplicitIssues(rec map[string]any, requested []int, class, reason, detail string) {
	if len(requested) == 0 {
		return
	}
	rows := make([]dispatchWaveIssueRefusal, 0, len(requested))
	for _, issue := range requested {
		rows = append(rows, dispatchWaveIssueRefusal{Issue: issue, Class: class, Reason: reason, Detail: detail})
	}
	dispatchWaveAttachExplicitIssueReceipt(rec, dispatchWavePrice{RequestedIssues: append([]int(nil), requested...), RefusedIssues: rows})
}

func dispatchWaveReadIntentHolds(root string, requested []int) (map[int]string, error) {
	holds := map[int]string{}
	if len(requested) == 0 {
		return holds, nil
	}
	wanted := dispatchWaveIntSet(requested)
	live, _, err := leaseref.NewInDir(root).LiveIntents(context.Background(), time.Now())
	if err != nil {
		return nil, err
	}
	for _, rec := range live {
		if !strings.HasPrefix(rec.Key, "issue-") {
			continue
		}
		issue, parseErr := strconv.Atoi(strings.TrimPrefix(rec.Key, "issue-"))
		if parseErr != nil || !wanted[issue] {
			continue
		}
		detail := fmt.Sprintf("issue #%d is claimed by %s", issue, firstString(strings.TrimSpace(rec.Holder), "an anonymous peer"))
		if rec.SessionID != "" {
			detail += " (session " + rec.SessionID + ")"
		}
		holds[issue] = detail
	}
	return holds, nil
}

func dispatchWaveApplyAuditIssueOutcomes(price dispatchWavePrice, rows []dispatchWaveExecutionAudit) dispatchWavePrice {
	if len(price.RequestedIssues) == 0 || len(rows) == 0 {
		return price
	}
	refusals := append([]dispatchWaveIssueRefusal(nil), price.RefusedIssues...)
	for _, row := range rows {
		if row.OK || row.Target.Issue <= 0 {
			continue
		}
		class := dispatchWaveIssueRefusalEligibility
		reason := firstString(strings.TrimSpace(row.Verdict), strings.ToUpper(strings.TrimSpace(row.Action)), dispatchWaveReasonIssueIneligible)
		switch {
		case reason == leaseref.ReasonIntentCollision:
			class, reason = dispatchWaveIssueRefusalIntent, leaseref.ReasonIntentCollision
		case row.Action == "in_flight_duplicate":
			class, reason = dispatchWaveIssueRefusalEligibility, dispatchWaveReasonIssueInFlight
		case strings.Contains(reason, "CAP") || strings.Contains(reason, "NO_SEATS"):
			class = dispatchWaveIssueRefusalCapacity
		case strings.Contains(reason, "NO_LANE") || row.Action == "no_lane":
			class = dispatchWaveIssueRefusalRouting
		}
		if row.Error != "" {
			reason = "WAVE_AUDIT_ERROR"
		}
		refusals = append(refusals, dispatchWaveIssueRefusal{
			Issue: row.Target.Issue, Class: class, Reason: reason,
			Detail: firstString(row.Reason, row.Error, row.Verdict, row.Action),
		})
	}
	price.RefusedIssues = dispatchWaveMergeRefusalRows(price.RequestedIssues, price.RefusedIssues, refusals)
	return dispatchWaveMergeIssueRefusals(price, nil)
}

func dispatchWaveMergeIssueRefusals(price dispatchWavePrice, extras []dispatchWaveIssueRefusal) dispatchWavePrice {
	if len(price.RequestedIssues) == 0 {
		return price
	}
	price.RefusedIssues = dispatchWaveMergeRefusalRows(price.RequestedIssues, price.RefusedIssues, extras)
	refused := map[int]dispatchWaveIssueRefusal{}
	for _, row := range price.RefusedIssues {
		refused[row.Issue] = row
	}
	runTargets := make([]dispatchWaveCandidate, 0, len(price.RunTargets))
	selected := map[int]bool{}
	for _, target := range price.RunTargets {
		if refused[target.Issue].Issue != 0 {
			continue
		}
		runTargets = append(runTargets, target)
		selected[target.Issue] = true
	}
	price.RunTargets = runTargets
	price.RunLanes = price.RunLanes[:0]
	price.SelectedIssues = price.SelectedIssues[:0]
	for _, target := range runTargets {
		price.RunLanes = append(price.RunLanes, target.Lane)
		price.SelectedIssues = append(price.SelectedIssues, target.Issue)
	}
	for i := range price.Candidates {
		price.Candidates[i].Selected = selected[price.Candidates[i].Issue]
		if row, ok := refused[price.Candidates[i].Issue]; ok {
			price.Candidates[i].Reason = row.Reason
		}
	}
	price.EffectiveCap = len(runTargets)
	price.LanesUtilized = len(runTargets)
	return price
}

func dispatchWaveMergeRefusalRows(requested []int, sets ...[]dispatchWaveIssueRefusal) []dispatchWaveIssueRefusal {
	byIssue := map[int]dispatchWaveIssueRefusal{}
	for _, rows := range sets {
		for _, row := range rows {
			if row.Issue > 0 && byIssue[row.Issue].Issue == 0 {
				byIssue[row.Issue] = row
			}
		}
	}
	return dispatchWaveRefusalRows(requested, byIssue)
}

func dispatchWaveRefusalRows(requested []int, byIssue map[int]dispatchWaveIssueRefusal) []dispatchWaveIssueRefusal {
	if len(byIssue) == 0 {
		return nil
	}
	out := make([]dispatchWaveIssueRefusal, 0, len(byIssue))
	seen := map[int]bool{}
	for _, issue := range requested {
		if row := byIssue[issue]; row.Issue != 0 {
			out = append(out, row)
			seen[issue] = true
		}
	}
	extra := make([]int, 0, len(byIssue)-len(out))
	for issue := range byIssue {
		if !seen[issue] {
			extra = append(extra, issue)
		}
	}
	sort.Ints(extra)
	for _, issue := range extra {
		out = append(out, byIssue[issue])
	}
	return out
}

func dispatchWaveIntSet(issues []int) map[int]bool {
	out := make(map[int]bool, len(issues))
	for _, issue := range issues {
		out[issue] = true
	}
	return out
}

func dispatchWaveExecutionPlanID(plan []dispatchWaveExecutionPlan) string {
	if len(plan) == 0 {
		return ""
	}
	return dispatchStablePlanID(plan)
}

func dispatchWaveExecutionPlans(root, backend, workKind, goal, goalProfile, waveID string, shortfall int, targets []dispatchWaveCandidate, lanes []dispatchtick.AccountWaveLane, recordLoop bool, codexLoopGate string, codexLoopGateSinceHours float64, codexLoopGateLimit int) []dispatchWaveExecutionPlan {
	limit := minInt(len(targets), len(lanes))
	if limit <= 0 {
		return nil
	}
	out := make([]dispatchWaveExecutionPlan, 0, limit)
	for i := 0; i < limit; i++ {
		target := targets[i]
		acct := accountFromWaveLane(lanes[i])
		mem := dispatchtick.Membership{Rank: i, WaveID: waveID, Size: limit, Shortfall: shortfall}
		args := dispatchWaveExecutionTickArgs(root, backend, workKind, goal, goalProfile, target, acct, mem, recordLoop, codexLoopGate, codexLoopGateSinceHours, codexLoopGateLimit)
		out = append(out, dispatchWaveExecutionPlan{
			Rank:                i,
			WaveID:              waveID,
			WaveSize:            limit,
			Shortfall:           shortfall,
			Backend:             backend,
			WorkKind:            workKind,
			Goal:                goal,
			GoalProfile:         goalProfile,
			Target:              dispatchWaveLaunchTarget(target),
			Account:             dispatchtick.AccountSidecar(acct),
			RecordLoop:          recordLoop,
			DispatchTickArgs:    args,
			DispatchTickCommand: append([]string{"fak", "dispatch", "tick"}, args...),
		})
	}
	return out
}

func dispatchWaveExecutionTickArgs(root, backend, workKind, goal, goalProfile string, target dispatchWaveCandidate, account dispatchtick.Account, membership dispatchtick.Membership, recordLoop bool, codexLoopGate string, codexLoopGateSinceHours float64, codexLoopGateLimit int) []string {
	args := []string{"--workspace", root, "--backend", backend}
	if strings.TrimSpace(workKind) != "" {
		args = append(args, "--work-kind", workKind)
	}
	if strings.TrimSpace(goal) != "" {
		args = append(args, "--goal", strings.TrimSpace(goal))
	}
	if strings.TrimSpace(goalProfile) != "" && strings.TrimSpace(goalProfile) != dispatchGoalProfileThroughput {
		args = append(args, "--goal-profile", strings.TrimSpace(goalProfile))
	}
	args = append(args, dispatchTickArgsForLaunchTarget(target)...)
	if !recordLoop {
		args = append(args, "--no-loop-ledger")
	}
	if backend == "codex" {
		args = append(args,
			"--codex-loop-gate", firstString(strings.TrimSpace(codexLoopGate), dispatchCodexLoopGateDefaultThreshold()),
			"--codex-loop-gate-since-hours", fmt.Sprint(codexLoopGateSinceHours),
			"--codex-loop-gate-limit", fmt.Sprint(codexLoopGateLimit),
		)
	}
	if account.Tag != "" {
		args = append(args, "--account-tag", account.Tag)
	}
	if account.Tier != nil {
		args = append(args, "--account-tier", fmt.Sprint(account.Tier))
	}
	if account.Model != "" {
		args = append(args, "--account-model", account.Model)
	}
	if account.Dir != "" {
		args = append(args, "--account-dir", account.Dir)
	}
	if membership.WaveID != "" {
		args = append(args,
			"--wave-id", membership.WaveID,
			"--wave-rank", fmt.Sprint(membership.Rank),
			"--wave-size", fmt.Sprint(membership.Size),
			"--wave-shortfall", fmt.Sprint(membership.Shortfall),
		)
	}
	return args
}

func dispatchWaveLaneSerialWaveCount(candidates []dispatchWaveCandidate) int {
	if len(candidates) == 0 {
		return 0
	}
	return dispatchLaneSerialWaveCount(dispatchCandidateKeys(candidates,
		func(cand dispatchWaveCandidate) string {
			return dispatchLaneSerialKey(cand.Lane, cand.LeaseID, cand.ID)
		}))
}

func dispatchWaveRouteStepBudget(route dispatchtick.IssueRoute) int {
	if route.ExpectedSteps > 0 {
		return route.ExpectedSteps
	}
	return 1
}

func dispatchWaveLaneStepBudget(grp dispatchtick.RouterLaneGroup) int {
	if grp.StepBudget > 0 {
		return grp.StepBudget
	}
	if grp.Count > 0 {
		return grp.Count
	}
	return len(grp.Issues)
}

func dispatchWaveStepBudget(candidates []dispatchWaveCandidate) int {
	total := 0
	for _, cand := range candidates {
		if cand.StepBudget > 0 {
			total += cand.StepBudget
		} else {
			total++
		}
	}
	return total
}

func dispatchWavePct(n, d int) int {
	if d <= 0 {
		return 0
	}
	return int(float64(n)*100/float64(d) + 0.5)
}

func waveCandidateID(lane string, issue int) string {
	if issue > 0 {
		return fmt.Sprintf("%s#%d", lane, issue)
	}
	return lane
}

func dispatchLaneLeaseID(lane string) string {
	return "resolve-" + cleanDispatchLeaseToken(lane)
}

func dispatchIssueLeaseID(lane string, issue int) string {
	return fmt.Sprintf("resolve-%s-%d", cleanDispatchLeaseToken(lane), issue)
}

func cleanDispatchLeaseToken(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "unknown"
	}
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-.")
}

func firstLaunchableIssue(nums []int, live, cooled, excluded map[int]bool) (int, bool) {
	for _, n := range nums {
		if !live[n] && !cooled[n] && !excluded[n] {
			return n, true
		}
	}
	return 0, false
}
