package main

// Issue-to-lane routing for the dispatch tick: fetch an issue's live state from
// GitHub (labels, state, progress-comment commit audits), load the lane taxonomy
// from dos.toml, pick the lane whose declared tree owns the issue, and assemble
// the prompt a spawned resolution worker starts from. Split out of
// dispatch_tick.go along this concern seam so the dispatch surface stays
// steerable as new verbs land (steerability dispatch_god_file).
// Behavior-preserving code motion -- same package, no logic change.
import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

type dispatchIssueInfo struct {
	Number     int
	Title      string
	Body       string
	Labels     []string
	State      string
	FetchError string
}

var dispatchFetchIssue = dispatchFetchIssueGH
var dispatchRouteIssues = dispatchRouteIssuesNative

func dispatchPrompt(root string, _ io.Writer, issue int, lane string) (map[string]any, error) {
	inf := dispatchFetchIssue(root, issue)
	roles, roleErr := branchrole.Load(root)
	rec := dispatchtick.BuildIssuePrompt(dispatchtick.IssuePromptInput{
		Number:            firstInt(inf.Number, issue),
		Title:             inf.Title,
		Body:              inf.Body,
		Labels:            inf.Labels,
		Lane:              lane,
		Workspace:         root,
		DevelopmentBranch: roles.DevelopmentBranch,
		FetchError:        inf.FetchError,
		ResumeWitness: dispatchtick.ResumeWitnessState{
			LastCommitAudit:   dispatchLastCommitAudit(root, issue),
			LastRouteDecision: dispatchLastRouteDecision(issue, lane),
			LastIssueStatus:   dispatchLastIssueStatus(inf.State),
		},
	})
	out := map[string]any{
		"schema":             rec.Schema,
		"issue":              rec.Issue,
		"lane":               rec.Lane,
		"title":              rec.Title,
		"body":               inf.Body,
		"fetch_error":        rec.FetchError,
		"prompt":             rec.Prompt,
		"prompt_chars":       rec.PromptChars,
		"development_branch": roles.DevelopmentBranch,
	}
	if roleErr != nil {
		out["branch_role_error"] = roleErr.Error()
	}
	return out, nil
}

func dispatchFetchIssueGH(root string, issue int) dispatchIssueInfo {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "issue", "view", strconv.Itoa(issue), "--json", "number,title,body,labels,state")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return dispatchIssueInfo{Number: issue, FetchError: truncateString(strings.TrimSpace(string(out)), 300)}
	}
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		return dispatchIssueInfo{Number: issue, FetchError: "gh issue view produced no JSON"}
	}
	n := dispatchMapInt(doc, "number")
	if n == 0 {
		n = issue
	}
	return dispatchIssueInfo{
		Number: n,
		Title:  dispatchMapString(doc, "title"),
		Body:   dispatchMapString(doc, "body"),
		Labels: dispatchIssueLabels(doc["labels"]),
		State:  dispatchMapString(doc, "state"),
	}
}

func dispatchIssueLabels(raw any) []string {
	out := []string{}
	for _, item := range anySlice(raw) {
		if m, ok := item.(map[string]any); ok {
			if name := dispatchMapString(m, "name"); name != "" {
				out = append(out, name)
			}
		}
	}
	return out
}

func dispatchLastCommitAudit(root string, issue int) string {
	if issue <= 0 {
		return ""
	}
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	rows := dispatchProgressReadRows(runsDir)
	for i := len(rows) - 1; i >= 0; i-- {
		row := rows[i]
		if !dispatchProgressRowMentionsIssue(row, issue) {
			continue
		}
		if summary := dispatchProgressCommitAuditSummary(row); summary != "" {
			return summary
		}
	}
	return ""
}

func dispatchProgressRowMentionsIssue(row map[string]any, issue int) bool {
	if dispatchMapInt(row, "issue") == issue || dispatchMapInt(row, "target_issue") == issue {
		return true
	}
	for _, n := range dispatchProgressIntSlice(row["witnessed_numbers"]) {
		if n == issue {
			return true
		}
	}
	closeResult := mapAt(row, "close_result")
	return dispatchMapInt(closeResult, "issue") == issue || dispatchMapInt(closeResult, "number") == issue
}

func dispatchProgressCommitAuditSummary(row map[string]any) string {
	closeResult := mapAt(row, "close_result")
	if len(closeResult) > 0 {
		parts := []string{"commit-audit close_result"}
		if _, ok := closeResult["ok"]; ok {
			parts = append(parts, "ok="+strconv.FormatBool(dispatchMapBool(closeResult, "ok")))
		}
		if verdict := firstString(
			dispatchMapString(closeResult, "verdict"),
			dispatchMapString(closeResult, "status"),
			dispatchMapString(closeResult, "reason"),
		); verdict != "" {
			parts = append(parts, "verdict="+verdict)
		}
		if sha := firstString(
			dispatchMapString(closeResult, "commit_sha"),
			dispatchMapString(closeResult, "sha"),
			dispatchMapString(closeResult, "commit"),
		); sha != "" {
			parts = append(parts, "sha="+sha)
		}
		if reason := firstString(
			dispatchMapString(closeResult, "blocker_reason"),
			dispatchMapString(closeResult, "error"),
		); reason != "" {
			parts = append(parts, "reason="+reason)
		}
		return strings.Join(parts, " ")
	}
	if errText := dispatchMapString(row, "audit_error"); errText != "" {
		return "commit-audit unavailable: " + errText
	}
	if len(dispatchProgressIntSlice(row["witnessed_numbers"])) > 0 {
		return "commit-audit witnessed issue still open"
	}
	return ""
}

func dispatchLastRouteDecision(issue int, lane string) string {
	lane = strings.TrimSpace(lane)
	if lane == "" || issue <= 0 {
		return ""
	}
	return fmt.Sprintf("lane=%s target=#%d", lane, issue)
}

func dispatchLastIssueStatus(state string) string {
	state = strings.TrimSpace(state)
	if state == "" {
		return ""
	}
	return strings.ToUpper(state)
}

func pickDispatchLane(root string, stderr io.Writer, explicit string, exclude map[string]bool, preferNewest bool, generation string) (dispatchLanePick, error) {
	router, err := dispatchRouteIssues(root, stderr)
	if err != nil {
		return dispatchLanePick{}, err
	}
	numsByLane := map[string][]int{}
	treesByLane := map[string][]string{}
	counts := map[string]int{}
	stepBudgets := map[string]int{}
	for lane, info := range router.Lanes {
		nums := append([]int(nil), info.Issues...)
		treesByLane[lane] = append([]string(nil), info.Tree...)
		// Order the lane's open issues PRIORITY-first, then by recency (#1395), so
		// PickTargetIssue (which takes the first not-skipped) drains the heaviest
		// priority/P* work before newer unlabeled noise: an old priority/P1 outranks
		// a fresh unlabeled filing. Ties fall back to the by-number recency order --
		// oldest-first by default (GitHub issue numbers are monotonic in creation
		// time, so the dispatcher drains the oldest backlog instead of forever
		// chasing the newest filing), newest-first under --prefer-newest. When no
		// candidate carries a priority/* label every weight is equal and the order
		// is byte-for-byte the old by-number order. This is safe ("when reasonable")
		// because the anti-churn cooldown (recentlyAttemptedIssues) advances past an
		// issue a worker could not land rather than re-storming it every tick.
		//
		// Generation-aware on top (docs/generation-loop-scheduling.md): when a
		// candidate carries a gen/* label, or --generation names an explicit
		// horizon, gen/now and gen/next stay launchable by default while
		// gen/second-next, gen/future, and unclassified issues are held. The gate
		// stays OFF for an ordinary, generation-blind lane, so this never holds
		// the backlog just because none of it happens to carry a gen/* label.
		cands := make([]dispatchtick.GenerationCandidate, len(nums))
		for i, n := range nums {
			weight := dispatchtick.PriorityWeightDefault
			if w, ok := info.Priority[n]; ok {
				weight = w
			}
			cands[i] = dispatchtick.GenerationCandidate{Number: n, Weight: weight, Generation: info.Generation[n]}
		}
		numsByLane[lane] = dispatchtick.OrderEligibleGenerationCandidates(cands, generation, preferNewest)
		counts[lane] = len(nums)
		stepBudget := info.StepBudget
		if stepBudget <= 0 {
			stepBudget = len(nums)
		}
		stepBudgets[lane] = stepBudget
	}
	chosen := strings.TrimSpace(explicit)
	var selfSourceHeld []string
	if chosen == "" {
		// #1397: skip fak's own-source lanes (cmd/** + internal/**) BEFORE the
		// busiest-by-step-budget pick when this tick is guarded. On a guarded trunk the
		// backlog is dominated by self-source internal/** lanes, so a picker that chose
		// the busiest lane and only THEN ran SelfModifyHoldForPick would HOLD every tick
		// and surface nothing -- even though docs/tools/.github/examples carry shippable
		// work. Skipping them here lets the auto-pick land on a shippable lane. The
		// EXPLICIT-lane path (explicit != "") is deliberately untouched: an operator who
		// names a self-source lane must still reach the post-pick SELF_MODIFY hold.
		guarded := !guardDisabled()
		bestStepBudget := -1
		bestCount := -1
		for lane, nums := range numsByLane {
			if exclude[lane] {
				continue
			}
			if !dispatchtick.LaneDispatchableUnderGuard(guarded, treesByLane[lane]) {
				selfSourceHeld = append(selfSourceHeld, lane)
				continue
			}
			stepBudget := stepBudgets[lane]
			if stepBudget > bestStepBudget ||
				(stepBudget == bestStepBudget && len(nums) > bestCount) ||
				(stepBudget == bestStepBudget && len(nums) == bestCount && lane < chosen) {
				chosen = lane
				bestStepBudget = stepBudget
				bestCount = len(nums)
			}
		}
		sort.Strings(selfSourceHeld)
	}
	excluded := make([]string, 0, len(exclude))
	for lane := range exclude {
		excluded = append(excluded, lane)
	}
	sort.Strings(excluded)
	tree := treesByLane[chosen]
	if len(tree) == 0 && chosen != "" {
		tree = []string{fmt.Sprintf("internal/%s/**", chosen)}
	}
	return dispatchLanePick{
		Lane:             chosen,
		Numbers:          numsByLane[chosen],
		ByLaneCount:      counts,
		ByLaneStepBudget: stepBudgets,
		ExcludedLanes:    excluded,
		Tree:             tree,
		RouterError:      dispatchRouterError(router),
		SelfSourceHeld:   selfSourceHeld,
	}, nil
}

func dispatchRouteIssuesNative(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
	const issueLimit = 1000
	taxonomy, taxErr := dispatchLaneTaxonomy(root)
	issues, issueErr := dispatchFetchOpenIssues(root, issueLimit)
	fetchErrs := []string{}
	if taxErr != nil {
		fetchErrs = append(fetchErrs, taxErr.Error())
	}
	if issueErr != nil {
		fetchErrs = append(fetchErrs, issueErr.Error())
	}
	return dispatchtick.RouteIssues(dispatchtick.RouterInput{
		Workspace:  root,
		Taxonomy:   taxonomy,
		Issues:     issues,
		IssueLimit: issueLimit,
		FetchError: strings.Join(fetchErrs, "; "),
	}), nil
}

var dispatchLoadLaneTaxonomy = func(root string) (dispatchtick.LaneTaxonomy, error) {
	if taxonomy, err := dispatchLaneTaxonomy(root); err == nil && len(taxonomy.Trees) > 0 {
		return taxonomy, nil
	}
	return dispatchLaneTaxonomyFromFile(root)
}

func dispatchLaneTaxonomy(root string) (dispatchtick.LaneTaxonomy, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "dos", "doctor", "--workspace", root, "--json")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	doc, perr := lastJSONObject(out)
	if perr != nil {
		if err != nil {
			return dispatchtick.LaneTaxonomy{}, fmt.Errorf("dos doctor: %w (%s)", err, strings.TrimSpace(string(out)))
		}
		return dispatchtick.LaneTaxonomy{}, fmt.Errorf("dos doctor produced no JSON")
	}
	lanes := mapAt(doc, "lanes")
	taxonomy := dispatchtick.LaneTaxonomy{
		Concurrent: stringSlice(lanes["concurrent"]),
		Trees:      map[string][]string{},
	}
	if raw, ok := lanes["trees"].(map[string]any); ok {
		for lane, globs := range raw {
			taxonomy.Trees[lane] = stringSlice(globs)
		}
	}
	return taxonomy, nil
}

func dispatchLaneTaxonomyFromFile(root string) (dispatchtick.LaneTaxonomy, error) {
	raw, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		return dispatchtick.LaneTaxonomy{}, err
	}
	taxonomy := dispatchtick.LaneTaxonomy{Trees: map[string][]string{}}
	section := ""
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "["), "]"))
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		values := parseDispatchTomlStringArray(parts[1])
		switch section {
		case "lanes":
			if key == "concurrent" {
				taxonomy.Concurrent = values
			}
		case "lanes.trees":
			if key != "" {
				taxonomy.Trees[key] = values
			}
		}
	}
	return taxonomy, nil
}

func parseDispatchTomlStringArray(raw string) []string {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "[")
	raw = strings.TrimSuffix(raw, "]")
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.Trim(strings.TrimSpace(part), `"`)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func dispatchFetchOpenIssues(root string, limit int) ([]dispatchtick.Issue, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "issue", "list", "--state", "open", "--limit", strconv.Itoa(limit), "--json", "number,title,labels,body")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	var issues []dispatchtick.Issue
	if uerr := json.Unmarshal(out, &issues); uerr != nil {
		if err != nil {
			return nil, fmt.Errorf("gh issue list: %w (%s)", err, strings.TrimSpace(string(out)))
		}
		return nil, fmt.Errorf("gh issue list produced invalid JSON: %w", uerr)
	}
	return issues, nil
}

func dispatchRouterError(router dispatchtick.RouterPayload) string {
	if router.OK {
		return ""
	}
	return router.Reason
}
