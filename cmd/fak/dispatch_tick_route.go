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
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/dispatchcache"
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

const dispatchRoutedBacklogTTL = 5 * time.Second

var dispatchRoutedBacklogCache = dispatchcache.New[dispatchtick.RouterPayload](time.Now)

const dispatchLaneQueueTTL = 5 * time.Minute

func dispatchLaneQueuePath(root string) string {
	return filepath.Join(root, ".fak", "dispatch", "lane-queues.json")
}

func persistDispatchLaneQueues(root, key string, payload dispatchtick.RouterPayload, now time.Time) error {
	lanes := make(map[string][]int, len(payload.Lanes))
	for lane, group := range payload.Lanes {
		lanes[lane] = append([]int(nil), group.Issues...)
	}
	return dispatchcache.WriteQueues(dispatchLaneQueuePath(root), key, lanes, now)
}

// dispatchDuplicateRiskCache lives across ticks so RouteIssues only reruns the
// O(n^2) duplicate-risk scan when the routable backlog's content hash actually
// changes (#4171). Each call re-hashes, so a stale key can never return a wrong
// map; output stays byte-identical to the always-recompute path.
var dispatchDuplicateRiskCache = &dispatchtick.DuplicateRiskCache{}

func dispatchPrompt(root string, _ io.Writer, issue int, lane string, cached ...dispatchIssueInfo) (map[string]any, error) {
	// #4167: prefer the router-fetched row (title/body/labels already in hand from the
	// tick's list/view fetch) so the hot path skips a redundant second `gh issue view`.
	// Fall back to the live fetch only when no cached row was threaded in or its body is
	// absent -- an unrouted `--target-issue N` (no cache hit), or the list fetch errored.
	// A routed issue is open, so the cached row carries State=OPEN for the resume witness.
	var inf dispatchIssueInfo
	if len(cached) > 0 && strings.TrimSpace(cached[0].Body) != "" {
		inf = cached[0]
	} else {
		inf = dispatchFetchIssue(root, issue)
	}
	roles, roleErr := branchrole.Load(root)
	contract := dispatchtick.ParseObjectiveContract(inf.Body)
	contract.ObjectiveID = fmt.Sprintf("issue-%d", firstInt(inf.Number, issue))
	rec := dispatchtick.BuildIssuePrompt(dispatchtick.IssuePromptInput{
		Number:            firstInt(inf.Number, issue),
		Title:             inf.Title,
		Body:              inf.Body,
		ObjectiveContract: contract,
		Labels:            inf.Labels,
		Lane:              lane,
		Workspace:         root,
		DevelopmentBranch: roles.DevelopmentBranch,
		FetchError:        inf.FetchError,
		ResumeWitness: dispatchtick.ResumeWitnessState{
			LastCommitAudit:   dispatchLastCommitAudit(root, issue),
			LastRouteDecision: dispatchLastRouteDecision(issue, lane, dispatchTickView),
			LastIssueStatus:   dispatchLastIssueStatus(inf.State),
		},
	})
	out := map[string]any{
		"schema":             rec.Schema,
		"issue":              rec.Issue,
		"lane":               rec.Lane,
		"title":              rec.Title,
		"body":               inf.Body,
		"labels":             inf.Labels,
		"objective_contract": contract,
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

// dispatchStringSlice reads a []string out of a prompt-record field, tolerating both the
// in-process []string dispatchPrompt stores and a []any that survives a JSON round-trip, so
// the tier-launch label read works whether the record was built in-process or rehydrated.
func dispatchStringSlice(raw any) []string {
	if ss, ok := raw.([]string); ok {
		return ss
	}
	out := []string{}
	for _, v := range anySlice(raw) {
		if s, ok := v.(string); ok {
			out = append(out, s)
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

func dispatchLastRouteDecision(issue int, lane, view string) string {
	lane = strings.TrimSpace(lane)
	if lane == "" || issue <= 0 {
		return ""
	}
	if view = strings.TrimSpace(view); view != "" {
		return fmt.Sprintf("view=%s lane=%s target=#%d", view, lane, issue)
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

func dispatchPersistedLanePick(root, lane string, now time.Time) (dispatchLanePick, bool) {
	const issueLimit = 1000
	key := dispatchcache.Key(root, dispatchTickView, issueLimit)
	q, ok := dispatchcache.ReadQueues(dispatchLaneQueuePath(root), key, dispatchLaneQueueTTL, now)
	if !ok || len(q.Lanes[lane]) == 0 {
		return dispatchLanePick{}, false
	}
	tree := []string(nil)
	if taxonomy, err := dispatchLoadLaneTaxonomy(root); err == nil {
		tree = append(tree, taxonomy.Trees[lane]...)
	}
	if len(tree) == 0 {
		tree = []string{fmt.Sprintf("internal/%s/**", lane)}
	}
	counts := make(map[string]int, len(q.Lanes))
	for name, issues := range q.Lanes {
		counts[name] = len(issues)
	}
	return dispatchLanePick{
		Lane: lane, Numbers: append([]int(nil), q.Lanes[lane]...), Tree: tree,
		ByLaneCount: counts, ByLaneStepBudget: counts, PathsByIssue: map[int][]string{},
		IssueByNumber: map[int]dispatchIssueInfo{}, View: strings.TrimSpace(dispatchTickView),
	}, true
}

func pickDispatchLane(root string, stderr io.Writer, explicit string, exclude map[string]bool, preferNewest bool, generation, goalProfile string, targetIssue int) (dispatchLanePick, error) {
	// A lane-pinned tick can consume the scorer's durable lane ordering directly.
	// It needs no whole-backlog regroup: live/cooldown filtering still happens in
	// resolveDispatchTickPick after this return. Auto-pick and target routing retain
	// the full router because they need cross-lane scores and per-issue scope.
	if lane := strings.TrimSpace(explicit); lane != "" && targetIssue == 0 && !preferNewest && strings.TrimSpace(generation) == "" {
		if pick, ok := dispatchPersistedLanePick(root, lane, time.Now()); ok {
			return pick, nil
		}
	}
	router, err := dispatchRouteIssues(root, stderr)
	if err != nil {
		return dispatchLanePick{}, err
	}
	// When no lane is pinned but a specific target issue is requested, route THAT
	// issue to its own lane before the busiest-lane auto-pick runs. Without this a
	// `--target-issue N` with no `--lane` falls through to the largest-step-budget
	// lane (in practice the coarse `cmd` lane) and the issue is dispatched there,
	// colliding with whatever holds cmd — regardless of where the issue actually
	// belongs. The operator named the issue; the router already knows its lane, so
	// honor that as if it were an explicit `--lane`. An unrouted target (Lane == "")
	// finds nothing here and falls through to the unchanged busiest-pick.
	targetLaneRouted := ""
	if strings.TrimSpace(explicit) == "" && targetIssue > 0 {
		for _, route := range router.Issues {
			if route.Number == targetIssue && route.Lane != "" {
				targetLaneRouted = route.Lane
				break
			}
		}
	}
	numsByLane := map[string][]int{}
	treesByLane := map[string][]string{}
	priorityByLane := map[string]map[int]int{}
	counts := map[string]int{}
	stepBudgets := map[string]int{}
	for lane, info := range router.Lanes {
		nums := append([]int(nil), info.Issues...)
		treesByLane[lane] = append([]string(nil), info.Tree...)
		priorityByLane[lane] = map[int]int{}
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
			priorityByLane[lane][n] = weight
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
	if chosen == "" && targetLaneRouted != "" {
		chosen = targetLaneRouted
	}
	var selfSourceHeld []string
	if chosen == "" {
		// #1397: skip fak's TRUST-CRITICAL lanes (the adjudicator/policy/kernel/shipgate
		// the referee binds to) BEFORE the busiest-by-step-budget pick when this tick is
		// guarded, so a picker that chose the busiest lane and only THEN ran
		// SelfModifyHoldForPick can never HOLD a whole tick and surface nothing. Only that
		// narrow referee set is skipped -- gateway/agent/compute/cmd and the rest of
		// internal/** are guard-shippable and stay in the busiest-pick. The EXPLICIT-lane
		// path (explicit != "") is deliberately untouched: an operator who names a
		// trust-critical lane must still reach the post-pick SELF_MODIFY hold.
		guarded := !guardDisabled()
		laneCandidates := make([]dispatchtick.DispatchLaneCandidate, 0, len(numsByLane))
		maxStepBudget, maxCount := 0, 0
		for lane, nums := range numsByLane {
			if exclude[lane] {
				continue
			}
			if !dispatchtick.LaneDispatchableUnderGuard(guarded, treesByLane[lane]) {
				selfSourceHeld = append(selfSourceHeld, lane)
				continue
			}
			candidate := dispatchtick.DispatchLaneCandidate{
				Lane: lane, Priority: dispatchLaneTopPriority(nums, priorityByLane[lane]),
				StepBudget: stepBudgets[lane], Count: len(nums),
				Core: dispatchtick.IsCoreSourceLaneTree(treesByLane[lane]),
			}
			laneCandidates = append(laneCandidates, candidate)
			if candidate.StepBudget > maxStepBudget {
				maxStepBudget = candidate.StepBudget
			}
			if candidate.Count > maxCount {
				maxCount = candidate.Count
			}
		}
		highPriority := goalProfile == dispatchGoalProfileHighPriority
		orderedLanes := dispatchtick.DefaultDispatchLaneScorers(highPriority, maxStepBudget, maxCount).Order(laneCandidates)
		if len(orderedLanes) > 0 {
			chosen = orderedLanes[0].Lane
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
	// Carry each issue's declared file scope so a single-target tick can narrow to
	// it (parity with the wave path, which prices route.Paths per issue).
	pathsByIssue := map[int][]string{}
	// #4167: cache each routed issue's already-fetched row so dispatchPrompt can build the
	// worker prompt from the body the list/view fetch already returned, instead of a second
	// `gh issue view`. A routed issue is open, so State is OPEN for the resume witness. A
	// cache miss (unrouted --target-issue) leaves dispatchPrompt to fall back to the fetch.
	issueByNumber := map[int]dispatchIssueInfo{}
	for _, route := range router.Issues {
		if len(route.Paths) > 0 {
			pathsByIssue[route.Number] = append([]string(nil), route.Paths...)
		}
		issueByNumber[route.Number] = dispatchIssueInfo{
			Number: route.Number,
			Title:  route.Title,
			Body:   route.Body,
			Labels: append([]string(nil), route.Labels...),
			State:  "OPEN",
		}
	}
	return dispatchLanePick{
		Lane:               chosen,
		Numbers:            numsByLane[chosen],
		ByLaneCount:        counts,
		ByLaneStepBudget:   stepBudgets,
		ExcludedLanes:      excluded,
		Tree:               tree,
		PathsByIssue:       pathsByIssue,
		IssueByNumber:      issueByNumber,
		View:               router.View,
		ViewQuery:          router.ViewQuery,
		ViewDigest:         router.ViewDigest,
		ViewFallback:       router.ViewFallback,
		ViewFallbackReason: router.ViewFallbackReason,
		RouterError:        dispatchRouterError(router),
		SelfSourceHeld:     selfSourceHeld,
	}, nil
}

func dispatchLaneTopPriority(nums []int, weights map[int]int) int {
	if len(nums) == 0 {
		return -1
	}
	if w, ok := weights[nums[0]]; ok {
		return w
	}
	return dispatchtick.PriorityWeightDefault
}

func dispatchLaneBetterForGoal(profile string, priority, stepBudget, count int, core bool, lane string, bestPriority, bestStepBudget, bestCount int, bestCore bool, chosen string) bool {
	if strings.TrimSpace(lane) == "" {
		return false
	}
	if strings.TrimSpace(chosen) == "" {
		return true
	}
	if profile == dispatchGoalProfileHighPriority {
		if count <= 0 && bestCount > 0 {
			return false
		}
		if count > 0 && bestCount <= 0 {
			return true
		}
		if priority != bestPriority {
			return priority > bestPriority
		}
	}
	// Default the unattended wave toward fak's own guard-shippable core engineering
	// (cmd/** + internal/**, minus the trust-critical referee) instead of the coarse
	// docs/tools buckets. Under throughput this is the LEADING term: a trunk-safe core
	// leaf outranks a docs/tools lane regardless of step budget, so "richest-first" can no
	// longer starve fragmented per-leaf core work behind the big docs/tools buckets. Under
	// high-priority it sits BELOW the priority-label compare, so a genuinely urgent
	// docs/tools issue still wins.
	if core != bestCore {
		return core
	}
	if stepBudget != bestStepBudget {
		return stepBudget > bestStepBudget
	}
	if count != bestCount {
		return count > bestCount
	}
	return lane < chosen
}

// dispatchDefaultView is the named issue-view (.github/issue-views.json) the
// unattended `fak dispatch tick` scopes its issue selection to by default: the
// operator-marked board/milestone focus (#1411). `--view ""` disables the
// scoping; a missing, unresolvable, or empty view fail-softs to the full open
// backlog so the tick never starves (parity with issue_lane_router.py --view).
const dispatchDefaultView = "current"

// dispatchTickView is the view slug the native router scopes its open-issue
// fetch to. A package seam rather than a dispatchRouteIssues parameter so the
// seam's many stubs keep their signature; only the tick writes it (from its
// --view flag), so every other dispatch verb keeps the full open backlog.
var dispatchTickView = ""

// Seams so a test can drive the view/backlog selection without gh.
var dispatchFetchViewIssues = dispatchFetchViewIssuesGH
var dispatchFetchBacklogIssues = dispatchFetchOpenIssues

type dispatchBacklogDelta struct {
	Issues    []dispatchtick.Issue
	Closed    []int
	Watermark time.Time
}

var dispatchFetchBacklogDeltaIssues = dispatchFetchBacklogDeltaGH

func dispatchBacklogSnapshotPath(root string) string {
	return filepath.Join(root, ".fak", "dispatch", "backlog.json")
}

func dispatchIssueRows(issues []dispatchtick.Issue) []dispatchcache.BacklogIssue {
	rows := make([]dispatchcache.BacklogIssue, 0, len(issues))
	for _, issue := range issues {
		if b, err := json.Marshal(issue); err == nil {
			rows = append(rows, dispatchcache.BacklogIssue{Number: issue.Number, Data: b})
		}
	}
	return rows
}
func dispatchRowsIssues(rows []dispatchcache.BacklogIssue) ([]dispatchtick.Issue, error) {
	out := make([]dispatchtick.Issue, 0, len(rows))
	for _, row := range rows {
		var issue dispatchtick.Issue
		if err := json.Unmarshal(row.Data, &issue); err != nil {
			return nil, err
		}
		out = append(out, issue)
	}
	return out, nil
}

func dispatchFetchBacklogIncremental(root string, limit int, now time.Time) ([]dispatchtick.Issue, error) {
	key := dispatchcache.Key(root, "", limit)
	path := dispatchBacklogSnapshotPath(root)
	if snap, ok := dispatchcache.ReadBacklog(path, key); ok {
		delta, err := dispatchFetchBacklogDeltaIssues(root, snap.Watermark, limit)
		if err == nil {
			watermark := delta.Watermark
			if watermark.IsZero() {
				watermark = now
			}
			// SyncBacklog merges and persists in one step so a quiet tick rewrites only the
			// watermark sidecar instead of the whole multi-megabyte issue array (#6092).
			merged, werr := dispatchcache.SyncBacklog(path, key, watermark, snap.Issues, dispatchIssueRows(delta.Issues), delta.Closed)
			if werr == nil {
				return dispatchRowsIssues(merged)
			}
		}
	}
	issues, err := dispatchFetchBacklogIssues(root, limit)
	if err != nil {
		return nil, err
	}
	_ = dispatchcache.WriteBacklog(path, key, now, dispatchIssueRows(issues))
	return issues, nil
}

func dispatchRouteIssuesNative(root string, stderr io.Writer) (dispatchtick.RouterPayload, error) {
	payload, err := dispatchRoutedBeforePrereqHold(root, stderr)
	if err != nil {
		return payload, err
	}
	// Operator pause hold (#5031): move each `fak steer pause`d unit's bound issue into the
	// skipped set under the existing BLOCKED_BY_HUMAN token before anything downstream picks.
	// Runs BEFORE the prereq hold on purpose -- a paused prerequisite stays in
	// SkippedHumanBlocked (still open), so a dependent of it remains correctly held. Reads the
	// ledger fresh each tick (never the routed cache), so a resume takes effect next tick.
	payload = holdSteerPausedForRoute(root, payload)
	// Dependency soft-hold: after the known-bad hold, hold back any dispatchable leaf whose
	// "depends-on:/blocked-by: #N" prerequisite is still an OPEN candidate this tick. Runs AFTER
	// known-bad on purpose -- a known-bad-held prerequisite stays in SkippedHumanBlocked (still
	// open), so a dependent of it remains correctly held. Fails open on a closed/absent prerequisite.
	payload = holdOpenPrereqForRoute(payload)
	fields := dispatchFetchProjectFields(root)
	for lane, group := range payload.Lanes {
		group.Priority, group.Issues = dispatchtick.MergeProjectFields(group.Priority, group.Issues, fields)
		group.Count = len(group.Issues)
		payload.Lanes[lane] = group
	}
	return payload, nil
}

// dispatchRoutedBeforePrereqHold fetches, routes, and applies the known-bad scope-hold, but NOT the
// dependency prereq hold -- so every routed candidate in payload.Issues still carries its BlockedBy
// edges. The prereq hold moves held dependents into SkippedHumanBlocked (a SkippedIssue drops the
// edges), so a consumer that needs the full dependency graph (fak dispatch graph) reads THIS payload,
// while the live tick reads dispatchRouteIssuesNative, which folds the hold on top.
func dispatchRoutedBeforePrereqHold(root string, stderr io.Writer) (dispatchtick.RouterPayload, error) {
	const issueLimit = 1000
	cacheKey := dispatchcache.Key(root, dispatchTickView, issueLimit)
	if payload, ok := dispatchRoutedBacklogCache.Get(cacheKey); ok {
		return payload, nil
	}
	taxonomy, taxErr := dispatchLaneTaxonomy(root)
	issues, injected, viewFallbackReason, issueErr := dispatchFetchScopedIssuesWithSignal(root, stderr, dispatchTickView, issueLimit)
	fetchErrs := []string{}
	if taxErr != nil {
		fetchErrs = append(fetchErrs, taxErr.Error())
	}
	if issueErr != nil {
		fetchErrs = append(fetchErrs, issueErr.Error())
	}
	payload := dispatchtick.RouteIssues(dispatchtick.RouterInput{
		Workspace:          root,
		Taxonomy:           taxonomy,
		Issues:             issues,
		IssueLimit:         issueLimit,
		Injected:           injected,
		FetchError:         strings.Join(fetchErrs, "; "),
		DuplicateRiskCache: dispatchDuplicateRiskCache,
	})
	payload.View = strings.TrimSpace(dispatchTickView)
	if payload.View != "" {
		if _, query, digest, err := dispatchViewQueryWithDigest(root, payload.View); err == nil {
			payload.ViewQuery, payload.ViewDigest = query, digest
		}
	}
	payload.ViewFallback = viewFallbackReason != ""
	payload.ViewFallbackReason = viewFallbackReason
	// W4 scope-hold (#2716): after routing, hold back ONLY the issues whose declared paths
	// intersect a live known-bad signature (internal/knownbad ledger, #2713). Disjoint
	// issues keep dispatching. Fails open on a missing/broken ledger so it never stalls.
	payload = holdKnownBadForRoute(root, payload)
	// Persist the already-scored per-lane order so the next process/tick can pop a
	// lane head without reconstructing ordering from the whole backlog (#4170).
	_ = persistDispatchLaneQueues(root, cacheKey, payload, time.Now())
	dispatchRoutedBacklogCache.Put(cacheKey, payload, dispatchRoutedBacklogTTL)
	return payload, nil
}

// dispatchFetchScopedIssues fetches the open-issue set the router folds. A
// non-empty view names an issue-view slug: the view's slice drives routing
// (injected=true, so coverage reports the slice, not a full-fetch cap). It
// FAIL-SOFTS to the full open backlog when the view cannot be resolved or
// fetched, or when it yields no dispatchable issue -- the same two WARN
// branches tools/issue_lane_router.py --view ships, so an unattended tick
// never starves on a bad or still-empty `current` view.
func dispatchFetchScopedIssues(root string, stderr io.Writer, view string, limit int) ([]dispatchtick.Issue, bool, error) {
	issues, injected, _, err := dispatchFetchScopedIssuesWithSignal(root, stderr, view, limit)
	return issues, injected, err
}

// Closed reason classes for the current-view fail-soft (#4172). When the tick scopes
// its fetch to a named issue-view but falls back to the full open backlog, exactly one
// of these names WHY: the view could not be read or resolved (view_unreadable), or it
// resolved to no dispatchable issue (view_empty). A small closed set keeps the signal
// machine-observable instead of buried in a free-text WARN line.
const (
	dispatchViewFailsoftUnreadable = "view_unreadable"
	dispatchViewFailsoftEmpty      = "view_empty"
)

// dispatchViewFailsoftRecord is the structured, machine-observable twin of the two
// stderr WARN branches in dispatchFetchScopedIssuesWithSignal: which view fell soft,
// under which closed reason class, and how many times it has fired this process. The
// fail-soft is otherwise INVISIBLE past the WARN line -- an operator watching a healthy
// full-backlog tick cannot tell the `current` view silently stopped scoping anything.
// Purely additive: the WARN text and the router payload's ViewFallbackReason are
// unchanged; a sibling leaf surfaces this record into the tick --json / metrics.
type dispatchViewFailsoftRecord struct {
	View   string `json:"view"`
	Reason string `json:"reason"`
	Count  int    `json:"count"`
}

var (
	dispatchViewFailsoftMu   sync.Mutex
	dispatchViewFailsoftLast dispatchViewFailsoftRecord
)

// recordDispatchViewFailsoft stamps the latest current-view fail-soft and bumps the
// process-lifetime count. Called at each WARN branch; safe under concurrent ticks.
func recordDispatchViewFailsoft(view, reason string) {
	dispatchViewFailsoftMu.Lock()
	defer dispatchViewFailsoftMu.Unlock()
	dispatchViewFailsoftLast.View = view
	dispatchViewFailsoftLast.Reason = reason
	dispatchViewFailsoftLast.Count++
}

// dispatchViewFailsoftSignal returns the latest fail-soft record. Reason is "" and Count
// 0 until the first fail-soft fires, so a caller can distinguish a scoped tick still
// routing its real view from one that has silently dropped to the full backlog.
func dispatchViewFailsoftSignal() dispatchViewFailsoftRecord {
	dispatchViewFailsoftMu.Lock()
	defer dispatchViewFailsoftMu.Unlock()
	return dispatchViewFailsoftLast
}

func dispatchFetchScopedIssuesWithSignal(root string, stderr io.Writer, view string, limit int) ([]dispatchtick.Issue, bool, string, error) {
	if stderr == nil {
		stderr = io.Discard
	}
	view = strings.TrimSpace(view)
	if view != "" {
		viewIssues, err := dispatchFetchViewIssues(root, view, limit)
		switch {
		case err != nil:
			fmt.Fprintf(stderr, "WARN: --view %q: %v; using full open backlog\n", view, err)
			recordDispatchViewFailsoft(view, dispatchViewFailsoftUnreadable)
		case !dispatchAnyDispatchable(viewIssues):
			fmt.Fprintf(stderr, "WARN: --view %q: no dispatchable issues; using full open backlog\n", view)
			recordDispatchViewFailsoft(view, dispatchViewFailsoftEmpty)
		default:
			return viewIssues, true, "", nil
		}
	}
	issues, err := dispatchFetchBacklogIncremental(root, limit, time.Now())
	reason := ""
	if view != "" {
		reason = "view unavailable or empty; full backlog used"
	}
	return issues, false, reason, err
}

func dispatchAnyDispatchable(issues []dispatchtick.Issue) bool {
	for _, issue := range issues {
		if dispatchtick.IsDispatchable(issue, "") {
			return true
		}
	}
	return false
}

// dispatchViewQuery resolves a view slug against .github/issue-views.json
// (the API-readable mirror of the GitHub saved views) to the repo and the
// issue-search query that materializes it.
func dispatchViewQuery(root, slug string) (repo, query string, err error) {
	repo, query, _, err = dispatchViewQueryWithDigest(root, slug)
	return
}

func dispatchViewQueryWithDigest(root, slug string) (repo, query, digest string, err error) {
	raw, err := os.ReadFile(filepath.Join(root, ".github", "issue-views.json"))
	if err != nil {
		return "", "", "", err
	}
	var cfg struct {
		Repo  string `json:"repo"`
		Views []struct {
			Slug  string `json:"slug"`
			Query string `json:"query"`
		} `json:"views"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return "", "", "", fmt.Errorf("issue-views.json: %w", err)
	}
	for _, v := range cfg.Views {
		if v.Slug == slug {
			if strings.TrimSpace(v.Query) == "" {
				return "", "", "", fmt.Errorf("view %q has an empty query", slug)
			}
			return cfg.Repo, v.Query, fmt.Sprintf("sha256:%x", sha256.Sum256(raw)), nil
		}
	}
	return "", "", "", fmt.Errorf("unknown view %q in issue-views.json", slug)
}

func dispatchFetchViewIssuesGH(root, slug string, limit int) ([]dispatchtick.Issue, error) {
	repo, query, err := dispatchViewQuery(root, slug)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	args := []string{"issue", "list"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	args = append(args, "--search", query, "--limit", strconv.Itoa(limit), "--json", "number,title,labels,body")
	cmd := exec.CommandContext(ctx, "gh", args...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	var issues []dispatchtick.Issue
	if uerr := json.Unmarshal(out, &issues); uerr != nil {
		if err != nil {
			return nil, fmt.Errorf("gh issue list --search: %w (%s)", err, strings.TrimSpace(string(out)))
		}
		return nil, fmt.Errorf("gh issue list --search produced invalid JSON: %w", uerr)
	}
	return issues, nil
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

func dispatchFetchBacklogDeltaGH(root string, watermark time.Time, limit int) (dispatchBacklogDelta, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	search := "updated:>=" + watermark.UTC().Format(time.RFC3339)
	cmd := exec.CommandContext(ctx, "gh", "issue", "list", "--state", "all", "--search", search, "--limit", strconv.Itoa(limit), "--json", "number,title,labels,body,state,updatedAt")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	var rows []struct {
		dispatchtick.Issue
		State     string    `json:"state"`
		UpdatedAt time.Time `json:"updatedAt"`
	}
	if uerr := json.Unmarshal(out, &rows); uerr != nil {
		if err != nil {
			return dispatchBacklogDelta{}, fmt.Errorf("gh issue delta: %w (%s)", err, strings.TrimSpace(string(out)))
		}
		return dispatchBacklogDelta{}, uerr
	}
	d := dispatchBacklogDelta{Watermark: time.Now().UTC()}
	for _, row := range rows {
		if row.UpdatedAt.After(d.Watermark) {
			d.Watermark = row.UpdatedAt
		}
		if strings.EqualFold(row.State, "OPEN") {
			d.Issues = append(d.Issues, row.Issue)
		} else {
			d.Closed = append(d.Closed, row.Number)
		}
	}
	return d, nil
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
