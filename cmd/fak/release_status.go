package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/releasestatus"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const releaseStatusSchema = "fleet-release-status/1"

var (
	releaseStatusSemverRE = regexp.MustCompile(`^v?(\d+)\.(\d+)\.(\d+)$`)
	releaseStatusStableRE = regexp.MustCompile(`^stable/.+`)
)

var releaseStatusRunJSON = releaseStatusRunPythonJSON
var releaseStatusNow = func() time.Time { return time.Now().UTC() }

type releaseStatusOptions struct {
	AsJSON           bool
	LimitCommits     int
	RequireCIGreen   bool
	Force            bool
	SkipCutPlan      bool
	SkipGH           bool
	StableWindowDays int
	CheckReady       bool
}

func runReleaseStatus(stdout, stderr io.Writer, argv []string) int {
	opts, code := parseReleaseStatusFlags(stderr, argv)
	if code != 0 {
		return code
	}
	root := repoRoot()
	status := buildReleaseStatus(root, opts)
	if opts.AsJSON {
		if err := writeIndentedJSON(stdout, status); err != nil {
			fmt.Fprintf(stderr, "fak release status: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, renderReleaseStatus(status))
	}
	if opts.CheckReady {
		rolling := releaseStatusMap(status["rolling"])
		decision := releaseStatusMap(rolling["decision"])
		if releaseStatusString(decision["decision"]) != "release" {
			return 1
		}
	}
	return 0
}

func parseReleaseStatusFlags(stderr io.Writer, argv []string) (releaseStatusOptions, int) {
	fs := flag.NewFlagSet("fak release status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	limitCommits := fs.Int("limit-commits", 50, "commit history budget")
	requireCI := fs.Bool("require-ci-green", false, "block release decisions without confirmed green CI")
	force := fs.Bool("force", false, "bypass only the substantive-commit floor")
	skipCutPlan := fs.Bool("skip-cut-plan", false, "do not run release_cut dry-run")
	skipGH := fs.Bool("skip-gh", false, "do not query GitHub release pages")
	stableWindowDays := fs.Int("stable-window-days", 3, "soak age required before stable candidate is ready")
	checkReady := fs.Bool("check-ready", false, "exit 1 unless the rolling release decision is release")
	if err := fs.Parse(argv); err != nil {
		return releaseStatusOptions{}, 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak release status: unexpected argument %q\n", fs.Arg(0))
		return releaseStatusOptions{}, 2
	}
	if *limitCommits <= 0 {
		fmt.Fprintln(stderr, "fak release status: --limit-commits must be > 0")
		return releaseStatusOptions{}, 2
	}
	if *stableWindowDays < 0 {
		fmt.Fprintln(stderr, "fak release status: --stable-window-days must be >= 0")
		return releaseStatusOptions{}, 2
	}
	return releaseStatusOptions{
		AsJSON:           *asJSON,
		LimitCommits:     *limitCommits,
		RequireCIGreen:   *requireCI,
		Force:            *force,
		SkipCutPlan:      *skipCutPlan,
		SkipGH:           *skipGH,
		StableWindowDays: *stableWindowDays,
		CheckReady:       *checkReady,
	}, 0
}

func buildReleaseStatus(root string, opts releaseStatusOptions) map[string]any {
	decision, contextPayload := releaseStatusDecision(root, opts)
	rollingTags := releaseStatusSemverTags(root, true)
	stable := releaseStatusStableSummary(root, rollingTags, opts.StableWindowDays)
	lastTag := releaseStatusString(contextPayload["last_tag"])
	if lastTag == "" && len(rollingTags) > 0 {
		lastTag = rollingTags[len(rollingTags)-1]
	}
	dirty := releaseStatusDirtySummary(root)
	ciDiag := releaseStatusCIDiagnosis(root)
	branchRegime := releaseStatusBranchRegimeMap(releaseStatusBranchRegime(root, lastTag))
	cutPlan := releaseStatusCutPlan(root, opts)
	status := map[string]any{
		"schema":             releaseStatusSchema,
		"root":               root,
		"development_branch": branchRegime["development_branch"],
		"development_head":   branchRegime["development_head"],
		"release_branch":     branchRegime["release_branch"],
		"release_head":       branchRegime["release_head"],
		"latest_tag":         branchRegime["latest_tag"],
		"promotion_blockers": branchRegime["promotion_blockers"],
		"head": map[string]any{
			"sha":    releaseStatusFirstString(releaseStatusString(contextPayload["head_sha"]), releaseStatusGitOutput(root, "rev-parse", "HEAD")),
			"branch": releaseStatusFirstString(releaseStatusString(contextPayload["current_branch"]), releaseStatusGitOutput(root, "rev-parse", "--abbrev-ref", "HEAD")),
			"dirty":  dirty,
		},
		"rolling": map[string]any{
			"last_tag":                releaseStatusNilIfEmpty(lastTag),
			"latest_any_tag":          releaseStatusNilIfEmpty(releaseStatusString(contextPayload["latest_any_tag"])),
			"commits_since_tag":       releaseStatusCommitsSinceTag(contextPayload, root, lastTag),
			"files_touched_since_tag": releaseStatusFilesSinceTag(contextPayload, root, lastTag),
			"tag_drift":               releaseStatusAnyOrNil(contextPayload["tag_drift"]),
			"ci_on_head":              releaseStatusAnyOrNil(contextPayload["ci_on_head"]),
			"ci_diagnosis":            ciDiag,
			"workflows_parse_ok":      releaseStatusAnyOrNil(contextPayload["workflows_parse_ok"]),
			"decision":                decision,
			"cut_plan":                cutPlan,
		},
		"github_release": releaseStatusGitHubReleaseView(root, lastTag, opts.SkipGH),
		"stable":         stable,
		"cadence":        releaseStatusCadence(root),
		"branch_regime":  branchRegime,
		"shadow_cutover": releaseStatusShadowCutover(branchRegime, cutPlan, releaseStatusPilot(root)),
	}
	action := releaseStatusNextAction(decision, stable, dirty, ciDiag)
	status["next_action"] = action
	for k, v := range releaseStatusLoopStatusFields(action) {
		status[k] = v
	}
	return status
}

func releaseStatusBranchRegime(root, latestTag string) releasestatus.BranchRegime {
	roles, roleErr := branchrole.Load(root)
	devHead := releaseStatusBranchHead(root, roles.DevelopmentBranch)
	releaseHead := releaseStatusBranchHead(root, roles.ReleaseBranch)
	releaseAhead, developmentAhead := releaseStatusBranchAhead(root, releaseHead, devHead)
	roleErrText := ""
	if roleErr != nil {
		roleErrText = roleErr.Error()
	}
	return releasestatus.FoldBranchRegime(releasestatus.BranchRegimeFacts{
		DevelopmentBranch: roles.DevelopmentBranch,
		DevelopmentHead:   devHead,
		ReleaseBranch:     roles.ReleaseBranch,
		ReleaseHead:       releaseHead,
		ReleaseSource:     roles.ReleaseSource,
		PublicFrontDoor:   roles.PublicFrontDoor,
		LatestTag:         latestTag,
		DevelopmentAhead:  developmentAhead,
		ReleaseAhead:      releaseAhead,
		DevelopmentCI:     releaseStatusBranchCI(root, roles.DevelopmentBranch, devHead),
		ReleaseLockHeld:   releaseStatusReleaseLockHeld(root),
		RoleError:         roleErrText,
	})
}

func releaseStatusBranchRegimeMap(r releasestatus.BranchRegime) map[string]any {
	return map[string]any{
		"development_branch":  r.DevelopmentBranch,
		"development_head":    releaseStatusNilIfEmpty(r.DevelopmentHead),
		"release_branch":      r.ReleaseBranch,
		"release_head":        releaseStatusNilIfEmpty(r.ReleaseHead),
		"release_source":      releaseStatusNilIfEmpty(r.ReleaseSource),
		"public_front_door":   releaseStatusNilIfEmpty(r.PublicFrontDoor),
		"latest_tag":          releaseStatusNilIfEmpty(r.LatestTag),
		"development_ahead":   r.DevelopmentAhead,
		"release_ahead":       r.ReleaseAhead,
		"drift":               r.Drift,
		"development_ci":      releaseStatusNilIfEmpty(r.DevelopmentCI),
		"promotion_candidate": releaseStatusNilIfEmpty(r.PromotionCandidate),
		"promotion_blocked":   r.PromotionBlocked,
		"promotion_blockers":  append([]string(nil), r.PromotionBlockers...),
		"release_lock_held":   r.ReleaseLockHeld,
		"next_action":         r.NextAction,
		"role_error":          releaseStatusNilIfEmpty(r.RoleError),
	}
}

// releaseStatusPilot reports the #1703 pilot-cohort lever: the declared pilot
// development branch (inert config) and whether THIS process opted in. It is
// visibility only — the shadow-cutover decision fold stays blocked on the real
// role split and proof bundle.
func releaseStatusPilot(root string) map[string]any {
	roles, err := branchrole.Load(root)
	pilot := map[string]any{
		"declared_branch": releaseStatusNilIfEmpty(roles.PilotDevelopmentBranch),
		"opt_in_env":      branchrole.PilotEnv,
		"opted_in":        branchrole.PilotOptedIn(),
		"active":          roles.PilotActive,
	}
	if err != nil {
		pilot["role_error"] = err.Error()
	}
	return pilot
}

func releaseStatusShadowCutover(branchRegime map[string]any, cutPlan map[string]any, pilot map[string]any) map[string]any {
	dev := releaseStatusString(branchRegime["development_branch"])
	release := releaseStatusString(branchRegime["release_branch"])
	source := releaseStatusString(branchRegime["release_source"])
	frontDoor := releaseStatusString(branchRegime["public_front_door"])
	promotionBlockers := releaseStatusStringSlice(branchRegime["promotion_blockers"])
	var blockers []string
	var proofGaps []string

	if dev == "" || release == "" || source == "" || frontDoor == "" {
		blockers = append(blockers, "BRANCH_ROLE_CONFIG_MISSING")
	}
	if dev != "" && release != "" && dev == release {
		blockers = append(blockers, "BRANCH_ROLES_NOT_SPLIT")
	}
	if source != "" && dev != "" && source != dev {
		blockers = append(blockers, "RELEASE_SOURCE_NOT_DEVELOPMENT_BRANCH")
	}
	if frontDoor != "" && release != "" && frontDoor != release {
		blockers = append(blockers, "PUBLIC_FRONT_DOOR_NOT_RELEASE_BRANCH")
	}
	for _, blocker := range promotionBlockers {
		blockers = append(blockers, "PROMOTION_BLOCKED_"+blocker)
	}

	cutSkipped := releaseStatusBool(cutPlan["skipped"])
	cutOK := releaseStatusBool(cutPlan["ok"])
	if cutSkipped {
		blockers = append(blockers, "RELEASE_PROMOTION_DRY_RUN_SKIPPED")
	} else if len(cutPlan) == 0 || !cutOK {
		blockers = append(blockers, "RELEASE_PROMOTION_DRY_RUN_REFUSED")
	}

	proofGaps = append(proofGaps,
		"DEV_RULESET_PROOF",
		"DEVELOPMENT_CI_ARTIFACT",
		"PILOT_COHORT_WITNESS",
		"FINAL_DECISION_RECORD",
	)
	decision := "ready_for_pilot"
	if len(blockers) > 0 || len(proofGaps) > 0 {
		decision = "hold"
	}
	out := map[string]any{
		"schema":         "fak.branch-regime.shadow-cutover.v1",
		"issue":          "#1703",
		"checklist":      "docs/branch-regime-shadow-cutover.md",
		"decision":       decision,
		"ready":          decision == "ready_for_pilot" && len(proofGaps) == 0,
		"blockers":       blockers,
		"proof_gaps":     proofGaps,
		"proof_commands": []string{"fak release status --json --require-ci-green --limit-commits 50", "fak workflow-audit --write-doc", "go test ./internal/workflowaudit -count=1", "fak release ship --json --source-branch dev --trunk main --base origin/dev"},
	}
	if len(pilot) > 0 {
		out["pilot"] = pilot
	}
	return out
}

func releaseStatusBranchHead(root, branch string) string {
	branch = strings.TrimSpace(branch)
	if branch == "" {
		return ""
	}
	for _, ref := range []string{
		branch,
		"refs/heads/" + branch,
		"origin/" + branch,
		"refs/remotes/origin/" + branch,
	} {
		if sha := releaseStatusGitOutput(root, "rev-parse", "--verify", ref+"^{commit}"); sha != "" {
			return strings.Fields(sha)[0]
		}
	}
	return ""
}

func releaseStatusBranchAhead(root, releaseHead, developmentHead string) (int, int) {
	releaseHead = strings.TrimSpace(releaseHead)
	developmentHead = strings.TrimSpace(developmentHead)
	if releaseHead == "" || developmentHead == "" || releaseHead == developmentHead {
		return 0, 0
	}
	out := releaseStatusGitOutput(root, "rev-list", "--left-right", "--count", releaseHead+"..."+developmentHead)
	fields := strings.Fields(out)
	if len(fields) != 2 {
		return 0, 0
	}
	releaseAhead, _ := strconv.Atoi(fields[0])
	developmentAhead, _ := strconv.Atoi(fields[1])
	return releaseAhead, developmentAhead
}

func releaseStatusBranchCI(root, branch, head string) string {
	if head != "" {
		row := releaseStatusCIRow(root, "run", "list", "--workflow", "ci.yml", "--commit", head, "--limit", "1", "--json", "status,conclusion,headSha,url")
		if len(row) > 0 {
			return releaseStatusFirstString(releaseStatusString(row["conclusion"]), releaseStatusString(row["status"]))
		}
	}
	if branch != "" {
		row := releaseStatusCIRow(root, "run", "list", "--workflow", "ci.yml", "--branch", branch, "--status", "completed", "--limit", "1", "--json", "status,conclusion,headSha,url")
		if len(row) > 0 {
			return releaseStatusFirstString(releaseStatusString(row["conclusion"]), releaseStatusString(row["status"]))
		}
	}
	return "unknown"
}

func releaseStatusCIRow(root string, args ...string) map[string]any {
	run, err := releaseStatusRunExternalJSON(root, 30*time.Second, "gh", args...)
	if err != nil {
		return nil
	}
	return releaseStatusFirstArrayObject(run)
}

func releaseStatusReleaseLockHeld(root string) bool {
	payload, _, err := releaseStatusRunJSON(root, 30*time.Second, "release_lock.py", "status")
	if err != nil {
		return false
	}
	return releaseStatusBool(payload["held"])
}

func releaseStatusDecision(root string, opts releaseStatusOptions) (map[string]any, map[string]any) {
	contextPayload, _, contextErr := releaseStatusRunJSON(root, 180*time.Second, "release_context.py", "--no-previews", "--limit-commits", strconv.Itoa(opts.LimitCommits))
	args := []string{"--json", "--limit-commits", strconv.Itoa(opts.LimitCommits)}
	if opts.RequireCIGreen {
		args = append(args, "--require-ci-green")
	}
	if opts.Force {
		args = append(args, "--force")
	}
	decision, code, decideErr := releaseStatusRunJSON(root, 180*time.Second, "release_decide.py", args...)
	if decideErr != nil || (code != 0 && code != 2) || decision == nil {
		errs := []any{}
		if contextErr != nil {
			errs = append(errs, contextErr.Error())
		}
		if decideErr != nil {
			errs = append(errs, decideErr.Error())
		}
		return map[string]any{"decision": "unknown", "reason": "release_decide failed", "errors": errs}, releaseStatusMapOrEmpty(contextPayload)
	}
	return decision, releaseStatusMapOrEmpty(contextPayload)
}

func releaseStatusCutPlan(root string, opts releaseStatusOptions) map[string]any {
	if opts.SkipCutPlan {
		return map[string]any{"skipped": true}
	}
	args := []string{"--json", "--allow-hold", "--limit-commits", strconv.Itoa(opts.LimitCommits)}
	if opts.RequireCIGreen {
		args = append(args, "--require-ci-green")
	}
	if opts.Force {
		args = append(args, "--force")
	}
	payload, code, err := releaseStatusRunJSON(root, 300*time.Second, "release_cut.py", args...)
	if err != nil || payload == nil {
		detail := ""
		if err != nil {
			detail = err.Error()
		}
		return map[string]any{"ok": false, "error": "release_cut emitted no JSON", "detail": detail}
	}
	payload["exit_code"] = code
	return payload
}

func releaseStatusRunPythonJSON(root string, timeout time.Duration, script string, args ...string) (map[string]any, int, error) {
	python := releasePython()
	scriptPath := filepath.Join(root, "tools", script)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, append([]string{scriptPath}, args...)...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	code := 0
	if err != nil {
		code = 1
		if ee, ok := err.(*exec.ExitError); ok {
			code = ee.ExitCode()
		}
	}
	var doc map[string]any
	if jerr := json.Unmarshal(out, &doc); jerr != nil || doc == nil {
		if err != nil {
			return nil, code, fmt.Errorf("%s: %w (%s)", script, err, strings.TrimSpace(string(out)))
		}
		return nil, code, fmt.Errorf("%s emitted no JSON: %s", script, strings.TrimSpace(string(out)))
	}
	return doc, code, nil
}

func releaseStatusDirtySummary(root string) map[string]any {
	raw := releaseStatusGitOutput(root, "status", "--porcelain=v1", "-z")
	entries := []string{}
	if raw != "" {
		for _, entry := range strings.Split(raw, "\x00") {
			if entry != "" {
				entries = append(entries, entry)
			}
		}
	}
	modified := []string{}
	untracked := []string{}
	for _, entry := range entries {
		if len(entry) < 4 {
			continue
		}
		path := entry[3:]
		if strings.HasPrefix(entry, "?? ") {
			untracked = append(untracked, path)
		} else {
			modified = append(modified, path)
		}
	}
	all := append(append([]string{}, modified...), untracked...)
	relevant := []string{}
	unrelated := []string{}
	for _, path := range all {
		if releaseStatusIsRelevantDirtyPath(path) {
			relevant = append(relevant, path)
		} else {
			unrelated = append(unrelated, path)
		}
	}
	return map[string]any{
		"clean":                  len(entries) == 0,
		"modified_count":         len(modified),
		"untracked_count":        len(untracked),
		"modified":               modified,
		"untracked":              untracked,
		"release_relevant_count": len(relevant),
		"release_relevant":       relevant,
		"unrelated_count":        len(unrelated),
		"unrelated":              unrelated,
	}
}

func releaseStatusIsRelevantDirtyPath(path string) bool {
	normalized := strings.ReplaceAll(path, "\\", "/")
	exact := map[string]bool{
		"VERSION":                    true,
		".claude/project.yaml":       true,
		"tools/safe_ff_sync.py":      true,
		"tools/safe_ff_sync_test.py": true,
	}
	if exact[normalized] {
		return true
	}
	for _, prefix := range []string{
		".claude/skills/release/",
		".claude/skills/stable-release/",
		".github/workflows/",
		"docs/releases/",
		"docs/stable-releases/",
		"tools/release_",
		"tools/stable_release_",
	} {
		if strings.HasPrefix(normalized, prefix) {
			return true
		}
	}
	return false
}

func releaseStatusCIDiagnosis(root string) map[string]any {
	head := releaseStatusGitOutput(root, "rev-parse", "HEAD")
	if head == "" {
		return map[string]any{"status": "unavailable", "reason": "could not resolve HEAD"}
	}
	run, err := releaseStatusRunExternalJSON(root, 30*time.Second, "gh", "run", "list", "--workflow", "ci.yml", "--commit", head, "--limit", "1", "--json", "databaseId,headSha,status,conclusion,displayTitle,url")
	scope := "head"
	row := releaseStatusFirstArrayObject(run)
	if err != nil || len(row) == 0 {
		run, err = releaseStatusRunExternalJSON(root, 30*time.Second, "gh", "run", "list", "--workflow", "ci.yml", "--branch", "main", "--status", "completed", "--limit", "1", "--json", "databaseId,headSha,status,conclusion,displayTitle,url")
		scope = "latest_trunk"
		row = releaseStatusFirstArrayObject(run)
	}
	if err != nil {
		return map[string]any{"status": "unavailable", "reason": err.Error()}
	}
	if len(row) == 0 {
		return map[string]any{"status": "none", "reason": "no ci.yml run found for HEAD or latest main"}
	}
	if releaseStatusString(row["conclusion"]) != "failure" {
		return map[string]any{"status": "not_failed", "scope": scope, "run": row, "reason": fmt.Sprintf("%s ci.yml run conclusion is %s", scope, releaseStatusFirstString(releaseStatusString(row["conclusion"]), releaseStatusString(row["status"])))}
	}
	return map[string]any{"status": "undifferentiated", "scope": scope, "run": row, "kind": "unknown", "action": "inspect_ci", "detail": "ci.yml is red, but no job annotation was available"}
}

func releaseStatusGitHubReleaseView(root, tag string, skip bool) map[string]any {
	if skip {
		return map[string]any{"status": "skipped", "tag": releaseStatusNilIfEmpty(tag)}
	}
	if tag == "" {
		return map[string]any{"status": "none", "tag": nil, "reason": "no rolling tag"}
	}
	doc, err := releaseStatusRunExternalJSONObject(root, 30*time.Second, "gh", "release", "view", tag, "--json", "tagName,url,publishedAt,isDraft,isPrerelease")
	if err == nil {
		doc["status"] = "present"
		return doc
	}
	reason := err.Error()
	lower := strings.ToLower(reason)
	if strings.Contains(lower, "not found") || strings.Contains(lower, "could not resolve") {
		return map[string]any{"status": "missing", "tag": tag, "reason": releaseStatusTail(reason, 300)}
	}
	return map[string]any{"status": "unknown", "tag": tag, "reason": releaseStatusTail(reason, 300)}
}

func releaseStatusCadence(root string) map[string]any {
	rel := filepath.Join(".github", "workflows", "release-cadence.yml")
	path := filepath.Join(root, rel)
	b, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{"present": false, "path": rel}
	}
	text := string(b)
	manualGate := strings.Contains(text, "inputs.dry_run == false") || strings.Contains(text, `inputs.dry_run }}" = "false"`)
	return map[string]any{
		"present":                true,
		"path":                   rel,
		"schedule":               strings.Contains(text, "schedule:"),
		"manual_dispatch":        strings.Contains(text, "workflow_dispatch:"),
		"dry_run_first":          strings.Contains(text, "dry_run:") && manualGate,
		"single_writer":          strings.Contains(text, "group: release-cadence") && strings.Contains(text, "cancel-in-progress: false"),
		"tag_after_green":        strings.Contains(text, "tools/release_tag.py") && strings.Contains(text, "--require-ci") && strings.Contains(text, "--wait-ci"),
		"checked_github_release": strings.Contains(text, "tools/release_publish.py") && strings.Contains(text, "--execute"),
	}
}

func releaseStatusStableSummary(root string, rollingTags []string, windowDays int) map[string]any {
	stable := releaseStatusStableTags(root)
	evidence := releaseStatusStableEvidence(root, stable)
	candidate := releaseStatusStableCandidate(root, rollingTags, stable, windowDays)
	if len(stable) == 0 {
		return map[string]any{
			"latest_stable":  nil,
			"stable_lag":     nil,
			"evidence":       evidence,
			"candidate":      candidate,
			"recommendation": releaseStatusStableRecommendation(0, candidate, true),
		}
	}
	latest := stable[0]
	stableVersion := releaseStatusString(latest["version"])
	stableTuple, ok := releaseStatusSemverTuple(stableVersion)
	if !ok {
		return map[string]any{
			"latest_stable":  latest,
			"stable_lag":     nil,
			"evidence":       evidence,
			"candidate":      candidate,
			"recommendation": "latest stable tag has no readable VERSION; inspect before promoting another stable",
		}
	}
	newer := []string{}
	for _, tag := range rollingTags {
		if releaseStatusSemverGreater(releaseStatusMustSemverTuple(tag), stableTuple) {
			newer = append(newer, tag)
		}
	}
	lag := len(newer)
	return map[string]any{
		"latest_stable":      latest,
		"stable_lag":         lag,
		"newer_rolling_tags": newer,
		"evidence":           evidence,
		"candidate":          candidate,
		"recommendation":     releaseStatusStableRecommendation(lag, candidate, false),
	}
}

func releaseStatusStableTags(root string) []map[string]any {
	out := releaseStatusGitOutput(root, "for-each-ref", "--sort=-creatordate", "--format=%(refname:short)|%(objectname)|%(*objectname)|%(creatordate:iso-strict)", "refs/tags/stable/")
	rows := []map[string]any{}
	for _, line := range strings.Split(out, "\n") {
		parts := strings.SplitN(line, "|", 4)
		if len(parts) != 4 || !releaseStatusStableRE.MatchString(parts[0]) {
			continue
		}
		sha := parts[2]
		if sha == "" {
			sha = parts[1]
		}
		rows = append(rows, map[string]any{
			"tag":        parts[0],
			"sha":        sha,
			"short_sha":  releaseStatusShortSHA(sha),
			"created_at": releaseStatusNilIfEmpty(parts[3]),
			"version":    releaseStatusVersionAtRef(root, sha),
		})
	}
	return rows
}

func releaseStatusStableEvidence(root string, stable []map[string]any) map[string]any {
	rows := []any{}
	failures := []any{}
	for _, tag := range stable {
		tagName := releaseStatusString(tag["tag"])
		codename := tagName
		if i := strings.Index(tagName, "/"); i >= 0 {
			codename = tagName[i+1:]
		}
		evidenceRel := filepath.ToSlash(filepath.Join("docs", "stable-releases", codename+".md"))
		issues := []string{}
		fields := map[string]string{}
		text := releaseStatusGitOutput(root, "show", "HEAD:"+evidenceRel)
		if text == "" {
			issues = append(issues, "missing committed evidence file "+evidenceRel)
		} else {
			fields = releaseStatusFrontmatter(text)
			if len(fields) == 0 {
				issues = append(issues, evidenceRel+" at HEAD has missing or unparseable frontmatter")
			}
		}
		expectedSHA := releaseStatusString(tag["sha"])
		candidateSHA := fields["candidate_sha"]
		if len(fields) > 0 && candidateSHA == "" {
			issues = append(issues, "frontmatter.candidate_sha missing")
		} else if len(fields) > 0 && !releaseStatusSHAMatches(candidateSHA, expectedSHA) {
			issues = append(issues, fmt.Sprintf("frontmatter.candidate_sha=%q does not match %s commit %s", candidateSHA, tagName, releaseStatusShortSHA(expectedSHA)))
		}
		underlying := fields["underlying_version"]
		if len(fields) > 0 && underlying == "" {
			issues = append(issues, "frontmatter.underlying_version missing")
		} else if len(fields) > 0 {
			if _, ok := releaseStatusSemverTuple(underlying); !ok {
				issues = append(issues, fmt.Sprintf("frontmatter.underlying_version=%q is not vX.Y.Z", underlying))
			} else if rollingSHA := releaseStatusTagSHA(root, underlying); rollingSHA == "" {
				issues = append(issues, "underlying rolling tag "+underlying+" does not exist")
			} else if !releaseStatusSHAMatches(rollingSHA, expectedSHA) {
				issues = append(issues, fmt.Sprintf("underlying rolling tag %s points at %s, not %s", underlying, releaseStatusShortSHA(rollingSHA), releaseStatusShortSHA(expectedSHA)))
			}
		}
		if len(fields) > 0 && fields["codename"] != "" && fields["codename"] != codename {
			issues = append(issues, fmt.Sprintf("frontmatter.codename=%q does not match %q", fields["codename"], codename))
		}
		row := map[string]any{"tag": tagName, "evidence_path": evidenceRel, "ok": len(issues) == 0, "issues": issues}
		rows = append(rows, row)
		for _, issue := range issues {
			failures = append(failures, map[string]any{"tag": tagName, "evidence_path": evidenceRel, "detail": issue})
		}
	}
	return map[string]any{"ok": len(failures) == 0, "checked": len(stable), "failures": failures, "rows": rows}
}

func releaseStatusStableCandidate(root string, rollingTags []string, stable []map[string]any, windowDays int) map[string]any {
	if len(rollingTags) == 0 {
		return map[string]any{"ready": false, "state": "no_candidate", "reason": "no reachable rolling vX.Y.Z tag exists", "window_days": windowDays}
	}
	candidateTag := rollingTags[len(rollingTags)-1]
	candidateSHA := releaseStatusTagSHA(root, candidateTag)
	row := map[string]any{"candidate_tag": candidateTag, "candidate_sha": releaseStatusNilIfEmpty(candidateSHA), "window_days": windowDays, "ready": false}
	if candidateSHA == "" {
		row["state"] = "unresolved"
		row["reason"] = "could not resolve " + candidateTag
		return row
	}
	for _, item := range stable {
		if releaseStatusSHAMatches(releaseStatusString(item["sha"]), candidateSHA) {
			row["state"] = "already_promoted"
			row["stable_tag"] = item["tag"]
			row["reason"] = fmt.Sprintf("%s is already promoted as %s", candidateTag, item["tag"])
			return row
		}
	}
	created := releaseStatusTagEpoch(root, candidateTag)
	if created == 0 {
		row["state"] = "unknown_age"
		row["reason"] = "could not read " + candidateTag + " commit time"
		return row
	}
	ageDays := float64(releaseStatusNow().Unix()-created) / 86400.0
	ageDays = float64(int64(ageDays*100+0.5)) / 100
	codename := releaseStatusSuggestStableCodename(root)
	row["age_days"] = ageDays
	row["suggested_codename"] = codename
	row["dry_run_command"] = fmt.Sprintf("python tools/stable_release_promote.py --codename %s --from %s --dry-run --json", codename, candidateTag)
	if ageDays < float64(windowDays) {
		row["state"] = "soaking"
		row["remaining_days"] = float64(int64((float64(windowDays)-ageDays)*100+0.5)) / 100
		row["reason"] = fmt.Sprintf("%s has soaked %gd of %dd", candidateTag, ageDays, windowDays)
		return row
	}
	row["ready"] = true
	row["state"] = "ready"
	row["reason"] = fmt.Sprintf("%s has soaked %gd; stable promotion preflight is due", candidateTag, ageDays)
	return row
}

func releaseStatusStableRecommendation(lag int, candidate map[string]any, noStable bool) string {
	state := releaseStatusString(candidate["state"])
	candidateTag := releaseStatusString(candidate["candidate_tag"])
	switch state {
	case "ready":
		return fmt.Sprintf("promote %s with /stable-release --codename %s", candidateTag, candidate["suggested_codename"])
	case "soaking":
		prefix := fmt.Sprintf("stable lags by %d rolling release(s)", lag)
		if noStable {
			prefix = "no stable tag exists"
		}
		return prefix + "; " + releaseStatusString(candidate["reason"])
	case "already_promoted":
		return "stable channel is current"
	}
	if noStable {
		return "no stable tag exists; promote a soaked rolling tag when one is known good"
	}
	if lag == 0 {
		return "stable channel is current"
	}
	return fmt.Sprintf("stable lags by %d rolling release(s); consider /stable-release after soak", lag)
}

func releaseStatusNextAction(decision, stable, dirty, ciDiag map[string]any) map[string]any {
	blockers := releaseStatusStringSlice(decision["blockers"])
	if releaseStatusString(decision["decision"]) == "release" {
		if !releaseStatusBool(dirty["clean"]) {
			return map[string]any{"kind": "cut_release_hot_tree", "detail": fmt.Sprintf("cut %s with `fak release ship --execute`; it uses a detached origin/main checkout and leaves this checkout's %d modified and %d untracked path(s) untouched", decision["next_version"], releaseStatusInt(dirty["modified_count"]), releaseStatusInt(dirty["untracked_count"]))}
		}
		return map[string]any{"kind": "cut_release", "detail": fmt.Sprintf("cut %s with release_cut, push, then tag after green CI", decision["next_version"])}
	}
	if releaseStatusDirtyRequiresCleanBeforeStatus(dirty) {
		return map[string]any{"kind": "clean_worktree", "detail": fmt.Sprintf("commit, shelve, or remove %d release-relevant dirty path(s) before treating release status as trunk evidence", releaseStatusInt(dirty["release_relevant_count"]))}
	}
	if releaseStatusContains(blockers, "CI_BASE_RED") {
		if releaseStatusString(ciDiag["action"]) == "fix_ci_billing" {
			return map[string]any{"kind": "fix_ci_billing", "detail": releaseStatusString(ciDiag["detail"])}
		}
		return map[string]any{"kind": "fix_ci", "detail": "fix current main ci.yml failure before cutting a release"}
	}
	if releaseStatusContains(blockers, "CI_RETRY_TO_GREEN") {
		return map[string]any{"kind": "pause_auto_release", "detail": "latest green ci.yml run was a retry; set FAK_AUTO_RELEASE=0 or confirm a fresh green run before cutting a release"}
	}
	if releaseStatusContains(blockers, "CI_BASE_NONE") || releaseStatusContains(blockers, "CI_STATE_UNKNOWN") {
		return map[string]any{"kind": "confirm_ci", "detail": "restore or confirm a green main ci.yml signal"}
	}
	if releaseStatusContains(blockers, "WORKFLOW_UNPARSEABLE") {
		return map[string]any{"kind": "fix_workflow", "detail": "repair GitHub workflow YAML before release"}
	}
	if releaseStatusContains(blockers, "VERSION_DRIFT") || releaseStatusContains(blockers, "VERSION_BEHIND_REACHABLE_TAG") {
		return map[string]any{"kind": "fix_version_topology", "detail": "reconcile VERSION and semver tag topology"}
	}
	if len(blockers) > 0 {
		return map[string]any{"kind": "hold", "detail": releaseStatusFirstString(releaseStatusString(decision["reason"]), blockers[0])}
	}
	evidence := releaseStatusMap(stable["evidence"])
	if v, ok := evidence["ok"].(bool); ok && !v {
		detail := "stable tag evidence is missing or inconsistent"
		failures, _ := evidence["failures"].([]any)
		if len(failures) > 0 {
			first := releaseStatusMap(failures[0])
			detail = fmt.Sprintf("repair %s: %s", first["tag"], first["detail"])
		}
		return map[string]any{"kind": "repair_stable_evidence", "detail": detail}
	}
	candidate := releaseStatusMap(stable["candidate"])
	if releaseStatusBool(candidate["ready"]) {
		return map[string]any{"kind": "promote_stable", "detail": fmt.Sprintf("promote %s with /stable-release --codename %s", candidate["candidate_tag"], candidate["suggested_codename"])}
	}
	if releaseStatusInt(stable["stable_lag"]) != 0 {
		return map[string]any{"kind": "consider_stable", "detail": stable["recommendation"]}
	}
	return map[string]any{"kind": "wait", "detail": releaseStatusFirstString(releaseStatusString(decision["reason"]), "nothing release-worthy pending")}
}

func releaseStatusLoopStatusFields(action map[string]any) map[string]any {
	kind := releaseStatusFirstString(releaseStatusString(action["kind"]), "unknown")
	detail := releaseStatusFirstString(releaseStatusString(action["detail"]), kind)
	actionKinds := map[string]bool{
		"clean_worktree":         true,
		"cut_release":            true,
		"fix_ci_billing":         true,
		"fix_ci":                 true,
		"confirm_ci":             true,
		"fix_workflow":           true,
		"fix_version_topology":   true,
		"repair_stable_evidence": true,
		"promote_stable":         true,
		"cut_release_hot_tree":   true,
		"hold":                   true,
		"consider_stable":        true,
	}
	ok := !actionKinds[kind]
	verdict := "OK"
	if !ok {
		verdict = "ACTION"
	}
	return map[string]any{"ok": ok, "verdict": verdict, "detail": detail}
}

func renderReleaseStatus(status map[string]any) string {
	rolling := releaseStatusMap(status["rolling"])
	decision := releaseStatusMap(rolling["decision"])
	stable := releaseStatusMap(status["stable"])
	action := releaseStatusMap(status["next_action"])
	branchRegime := releaseStatusMap(status["branch_regime"])
	shadowCutover := releaseStatusMap(status["shadow_cutover"])
	lines := []string{
		fmt.Sprintf("release-status: %s - %s", strings.ToUpper(releaseStatusFirstString(releaseStatusString(decision["decision"]), "unknown")), releaseStatusString(decision["reason"])),
		fmt.Sprintf("  last tag: %s", releaseStatusFirstString(releaseStatusString(rolling["last_tag"]), "(none)")),
		releaseStatusRenderBranchRegime(branchRegime),
		releaseStatusRenderShadowCutover(shadowCutover),
		fmt.Sprintf("  commits since tag: %d", releaseStatusInt(rolling["commits_since_tag"])),
		fmt.Sprintf("  next action: %s - %s", releaseStatusString(action["kind"]), releaseStatusString(action["detail"])),
	}
	if latest := releaseStatusMap(stable["latest_stable"]); len(latest) > 0 {
		lines = append(lines, fmt.Sprintf("  stable: %s (%s); lag=%v", latest["tag"], latest["version"], stable["stable_lag"]))
	} else {
		lines = append(lines, "  stable: none")
	}
	return strings.Join(lines, "\n")
}

func releaseStatusRenderBranchRegime(branchRegime map[string]any) string {
	if len(branchRegime) == 0 {
		return "  branch regime: unavailable"
	}
	dev := releaseStatusString(branchRegime["development_branch"])
	release := releaseStatusString(branchRegime["release_branch"])
	drift := releaseStatusString(branchRegime["drift"])
	devAhead := releaseStatusInt(branchRegime["development_ahead"])
	releaseAhead := releaseStatusInt(branchRegime["release_ahead"])
	blocked := releaseStatusBool(branchRegime["promotion_blocked"])
	blockers := releaseStatusStringSlice(branchRegime["promotion_blockers"])
	detail := fmt.Sprintf("%s vs %s: %s", dev, release, drift)
	switch drift {
	case "development_ahead":
		detail = fmt.Sprintf("%s is %d commit(s) ahead of %s", dev, devAhead, release)
	case "release_ahead":
		detail = fmt.Sprintf("%s is %d commit(s) ahead of %s", release, releaseAhead, dev)
	case "diverged":
		detail = fmt.Sprintf("%s and %s diverged (%s ahead=%d, %s ahead=%d)", dev, release, dev, devAhead, release, releaseAhead)
	case "no_drift":
		detail = fmt.Sprintf("%s and %s have no drift", dev, release)
	}
	if blocked {
		detail += "; promotion blocked"
		if len(blockers) > 0 {
			detail += ": " + strings.Join(blockers, ", ")
		}
	}
	return "  branch regime: " + detail
}

func releaseStatusRenderShadowCutover(shadowCutover map[string]any) string {
	if len(shadowCutover) == 0 {
		return "  shadow cutover: unavailable"
	}
	decision := releaseStatusFirstString(releaseStatusString(shadowCutover["decision"]), "unknown")
	blockers := releaseStatusStringSlice(shadowCutover["blockers"])
	proofGaps := releaseStatusStringSlice(shadowCutover["proof_gaps"])
	detail := decision
	if len(blockers) > 0 {
		detail += "; blocker: " + blockers[0]
		if len(blockers) > 1 {
			detail += fmt.Sprintf(" (+%d more)", len(blockers)-1)
		}
	} else if len(proofGaps) > 0 {
		detail += "; proof gap: " + proofGaps[0]
		if len(proofGaps) > 1 {
			detail += fmt.Sprintf(" (+%d more)", len(proofGaps)-1)
		}
	}
	pilot := releaseStatusMap(shadowCutover["pilot"])
	if declared := releaseStatusString(pilot["declared_branch"]); declared != "" {
		detail += fmt.Sprintf("; pilot lever: %s (opt-in %s=1", declared, releaseStatusFirstString(releaseStatusString(pilot["opt_in_env"]), "FLEET_BRANCH_PILOT"))
		if releaseStatusBool(pilot["active"]) {
			detail += ", ACTIVE in this process"
		}
		detail += ")"
	}
	return "  shadow cutover: " + detail
}

func releaseStatusSemverTags(root string, merged bool) []string {
	args := []string{"tag", "--sort=v:refname"}
	if merged {
		args = append(args, "--merged", "HEAD")
	}
	out := releaseStatusGitOutput(root, args...)
	tags := []string{}
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if releaseStatusSemverRE.MatchString(line) {
			tags = append(tags, line)
		}
	}
	sort.SliceStable(tags, func(i, j int) bool {
		return releaseStatusSemverLess(releaseStatusMustSemverTuple(tags[i]), releaseStatusMustSemverTuple(tags[j]))
	})
	return tags
}

func releaseStatusRunExternalJSON(root string, timeout time.Duration, name string, args ...string) (any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w (%s)", name, strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	var doc any
	if jerr := json.Unmarshal(out, &doc); jerr != nil {
		return nil, fmt.Errorf("%s %s emitted non-JSON: %s", name, strings.Join(args, " "), strings.TrimSpace(string(out)))
	}
	return doc, nil
}

func releaseStatusRunExternalJSONObject(root string, timeout time.Duration, name string, args ...string) (map[string]any, error) {
	doc, err := releaseStatusRunExternalJSON(root, timeout, name, args...)
	if err != nil {
		return nil, err
	}
	m, ok := doc.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s %s emitted non-object JSON", name, strings.Join(args, " "))
	}
	return m, nil
}

func releaseStatusGitOutput(root string, args ...string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func releaseStatusCommitsSinceTag(contextPayload map[string]any, root, lastTag string) int {
	if arr, ok := contextPayload["commits_since_tag"].([]any); ok {
		return len(arr)
	}
	if lastTag == "" {
		return 0
	}
	out := releaseStatusGitOutput(root, "rev-list", "--count", lastTag+"..HEAD")
	n, _ := strconv.Atoi(strings.TrimSpace(out))
	return n
}

func releaseStatusFilesSinceTag(contextPayload map[string]any, root, lastTag string) int {
	if arr, ok := contextPayload["files_touched_since_tag"].([]any); ok {
		return len(arr)
	}
	if lastTag == "" {
		return 0
	}
	out := releaseStatusGitOutput(root, "diff", "--name-only", lastTag+"..HEAD")
	count := 0
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) != "" {
			count++
		}
	}
	return count
}

func releaseStatusVersionAtRef(root, ref string) any {
	out := releaseStatusGitOutput(root, "show", ref+":VERSION")
	if out == "" {
		return nil
	}
	return strings.TrimSpace(strings.Split(out, "\n")[0])
}

func releaseStatusTagSHA(root, tag string) string {
	return releaseStatusGitOutput(root, "rev-list", "-n1", tag)
}

func releaseStatusTagEpoch(root, tag string) int64 {
	out := releaseStatusGitOutput(root, "log", "-1", "--format=%ct", tag)
	n, _ := strconv.ParseInt(strings.TrimSpace(out), 10, 64)
	return n
}

func releaseStatusSuggestStableCodename(root string) string {
	now := releaseStatusNow()
	prefix := now.Format("2006-01-stable")
	for i := 1; i < 100; i++ {
		codename := prefix
		if i > 1 {
			codename = fmt.Sprintf("%s-%d", prefix, i)
		}
		if !releaseStatusStableTagExists(root, codename) && !releaseStatusCommittedPathExists(root, filepath.ToSlash(filepath.Join("docs", "stable-releases", codename+".md"))) {
			return codename
		}
	}
	return fmt.Sprintf("%s-%d", prefix, now.Unix())
}

func releaseStatusStableTagExists(root, codename string) bool {
	return releaseStatusGitOutput(root, "rev-parse", "--verify", "--quiet", "refs/tags/stable/"+codename) != ""
}

func releaseStatusCommittedPathExists(root, rel string) bool {
	return releaseStatusGitOutput(root, "cat-file", "-e", "HEAD:"+rel) != ""
}

func releaseStatusFrontmatter(text string) map[string]string {
	if !strings.HasPrefix(text, "---\n") {
		return nil
	}
	end := strings.Index(text[4:], "\n---")
	if end < 0 {
		return nil
	}
	body := text[4 : 4+end]
	out := map[string]string{}
	for _, line := range strings.Split(body, "\n") {
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, " ") || !strings.Contains(line, ":") {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		key := strings.TrimSpace(parts[0])
		val := strings.Trim(strings.TrimSpace(parts[1]), `"'`)
		if key != "" {
			out[key] = val
		}
	}
	return out
}

func releaseStatusSemverTuple(s string) ([3]int, bool) {
	m := releaseStatusSemverRE.FindStringSubmatch(strings.TrimSpace(s))
	if m == nil {
		return [3]int{}, false
	}
	var out [3]int
	for i := 0; i < 3; i++ {
		n, err := strconv.Atoi(m[i+1])
		if err != nil {
			return [3]int{}, false
		}
		out[i] = n
	}
	return out, true
}

func releaseStatusMustSemverTuple(s string) [3]int {
	v, _ := releaseStatusSemverTuple(s)
	return v
}

func releaseStatusSemverLess(a, b [3]int) bool {
	for i := 0; i < 3; i++ {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return false
}

func releaseStatusSemverGreater(a, b [3]int) bool { return releaseStatusSemverLess(b, a) }

func releaseStatusSHAMatches(a, b string) bool {
	left := strings.ToLower(strings.TrimSpace(a))
	right := strings.ToLower(strings.TrimSpace(b))
	return left != "" && right != "" && (left == right || strings.HasPrefix(left, right) || strings.HasPrefix(right, left))
}

func releaseStatusDirtyRequiresCleanBeforeStatus(dirty map[string]any) bool {
	if releaseStatusBool(dirty["clean"]) {
		return false
	}
	if _, ok := dirty["release_relevant_count"]; !ok {
		return true
	}
	return releaseStatusInt(dirty["release_relevant_count"]) > 0
}

func releaseStatusFirstArrayObject(v any) map[string]any {
	switch x := v.(type) {
	case []any:
		if len(x) > 0 {
			return releaseStatusMap(x[0])
		}
	case map[string]any:
		if arr, ok := x["value"].([]any); ok && len(arr) > 0 {
			return releaseStatusMap(arr[0])
		}
	}
	return nil
}

func releaseStatusMap(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return nil
}

func releaseStatusMapOrEmpty(v map[string]any) map[string]any {
	if v == nil {
		return map[string]any{}
	}
	return v
}

func releaseStatusString(v any) string {
	if s, ok := v.(string); ok {
		return strings.TrimSpace(s)
	}
	return ""
}

func releaseStatusStringSlice(v any) []string {
	if typed, ok := v.([]string); ok {
		return append([]string(nil), typed...)
	}
	raw, _ := v.([]any)
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		if s := releaseStatusString(item); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func releaseStatusBool(v any) bool {
	b, _ := v.(bool)
	return b
}

func releaseStatusInt(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	case json.Number:
		n, _ := x.Int64()
		return int(n)
	}
	return 0
}

func releaseStatusNilIfEmpty(s string) any {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return s
}

func releaseStatusAnyOrNil(v any) any {
	if v == nil {
		return nil
	}
	return v
}

func releaseStatusFirstString(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func releaseStatusContains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

func releaseStatusShortSHA(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

func releaseStatusTail(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
