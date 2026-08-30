package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/workdelivery"
)

type releaseShipCommandRunner func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string)

var releaseShipRunCommand releaseShipCommandRunner = runReleaseShipCommand
var releaseShipMkdirTemp = os.MkdirTemp

const (
	releaseShipWorktreeAddCommandTimeout = 5 * time.Minute
	releaseShipCutCommandTimeout         = 30 * time.Minute
	releaseShipCommitResolveTimeout      = time.Minute
	releaseShipPRPlanCommandTimeout      = 2 * time.Minute
	releaseShipRemoteProbeCommandTimeout = 2 * time.Minute
	releaseShipGHPRCommandTimeout        = 2 * time.Minute
	releaseShipMergeBaseCommandTimeout   = time.Minute
	releaseShipGitPushCommandTimeout     = 10 * time.Minute
	releaseShipCIPollCommandTimeout      = time.Minute
	releaseShipTagCommandTimeout         = 45 * time.Minute
	releaseShipPublishCommandTimeout     = 10 * time.Minute
	releaseShipLockRenewGrace            = 5 * time.Minute
)

type releaseShipOptions struct {
	execute              bool
	asJSON               bool
	base                 string
	sourceBranch         string
	remote               string
	trunk                string
	workflow             string
	version              string
	limitCommits         int
	ttl                  int
	fetch                bool
	force                bool
	requireCI            bool
	waitCI               bool
	skipCI               bool
	skipDryRun           bool
	ciAppearTimeout      time.Duration
	keepWorktree         bool
	worktreeDir          string
	openPR               bool
	prTitle              string
	promotionBranch      string
	allowDirectPromotion bool
	pushRetries          int
	readinessFile        string
}

type releaseShipResult struct {
	OK                     bool                 `json:"ok"`
	DryRun                 bool                 `json:"dry_run"`
	Root                   string               `json:"root"`
	Base                   string               `json:"base"`
	BaseSHA                string               `json:"base_sha,omitempty"`
	SourceBranch           string               `json:"source_branch,omitempty"`
	SourceSHA              string               `json:"source_sha,omitempty"`
	SourceCI               map[string]any       `json:"source_ci,omitempty"`
	TargetBranch           string               `json:"target_branch,omitempty"`
	TargetSHA              string               `json:"target_sha,omitempty"`
	TargetAncestry         map[string]any       `json:"target_ancestry,omitempty"`
	TargetBranchCheck      map[string]any       `json:"target_branch_check,omitempty"`
	TargetBranchFinalCheck map[string]any       `json:"target_branch_final_check,omitempty"`
	Worktree               string               `json:"worktree,omitempty"`
	LockRoot               string               `json:"lock_root,omitempty"`
	ReleaseOwner           string               `json:"release_owner,omitempty"`
	Version                string               `json:"version,omitempty"`
	Tag                    string               `json:"tag,omitempty"`
	CommitSHA              string               `json:"commit_sha,omitempty"`
	Remote                 string               `json:"remote,omitempty"`
	Trunk                  string               `json:"trunk,omitempty"`
	PromotionBranch        string               `json:"promotion_branch,omitempty"`
	Cut                    map[string]any       `json:"cut,omitempty"`
	TagResult              map[string]any       `json:"tag_result,omitempty"`
	Publish                map[string]any       `json:"publish,omitempty"`
	ReleaseLock            map[string]any       `json:"release_lock,omitempty"`
	ReleaseLockRenewals    []map[string]any     `json:"release_lock_renewals,omitempty"`
	ReleaseLockRelease     map[string]any       `json:"release_lock_release,omitempty"`
	PushRetries            []map[string]any     `json:"push_retries,omitempty"`
	RemoteBranch           map[string]string    `json:"remote_branch,omitempty"`
	RemoteBranchPush       map[string]any       `json:"remote_branch_push,omitempty"`
	RemoteBranchFinalCheck map[string]any       `json:"remote_branch_final_check,omitempty"`
	Cleanup                map[string]any       `json:"cleanup,omitempty"`
	PromotionBranchPush    map[string]any       `json:"promotion_branch_push,omitempty"`
	PullRequest            map[string]any       `json:"pull_request,omitempty"`
	Warnings               []string             `json:"warnings,omitempty"`
	Errors                 []string             `json:"errors,omitempty"`
	CommandTail            map[string]string    `json:"command_tail,omitempty"`
	ExecutedCommands       []releaseShipCommand `json:"executed_commands,omitempty"`
}

type releaseShipCommand struct {
	CWD  string   `json:"cwd"`
	Name string   `json:"name"`
	Args []string `json:"args"`
}

func runReleaseShip(stdout, stderr io.Writer, argv []string) int {
	opts, err := parseReleaseShipOptions(stderr, argv)
	if err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		fmt.Fprintf(stderr, "fak release ship: %v\n", err)
		return 2
	}
	if opts.execute {
		if opts.readinessFile == "" {
			fmt.Fprintln(stderr, "fak release ship: --execute requires --readiness with explicit witnessed release-ready evidence")
			return 2
		}
		unit, receipt, readErr := readReleaseReadiness(opts.readinessFile)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak release ship: release readiness: %v\n", readErr)
			return 1
		}
		if _, readyErr := workdelivery.RequireReleaseReady(unit, &receipt); readyErr != nil {
			fmt.Fprintf(stderr, "fak release ship: %v\n", readyErr)
			return 1
		}
	}
	result := executeReleaseShip(opts)
	if opts.asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak release ship: encode json: %v\n", err)
			return 1
		}
	} else {
		renderReleaseShip(stdout, stderr, result)
	}
	if result.OK {
		return 0
	}
	return 1
}

func parseReleaseShipOptions(stderr io.Writer, argv []string) (releaseShipOptions, error) {
	fs := flag.NewFlagSet("fak release ship", flag.ContinueOnError)
	fs.SetOutput(stderr)
	opts := defaultReleaseShipOptions(repoRoot())
	base := fs.String("base", "", "ref to detach from (default: <remote>/<source-branch>)")
	fs.BoolVar(&opts.execute, "execute", false, "mutate: cut, push, tag, and publish; default is a detached dry-run plan")
	fs.BoolVar(&opts.asJSON, "json", false, "emit JSON")
	fs.StringVar(&opts.sourceBranch, "source-branch", opts.sourceBranch, "branch role to cut from")
	fs.StringVar(&opts.remote, "remote", opts.remote, "remote to fetch/push")
	fs.StringVar(&opts.trunk, "trunk", opts.trunk, "branch to push HEAD to")
	fs.StringVar(&opts.workflow, "workflow", opts.workflow, "GitHub Actions workflow checked before tagging")
	fs.StringVar(&opts.version, "version", "", "target X.Y.Z; default from release_decide")
	fs.IntVar(&opts.limitCommits, "limit-commits", opts.limitCommits, "commit window passed to release helpers")
	fs.IntVar(&opts.ttl, "ttl", opts.ttl, "minimum release lock TTL in seconds; long release phases extend it to their command budget")
	fs.BoolVar(&opts.fetch, "fetch", opts.fetch, "fetch remote source/target branches before creating the detached worktree")
	fs.BoolVar(&opts.force, "force", false, "pass --force to release_cut (only bypasses the substantive floor)")
	fs.BoolVar(&opts.requireCI, "require-ci", opts.requireCI, "require green CI before tagging")
	fs.BoolVar(&opts.waitCI, "wait-ci", opts.waitCI, "watch a pending CI run before tagging")
	fs.BoolVar(&opts.skipCI, "skip-ci", false, "skip CI folding in release_tag; intended for emergency/operator use")
	fs.BoolVar(&opts.skipDryRun, "skip-dry-run", opts.skipDryRun, "emergency escape hatch: skip the release-substrate dry-run witness at the cut (default runs it; it is tag-based and passes on a real bump). The tag step always reuses the cut's same-commit verdict.")
	fs.StringVar(&opts.readinessFile, "readiness", opts.readinessFile, "work-delivery JSON containing ready state and its explicit witnessed release-readiness receipt (required with --execute)")
	fs.DurationVar(&opts.ciAppearTimeout, "ci-appear-timeout", opts.ciAppearTimeout, "how long to wait for a just-pushed CI run to appear before release_tag folds it")
	fs.BoolVar(&opts.keepWorktree, "keep-worktree", false, "leave the detached worktree on disk")
	fs.StringVar(&opts.worktreeDir, "worktree-dir", "", "use this detached worktree path instead of a temp dir")
	fs.BoolVar(&opts.openPR, "open-pr", false, "open a source->trunk promotion PR via `gh pr create` with a prplan-folded body, instead of pushing straight to trunk; requires distinct --source-branch/--trunk")
	fs.StringVar(&opts.prTitle, "pr-title", "", "override the generated --open-pr title")
	fs.StringVar(&opts.promotionBranch, "promotion-branch", "", "branch to push the exact release-cut commit to before opening --open-pr; default fak/release/<tag>")
	fs.BoolVar(&opts.allowDirectPromotion, "allow-direct-promotion", false, "allow --execute to push a distinct --source-branch directly to --trunk; default requires --open-pr for hot-tree safety")
	fs.IntVar(&opts.pushRetries, "push-retries", opts.pushRetries, "same-trunk hot path: re-cut onto the advanced trunk tip and retry the fast-forward push up to N times when a peer pushes during the cut (never force-pushes)")
	if err := fs.Parse(argv); err != nil {
		return opts, err
	}
	opts.promotionBranch = strings.TrimPrefix(strings.TrimSpace(opts.promotionBranch), "refs/heads/")
	if strings.TrimSpace(*base) != "" {
		opts.base = strings.TrimSpace(*base)
	} else {
		opts.base = opts.remote + "/" + opts.sourceBranch
	}
	if fs.NArg() != 0 {
		return opts, fmt.Errorf("unexpected argument %q", fs.Arg(0))
	}
	if opts.remote == "" || opts.trunk == "" || opts.base == "" || opts.sourceBranch == "" {
		return opts, fmt.Errorf("--remote, --trunk, --source-branch, and --base are required")
	}
	if opts.limitCommits <= 0 {
		return opts, fmt.Errorf("--limit-commits must be positive")
	}
	if opts.ttl <= 0 {
		return opts, fmt.Errorf("--ttl must be positive")
	}
	if opts.pushRetries < 0 {
		return opts, fmt.Errorf("--push-retries must not be negative")
	}
	if opts.openPR && opts.sourceBranch == opts.trunk {
		return opts, fmt.Errorf("--open-pr requires distinct --source-branch and --trunk (today's branch-role regime has both = %q; see docs/branch-regime-shadow-cutover.md)", opts.trunk)
	}
	if opts.openPR && opts.allowDirectPromotion {
		return opts, fmt.Errorf("--allow-direct-promotion is incompatible with --open-pr")
	}
	if releaseShipDirectPromotionRequiresOpenPR(opts) {
		return opts, fmt.Errorf("--execute with distinct --source-branch %q and --trunk %q must use --open-pr; pass --allow-direct-promotion only for an operator-approved direct hot-tree promotion", opts.sourceBranch, opts.trunk)
	}
	if opts.openPR && opts.promotionBranch == opts.trunk {
		return opts, fmt.Errorf("--promotion-branch must not equal --trunk %q", opts.trunk)
	}
	if opts.openPR && opts.promotionBranch != "" && opts.promotionBranch == opts.sourceBranch {
		return opts, fmt.Errorf("--promotion-branch must not equal live --source-branch %q", opts.sourceBranch)
	}
	if opts.openPR && opts.promotionBranch != "" {
		if reason := releaseShipInvalidPromotionBranchReason(opts.promotionBranch); reason != "" {
			return opts, fmt.Errorf("invalid --promotion-branch %q: %s", opts.promotionBranch, reason)
		}
	}
	return opts, nil
}

func releaseShipFetchBranches(opts releaseShipOptions) []string {
	seen := map[string]bool{}
	var out []string
	for _, branch := range []string{opts.sourceBranch, opts.trunk} {
		branch = strings.TrimSpace(branch)
		if branch == "" || seen[branch] {
			continue
		}
		seen[branch] = true
		out = append(out, branch)
	}
	return out
}

func defaultReleaseShipOptions(root string) releaseShipOptions {
	roles, err := branchrole.Load(root)
	if err != nil {
		roles = branchrole.Defaults()
	}
	opts := releaseShipOptions{
		sourceBranch: roles.ReleaseSource,
		remote:       "origin",
		trunk:        roles.ReleaseBranch,
		workflow:     "ci.yml",
		limitCommits: 50,
		ttl:          1800,
		fetch:        true,
		requireCI:    true,
		waitCI:       true,
		// skipDryRun defaults false: the cut runs the release-substrate dry-run
		// witness, which now reads the latest existing tag (not the just-bumped
		// VERSION) and so passes on a real version bump. The tag step reuses that
		// same-commit verdict (see runReleaseShipTag) rather than re-running it.
		ciAppearTimeout: 2 * time.Minute,
		pushRetries:     2,
	}
	return opts
}

func executeReleaseShip(opts releaseShipOptions) (result releaseShipResult) {
	root := repoRoot()
	result = releaseShipResult{
		DryRun:       !opts.execute,
		Root:         root,
		Base:         opts.base,
		SourceBranch: opts.sourceBranch,
		Remote:       opts.remote,
		Trunk:        opts.trunk,
		TargetBranch: opts.trunk,
		LockRoot:     root,
		CommandTail:  map[string]string{},
		RemoteBranch: map[string]string{},
	}
	if releaseShipDirectPromotionRequiresOpenPR(opts) {
		result.fail("direct_promotion_requires_open_pr", fmt.Sprintf("--execute with distinct --source-branch %q and --trunk %q must use --open-pr or --allow-direct-promotion", opts.sourceBranch, opts.trunk))
		return finishReleaseShip(result)
	}
	owner := releaseShipOwner()
	result.ReleaseOwner = owner
	env := releaseShipEnv(root, owner)
	lockAcquired := false
	defer func() {
		if lockAcquired {
			result.ReleaseLockRelease = runReleaseShipLockRelease(&result, root, env)
			if ok, _ := result.ReleaseLockRelease["ok"].(bool); !ok {
				result.Warnings = append(result.Warnings, "release lock release failed: "+jsonTail(result.ReleaseLockRelease))
			}
		}
	}()

	if opts.fetch {
		for _, branch := range releaseShipFetchBranches(opts) {
			refspec := fmt.Sprintf("refs/heads/%s:refs/remotes/%s/%s", branch, opts.remote, branch)
			if code, out := releaseShipCmd(&result, root, "git", []string{"fetch", opts.remote, refspec}, nil, 5*time.Minute); code != 0 {
				result.fail("fetch_failed", out)
				return finishReleaseShip(result)
			}
		}
	}
	code, out := releaseShipCmd(&result, root, "git", []string{"rev-parse", "--verify", opts.base + "^{commit}"}, nil, time.Minute)
	if code != 0 {
		result.fail("base_unresolvable", out)
		return finishReleaseShip(result)
	}
	result.BaseSHA = strings.TrimSpace(out)
	result.SourceSHA = result.BaseSHA
	if opts.requireCI && !opts.skipCI {
		result.SourceCI = releaseShipSourceCI(&result, root, opts, result.SourceSHA)
		if !releaseShipPayloadOK(result.SourceCI) && opts.execute {
			result.fail("source_ci_unconfirmed", jsonTail(result.SourceCI))
			return finishReleaseShip(result)
		}
	}
	if releaseShipNeedsTargetAncestry(opts) {
		result.TargetAncestry = releaseShipTargetAncestry(&result, root, opts, result.SourceSHA)
		result.TargetSHA = stringFromAny(result.TargetAncestry["target_sha"])
		if !releaseShipPayloadOK(result.TargetAncestry) {
			result.fail("target_non_fast_forward", jsonTail(result.TargetAncestry))
			return finishReleaseShip(result)
		}
	}

	if opts.execute {
		lock := runReleaseShipLockAcquire(&result, root, env, opts)
		result.ReleaseLock = lock
		if ok, _ := lock["ok"].(bool); !ok {
			result.fail("release_lock_refused", jsonTail(lock))
			return finishReleaseShip(result)
		}
		lockAcquired = true
	}

	wt, err := releaseShipWorktreeDir(root, opts)
	if err != nil {
		result.fail("worktree_dir_failed", err.Error())
		return finishReleaseShip(result)
	}
	result.Worktree = wt
	worktreeAdded := false

	code, out = releaseShipCmd(&result, root, "git", []string{"worktree", "add", "--detach", wt, result.BaseSHA}, nil, releaseShipWorktreeAddCommandTimeout)
	if code != 0 {
		if opts.worktreeDir == "" {
			_ = os.Remove(wt)
		}
		result.fail("worktree_add_failed", out)
		return finishReleaseShip(result)
	}
	worktreeAdded = true

	cut := runReleaseShipCut(&result, wt, releaseShipPromotionEnv(env, result), opts)
	result.Cut = cut
	if ok, _ := cut["ok"].(bool); !ok {
		if existing := recoverReleaseShipExistingCut(&result, wt, opts, cut); existing != nil {
			cut = existing
			result.Cut = cut
		} else {
			result.fail("release_cut_refused", jsonTail(cut))
			return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
		}
	}
	result.Version = stringFromAny(cut["version"])
	result.Tag = stringFromAny(cut["tag"])
	result.CommitSHA = stringFromAny(cut["commit_sha"])
	if opts.openPR {
		result.PromotionBranch = releaseShipPromotionBranch(opts, result)
	}
	if opts.openPR && result.CommitSHA != "" {
		result.PullRequest = buildReleaseShipPRPreview(&result, wt, opts)
	}
	if !opts.execute {
		result.OK = true
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	if result.CommitSHA == "" {
		code, out = releaseShipCmd(&result, wt, "git", []string{"rev-parse", "HEAD"}, nil, releaseShipCommitResolveTimeout)
		if code != 0 {
			result.fail("release_commit_unresolvable", out)
			return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
		}
		result.CommitSHA = strings.TrimSpace(out)
	}
	if result.Version == "" || result.Tag == "" || result.CommitSHA == "" {
		result.fail("release_cut_missing_outputs", jsonTail(cut))
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}

	if opts.openPR {
		result.PullRequest = runReleaseShipOpenPRExecute(&result, wt, env, opts, result.PullRequest)
		if ok, _ := result.PullRequest["ok"].(bool); !ok {
			result.fail("open_pr_failed", jsonTail(result.PullRequest))
			return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
		}
		result.OK = true
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}

	if !executeReleaseShipTrunkPush(&result, root, wt, env, opts) {
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	remoteSHA := verifyReleaseShipRemote(&result, wt, opts, result.CommitSHA)
	if remoteSHA == "" {
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	if opts.requireCI && !opts.skipCI && opts.ciAppearTimeout > 0 {
		waitReleaseShipCIAppears(&result, wt, opts, result.CommitSHA)
	}
	finalRemote := releaseShipRemoteBranchCheck(&result, wt, opts, result.CommitSHA, "remote_branch_changed_before_tag")
	result.RemoteBranchFinalCheck = finalRemote
	if !releaseStatusBool(finalRemote["ok"]) {
		result.fail("remote_branch_changed_before_tag", jsonTail(finalRemote))
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	if !renewReleaseShipLockForPhase(&result, root, env, opts, "before_tag", releaseShipTagCommandTimeout) {
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}

	tag := runReleaseShipTag(&result, wt, env, opts)
	result.TagResult = tag
	if ok, _ := tag["ok"].(bool); !ok {
		result.fail("release_tag_refused", jsonTail(tag))
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	if !renewReleaseShipLockForPhase(&result, root, env, opts, "before_publish", releaseShipPublishCommandTimeout) {
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	publish := runReleaseShipPublish(&result, wt, env, result.Version)
	result.Publish = publish
	if ok, _ := publish["ok"].(bool); !ok {
		result.fail("release_publish_refused", jsonTail(publish))
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	result.OK = true
	return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
}

func runReleaseShipCut(result *releaseShipResult, wt string, env []string, opts releaseShipOptions) map[string]any {
	args := []string{releaseShipScript(wt, "release_cut.py"), "--json", "--allow-stale-upstream", "--limit-commits", strconv.Itoa(opts.limitCommits)}
	if opts.execute {
		args = append(args, "--execute", "--lock-already-held")
	} else {
		args = append(args, "--allow-hold")
	}
	if opts.version != "" {
		args = append(args, "--version", opts.version)
	}
	if opts.force {
		args = append(args, "--force")
	}
	if opts.requireCI {
		args = append(args, "--require-ci-green")
	}
	if opts.skipDryRun {
		args = append(args, "--skip-dry-run")
	}
	return releaseShipJSONCommand(result, wt, releaseShipPython(), args, env, releaseShipCutCommandTimeout, "cut")
}

func recoverReleaseShipExistingCut(result *releaseShipResult, wt string, opts releaseShipOptions, cut map[string]any) map[string]any {
	if !releaseShipExistingCutRecoveryAllowed(opts, cut) {
		return nil
	}
	code, out := releaseShipCmd(result, wt, "git", []string{"show", "HEAD:VERSION"}, nil, time.Minute)
	if code != 0 {
		return nil
	}
	version := strings.TrimSpace(out)
	versionTuple, ok := releaseStatusSemverTuple(version)
	if !ok {
		return nil
	}
	if opts.version != "" && strings.TrimPrefix(strings.TrimSpace(opts.version), "v") != version {
		return nil
	}
	tag := "v" + version
	note := "docs/releases/" + tag + ".md"
	code, out = releaseShipCmd(result, wt, "git", []string{"cat-file", "-e", "HEAD:" + note}, nil, time.Minute)
	if code != 0 {
		return nil
	}
	code, tagOut := releaseShipCmd(result, wt, "git", []string{"rev-list", "-n1", tag}, nil, time.Minute)
	if code == 0 {
		tagSHA := strings.TrimSpace(tagOut)
		if !sameSHA(tagSHA, result.BaseSHA) {
			return nil
		}
		return map[string]any{
			"ok":           true,
			"existing_cut": true,
			"reason":       "release_tag_already_points_at_head",
			"version":      version,
			"tag":          tag,
			"commit_sha":   result.BaseSHA,
			"notes_file":   note,
			"tag_sha":      tagSHA,
			"recovered_from": map[string]any{
				"aborted": stringFromAny(cut["aborted"]),
				"detail":  stringFromAny(cut["detail"]),
			},
		}
	}
	code, tagsOut := releaseShipCmd(result, wt, "git", []string{"tag", "--sort=-v:refname"}, nil, time.Minute)
	if code != 0 {
		return nil
	}
	latest := ""
	for _, line := range strings.Split(tagsOut, "\n") {
		candidate := strings.TrimSpace(line)
		if _, ok := releaseStatusSemverTuple(candidate); ok {
			latest = candidate
			break
		}
	}
	if latest == "" {
		return map[string]any{
			"ok":           true,
			"existing_cut": true,
			"reason":       "version_ahead_of_latest_tag",
			"version":      version,
			"tag":          tag,
			"commit_sha":   result.BaseSHA,
			"notes_file":   note,
			"latest_tag":   "",
			"recovered_from": map[string]any{
				"aborted": stringFromAny(cut["aborted"]),
				"detail":  stringFromAny(cut["detail"]),
			},
		}
	}
	latestTuple, ok := releaseStatusSemverTuple(latest)
	if !ok || !releaseStatusSemverGreater(versionTuple, latestTuple) {
		return nil
	}
	return map[string]any{
		"ok":           true,
		"existing_cut": true,
		"reason":       "version_ahead_of_latest_tag",
		"version":      version,
		"tag":          tag,
		"commit_sha":   result.BaseSHA,
		"notes_file":   note,
		"latest_tag":   latest,
		"recovered_from": map[string]any{
			"aborted": stringFromAny(cut["aborted"]),
			"detail":  stringFromAny(cut["detail"]),
		},
	}
}

func releaseShipExistingCutRecoveryAllowed(opts releaseShipOptions, cut map[string]any) bool {
	if !opts.execute || opts.openPR || strings.TrimSpace(opts.sourceBranch) == "" || opts.sourceBranch != opts.trunk {
		return false
	}
	if stringFromAny(cut["aborted"]) != "git commit failed" {
		return false
	}
	detail := strings.ToLower(stringFromAny(cut["detail"]))
	return strings.Contains(detail, "nothing to commit") ||
		strings.Contains(detail, "working tree clean") ||
		strings.Contains(detail, "no changes added to commit")
}

func runReleaseShipTag(result *releaseShipResult, wt string, env []string, opts releaseShipOptions) map[string]any {
	args := []string{
		releaseShipScript(wt, "release_tag.py"),
		"--version", result.Version,
		"--ref", result.CommitSHA,
		"--workflow", opts.workflow,
		"--remote", opts.remote,
		"--trunk", "",
		"--execute",
		"--push",
		"--json",
	}
	if opts.skipCI {
		args = append(args, "--skip-ci")
	}
	if opts.requireCI {
		args = append(args, "--require-ci")
	}
	if opts.waitCI {
		args = append(args, "--wait-ci")
	}
	// Always skip the tag's own dry-run witness. --ref is result.CommitSHA — the exact
	// commit release_cut just created and we pushed to trunk — and the cut step already
	// ran release_dry_run on it. Re-running the full substrate suite here would re-clone
	// and re-verify byte-identical tree content. The cut is the single witness in the
	// ship chain; --skip-dry-run on ship only governs whether the cut runs it.
	args = append(args, "--skip-dry-run")
	if opts.execute {
		args = append(args, "--lock-already-held")
	}
	return releaseShipJSONCommand(result, wt, releaseShipPython(), args, env, releaseShipTagCommandTimeout, "tag")
}

func runReleaseShipPublish(result *releaseShipResult, wt string, env []string, version string) map[string]any {
	args := []string{releaseShipScript(wt, "release_publish.py"), "--version", version, "--execute", "--json"}
	return releaseShipJSONCommand(result, wt, releaseShipPython(), args, env, releaseShipPublishCommandTimeout, "publish")
}

// buildReleaseShipPRPreview folds the promotion range (old trunk tip ..
// cut commit) into the same PR-unit shape fak release prplan renders, so an
// --open-pr caller can see the exact title/body it would post before any
// push or gh call happens -- in a dry run this is the whole result.
func buildReleaseShipPRPreview(result *releaseShipResult, wt string, opts releaseShipOptions) map[string]any {
	baseSHA := result.TargetSHA
	headSHA := result.CommitSHA
	if baseSHA == "" || headSHA == "" || sameSHA(baseSHA, headSHA) {
		return map[string]any{"ok": false, "reason": "empty_promotion_range", "base_sha": baseSHA, "head_sha": headSHA}
	}
	headBranch := result.PromotionBranch
	if headBranch == "" {
		headBranch = opts.sourceBranch
	}
	code, out := releaseShipCmd(result, wt, "git", []string{
		"log", "--no-merges", "--name-only", "--format=%x1e%H%x1f%s%x1f%b%x1f", baseSHA + ".." + headSHA,
	}, nil, releaseShipPRPlanCommandTimeout)
	if code != 0 {
		return map[string]any{"ok": false, "reason": "promotion_log_failed", "tail": tail(out)}
	}
	commits := parsePRPlanLog(out)
	units, unstamped := foldPRPlanUnits(commits)
	checkOK := len(unstamped) == 0
	plan := map[string]any{
		"schema":          releasePRPlanSchema,
		"base":            opts.trunk,
		"base_sha":        baseSHA,
		"head":            headBranch,
		"head_sha":        headSHA,
		"range":           baseSHA + ".." + headSHA,
		"commit_count":    len(commits),
		"unit_count":      len(units),
		"unstamped_count": len(unstamped),
		"units":           units,
		"unstamped":       unstamped,
		"check_ok":        checkOK,
	}
	title := strings.TrimSpace(opts.prTitle)
	if title == "" {
		title = fmt.Sprintf("Promote %s -> %s: %d commit(s), %d lane unit(s) (%s)", opts.sourceBranch, opts.trunk, len(commits), len(units), result.Tag)
	}
	return map[string]any{
		"ok":              true,
		"base":            opts.trunk,
		"head":            headBranch,
		"source_branch":   opts.sourceBranch,
		"source_sha":      result.SourceSHA,
		"title":           title,
		"body":            renderPRPlanMarkdown(plan, 20),
		"commit_count":    len(commits),
		"unit_count":      len(units),
		"unstamped_count": len(unstamped),
		"check_ok":        checkOK,
	}
}

// runReleaseShipOpenPRExecute pushes the cut commit onto a release-owned
// promotion branch (so the live source can keep moving after the cut) and
// opens the promotion-branch->trunk PR via `gh pr create`
// with the prplan-folded preview body -- the "PRs managed in advance" a human
// operator reviews and merges through the native GitHub UI. Tag/publish are
// deliberately not run here: they belong to a later `fak release ship` run
// against the merged trunk SHA.
func runReleaseShipOpenPRExecute(result *releaseShipResult, wt string, env []string, opts releaseShipOptions, preview map[string]any) map[string]any {
	if ok, _ := preview["ok"].(bool); !ok {
		return map[string]any{"ok": false, "reason": "no_promotion_range", "preview": preview}
	}
	if checkOK, _ := preview["check_ok"].(bool); !checkOK {
		return map[string]any{"ok": false, "reason": "unstamped_commits_in_range", "preview": preview}
	}
	target := releaseShipRemoteTargetCheck(result, wt, opts)
	result.TargetBranchCheck = target
	if !releaseStatusBool(target["ok"]) {
		return target
	}
	headBranch := result.PromotionBranch
	if headBranch == "" {
		headBranch = releaseShipPromotionBranch(opts, *result)
		result.PromotionBranch = headBranch
	}
	pushed := releaseShipPushPromotionBranch(result, wt, env, opts, headBranch)
	result.PromotionBranchPush = pushed
	if !releaseStatusBool(pushed["ok"]) {
		return pushed
	}
	finalTarget := releaseShipRemoteTargetCheck(result, wt, opts)
	result.TargetBranchFinalCheck = finalTarget
	if !releaseStatusBool(finalTarget["ok"]) {
		if reason := stringFromAny(finalTarget["reason"]); reason != "" {
			finalTarget["base_reason"] = reason
		}
		finalTarget["reason"] = "target_branch_changed_after_promotion_push"
		return finalTarget
	}
	body, _ := preview["body"].(string)
	bodyFile, err := os.CreateTemp("", "fak-release-pr-*.md")
	if err != nil {
		return map[string]any{"ok": false, "reason": "body_tempfile_failed", "detail": err.Error()}
	}
	defer os.Remove(bodyFile.Name())
	if _, err := bodyFile.WriteString(body); err != nil {
		bodyFile.Close()
		return map[string]any{"ok": false, "reason": "body_write_failed", "detail": err.Error()}
	}
	bodyFile.Close()
	title, _ := preview["title"].(string)

	existing := releaseShipFindOpenPR(result, wt, env, opts, headBranch)
	if ok, _ := existing["ok"].(bool); !ok {
		return existing
	}
	if found, _ := existing["found"].(bool); found {
		target := releaseShipExistingPRTarget(existing)
		if target == "" {
			return map[string]any{"ok": false, "reason": "existing_pr_missing_identifier", "existing": existing}
		}
		args := []string{"pr", "edit", target, "--title", title, "--body-file", bodyFile.Name()}
		code, out := releaseShipCmd(result, wt, "gh", args, env, releaseShipGHPRCommandTimeout)
		if code != 0 {
			return map[string]any{"ok": false, "reason": "gh_pr_edit_failed", "existing": existing, "tail": tail(out)}
		}
		return map[string]any{
			"ok":             true,
			"existing":       true,
			"updated":        true,
			"url":            stringFromAny(existing["url"]),
			"number":         existing["number"],
			"base":           opts.trunk,
			"head":           headBranch,
			"source_branch":  opts.sourceBranch,
			"source_sha":     result.SourceSHA,
			"title":          title,
			"commit_count":   preview["commit_count"],
			"unit_count":     preview["unit_count"],
			"target_check":   finalTarget,
			"promotion_push": pushed,
		}
	}

	args := []string{"pr", "create", "--base", opts.trunk, "--head", headBranch, "--title", title, "--body-file", bodyFile.Name()}
	code, out := releaseShipCmd(result, wt, "gh", args, env, releaseShipGHPRCommandTimeout)
	if code != 0 {
		return map[string]any{"ok": false, "reason": "gh_pr_create_failed", "tail": tail(out)}
	}
	return map[string]any{
		"ok":             true,
		"created":        true,
		"existing":       false,
		"url":            releaseShipLastNonEmptyLine(out),
		"base":           opts.trunk,
		"head":           headBranch,
		"source_branch":  opts.sourceBranch,
		"source_sha":     result.SourceSHA,
		"title":          title,
		"commit_count":   preview["commit_count"],
		"unit_count":     preview["unit_count"],
		"target_check":   finalTarget,
		"promotion_push": pushed,
	}
}

func releaseShipPromotionBranch(opts releaseShipOptions, result releaseShipResult) string {
	if branch := strings.TrimSpace(opts.promotionBranch); branch != "" {
		return branch
	}
	name := strings.TrimSpace(result.Tag)
	if name == "" {
		name = strings.TrimSpace(result.Version)
	}
	if name == "" {
		name = releaseStatusShortSHA(result.CommitSHA)
	}
	return "fak/release/" + releaseShipRefComponent(name)
}

func releaseShipInvalidPromotionBranchReason(branch string) string {
	branch = strings.TrimSpace(branch)
	if branch == "" {
		return "empty branch name"
	}
	if branch == "@" {
		return "single @ is not a valid branch name"
	}
	if strings.HasPrefix(branch, "/") || strings.HasSuffix(branch, "/") {
		return "must not start or end with slash"
	}
	if strings.Contains(branch, "//") {
		return "must not contain empty path components"
	}
	if strings.Contains(branch, "..") {
		return "must not contain consecutive dots"
	}
	if strings.Contains(branch, "@{") {
		return "must not contain @{"
	}
	if strings.HasSuffix(branch, ".") {
		return "must not end with dot"
	}
	for _, r := range branch {
		if r <= ' ' || r == 0x7f {
			return "must not contain whitespace or control characters"
		}
		switch r {
		case '~', '^', ':', '?', '*', '[', '\\':
			return fmt.Sprintf("must not contain %q", r)
		}
	}
	for _, component := range strings.Split(branch, "/") {
		if component == "" {
			return "must not contain empty path components"
		}
		if strings.HasPrefix(component, ".") {
			return "path component must not start with dot"
		}
		if strings.HasSuffix(strings.ToLower(component), ".lock") {
			return "path component must not end with .lock"
		}
	}
	return ""
}

func releaseShipRefComponent(s string) string {
	s = strings.TrimPrefix(strings.TrimSpace(s), "refs/tags/")
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		ok := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '.' || r == '_' || r == '-'
		if ok {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(b.String(), ".-")
	for strings.Contains(out, "..") {
		out = strings.ReplaceAll(out, "..", ".-")
	}
	out = strings.Trim(out, ".-")
	for strings.HasSuffix(strings.ToLower(out), ".lock") {
		out = out[:len(out)-len(".lock")]
		out = strings.Trim(out, ".-")
	}
	if out == "" {
		return "untagged"
	}
	return out
}

func releaseShipRemoteTargetCheck(result *releaseShipResult, wt string, opts releaseShipOptions) map[string]any {
	ref := "refs/heads/" + opts.trunk
	expected := strings.TrimSpace(result.TargetSHA)
	if expected == "" {
		return map[string]any{"ok": false, "reason": "target_sha_missing", "remote": opts.remote, "branch": opts.trunk, "ref": ref}
	}
	code, out := releaseShipCmd(result, wt, "git", []string{"ls-remote", opts.remote, ref}, nil, releaseShipRemoteProbeCommandTimeout)
	if code != 0 {
		return map[string]any{"ok": false, "reason": "target_branch_probe_failed", "remote": opts.remote, "branch": opts.trunk, "ref": ref, "expected_sha": expected, "tail": tail(out)}
	}
	fields := strings.Fields(out)
	if len(fields) == 0 {
		return map[string]any{"ok": false, "reason": "target_branch_missing", "remote": opts.remote, "branch": opts.trunk, "ref": ref, "expected_sha": expected}
	}
	got := fields[0]
	payload := map[string]any{
		"ok":           sameSHA(got, expected),
		"remote":       opts.remote,
		"branch":       opts.trunk,
		"ref":          ref,
		"sha":          got,
		"expected_sha": expected,
	}
	if !releaseStatusBool(payload["ok"]) {
		payload["reason"] = "target_branch_changed_after_preview"
	}
	return payload
}

func releaseShipPushTrunk(result *releaseShipResult, wt string, opts releaseShipOptions) map[string]any {
	ref := "refs/heads/" + opts.trunk
	expected := releaseShipTrunkLeaseExpected(*result)
	if expected == "" {
		return map[string]any{"ok": false, "reason": "trunk_lease_expected_sha_missing", "remote": opts.remote, "branch": opts.trunk, "ref": ref}
	}
	code, out := releaseShipCmd(result, wt, "git", []string{"merge-base", "--is-ancestor", expected, result.CommitSHA}, nil, releaseShipMergeBaseCommandTimeout)
	if code != 0 {
		return map[string]any{
			"ok":                 false,
			"reason":             "release_commit_not_descendant_of_trunk",
			"remote":             opts.remote,
			"branch":             opts.trunk,
			"ref":                ref,
			"lease_expected_sha": expected,
			"release_commit_sha": result.CommitSHA,
			"tail":               tail(out),
		}
	}
	args := []string{"push", opts.remote, "--force-with-lease=" + ref + ":" + expected, "HEAD:" + ref}
	code, out = releaseShipCmd(result, wt, "git", args, nil, releaseShipGitPushCommandTimeout)
	if code != 0 {
		return map[string]any{
			"ok":                 false,
			"reason":             "push_trunk_failed",
			"remote":             opts.remote,
			"branch":             opts.trunk,
			"ref":                ref,
			"lease_expected_sha": expected,
			"release_commit_sha": result.CommitSHA,
			"tail":               tail(out),
		}
	}
	return map[string]any{
		"ok":                 true,
		"remote":             opts.remote,
		"branch":             opts.trunk,
		"ref":                ref,
		"lease_expected_sha": expected,
		"release_commit_sha": result.CommitSHA,
		"used_force_lease":   true,
	}
}

// executeReleaseShipTrunkPush pushes the release-cut commit to trunk with a
// force-with-lease, and on the same-trunk hot path re-cuts onto an
// advanced trunk tip and retries when a peer fast-forwards trunk mid-cut. It
// never force-pushes (each retry leases against the exact observed tip), never
// rebases past a rewritten trunk, and returns true only after a verified push --
// so tag/publish still run only against a landed release commit.
func executeReleaseShipTrunkPush(result *releaseShipResult, root, wt string, env []string, opts releaseShipOptions) bool {
	for attempt := 0; ; attempt++ {
		push := releaseShipPushTrunk(result, wt, opts)
		result.RemoteBranchPush = push
		if releaseStatusBool(push["ok"]) {
			return true
		}
		if !releaseShipPushRetryable(opts, push) || attempt >= opts.pushRetries {
			result.fail("push_trunk_failed", jsonTail(push))
			return false
		}
		// Renew before the (possibly slow) re-cut so the lease survives the retry;
		// a lost lock surfaces as release_lock_renew_failed and aborts.
		if !renewReleaseShipLockForPhase(result, root, env, opts,
			fmt.Sprintf("push_retry_%d", attempt+1),
			releaseShipCutCommandTimeout+releaseShipGitPushCommandTimeout) {
			return false
		}
		rebase := releaseShipRebaseCutOntoTrunk(result, root, wt, env, opts)
		result.PushRetries = append(result.PushRetries, rebase)
		if !releaseStatusBool(rebase["ok"]) {
			result.fail("push_trunk_retry_failed", jsonTail(rebase))
			return false
		}
	}
}

// releaseShipPushRetryable gates the rebase-and-retry to the same-trunk hot path
// (source == trunk, the default main-only regime) and only for a lease-mismatch
// push refusal -- a peer advancing trunk during the cut. Distinct-branch
// promotion (--open-pr / --allow-direct-promotion) keeps the refuse behavior.
func releaseShipPushRetryable(opts releaseShipOptions, push map[string]any) bool {
	if opts.pushRetries <= 0 || opts.openPR {
		return false
	}
	if opts.sourceBranch == "" || opts.sourceBranch != opts.trunk {
		return false
	}
	return stringFromAny(push["reason"]) == "push_trunk_failed"
}

// releaseShipRebaseCutOntoTrunk re-fetches trunk, proves the new tip is a
// fast-forward of the base we cut from (a peer added commits; history was not
// rewritten), re-witnesses source CI on that new tip, resets the detached
// worktree to it, and re-cuts the VERSION+note commit onto it -- the automated
// form of the manual /release "fast-forward your single release commit on top"
// recovery. It updates the result's base/commit identity so the next push leases
// against the new tip. It fails closed on a diverged trunk, an unchanged trunk (a
// transient push error rather than a peer advance), a peer tip whose CI is not
// confirmed green, or a refused re-cut. Re-witnessing CI here keeps the shipping
// commit's green-CI guarantee honest: the pre-flight only confirmed the original
// base, so without this the retry would tag -- and its FAK_RELEASE_SOURCE_CI
// witness would claim green for -- a peer tip that was never checked.
func releaseShipRebaseCutOntoTrunk(result *releaseShipResult, root, wt string, env []string, opts releaseShipOptions) map[string]any {
	oldBase := result.BaseSHA
	refspec := fmt.Sprintf("refs/heads/%s:refs/remotes/%s/%s", opts.trunk, opts.remote, opts.trunk)
	if code, out := releaseShipCmd(result, root, "git", []string{"fetch", opts.remote, refspec}, nil, 5*time.Minute); code != 0 {
		return map[string]any{"ok": false, "reason": "retry_fetch_failed", "old_base": oldBase, "tail": tail(out)}
	}
	code, out := releaseShipCmd(result, root, "git", []string{"rev-parse", "--verify", opts.remote + "/" + opts.trunk + "^{commit}"}, nil, time.Minute)
	if code != 0 {
		return map[string]any{"ok": false, "reason": "retry_base_unresolvable", "old_base": oldBase, "tail": tail(out)}
	}
	newBase := strings.TrimSpace(out)
	if sameSHA(newBase, oldBase) {
		return map[string]any{"ok": false, "reason": "retry_trunk_unchanged", "old_base": oldBase, "new_base": newBase}
	}
	if code, out := releaseShipCmd(result, wt, "git", []string{"merge-base", "--is-ancestor", oldBase, newBase}, nil, releaseShipMergeBaseCommandTimeout); code != 0 {
		return map[string]any{"ok": false, "reason": "retry_trunk_diverged", "old_base": oldBase, "new_base": newBase, "tail": tail(out)}
	}
	if code, out := releaseShipCmd(result, wt, "git", []string{"reset", "--hard", newBase}, nil, time.Minute); code != 0 {
		return map[string]any{"ok": false, "reason": "retry_reset_failed", "old_base": oldBase, "new_base": newBase, "tail": tail(out)}
	}
	result.BaseSHA = newBase
	result.SourceSHA = newBase
	// Re-witness source CI on the peer's new tip before re-cutting onto it. The
	// pre-flight only confirmed the original base; the peer tip we now ship may be
	// pending or red. Refresh result.SourceCI so the emitted witness and the
	// FAK_RELEASE_SOURCE_CI env handed to the re-cut describe the tip we actually
	// tag -- and refuse (fail-closed, symmetric with the pre-flight) if it is not
	// green. On a hot trunk the peer tip may legitimately be mid-CI, so this
	// sometimes defers the release to the next run; that is the correct behavior.
	if opts.requireCI && !opts.skipCI {
		result.SourceCI = releaseShipSourceCI(result, root, opts, newBase)
		if !releaseShipPayloadOK(result.SourceCI) {
			return map[string]any{"ok": false, "reason": "retry_source_ci_unconfirmed", "old_base": oldBase, "new_base": newBase, "detail": jsonTail(result.SourceCI)}
		}
	}
	cut := runReleaseShipCut(result, wt, releaseShipPromotionEnv(env, *result), opts)
	result.Cut = cut
	if ok, _ := cut["ok"].(bool); !ok {
		return map[string]any{"ok": false, "reason": "retry_cut_refused", "old_base": oldBase, "new_base": newBase, "detail": jsonTail(cut)}
	}
	result.Version = stringFromAny(cut["version"])
	result.Tag = stringFromAny(cut["tag"])
	commit := stringFromAny(cut["commit_sha"])
	if commit == "" {
		c, o := releaseShipCmd(result, wt, "git", []string{"rev-parse", "HEAD"}, nil, releaseShipCommitResolveTimeout)
		if c != 0 {
			return map[string]any{"ok": false, "reason": "retry_commit_unresolvable", "old_base": oldBase, "new_base": newBase, "tail": tail(o)}
		}
		commit = strings.TrimSpace(o)
	}
	result.CommitSHA = commit
	return map[string]any{"ok": true, "old_base": oldBase, "new_base": newBase, "commit_sha": commit, "version": result.Version, "tag": result.Tag}
}

func releaseShipTrunkLeaseExpected(result releaseShipResult) string {
	if sha := strings.TrimSpace(result.TargetSHA); sha != "" {
		return sha
	}
	return strings.TrimSpace(result.BaseSHA)
}

func releaseShipPushPromotionBranch(result *releaseShipResult, wt string, env []string, opts releaseShipOptions, branch string) map[string]any {
	ref := "refs/heads/" + branch
	code, out := releaseShipCmd(result, wt, "git", []string{"ls-remote", opts.remote, ref}, env, releaseShipRemoteProbeCommandTimeout)
	if code != 0 {
		return map[string]any{"ok": false, "reason": "promotion_branch_probe_failed", "branch": branch, "tail": tail(out)}
	}
	existing := ""
	if fields := strings.Fields(out); len(fields) > 0 {
		existing = fields[0]
	}
	args := []string{"push", opts.remote, "--force-with-lease=" + ref + ":" + existing}
	args = append(args, "HEAD:"+ref)
	code, out = releaseShipCmd(result, wt, "git", args, env, releaseShipGitPushCommandTimeout)
	if code != 0 {
		return map[string]any{
			"ok":                    false,
			"reason":                "push_promotion_branch_failed",
			"branch":                branch,
			"previous_sha":          existing,
			"lease_expected_absent": existing == "",
			"lease_expected_sha":    existing,
			"tail":                  tail(out),
		}
	}
	code, out = releaseShipCmd(result, wt, "git", []string{"ls-remote", opts.remote, ref}, env, releaseShipRemoteProbeCommandTimeout)
	if code != 0 {
		return map[string]any{"ok": false, "reason": "promotion_branch_verify_failed", "branch": branch, "tail": tail(out)}
	}
	fields := strings.Fields(out)
	if len(fields) == 0 {
		return map[string]any{"ok": false, "reason": "promotion_branch_missing_after_push", "branch": branch}
	}
	got := fields[0]
	if !sameSHA(got, result.CommitSHA) {
		return map[string]any{"ok": false, "reason": "promotion_branch_mismatch", "branch": branch, "sha": got, "want": result.CommitSHA}
	}
	return map[string]any{
		"ok":                    true,
		"branch":                branch,
		"sha":                   got,
		"previous_sha":          existing,
		"used_force_lease":      true,
		"lease_expected_absent": existing == "",
		"lease_expected_sha":    existing,
		"source_branch":         opts.sourceBranch,
		"source_sha":            result.SourceSHA,
		"release_commit_sha":    result.CommitSHA,
	}
}

func releaseShipFindOpenPR(result *releaseShipResult, wt string, env []string, opts releaseShipOptions, headBranch string) map[string]any {
	args := []string{"pr", "list", "--base", opts.trunk, "--head", headBranch, "--state", "open", "--json", "number,url,title,headRefName,baseRefName", "--limit", "1"}
	code, out := releaseShipCmd(result, wt, "gh", args, env, releaseShipGHPRCommandTimeout)
	if code != 0 {
		return map[string]any{"ok": false, "reason": "gh_pr_list_failed", "tail": tail(out)}
	}
	var rows []map[string]any
	if err := json.Unmarshal([]byte(out), &rows); err != nil {
		return map[string]any{"ok": false, "reason": "gh_pr_list_non_json", "tail": tail(out)}
	}
	if len(rows) == 0 {
		return map[string]any{"ok": true, "found": false}
	}
	row := rows[0]
	return map[string]any{
		"ok":     true,
		"found":  true,
		"pr":     row,
		"url":    stringFromAny(row["url"]),
		"number": row["number"],
		"title":  stringFromAny(row["title"]),
	}
}

func releaseShipExistingPRTarget(existing map[string]any) string {
	if url := stringFromAny(existing["url"]); url != "" {
		return url
	}
	switch n := existing["number"].(type) {
	case float64:
		return strconv.Itoa(int(n))
	case int:
		return strconv.Itoa(n)
	case string:
		return strings.TrimSpace(n)
	default:
		return ""
	}
}

func releaseShipLastNonEmptyLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if line := strings.TrimSpace(lines[i]); line != "" {
			return line
		}
	}
	return ""
}

func runReleaseShipLockAcquire(result *releaseShipResult, root string, env []string, opts releaseShipOptions) map[string]any {
	args := []string{
		releaseShipScript(root, "release_lock.py"),
		"acquire",
		"--ttl", strconv.Itoa(releaseShipLockTTLForBudget(opts, releaseShipInitialLockBudget(opts))),
		"--note", "fak release ship",
	}
	return releaseShipJSONCommand(result, root, releaseShipPython(), args, env, time.Minute, "release_lock")
}

func renewReleaseShipLockForPhase(result *releaseShipResult, root string, env []string, opts releaseShipOptions, label string, phaseBudget time.Duration) bool {
	renew := runReleaseShipLockRenew(result, root, env, releaseShipLockTTLForBudget(opts, phaseBudget), label)
	if !releaseStatusBool(renew["ok"]) {
		result.fail("release_lock_renew_failed", jsonTail(renew))
		return false
	}
	return true
}

func runReleaseShipLockRenew(result *releaseShipResult, root string, env []string, ttl int, label string) map[string]any {
	args := []string{
		releaseShipScript(root, "release_lock.py"),
		"renew",
		"--ttl", strconv.Itoa(ttl),
	}
	payload := releaseShipJSONCommand(result, root, releaseShipPython(), args, env, time.Minute, "release_lock_renew_"+label)
	payload["label"] = label
	payload["requested_ttl_s"] = ttl
	result.ReleaseLockRenewals = append(result.ReleaseLockRenewals, payload)
	return payload
}

func releaseShipInitialLockBudget(opts releaseShipOptions) time.Duration {
	budget := releaseShipWorktreeAddCommandTimeout + releaseShipCutCommandTimeout + releaseShipCommitResolveTimeout
	if opts.openPR {
		promotionPushBudget := releaseShipRemoteProbeCommandTimeout + releaseShipGitPushCommandTimeout + releaseShipRemoteProbeCommandTimeout
		targetChecksBudget := 2 * releaseShipRemoteProbeCommandTimeout
		prBudget := releaseShipPRPlanCommandTimeout + 2*releaseShipGHPRCommandTimeout
		return budget + prBudget + targetChecksBudget + promotionPushBudget
	}
	budget += releaseShipMergeBaseCommandTimeout + releaseShipGitPushCommandTimeout
	budget += 2 * releaseShipRemoteProbeCommandTimeout
	if opts.requireCI && !opts.skipCI && opts.ciAppearTimeout > 0 {
		budget += opts.ciAppearTimeout + releaseShipCIPollCommandTimeout
	}
	return budget
}

func releaseShipLockTTLForBudget(opts releaseShipOptions, phaseBudget time.Duration) int {
	minTTL := int((phaseBudget + releaseShipLockRenewGrace).Seconds())
	if opts.ttl > minTTL {
		return opts.ttl
	}
	return minTTL
}

func runReleaseShipLockRelease(result *releaseShipResult, root string, env []string) map[string]any {
	args := []string{releaseShipScript(root, "release_lock.py"), "release"}
	return releaseShipJSONCommand(result, root, releaseShipPython(), args, env, time.Minute, "release_lock_release")
}

func releaseShipJSONCommand(result *releaseShipResult, cwd, name string, args []string, env []string, timeout time.Duration, label string) map[string]any {
	code, out := releaseShipCmd(result, cwd, name, args, env, timeout)
	payload := map[string]any{}
	if err := json.Unmarshal([]byte(out), &payload); err != nil {
		payload = map[string]any{
			"ok":        false,
			"exit_code": code,
			"error":     "command emitted non-json output",
			"tail":      tail(out),
		}
	} else {
		payload["exit_code"] = code
	}
	if code != 0 {
		if _, ok := payload["ok"]; !ok {
			payload["ok"] = false
		}
	}
	result.CommandTail[label] = tail(out)
	return payload
}

func verifyReleaseShipRemote(result *releaseShipResult, wt string, opts releaseShipOptions, want string) string {
	check := releaseShipRemoteBranchCheck(result, wt, opts, want, "remote_branch_mismatch")
	if !releaseStatusBool(check["ok"]) {
		result.fail(stringFromAny(check["reason"]), jsonTail(check))
		return ""
	}
	got := stringFromAny(check["sha"])
	result.RemoteBranch = map[string]string{
		"remote": opts.remote,
		"trunk":  opts.trunk,
		"sha":    got,
	}
	return got
}

func releaseShipRemoteBranchCheck(result *releaseShipResult, wt string, opts releaseShipOptions, want string, mismatchReason string) map[string]any {
	ref := "refs/heads/" + opts.trunk
	want = strings.TrimSpace(want)
	if want == "" {
		return map[string]any{"ok": false, "reason": "remote_branch_expected_sha_missing", "remote": opts.remote, "trunk": opts.trunk, "ref": ref}
	}
	code, out := releaseShipCmd(result, wt, "git", []string{"ls-remote", opts.remote, ref}, nil, releaseShipRemoteProbeCommandTimeout)
	if code != 0 {
		return map[string]any{"ok": false, "reason": "remote_verify_failed", "remote": opts.remote, "trunk": opts.trunk, "ref": ref, "expected_sha": want, "tail": tail(out)}
	}
	fields := strings.Fields(out)
	if len(fields) == 0 {
		return map[string]any{"ok": false, "reason": "remote_branch_missing", "remote": opts.remote, "trunk": opts.trunk, "ref": ref, "expected_sha": want}
	}
	got := fields[0]
	payload := map[string]any{
		"remote":       opts.remote,
		"trunk":        opts.trunk,
		"ref":          ref,
		"sha":          got,
		"expected_sha": want,
	}
	if sameSHA(got, want) {
		payload["ok"] = true
		return payload
	}
	// The observed trunk tip is no longer exactly our release commit. On a hot
	// trunk a peer can fast-forward the branch past the release commit while we
	// wait for its CI run to appear -- that is benign: the release commit is still
	// on trunk and the tag is content-addressed to it, so tag/publish stay
	// correct. Only a tip that no longer contains the release commit (a rewrite
	// that orphaned it) is a genuine mismatch. Distinguish the two with an
	// ancestry probe; fail closed if it cannot confirm the commit is still there.
	if releaseShipCommitStillOnTrunk(result, wt, opts, want) {
		payload["ok"] = true
		payload["fast_forwarded"] = true
		return payload
	}
	payload["ok"] = false
	payload["reason"] = mismatchReason
	return payload
}

// releaseShipCommitStillOnTrunk fetches the remote trunk and reports whether
// want is still an ancestor of (reachable from) the current trunk tip -- i.e.
// the release commit survived a peer's fast-forward. A fetch failure or a
// non-ancestor tip both answer false, so the caller refuses on a genuine
// rewrite rather than an ordinary peer advance.
func releaseShipCommitStillOnTrunk(result *releaseShipResult, wt string, opts releaseShipOptions, want string) bool {
	refspec := fmt.Sprintf("refs/heads/%s:refs/remotes/%s/%s", opts.trunk, opts.remote, opts.trunk)
	if code, _ := releaseShipCmd(result, wt, "git", []string{"fetch", opts.remote, refspec}, nil, 5*time.Minute); code != 0 {
		return false
	}
	code, _ := releaseShipCmd(result, wt, "git", []string{"merge-base", "--is-ancestor", want, opts.remote + "/" + opts.trunk}, nil, releaseShipMergeBaseCommandTimeout)
	return code == 0
}

func waitReleaseShipCIAppears(result *releaseShipResult, wt string, opts releaseShipOptions, sha string) {
	deadline := time.Now().Add(opts.ciAppearTimeout)
	args := []string{"run", "list", "--workflow", opts.workflow, "--commit", sha, "--limit", "1", "--json", "databaseId,status,conclusion,url"}
	for {
		code, out := releaseShipCmd(result, wt, "gh", args, nil, releaseShipCIPollCommandTimeout)
		if code != 0 {
			result.Warnings = append(result.Warnings, "gh run list failed while waiting for CI to appear: "+tail(out))
			return
		}
		var rows []map[string]any
		if err := json.Unmarshal([]byte(out), &rows); err != nil {
			result.Warnings = append(result.Warnings, "gh run list emitted non-json output while waiting for CI")
			return
		}
		if len(rows) > 0 {
			return
		}
		if time.Now().After(deadline) {
			result.Warnings = append(result.Warnings, "no CI run appeared before ci-appear-timeout; release_tag will make the final CI decision")
			return
		}
		time.Sleep(5 * time.Second)
	}
}

func releaseShipSourceCI(result *releaseShipResult, root string, opts releaseShipOptions, sha string) map[string]any {
	if strings.TrimSpace(sha) == "" {
		return map[string]any{"ok": false, "status": "missing_source_sha"}
	}
	args := []string{"run", "list", "--workflow", opts.workflow, "--commit", sha, "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha"}
	code, out := releaseShipCmd(result, root, "gh", args, nil, time.Minute)
	if code != 0 {
		return map[string]any{"ok": false, "status": "unavailable", "tail": tail(out)}
	}
	var rows []map[string]any
	if err := json.Unmarshal([]byte(out), &rows); err != nil {
		return map[string]any{"ok": false, "status": "non_json", "tail": tail(out)}
	}
	if len(rows) == 0 {
		return map[string]any{"ok": false, "status": "missing", "source_sha": sha}
	}
	row := rows[0]
	conclusion := strings.ToLower(stringFromAny(row["conclusion"]))
	workflowStatus := strings.ToLower(stringFromAny(row["status"]))
	state := releaseShipCIState(workflowStatus, conclusion)
	return map[string]any{
		"ok":              state == "green",
		"status":          state,
		"workflow_status": workflowStatus,
		"conclusion":      conclusion,
		"source_sha":      sha,
		"run":             row,
	}
}

func releaseShipCIState(status, conclusion string) string {
	if status == "queued" || status == "in_progress" || status == "requested" || status == "waiting" || status == "pending" {
		return "pending"
	}
	switch conclusion {
	case "success":
		return "green"
	case "cancelled", "stale", "skipped":
		return "cancelled_or_starved"
	case "failure", "timed_out", "action_required", "startup_failure":
		return "failed"
	case "":
		return "missing"
	default:
		return "failed"
	}
}

// releaseShipPayloadOK reads the "ok" verdict out of a release-ship gate payload. An absent
// payload is NOT ok: a gate that never ran must never read as a gate that passed -- which is
// the whole reason these two gates need a reader of their own rather than a bare
// payload["ok"] assertion. The source-CI and the target-ancestry gates publish the same
// shaped map and both can be skipped, so one reader serves both.
func releaseShipPayloadOK(payload map[string]any) bool {
	if payload == nil {
		return false
	}
	ok, _ := payload["ok"].(bool)
	return ok
}

func releaseShipNeedsTargetAncestry(opts releaseShipOptions) bool {
	source := strings.TrimSpace(opts.sourceBranch)
	target := strings.TrimSpace(opts.trunk)
	return source != "" && target != "" && source != target
}

func releaseShipDirectPromotionRequiresOpenPR(opts releaseShipOptions) bool {
	source := strings.TrimSpace(opts.sourceBranch)
	target := strings.TrimSpace(opts.trunk)
	return opts.execute && !opts.openPR && !opts.allowDirectPromotion && source != "" && target != "" && source != target
}

func releaseShipTargetAncestry(result *releaseShipResult, root string, opts releaseShipOptions, sourceSHA string) map[string]any {
	targetRef := opts.remote + "/" + opts.trunk
	if strings.TrimSpace(sourceSHA) == "" {
		return map[string]any{"ok": false, "status": "missing_source_sha", "target_ref": targetRef}
	}
	code, out := releaseShipCmd(result, root, "git", []string{"rev-parse", "--verify", targetRef + "^{commit}"}, nil, time.Minute)
	if code != 0 {
		return map[string]any{"ok": false, "status": "target_unresolvable", "target_ref": targetRef, "tail": tail(out)}
	}
	targetSHA := strings.TrimSpace(out)
	payload := map[string]any{
		"target_ref": targetRef,
		"target_sha": targetSHA,
		"source_ref": opts.sourceBranch,
		"source_sha": sourceSHA,
	}
	if sameSHA(targetSHA, sourceSHA) {
		payload["ok"] = true
		payload["status"] = "same"
		return payload
	}
	code, out = releaseShipCmd(result, root, "git", []string{"merge-base", "--is-ancestor", targetSHA, sourceSHA}, nil, time.Minute)
	switch code {
	case 0:
		payload["ok"] = true
		payload["status"] = "ancestor"
	case 1:
		payload["ok"] = false
		payload["status"] = "non_fast_forward"
	default:
		payload["ok"] = false
		payload["status"] = "ancestor_check_failed"
		payload["tail"] = tail(out)
	}
	return payload
}

func cleanupReleaseShipWorktree(root, wt string) map[string]any {
	code, out := releaseShipRunCommand(root, "git", []string{"worktree", "remove", "--force", wt}, nil, 5*time.Minute)
	payload := map[string]any{
		"ok":        code == 0,
		"path":      wt,
		"exit_code": code,
	}
	if code != 0 {
		payload["tail"] = tail(out)
	}
	pruneCode, pruneOut := releaseShipRunCommand(root, "git", []string{"worktree", "prune"}, nil, time.Minute)
	payload["prune_exit_code"] = pruneCode
	if pruneCode != 0 {
		payload["prune_tail"] = tail(pruneOut)
	}
	return payload
}

func readReleaseReadiness(path string) (workdelivery.WorkUnit, workdelivery.Receipt, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return workdelivery.WorkUnit{}, workdelivery.Receipt{}, err
	}
	var envelope struct {
		Unit          workdelivery.WorkUnit                `json:"unit"`
		Receipt       workdelivery.Receipt                 `json:"receipt"`
		Qualification *installedLaunchQualificationReceipt `json:"installed_launch_qualification,omitempty"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return workdelivery.WorkUnit{}, workdelivery.Receipt{}, err
	}
	if err := envelope.Unit.Validate(); err != nil {
		return workdelivery.WorkUnit{}, workdelivery.Receipt{}, err
	}
	if err := validateInstalledLaunchQualification(envelope.Qualification); err != nil {
		return workdelivery.WorkUnit{}, workdelivery.Receipt{}, err
	}
	return envelope.Unit, envelope.Receipt, nil
}
