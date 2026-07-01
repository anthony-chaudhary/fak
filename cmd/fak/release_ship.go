package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
)

type releaseShipCommandRunner func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string)

var releaseShipRunCommand releaseShipCommandRunner = runReleaseShipCommand
var releaseShipMkdirTemp = os.MkdirTemp

type releaseShipOptions struct {
	execute         bool
	asJSON          bool
	base            string
	sourceBranch    string
	remote          string
	trunk           string
	workflow        string
	version         string
	limitCommits    int
	ttl             int
	fetch           bool
	force           bool
	requireCI       bool
	waitCI          bool
	skipCI          bool
	skipDryRun      bool
	ciAppearTimeout time.Duration
	keepWorktree    bool
	worktreeDir     string
}

type releaseShipResult struct {
	OK                 bool                 `json:"ok"`
	DryRun             bool                 `json:"dry_run"`
	Root               string               `json:"root"`
	Base               string               `json:"base"`
	BaseSHA            string               `json:"base_sha,omitempty"`
	SourceBranch       string               `json:"source_branch,omitempty"`
	SourceSHA          string               `json:"source_sha,omitempty"`
	TargetBranch       string               `json:"target_branch,omitempty"`
	Worktree           string               `json:"worktree,omitempty"`
	LockRoot           string               `json:"lock_root,omitempty"`
	ReleaseOwner       string               `json:"release_owner,omitempty"`
	Version            string               `json:"version,omitempty"`
	Tag                string               `json:"tag,omitempty"`
	CommitSHA          string               `json:"commit_sha,omitempty"`
	Remote             string               `json:"remote,omitempty"`
	Trunk              string               `json:"trunk,omitempty"`
	Cut                map[string]any       `json:"cut,omitempty"`
	TagResult          map[string]any       `json:"tag_result,omitempty"`
	Publish            map[string]any       `json:"publish,omitempty"`
	ReleaseLock        map[string]any       `json:"release_lock,omitempty"`
	ReleaseLockRelease map[string]any       `json:"release_lock_release,omitempty"`
	RemoteBranch       map[string]string    `json:"remote_branch,omitempty"`
	Cleanup            map[string]any       `json:"cleanup,omitempty"`
	Warnings           []string             `json:"warnings,omitempty"`
	Errors             []string             `json:"errors,omitempty"`
	CommandTail        map[string]string    `json:"command_tail,omitempty"`
	ExecutedCommands   []releaseShipCommand `json:"executed_commands,omitempty"`
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
	fs.IntVar(&opts.ttl, "ttl", opts.ttl, "release lock TTL in seconds")
	fs.BoolVar(&opts.fetch, "fetch", opts.fetch, "fetch remote source/target branches before creating the detached worktree")
	fs.BoolVar(&opts.force, "force", false, "pass --force to release_cut (only bypasses the substantive floor)")
	fs.BoolVar(&opts.requireCI, "require-ci", opts.requireCI, "require green CI before tagging")
	fs.BoolVar(&opts.waitCI, "wait-ci", opts.waitCI, "watch a pending CI run before tagging")
	fs.BoolVar(&opts.skipCI, "skip-ci", false, "skip CI folding in release_tag; intended for emergency/operator use")
	fs.BoolVar(&opts.skipDryRun, "skip-dry-run", opts.skipDryRun, "pass --skip-dry-run to cut/tag while the dry-run witness still requires a preexisting tag")
	fs.DurationVar(&opts.ciAppearTimeout, "ci-appear-timeout", opts.ciAppearTimeout, "how long to wait for a just-pushed CI run to appear before release_tag folds it")
	fs.BoolVar(&opts.keepWorktree, "keep-worktree", false, "leave the detached worktree on disk")
	fs.StringVar(&opts.worktreeDir, "worktree-dir", "", "use this detached worktree path instead of a temp dir")
	if err := fs.Parse(argv); err != nil {
		return opts, err
	}
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
		sourceBranch:    roles.ReleaseSource,
		remote:          "origin",
		trunk:           roles.ReleaseBranch,
		workflow:        "ci.yml",
		limitCommits:    50,
		ttl:             1800,
		fetch:           true,
		requireCI:       true,
		waitCI:          true,
		skipDryRun:      true,
		ciAppearTimeout: 2 * time.Minute,
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

	if opts.execute {
		lock := runReleaseShipLockAcquire(&result, root, env, opts)
		result.ReleaseLock = lock
		if ok, _ := lock["ok"].(bool); !ok {
			result.fail("release_lock_refused", jsonTail(lock))
			return finishReleaseShip(result)
		}
		lockAcquired = true
	}

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

	wt, err := releaseShipWorktreeDir(root, opts)
	if err != nil {
		result.fail("worktree_dir_failed", err.Error())
		return finishReleaseShip(result)
	}
	result.Worktree = wt
	worktreeAdded := false

	code, out = releaseShipCmd(&result, root, "git", []string{"worktree", "add", "--detach", wt, opts.base}, nil, 5*time.Minute)
	if code != 0 {
		if opts.worktreeDir == "" {
			_ = os.Remove(wt)
		}
		result.fail("worktree_add_failed", out)
		return finishReleaseShip(result)
	}
	worktreeAdded = true

	cut := runReleaseShipCut(&result, wt, env, opts)
	result.Cut = cut
	if ok, _ := cut["ok"].(bool); !ok {
		result.fail("release_cut_refused", jsonTail(cut))
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	result.Version = stringFromAny(cut["version"])
	result.Tag = stringFromAny(cut["tag"])
	result.CommitSHA = stringFromAny(cut["commit_sha"])
	if !opts.execute {
		result.OK = true
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	if result.CommitSHA == "" {
		code, out = releaseShipCmd(&result, wt, "git", []string{"rev-parse", "HEAD"}, nil, time.Minute)
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

	if code, out = releaseShipCmd(&result, wt, "git", []string{"push", opts.remote, "HEAD:refs/heads/" + opts.trunk}, nil, 10*time.Minute); code != 0 {
		result.fail("push_trunk_failed", out)
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	remoteSHA := verifyReleaseShipRemote(&result, wt, opts, result.CommitSHA)
	if remoteSHA == "" {
		return finishReleaseShipWithCleanup(&result, root, opts, worktreeAdded)
	}
	if opts.requireCI && !opts.skipCI && opts.ciAppearTimeout > 0 {
		waitReleaseShipCIAppears(&result, wt, opts, result.CommitSHA)
	}

	tag := runReleaseShipTag(&result, wt, env, opts)
	result.TagResult = tag
	if ok, _ := tag["ok"].(bool); !ok {
		result.fail("release_tag_refused", jsonTail(tag))
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
	return releaseShipJSONCommand(result, wt, releaseShipPython(), args, env, 30*time.Minute, "cut")
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
	if opts.skipDryRun {
		args = append(args, "--skip-dry-run")
	}
	if opts.execute {
		args = append(args, "--lock-already-held")
	}
	return releaseShipJSONCommand(result, wt, releaseShipPython(), args, env, 45*time.Minute, "tag")
}

func runReleaseShipPublish(result *releaseShipResult, wt string, env []string, version string) map[string]any {
	args := []string{releaseShipScript(wt, "release_publish.py"), "--version", version, "--execute", "--json"}
	return releaseShipJSONCommand(result, wt, releaseShipPython(), args, env, 10*time.Minute, "publish")
}

func runReleaseShipLockAcquire(result *releaseShipResult, root string, env []string, opts releaseShipOptions) map[string]any {
	args := []string{
		releaseShipScript(root, "release_lock.py"),
		"acquire",
		"--ttl", strconv.Itoa(opts.ttl),
		"--note", "fak release ship",
	}
	return releaseShipJSONCommand(result, root, releaseShipPython(), args, env, time.Minute, "release_lock")
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
	code, out := releaseShipCmd(result, wt, "git", []string{"ls-remote", opts.remote, "refs/heads/" + opts.trunk}, nil, 2*time.Minute)
	if code != 0 {
		result.fail("remote_verify_failed", out)
		return ""
	}
	fields := strings.Fields(out)
	if len(fields) == 0 {
		result.fail("remote_branch_missing", out)
		return ""
	}
	got := fields[0]
	result.RemoteBranch = map[string]string{
		"remote": opts.remote,
		"trunk":  opts.trunk,
		"sha":    got,
	}
	if !sameSHA(got, want) {
		result.fail("remote_branch_mismatch", fmt.Sprintf("remote %s/%s is %s, want %s", opts.remote, opts.trunk, got, want))
		return ""
	}
	return got
}

func waitReleaseShipCIAppears(result *releaseShipResult, wt string, opts releaseShipOptions, sha string) {
	deadline := time.Now().Add(opts.ciAppearTimeout)
	args := []string{"run", "list", "--workflow", opts.workflow, "--commit", sha, "--limit", "1", "--json", "databaseId,status,conclusion,url"}
	for {
		code, out := releaseShipCmd(result, wt, "gh", args, nil, time.Minute)
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

func releaseShipCmd(result *releaseShipResult, cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
	result.ExecutedCommands = append(result.ExecutedCommands, releaseShipCommand{
		CWD:  cwd,
		Name: name,
		Args: append([]string(nil), args...),
	})
	return releaseShipRunCommand(cwd, name, args, env, timeout)
}

func runReleaseShipCommand(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
	cmd := exec.Command(name, args...)
	cmd.Dir = cwd
	if env != nil {
		cmd.Env = env
	}
	var combined strings.Builder
	cmd.Stdout = &combined
	cmd.Stderr = &combined
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	done := make(chan error, 1)
	if err := cmd.Start(); err != nil {
		return 127, err.Error()
	}
	go func() { done <- cmd.Wait() }()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case err := <-done:
		if err != nil {
			if ee, ok := err.(*exec.ExitError); ok {
				return ee.ExitCode(), combined.String()
			}
			return 127, combined.String() + err.Error()
		}
		return 0, combined.String()
	case <-timer.C:
		_ = cmd.Process.Kill()
		err := <-done
		if err != nil {
			return 124, combined.String() + fmt.Sprintf("\n(timed out after %s)", timeout)
		}
		return 124, combined.String() + fmt.Sprintf("\n(timed out after %s)", timeout)
	}
}

func releaseShipWorktreeDir(root string, opts releaseShipOptions) (string, error) {
	if opts.worktreeDir != "" {
		if filepath.IsAbs(opts.worktreeDir) {
			return filepath.Clean(opts.worktreeDir), nil
		}
		return filepath.Join(root, opts.worktreeDir), nil
	}
	parent := os.TempDir()
	dir, err := releaseShipMkdirTemp(parent, "fak-release-ship-*")
	if err != nil {
		return "", err
	}
	return dir, nil
}

func releaseShipPython() string {
	if p := strings.TrimSpace(os.Getenv("FAK_PYTHON")); p != "" {
		return p
	}
	return "python"
}

func releaseShipScript(root, script string) string {
	return filepath.Join(root, "tools", script)
}

func releaseShipOwner() string {
	for _, key := range []string{"FAK_RELEASE_OWNER", "CLAUDE_CODE_SESSION_ID"} {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			return value
		}
	}
	return fmt.Sprintf("fak-release-ship-%d-%d", os.Getpid(), time.Now().UnixNano())
}

func releaseShipEnv(lockRoot, owner string) []string {
	env := os.Environ()
	env = setEnv(env, "FAK_RELEASE_LOCK_ROOT", lockRoot)
	env = setEnv(env, "FAK_RELEASE_OWNER", owner)
	return env
}

func setEnv(env []string, key, value string) []string {
	prefix := key + "="
	out := make([]string, 0, len(env)+1)
	found := false
	for _, item := range env {
		if strings.HasPrefix(item, prefix) {
			out = append(out, prefix+value)
			found = true
		} else {
			out = append(out, item)
		}
	}
	if !found {
		out = append(out, prefix+value)
	}
	return out
}

func (r *releaseShipResult) fail(kind, detail string) {
	r.Errors = append(r.Errors, kind+": "+tail(detail))
	r.OK = false
}

func finishReleaseShip(result releaseShipResult) releaseShipResult {
	if len(result.Errors) > 0 {
		result.OK = false
	}
	if len(result.CommandTail) == 0 {
		result.CommandTail = nil
	}
	if len(result.RemoteBranch) == 0 {
		result.RemoteBranch = nil
	}
	if len(result.ExecutedCommands) == 0 {
		result.ExecutedCommands = nil
	}
	return result
}

func finishReleaseShipWithCleanup(result *releaseShipResult, root string, opts releaseShipOptions, worktreeAdded bool) releaseShipResult {
	if worktreeAdded && !opts.keepWorktree {
		cleanup := cleanupReleaseShipWorktree(root, result.Worktree)
		result.Cleanup = cleanup
		if ok, _ := cleanup["ok"].(bool); !ok {
			result.Warnings = append(result.Warnings, "detached worktree cleanup failed")
		}
	} else if opts.keepWorktree && worktreeAdded {
		result.Cleanup = map[string]any{"ok": true, "kept": true, "path": result.Worktree}
	}
	return finishReleaseShip(*result)
}

func renderReleaseShip(stdout, stderr io.Writer, result releaseShipResult) {
	if result.OK {
		fmt.Fprintf(stdout, "release-ship: OK")
		if result.Tag != "" {
			fmt.Fprintf(stdout, " %s", result.Tag)
		}
		fmt.Fprintln(stdout)
	} else {
		fmt.Fprintf(stderr, "release-ship: REFUSED")
		if result.Tag != "" {
			fmt.Fprintf(stderr, " %s", result.Tag)
		}
		fmt.Fprintln(stderr)
	}
	if result.Worktree != "" {
		fmt.Fprintf(stdout, "  worktree: %s\n", result.Worktree)
	}
	if result.CommitSHA != "" {
		fmt.Fprintf(stdout, "  commit: %s\n", result.CommitSHA)
	}
	if result.RemoteBranch != nil {
		fmt.Fprintf(stdout, "  pushed: %s/%s %s\n", result.RemoteBranch["remote"], result.RemoteBranch["trunk"], result.RemoteBranch["sha"])
	}
	if result.Publish != nil {
		if gh, ok := result.Publish["github_release"].(map[string]any); ok {
			if url := stringFromAny(gh["url"]); url != "" {
				fmt.Fprintf(stdout, "  github: %s\n", url)
			} else if status := stringFromAny(gh["status"]); status != "" {
				fmt.Fprintf(stdout, "  github: %s\n", status)
			}
		}
	}
	for _, warning := range result.Warnings {
		fmt.Fprintf(stderr, "  warning: %s\n", warning)
	}
	for _, err := range result.Errors {
		fmt.Fprintf(stderr, "  ERROR: %s\n", err)
	}
}

func stringFromAny(value any) string {
	if s, ok := value.(string); ok {
		return s
	}
	return ""
}

func sameSHA(a, b string) bool {
	a = strings.TrimSpace(strings.ToLower(a))
	b = strings.TrimSpace(strings.ToLower(b))
	return a != "" && b != "" && (a == b || strings.HasPrefix(a, b) || strings.HasPrefix(b, a))
}

func jsonTail(value map[string]any) string {
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return tail(string(raw))
}

func tail(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= 500 {
		return s
	}
	return s[len(s)-500:]
}
