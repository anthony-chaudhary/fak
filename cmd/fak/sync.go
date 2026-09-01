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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

const (
	syncExitOK       = 0
	syncExitUsage    = 2
	syncExitRefused  = 3
	syncExitInternal = 4
)

var syncAheadAudit = defaultSyncAheadAudit
var syncWorktree = defaultSyncWorktree
var (
	syncAssess        = safesync.Assess
	syncSafePush      = safesync.SafePush
	syncCaptureSource = func(repo string) (string, error) { return gitOut(repo, "rev-parse", "HEAD") }
)

func runSync(stdout, stderr io.Writer, argv []string) int {
	command := "check"
	if len(argv) > 0 {
		switch argv[0] {
		case "check", "apply", "push", "drain":
			command = argv[0]
			argv = argv[1:]
		case "help", "-h", "--help":
			syncUsage(stdout)
			return syncExitOK
		default:
			if !strings.HasPrefix(argv[0], "-") {
				fmt.Fprintf(stderr, "fak sync: unknown command %q (want check, apply, push, or drain)\n", argv[0])
				syncUsage(stderr)
				return syncExitUsage
			}
		}
	}

	fs := flag.NewFlagSet("sync "+command, flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "sync")
	repo := fs.String("repo", ".", "repo path (default: cwd)")
	remote := fs.String("remote", "origin", "remote name")
	branch := fs.String("branch", "", "branch to sync (default: current branch)")
	fetch := fs.Bool("fetch", false, "git fetch <remote> <branch> before assessing")
	retries := fs.Int("retries", 3, "push: total attempts before giving up on a moving trunk")
	queueFile := fs.String("queue-file", "", "drain: stranded-commit queue file (default: <repo>/.fak/sync-drain-queue.json)")
	budgetDefault := 60 * time.Second
	budgetHelp := "drain: trunk-green build budget for the window verdict"
	if command == "push" {
		budgetDefault = safesync.DefaultPushVelocityBudget
		budgetHelp = "push: responsiveness budget used for the safety-qualified velocity score"
	}
	if command == "apply" {
		budgetDefault = safesync.DefaultPushVelocityBudget
		budgetHelp = "apply: responsiveness budget used for the effect-qualified velocity score"
	}
	budget := fs.Duration("budget", budgetDefault, budgetHelp)
	asJSON := fs.Bool("json", false, "emit the assessment as JSON")
	resumeToken := fs.String("resume-token", "", "check: operation-bound token emitted by a blocked PUBLIC_LEAK preflight")
	var recheckPaths pathList
	fs.Var(&recheckPaths, "recheck-path", "check: recheck PUBLIC_LEAK only for this repo-relative repair path (repeatable)")
	if err := fs.Parse(argv); err != nil {
		return syncExitUsage
	}
	if command != "check" && (*resumeToken != "" || len(recheckPaths) > 0) {
		fmt.Fprintf(stderr, "fak sync: --resume-token and --recheck-path are check-only options\n")
		return syncExitUsage
	}
	normalizedRecheckPaths, normalizeErr := normalizeSyncRecheckPaths(recheckPaths)
	if normalizeErr != nil {
		fmt.Fprintf(stderr, "fak sync: %v\n", normalizeErr)
		return syncExitUsage
	}
	recheckPaths = normalizedRecheckPaths

	// push is the push-side sibling of check/apply: a safe `git push` that retries a
	// transient non-fast-forward race (a peer landed between fetch and push, but HEAD
	// already contains origin) and stops with a clear next step when genuinely behind.
	if command == "push" {
		if err := validatePushVelocityBudget(*budget); err != nil {
			fmt.Fprintf(stderr, "fak sync: %v\n", err)
			return syncExitUsage
		}
		repoPath := pathutil.ExpandTilde(*repo)
		sourceSHA, err := syncCaptureSource(repoPath)
		if err != nil || strings.TrimSpace(sourceSHA) == "" {
			fmt.Fprintf(stderr, "fak sync: capture push source: %v\n", err)
			return syncExitInternal
		}
		res, err := syncSafePush(context.Background(), safesync.PushOptions{
			Repo:           repoPath,
			Remote:         *remote,
			Branch:         *branch,
			SourceRef:      strings.TrimSpace(sourceSHA),
			TargetRef:      syncTargetRef(*branch),
			MaxRetries:     *retries,
			VelocityBudget: *budget,
		})
		res = annotatePushWorktree(context.Background(), res, repoPath)
		if err != nil {
			if *asJSON {
				if writeErr := writeIndentedJSON(stdout, res); writeErr != nil {
					fmt.Fprintf(stderr, "fak sync: %v\n", writeErr)
					return syncExitInternal
				}
			} else {
				renderSyncPush(stdout, res)
			}
			fmt.Fprintf(stderr, "fak sync: %v\n", err)
			return syncExitInternal
		}
		if *asJSON {
			if err := writeIndentedJSON(stdout, res); err != nil {
				fmt.Fprintf(stderr, "fak sync: %v\n", err)
				return syncExitInternal
			}
		} else {
			renderSyncPush(stdout, res)
		}
		if res.Pushed {
			return syncExitOK
		}
		return syncExitRefused
	}

	// drain is the release valve for commits stranded by a red-trunk push refusal (#3617): it
	// queues them, polls the trunk-green quiescent window (reusing the pre-push witness), and
	// flushes in one push when green — backing off, not blind-retrying, while red.
	if command == "drain" {
		repoPath := pathutil.ExpandTilde(*repo)
		qp := *queueFile
		if strings.TrimSpace(qp) == "" {
			qp = filepath.Join(repoPath, ".fak", "sync-drain-queue.json")
		}
		return runSyncDrain(stdout, stderr, syncDrainConfig{
			repo:      repoPath,
			remote:    *remote,
			branch:    *branch,
			queuePath: qp,
			asJSON:    *asJSON,
			budget:    *budget,
		})
	}

	opts := safesync.Options{
		Repo:                pathutil.ExpandTilde(*repo),
		Remote:              *remote,
		Branch:              *branch,
		Fetch:               *fetch,
		ApplyVelocityBudget: *budget,
	}
	var (
		info safesync.Assessment
		err  error
	)
	if command == "apply" {
		info, err = safesync.Apply(context.Background(), opts)
	} else {
		info, err = syncAssess(context.Background(), opts)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak sync: %v\n", err)
		return syncExitInternal
	}
	if info.State == safesync.StateAhead {
		info = annotateAheadPushAudit(context.Background(), info, pathutil.ExpandTilde(*repo), *remote)
	}
	info = annotateSyncWorktree(context.Background(), info, pathutil.ExpandTilde(*repo))

	var publicLeak *syncPublicLeakPreflight
	if command == "check" {
		repoPath := pathutil.ExpandTilde(*repo)
		expectedToken := syncPublicLeakOperationToken(repoPath, *remote, info)
		if *resumeToken != "" && *resumeToken != expectedToken {
			fmt.Fprintln(stderr, "fak sync: PUBLIC_LEAK resume token does not match this repo, branch, HEAD, and remote target; rerun `fak sync check` to start a new operation")
			return syncExitRefused
		}
		preflight, scanErr := assessSyncPublicLeak(repoPath, *remote, info, recheckPaths)
		if scanErr != nil {
			fmt.Fprintf(stderr, "fak sync: PUBLIC_LEAK preflight: %v\n", scanErr)
			return syncExitInternal
		}
		preflight.ResumeValidated = *resumeToken != ""
		if len(preflight.Findings) > 0 || len(recheckPaths) > 0 || preflight.ResumeValidated {
			publicLeak = &preflight
		}
	}

	if *asJSON {
		var report any = info
		if publicLeak != nil {
			report = syncCheckReport{Assessment: info, PublicLeak: publicLeak}
		}
		if err := writeIndentedJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "fak sync: %v\n", err)
			return syncExitInternal
		}
	} else {
		renderSync(stdout, command, info)
		if publicLeak != nil {
			renderSyncPublicLeak(stdout, *publicLeak)
		}
	}

	// A targeted recheck answers only whether the named repair paths are now clean. It
	// deliberately does not reinterpret the branch-sync state; the emitted resume command
	// performs a fresh whole-candidate scan before returning to the original operation.
	if command == "check" && len(recheckPaths) > 0 {
		if publicLeak != nil && publicLeak.BlockingCount > 0 {
			return syncExitRefused
		}
		return syncExitOK
	}
	if publicLeak != nil && publicLeak.BlockingCount > 0 {
		return syncExitRefused
	}

	if info.State == safesync.StateInSync {
		return syncExitOK
	}
	if command == "apply" {
		if info.Applied {
			return syncExitOK
		}
		return syncExitRefused
	}
	if info.OK {
		return syncExitOK
	}
	return syncExitRefused
}

func syncTargetRef(branch string) string {
	branch = strings.TrimSpace(branch)
	if branch == "" {
		return ""
	}
	return "refs/heads/" + branch
}

func syncUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak sync [check] [--repo DIR] [--remote origin] [--branch B] [--fetch] [--json]
                        [--recheck-path PATH ...] [--resume-token TOKEN]
  fak sync apply   [--repo DIR] [--remote origin] [--branch B] [--fetch] [--json]
  fak sync push    [--repo DIR] [--remote origin] [--branch B] [--retries N] [--budget 5s] [--json]
  fak sync drain   [--repo DIR] [--remote origin] [--branch B] [--queue-file F] [--budget D] [--json]

Safe shared-trunk git for dirty worktrees. check is read-only except for optional
--fetch. It runs PUBLIC_LEAK before commit time, classifies candidate findings against
the remote baseline, and emits collision-safe repair slices, a targeted recheck, and an
operation-bound resume token. A resume always reruns the whole gate; the token cannot
bypass or soften it. apply runs the fast-forward only when every path Git would write is clean at
HEAD or already byte-identical to the remote-tracking version. push pushes the branch
and retries a TRANSIENT non-fast-forward race (a peer landed between fetch and push,
but HEAD already contains origin); on a genuine behind/diverged state it stops with a
clear integrate-then-push next step. Its velocity evidence scores only a published
safe push against the declared --budget; refusals/errors retain timing but are UNSCORED.
apply uses only git merge --ff-only
--no-autostash --no-overwrite-ignore against the immutable SHA that check assessed;
a last-moment worktree change is a refusal, never pre-cleaned. drain is the release valve for commits stranded by
a red-trunk push refusal: it queues the ahead commits, polls the trunk-green quiescent
window (reusing the pre-push build witness), flushes in one push when green, and backs
off — not blind-retries — while red. None of these run git pull, stash, reset --hard,
clean, add, a non-fast-forward merge, or --force.
`)
}

// renderSyncPush is the human view of a SafePush outcome.
func renderSyncPush(w io.Writer, res safesync.PushResult) {
	if res.Pushed {
		attempts := "1 attempt"
		if res.Attempts != 1 {
			attempts = fmt.Sprintf("%d attempts", res.Attempts)
		}
		fmt.Fprintf(w, "pushed %s -> %s/%s (%s)\n", res.Branch, res.Remote, res.Branch, attempts)
		renderSyncPushVelocity(w, res.Velocity)
		renderWorktree(w, res.Worktree)
		return
	}
	label := "REFUSED"
	if res.Reason == safesync.PushReasonInternal {
		label = "ERROR"
	}
	fmt.Fprintf(w, "[%s] not pushed (%s): %s\n", label, res.Reason, res.Detail)
	renderSyncPushVelocity(w, res.Velocity)
	renderWorktree(w, res.Worktree)
}

func renderSync(w io.Writer, command string, info safesync.Assessment) {
	defer func() {
		if command == "apply" {
			renderSyncApplyVelocity(w, info.ApplyVelocity)
		}
	}()
	switch info.State {
	case safesync.StateInSync:
		fmt.Fprintln(w, "in sync: local branch already matches the remote; nothing to do")
		renderSyncWorktree(w, info)
	case safesync.StateAhead:
		fmt.Fprintf(w, "%s: %s\n", info.State, info.Reason)
		if info.PushAudit != nil && !info.PushAudit.OK {
			fmt.Fprintf(w, "  pre-push audit: BLOCKED (%d residual claim(s) in %s)\n", len(info.PushAudit.Residuals), info.PushAudit.Range)
			for _, r := range info.PushAudit.Residuals {
				subject := r.Subject
				if subject == "" {
					subject = "(subject unavailable)"
				}
				fmt.Fprintf(w, "    RESIDUAL  %s  %s  %s\n", short(r.SHA), r.Witness, subject)
				if r.Reason != "" {
					fmt.Fprintf(w, "              %s\n", r.Reason)
				}
			}
		}
		renderSyncWorktree(w, info)
	case safesync.StateDiverged, safesync.StateNoRemoteRef:
		fmt.Fprintf(w, "%s: %s\n", info.State, info.Reason)
		renderSyncWorktree(w, info)
	case safesync.StateBehind:
		status := "REFUSED"
		if info.Applied {
			status = "applied"
		} else if info.OK && command == "check" {
			status = "SAFE"
		}
		fmt.Fprintf(w, "[%s] behind %s: %d fast-forward path(s), %d identical, %d divergent\n",
			status, info.TargetRef, info.WriteCount, len(info.Identical), len(info.Divergent))
		if info.Reason != "" {
			fmt.Fprintf(w, "  %s\n", info.Reason)
		}
		for _, e := range info.Divergent {
			fmt.Fprintf(w, "    DIVERGES  %s  %s\n", e.Status, e.Path)
		}
		if info.Applied {
			fmt.Fprintf(w, "  HEAD -> %s (novel local work on other paths preserved)\n", short(info.NewHead))
		}
		renderSyncWorktree(w, info)
	default:
		fmt.Fprintf(w, "%s: %s\n", info.State, info.Reason)
		renderSyncWorktree(w, info)
	}
}

func renderSyncWorktree(w io.Writer, info safesync.Assessment) {
	renderWorktree(w, info.Worktree)
}

func renderWorktree(w io.Writer, wt *safesync.Worktree) {
	if wt == nil || !wt.Dirty {
		return
	}
	fmt.Fprintf(w, "worktree dirty: %d path(s) across %d lane(s), %d no-lane, %d junk\n",
		wt.TotalDirty, wt.Lanes, wt.NoLane, wt.Junk)
	if len(wt.JunkPaths) > 0 {
		fmt.Fprintf(w, "  junk: %s\n", pathPreview(wt.JunkPaths, 5))
	}
	if wt.NextAction != "" {
		fmt.Fprintf(w, "  next: %s\n", wt.NextAction)
	}
}

func pathPreview(paths []string, limit int) string {
	if limit <= 0 || len(paths) <= limit {
		return strings.Join(paths, ", ")
	}
	return fmt.Sprintf("%s, +%d more", strings.Join(paths[:limit], ", "), len(paths)-limit)
}

func annotateSyncWorktree(ctx context.Context, info safesync.Assessment, repo string) safesync.Assessment {
	wt := lookupSyncWorktree(ctx, repo)
	if wt == nil {
		return info
	}
	info.Worktree = wt
	return info
}

func annotatePushWorktree(ctx context.Context, res safesync.PushResult, repo string) safesync.PushResult {
	wt := lookupSyncWorktree(ctx, repo)
	if wt == nil {
		return res
	}
	res.Worktree = wt
	return res
}

func lookupSyncWorktree(ctx context.Context, repo string) *safesync.Worktree {
	wt, ok := syncWorktree(ctx, repo)
	if !ok || wt.TotalDirty == 0 {
		return nil
	}
	return &wt
}

func defaultSyncWorktree(ctx context.Context, repo string) (safesync.Worktree, bool) {
	entries, err := gitStatusDirty(ctx, repo)
	if err != nil {
		return safesync.Worktree{}, false
	}
	if len(entries) == 0 {
		return safesync.Worktree{}, true
	}
	plan := classifyDirty(entries, hooksLaneResolver(repo), originProbeFor(ctx, repo))
	return safesync.Worktree{
		Dirty:        true,
		TotalDirty:   plan.TotalDirty,
		Stampable:    stampableCount(plan),
		Lanes:        len(plan.Groups),
		NoLane:       len(plan.NoLane),
		Junk:         len(plan.Junk),
		JunkPaths:    sweepEntryPaths(plan.Junk),
		OldestPath:   plan.OldestDirtyPath,
		OldestAgeSec: plan.OldestDirtyAgeSeconds,
		NextAction:   plan.NextAction,
	}, true
}

func sweepEntryPaths(entries []sweepEntry) []string {
	if len(entries) == 0 {
		return nil
	}
	paths := make([]string, 0, len(entries))
	for _, e := range entries {
		paths = append(paths, e.Path)
	}
	return paths
}

func annotateAheadPushAudit(ctx context.Context, info safesync.Assessment, repo, remote string) safesync.Assessment {
	audit, ok := syncAheadAudit(ctx, repo, info.TargetRef)
	if !ok {
		return info
	}
	info.PushAudit = &audit
	if !audit.OK && len(audit.Residuals) > 0 {
		branch := info.Branch
		if branch == "" {
			branch = "main"
		}
		if remote == "" {
			remote = "origin"
		}
		info.Reason = fmt.Sprintf("local branch is ahead of remote, but the pre-push audit would block on %d residual claim(s); repair or get an operator decision before running `fak sync push --remote %s --branch %s`", len(audit.Residuals), remote, branch)
	}
	return info
}

type syncCommitAuditRow struct {
	SHA       string `json:"sha"`
	Verdict   string `json:"verdict"`
	ClaimKind string `json:"claim_kind"`
	Witness   string `json:"witness"`
	Reason    string `json:"reason"`
}

func defaultSyncAheadAudit(ctx context.Context, repo, targetRef string) (safesync.PushAudit, bool) {
	if strings.TrimSpace(repo) == "" {
		repo = "."
	}
	if _, err := os.Stat(filepath.Join(repo, "dos.toml")); err != nil {
		return safesync.PushAudit{}, false
	}
	if _, err := exec.LookPath("dos"); err != nil {
		return safesync.PushAudit{}, false
	}
	rangeSpec := targetRef + "..HEAD"
	cmd := exec.CommandContext(ctx, "dos", "commit-audit", "--json", rangeSpec)
	cmd.Dir = repo
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		if exit, ok := err.(*exec.ExitError); ok && len(out) == 0 {
			out = exit.Stderr
		}
	}
	rows, ok := parseSyncCommitAuditRows(out)
	if !ok {
		return safesync.PushAudit{}, false
	}
	audit := safesync.PushAudit{OK: true, Range: rangeSpec}
	for _, row := range rows {
		if row.Verdict != "CLAIM_UNWITNESSED" {
			continue
		}
		audit.OK = false
		audit.Residuals = append(audit.Residuals, safesync.PushAuditResidual{
			SHA:       row.SHA,
			Subject:   syncCommitSubject(ctx, repo, row.SHA),
			Verdict:   row.Verdict,
			ClaimKind: row.ClaimKind,
			Witness:   row.Witness,
			Reason:    row.Reason,
		})
	}
	return audit, true
}

func parseSyncCommitAuditRows(raw []byte) ([]syncCommitAuditRow, bool) {
	raw = []byte(strings.TrimSpace(string(raw)))
	if len(raw) == 0 {
		return nil, false
	}
	var rows []syncCommitAuditRow
	if err := json.Unmarshal(raw, &rows); err == nil {
		return rows, true
	}
	var one syncCommitAuditRow
	if err := json.Unmarshal(raw, &one); err == nil {
		return []syncCommitAuditRow{one}, true
	}
	return nil, false
}

func syncCommitSubject(ctx context.Context, repo, sha string) string {
	if strings.TrimSpace(sha) == "" {
		return ""
	}
	cmd := exec.CommandContext(ctx, "git", "log", "-1", "--format=%s", sha)
	cmd.Dir = repo
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
