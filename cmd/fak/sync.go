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

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

const (
	syncExitOK       = 0
	syncExitUsage    = 2
	syncExitRefused  = 3
	syncExitInternal = 4
)

func cmdSync(argv []string) { os.Exit(runSync(os.Stdout, os.Stderr, argv)) }

var syncAheadAudit = defaultSyncAheadAudit
var syncWorktree = defaultSyncWorktree

func runSync(stdout, stderr io.Writer, argv []string) int {
	command := "check"
	if len(argv) > 0 {
		switch argv[0] {
		case "check", "apply", "push":
			command = argv[0]
			argv = argv[1:]
		case "help", "-h", "--help":
			syncUsage(stdout)
			return syncExitOK
		default:
			if !strings.HasPrefix(argv[0], "-") {
				fmt.Fprintf(stderr, "fak sync: unknown command %q (want check, apply, or push)\n", argv[0])
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
	asJSON := fs.Bool("json", false, "emit the assessment as JSON")
	if err := fs.Parse(argv); err != nil {
		return syncExitUsage
	}

	// push is the push-side sibling of check/apply: a safe `git push` that retries a
	// transient non-fast-forward race (a peer landed between fetch and push, but HEAD
	// already contains origin) and stops with a clear next step when genuinely behind.
	if command == "push" {
		repoPath := pathutil.ExpandTilde(*repo)
		res, err := safesync.SafePush(context.Background(), safesync.PushOptions{
			Repo:       repoPath,
			Remote:     *remote,
			Branch:     *branch,
			MaxRetries: *retries,
		})
		if err != nil {
			fmt.Fprintf(stderr, "fak sync: %v\n", err)
			return syncExitInternal
		}
		res = annotatePushWorktree(context.Background(), res, repoPath)
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

	opts := safesync.Options{
		Repo:   pathutil.ExpandTilde(*repo),
		Remote: *remote,
		Branch: *branch,
		Fetch:  *fetch,
	}
	var (
		info safesync.Assessment
		err  error
	)
	if command == "apply" {
		info, err = safesync.Apply(context.Background(), opts)
	} else {
		info, err = safesync.Assess(context.Background(), opts)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak sync: %v\n", err)
		return syncExitInternal
	}
	if info.State == safesync.StateAhead {
		info = annotateAheadPushAudit(context.Background(), info, pathutil.ExpandTilde(*repo), *remote)
	}
	info = annotateSyncWorktree(context.Background(), info, pathutil.ExpandTilde(*repo))

	if *asJSON {
		if err := writeIndentedJSON(stdout, info); err != nil {
			fmt.Fprintf(stderr, "fak sync: %v\n", err)
			return syncExitInternal
		}
	} else {
		renderSync(stdout, command, info)
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

func syncUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak sync [check] [--repo DIR] [--remote origin] [--branch B] [--fetch] [--json]
  fak sync apply   [--repo DIR] [--remote origin] [--branch B] [--fetch] [--json]
  fak sync push    [--repo DIR] [--remote origin] [--branch B] [--retries N] [--json]

Safe shared-trunk git for dirty worktrees. check is read-only except for optional
--fetch. apply runs the fast-forward only when every path Git would write is clean at
HEAD or already byte-identical to the remote-tracking version. push pushes the branch
and retries a TRANSIENT non-fast-forward race (a peer landed between fetch and push,
but HEAD already contains origin); on a genuine behind/diverged state it stops with a
clear integrate-then-push next step. None of these run git pull, stash, reset --hard,
clean, add, merge, or --force.
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
		renderWorktree(w, res.Worktree)
		return
	}
	fmt.Fprintf(w, "[REFUSED] not pushed (%s): %s\n", res.Reason, res.Detail)
	renderWorktree(w, res.Worktree)
}

func renderSync(w io.Writer, command string, info safesync.Assessment) {
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
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
