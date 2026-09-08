package safecommit

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// runLockRidingMutation runs one lock-riding git mutation (the pathspec-scoped add, then
// the commit — extracted verbatim from CommitWith, shared by both steps) and maps a
// non-zero exit to the transient-vs-permanent split: a peer's raw git holding index.lock
// (or a ref lock) is TRANSIENT contention — after the in-place retries are spent it
// surfaces as the retryable LOCK_BUSY, never the halt-class HOOK_REFUSED, which is
// reserved for a genuine hook refusal. The returned err is non-nil only when git itself
// could not be executed; on success both reason and err are empty.
func runLockRidingMutation(ctx context.Context, run Runner, dir string, args []string) (reason, detail string, err error) {
	if ctx.Err() != nil {
		return ReasonCommitStalled, "git mutation aborted: " + ctx.Err().Error(), nil
	}
	out, code, rerr := runRidingLockContention(ctx, run, dir, args...)
	if ctx.Err() != nil && (rerr != nil || code != 0) {
		return ReasonCommitStalled, "git mutation timed out: " + ctx.Err().Error(), nil
	}
	// Stale-index-lock auto-recovery (#3915): contention that outlives the in-place
	// retries and names the INDEX lock may be a crashed writer's abandoned lock,
	// which never clears on its own and would otherwise force a manual `rm`. Reap it
	// when it is provably stale and retry once; when it is fresh (a live git may
	// hold it), report a precise "another git process is active" message instead of
	// git's generic crash text. Any other failure class is unaffected.
	if rerr == nil && code != 0 && isIndexLockContention(out) {
		if r, d, e, handled := recoverStaleIndexLock(ctx, run, dir, args); handled {
			return r, d, e
		}
	}
	return classifyMutation(out, code, rerr)
}

// classifyMutation maps a finished git mutation (out, exit code, exec err) to the
// safecommit reason/detail split: an exec failure is an infrastructure error; a
// clean exit is success; a lock-contention non-zero is the retryable LOCK_BUSY;
// anything else is the halt-class HOOK_REFUSED.
func classifyMutation(out string, code int, rerr error) (reason, detail string, err error) {
	if rerr != nil {
		if errors.Is(rerr, context.DeadlineExceeded) || errors.Is(rerr, context.Canceled) {
			return ReasonCommitStalled, "git mutation timed out: " + rerr.Error(), nil
		}
		return "", "", fmt.Errorf("safecommit: git not executable: %w", rerr)
	}
	if code == 0 {
		return "", "", nil
	}
	if isGitLockContention(out) {
		return ReasonLockBusy, trimDetail(out), nil
	}
	return ReasonHookRefused, trimDetail(out), nil
}

func pushVerifiedCommit(ctx context.Context, run Runner, dir, trunk, sha string) (safesync.PushResult, error) {
	sha = strings.TrimSpace(sha)
	if sha == "" {
		return safesync.PushResult{Reason: safesync.PushReasonError, Detail: "verified commit SHA missing; not pushed"}, nil
	}
	remote := gitConfigValue(ctx, run, dir, "branch."+trunk+".remote")
	if remote == "" {
		remote = "origin"
	}
	mergeRef := gitConfigValue(ctx, run, dir, "branch."+trunk+".merge")
	if mergeRef == "" {
		mergeRef = "refs/heads/" + trunk
	}
	return safesync.SafePush(ctx, safesync.PushOptions{
		Repo:      dir,
		Remote:    remote,
		Branch:    branchFromMergeRef(mergeRef, trunk),
		SourceRef: sha,
		TargetRef: mergeRef,
		Runner:    safeSyncRunner(run),
	})
}

// applyVerifiedPush performs step (8): the optional push of the already-verified commit. It
// pushes only when opts.Push is set, by exact SHA refspec through pushVerifiedCommit, and maps
// a rejected push to ReasonPushRejected (a value, never a force-push). When the push is rejected
// due to safe disjoint divergence and !opts.DisableAutoReconcile, it attempts auto-reconciliation
// through safesync (#12078).
func applyVerifiedPush(ctx context.Context, run Runner, opts Options, trunk string, res Result) (Result, error) {
	if opts.Push {
		pushed, err := pushVerifiedCommit(ctx, run, opts.Dir, trunk, res.SHA)
		if err != nil {
			return res, err
		}
		if !pushed.Pushed {
			isDisjoint := pushed.Reason == safesync.ReasonDivergedDisjoint ||
				(pushed.Divergence == string(safesync.PushDiverged) && pushed.Reason == safesync.ReasonDivergedDisjoint)
			if isDisjoint && !opts.DisableAutoReconcile {
				remote := gitConfigValue(ctx, run, opts.Dir, "branch."+trunk+".remote")
				if remote == "" {
					remote = "origin"
				}
				branch := branchFromMergeRef(gitConfigValue(ctx, run, opts.Dir, "branch."+trunk+".merge"), trunk)
				pktOpts := safesync.PacketOptions{
					Repo:    opts.Dir,
					Remote:  remote,
					Branch:  branch,
					Runner:  safeSyncRunner(run),
					Session: opts.SessionID,
				}
				pkt, pktErr := safesync.BuildReconciliationPacket(ctx, pktOpts)
				if pktErr == nil && pkt != nil && pkt.Dispatchable &&
					pkt.Disposition == safesync.DispositionSafeDisjoint {
					execOpts := safesync.ExecuteOptions{
						Repo:           opts.Dir,
						Remote:         remote,
						Branch:         branch,
						Runner:         safeSyncRunner(run),
						WriterLeaseTTL: safesync.DefaultWriterLeaseTTL,
						Session:        opts.SessionID,
					}
					receipt, _ := safesync.ExecutePacket(ctx, pkt, execOpts)
					if receipt != nil && receipt.Pushed {
						res.Pushed = true
						return res, nil
					}
				}
			}
			res.Reason = ReasonPushRejected
			res.Detail = trimDetail(pushed.Detail)
			if res.Detail == "" {
				res.Detail = pushed.Reason
			}
			return res, nil
		}
		res.Pushed = true
	}

	return res, nil
}

func safeSyncRunner(run Runner) safesync.Runner {
	return func(ctx context.Context, repo string, args ...string) safesync.RunResult {
		out, code, err := run(ctx, repo, args...)
		b := []byte(out)
		return safesync.RunResult{Stdout: b, Stderr: b, Code: code, Err: err}
	}
}

func branchFromMergeRef(mergeRef, fallback string) string {
	const prefix = "refs/heads/"
	mergeRef = strings.TrimSpace(mergeRef)
	if branch, ok := strings.CutPrefix(mergeRef, prefix); ok && strings.TrimSpace(branch) != "" {
		return branch
	}
	return fallback
}

func gitConfigValue(ctx context.Context, run Runner, dir, key string) string {
	out, code, err := run(ctx, dir, "config", "--get", key)
	if err != nil || code != 0 {
		return ""
	}
	return strings.TrimSpace(out)
}

// headSHA runs `git rev-parse HEAD` in dir and returns the trimmed SHA. A non-zero exit
// is not an error — an empty repo has no HEAD yet — it simply yields "". A failure to
// exec git itself is the infrastructure error the caller returns as-is.
func headSHA(ctx context.Context, run Runner, dir string) (string, error) {
	head, code, err := run(ctx, dir, "rev-parse", "HEAD")
	if err != nil {
		return "", fmt.Errorf("safecommit: git not executable: %w", err)
	}
	if code != 0 {
		return "", nil
	}
	return strings.TrimSpace(head), nil
}
