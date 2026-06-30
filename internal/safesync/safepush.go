package safesync

import (
	"context"
	"strconv"
	"strings"
)

// safepush.go — the SAFE PUSH retry for the hot shared trunk, the push-side sibling
// of Assess/Apply. `git push` to a constantly-moving trunk is routinely rejected
// non-fast-forward because a peer landed between your last fetch and your push — even
// when your local HEAD already CONTAINS origin (a TRANSIENT race that clears on a
// re-fetch + re-push, exactly as observed by hand on this trunk). SafePush wraps that
// dance: push; on a non-ff rejection, fetch and RE-CLASSIFY HEAD vs the remote ref;
// if the remote is now an ANCESTOR of HEAD (we are strictly ahead — the race), retry
// the push; if we are genuinely BEHIND/DIVERGED, STOP with a clear integrate-then-push
// next step rather than risk an auto-merge into a dirty shared tree. It NEVER
// force-pushes, merges, resets, stashes, or autostashes — every action is
// non-destructive, the same discipline Apply holds on the pull side.

// PushDivergence classifies HEAD vs the remote ref after a fetch, for the push retry.
type PushDivergence string

const (
	// PushAhead: the remote ref is an ancestor of HEAD (or equal) — HEAD already
	// contains it, so the non-ff rejection was a transient race and a re-push is safe.
	PushAhead PushDivergence = "ahead"
	// PushBehind: HEAD is an ancestor of the remote ref — integrate it first.
	PushBehind PushDivergence = "behind"
	// PushDiverged: neither is an ancestor — both moved; integrate first.
	PushDiverged PushDivergence = "diverged"
)

// PushAction is the next step the retry loop takes after a non-ff rejection.
type PushAction string

const (
	PushRetry PushAction = "retry" // transient race — HEAD already contains the remote; re-push
	PushStop  PushAction = "stop"  // genuine divergence — integrate in place, never auto-merge here
)

// DecidePush is the PURE core of the retry: given the post-fetch divergence, choose
// whether to re-push (the rejection was a race) or stop (real integration needed). It
// is exported and pure so the policy is unit-tested without a git remote.
func DecidePush(div PushDivergence) PushAction {
	if div == PushAhead {
		return PushRetry
	}
	return PushStop
}

// PushOptions configures SafePush.
type PushOptions struct {
	Repo       string
	Remote     string
	Branch     string // default: current branch
	MaxRetries int    // total push attempts; default 3
	Runner     Runner `json:"-"`
}

// Push reason constants for PushResult.Reason ("" means pushed).
const (
	PushReasonBehind     = "BEHIND"            // genuinely behind/diverged — integrate then re-push
	PushReasonError      = "PUSH_ERROR"        // a rejection that is NOT non-fast-forward (hook/auth/network)
	PushReasonExhausted  = "RETRIES_EXHAUSTED" // still racing after MaxRetries — the trunk is moving fast
	PushReasonGitMissing = "GIT_UNAVAILABLE"   // git/fetch could not run
)

// PushResult is the structured outcome of SafePush.
type PushResult struct {
	Pushed     bool   `json:"pushed"`
	Attempts   int    `json:"attempts"`
	Branch     string `json:"branch,omitempty"`
	Remote     string `json:"remote,omitempty"`
	Reason     string `json:"reason,omitempty"`     // "" | one of the PushReason* constants
	Divergence string `json:"divergence,omitempty"` // last classified divergence on a non-ff
	Detail     string `json:"detail,omitempty"`
}

// SafePush pushes repo's branch to remote, retrying a TRANSIENT non-ff rejection (a
// re-fetch shows HEAD already contains the remote) up to MaxRetries times. A genuine
// behind/diverged state returns Reason=BEHIND with a clear integrate-then-push next
// step; it never integrates for you. Non-destructive: only push + fetch + read-only
// merge-base; never force/merge/reset/stash. err is returned only when a read-only git
// query (branch resolution / merge-base) cannot run; recoverable push outcomes are
// reported through PushResult.Reason.
func SafePush(ctx context.Context, opts PushOptions) (PushResult, error) {
	run := opts.Runner
	if run == nil {
		run = RealRunner
	}
	repo := strings.TrimSpace(opts.Repo)
	if repo == "" {
		repo = "."
	}
	remote := strings.TrimSpace(opts.Remote)
	if remote == "" {
		remote = "origin"
	}
	branch := strings.TrimSpace(opts.Branch)
	if branch == "" {
		b, err := currentBranch(ctx, run, repo)
		if err != nil {
			return PushResult{}, err
		}
		branch = b
	}
	max := opts.MaxRetries
	if max <= 0 {
		max = 3
	}
	res := PushResult{Branch: branch, Remote: remote}
	remoteRef := remote + "/" + branch

	for attempt := 1; attempt <= max; attempt++ {
		res.Attempts = attempt
		pr := run(ctx, repo, "push", remote, branch)
		if pr.Err != nil {
			res.Reason = PushReasonGitMissing
			res.Detail = pr.Err.Error()
			return res, nil
		}
		if pr.Code == 0 {
			res.Reason = ""
			res.Pushed = true
			return res, nil
		}
		msg := runDetail(pr)
		if !isNonFastForward(msg) {
			res.Reason = PushReasonError
			res.Detail = pushFirstLine(msg)
			return res, nil
		}
		// Non-ff: fetch the remote ref, then re-classify HEAD against it.
		if fr := run(ctx, repo, "fetch", remote, branch); fr.Err != nil || fr.Code != 0 {
			res.Reason = PushReasonGitMissing
			res.Detail = "fetch " + remoteRef + " failed: " + pushFirstLine(runDetail(fr))
			return res, nil
		}
		div, err := classifyPushDivergence(ctx, run, repo, remoteRef)
		if err != nil {
			return res, err
		}
		res.Divergence = string(div)
		if DecidePush(div) == PushStop {
			res.Reason = PushReasonBehind
			res.Detail = "behind " + remoteRef + "; integrate in place (`fak sync apply` or `git merge " + remoteRef + "`) then re-run — never force-push"
			return res, nil
		}
		// PushRetry: the rejection was a race (HEAD already contains the remote); loop.
	}
	res.Reason = PushReasonExhausted
	res.Detail = "push still rejected after " + strconv.Itoa(max) + " attempts; the trunk is moving fast — retry shortly"
	return res, nil
}

// classifyPushDivergence compares HEAD to the (already fetched) remote ref.
func classifyPushDivergence(ctx context.Context, run Runner, repo, remoteRef string) (PushDivergence, error) {
	remoteInHead, err := isAncestor(ctx, run, repo, remoteRef, "HEAD")
	if err != nil {
		return "", err
	}
	if remoteInHead {
		return PushAhead, nil // remote is an ancestor of HEAD (or equal): the rejection was a race
	}
	headInRemote, err := isAncestor(ctx, run, repo, "HEAD", remoteRef)
	if err != nil {
		return "", err
	}
	if headInRemote {
		return PushBehind, nil
	}
	return PushDiverged, nil
}

// isNonFastForward reports whether git push output is a non-fast-forward rejection (a
// peer moved the ref) — the only class SafePush retries — as opposed to a hook refusal,
// an auth failure, or a network error, which must surface as-is.
func isNonFastForward(out string) bool {
	l := strings.ToLower(out)
	switch {
	case strings.Contains(l, "non-fast-forward"):
		return true
	case strings.Contains(l, "[rejected]") && (strings.Contains(l, "fetch first") || strings.Contains(l, "behind")):
		return true
	case strings.Contains(l, "updates were rejected because the"):
		return true
	default:
		return false
	}
}

// runDetail returns the stderr (or stdout fallback) of a RunResult, trimmed.
func runDetail(r RunResult) string {
	d := strings.TrimSpace(string(r.Stderr))
	if d == "" {
		d = strings.TrimSpace(string(r.Stdout))
	}
	return d
}

// pushFirstLine returns the first non-empty line of s (push rejections are multi-line;
// the headline is the actionable part for a one-line CLI/JSON detail).
func pushFirstLine(s string) string {
	for _, ln := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(ln); t != "" {
			return t
		}
	}
	return ""
}
