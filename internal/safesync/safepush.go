package safesync

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
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
	Repo           string
	Remote         string
	Branch         string // default: current branch
	SourceRef      string // optional exact source to push, e.g. a verified commit SHA
	TargetRef      string // optional destination ref when SourceRef is set; default refs/heads/<branch>
	MaxRetries     int    // total push attempts; default 3
	VelocityBudget time.Duration
	Runner         Runner           `json:"-"`
	Now            func() time.Time `json:"-"` // injectable end-to-end wall clock
}

// DefaultPushVelocityBudget is the declared responsiveness SLO used when a
// caller does not supply one. It is an operator budget, not a comparative
// performance claim or a correctness threshold.
const DefaultPushVelocityBudget = 5 * time.Second

// Push reason constants for PushResult.Reason ("" means pushed).
const (
	// PushReasonBehind is a compatibility alias for ReasonBehindFastForwardable.
	// Deprecated: use ReasonBehindFastForwardable or the matching typed divergence reason.
	PushReasonBehind      = ReasonBehindFastForwardable // genuinely behind/diverged — integrate then re-push
	PushReasonError       = "PUSH_ERROR"                // a rejection that is NOT non-fast-forward or transient network (hook/auth)
	PushReasonExhausted   = "RETRIES_EXHAUSTED"         // still racing after MaxRetries — the trunk is moving fast
	PushReasonGitMissing  = "GIT_UNAVAILABLE"           // git/fetch could not run
	PushReasonUnreachable = "REMOTE_UNREACHABLE"        // a transient network failure persisted through every retry
	PushReasonCancelled   = "CANCELLED"                 // the caller's ctx was cancelled mid-backoff; no further attempt was made
	PushReasonInternal    = "INTERNAL_ERROR"            // a read-only Git classification/query failed; effect is indeterminate
)

// PushResult is the structured outcome of SafePush.
type PushResult struct {
	Pushed     bool         `json:"pushed"`
	Attempts   int          `json:"attempts"`
	Branch     string       `json:"branch,omitempty"`
	Remote     string       `json:"remote,omitempty"`
	Reason     string       `json:"reason,omitempty"`     // "" | one of the PushReason* constants
	Divergence string       `json:"divergence,omitempty"` // last classified divergence on a non-ff
	Detail     string       `json:"detail,omitempty"`
	Worktree   *Worktree    `json:"worktree,omitempty"`
	Velocity   PushVelocity `json:"velocity"`
}

// PushVelocity is end-to-end, safety-qualified push timing. A numeric score is
// present only when SafePush actually published the requested ref. Refusals and
// errors keep their elapsed/budget evidence but are UNSCORED, so a 1ms failure
// can never look like high ship velocity.
type PushVelocity struct {
	Qualified   bool     `json:"qualified"`
	ElapsedMS   int64    `json:"elapsed_ms"`
	BudgetMS    int64    `json:"budget_ms"`
	BudgetRatio float64  `json:"budget_ratio"`
	Score       *int     `json:"score"`
	Grade       string   `json:"grade"`
	Notes       []string `json:"notes"`
}

// SafePush pushes repo's branch (or SourceRef:TargetRef when SourceRef is set) to remote,
// retrying a TRANSIENT non-ff rejection (a re-fetch shows the pushed source already
// contains the remote) up to MaxRetries times. A genuine
// behind/diverged state returns Reason=BEHIND with a clear integrate-then-push next
// step; it never integrates for you. Non-destructive: only push + fetch + read-only
// merge-base; never force/merge/reset/stash. err is returned only when a read-only git
// query (branch resolution / merge-base) cannot run; recoverable push outcomes are
// reported through PushResult.Reason.
func SafePush(ctx context.Context, opts PushOptions) (res PushResult, err error) {
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	budget := opts.VelocityBudget
	if budget == 0 {
		budget = DefaultPushVelocityBudget
	}
	scoreBudget := budget
	if scoreBudget < time.Millisecond {
		scoreBudget = time.Millisecond
	}
	started := now()
	defer func() {
		elapsed := now().Sub(started)
		if elapsed < 0 {
			elapsed = 0
		}
		if err != nil && res.Reason == "" {
			res.Reason = PushReasonInternal
			res.Detail = err.Error()
		}
		res.Velocity = ScorePushVelocity(res, elapsed, scoreBudget, err)
	}()
	if budget < time.Millisecond {
		return res, fmt.Errorf("push velocity budget must be at least 1ms")
	}
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
	res = PushResult{Branch: branch, Remote: remote}
	remoteRef := remote + "/" + branch
	pushArgs := safePushArgs(remote, branch, opts.SourceRef, opts.TargetRef)
	compareRef := safePushCompareRef(opts.SourceRef)

	lastNetDetail := "" // non-empty when the most recent failure was transient network
	for attempt := 1; attempt <= max; attempt++ {
		res.Attempts = attempt
		pr := run(ctx, repo, pushArgs...)
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
		if isTransientPushNetwork(msg) {
			// A network blip (hung-up remote, timeout, DNS wobble, upstream 5xx) is as
			// transient as the non-ff race — retry after a backoff. No fetch first: if
			// the network is down the fetch fails too and would misreport GIT_UNAVAILABLE.
			lastNetDetail = pushFirstLine(msg)
			if attempt < max {
				if werr := pushBackoff(ctx, attempt); werr != nil {
					return cancelledResult(res, werr), nil
				}
			}
			continue
		}
		if !isNonFastForward(msg) {
			// A long pre-push hook can lose a race in the useful direction: another
			// worker publishes this shared-trunk tip while our hook is still running,
			// after which git reports the hook's stale failure even though the requested
			// source is now on the remote. Reconcile once before calling that a refusal.
			if fr := run(ctx, repo, "fetch", remote, branch); fr.Err == nil && fr.Code == 0 {
				remoteContainsSource, aerr := isAncestor(ctx, run, repo, compareRef, remoteRef)
				if aerr == nil && remoteContainsSource {
					res.Pushed = true
					res.Detail = "remote contains requested ref after concurrent publication"
					return res, nil
				}
			}
			res.Reason = PushReasonError
			res.Detail = pushHeadline(msg)
			return res, nil
		}
		lastNetDetail = ""
		// Non-ff: fetch the remote ref, then re-classify the pushed source against it.
		if fr := run(ctx, repo, "fetch", remote, branch); fr.Err != nil || fr.Code != 0 {
			fmsg := runDetail(fr)
			if fr.Err == nil && isTransientPushNetwork(fmsg) {
				// The fetch lost the same network blip; ride it out like the push.
				lastNetDetail = pushFirstLine(fmsg)
				if attempt < max {
					if werr := pushBackoff(ctx, attempt); werr != nil {
						return cancelledResult(res, werr), nil
					}
				}
				continue
			}
			res.Reason = PushReasonGitMissing
			res.Detail = "fetch " + remoteRef + " failed: " + pushFirstLine(fmsg)
			return res, nil
		}
		div, err := classifyPushDivergence(ctx, run, repo, compareRef, remoteRef)
		if err != nil {
			return res, err
		}
		res.Divergence = string(div)
		if DecidePush(div) == PushStop {
			switch div {
			case PushBehind:
				res.Reason = ReasonBehindFastForwardable
				res.Detail = "behind " + remoteRef + "; run `fak sync apply --fetch --remote " + remote + " --branch " + branch + "` to fast-forward only when the write set is clean, then re-run `fak sync push`; never force-push, stash, reset, or raw-merge peer work"
			case PushDiverged:
				res.Reason = classifyDivergedPaths(ctx, run, repo, compareRef, remoteRef)
				if res.Reason == ReasonDivergedDisjoint {
					res.Detail = "diverged from " + remoteRef + " with disjoint paths; run `fak sync check --fetch --remote " + remote + " --branch " + branch + "` to preview integration"
				} else {
					res.Detail = "diverged from " + remoteRef + " with overlapping paths; resolve conflicting changes in place before pushing"
				}
			default:
				res.Reason = ReasonBehindFastForwardable
				res.Detail = "behind " + remoteRef + "; run `fak sync apply --fetch --remote " + remote + " --branch " + branch + "` to fast-forward only when the write set is clean, then re-run `fak sync push`; never force-push, stash, reset, or raw-merge peer work"
			}
			return res, nil
		}
		// PushRetry: the rejection was a race (HEAD already contains the remote).
		// Back off before re-pushing: under high concurrency several peers lose the
		// SAME race at the same instant, and an immediate lockstep re-push just
		// re-collides on the still-moving trunk (and hammers the remote). No sleep
		// after the FINAL attempt — there is nothing left to wait for.
		if attempt < max {
			if werr := pushBackoff(ctx, attempt); werr != nil {
				return cancelledResult(res, werr), nil
			}
		}
	}
	if lastNetDetail != "" {
		res.Reason = PushReasonUnreachable
		res.Detail = "transient network failure persisted after " + strconv.Itoa(max) + " attempts (" + lastNetDetail + "); check connectivity and retry shortly"
		return res, nil
	}
	res.Reason = PushReasonExhausted
	res.Detail = "push still rejected after " + strconv.Itoa(max) + " attempts; the trunk is moving fast — retry shortly"
	return res, nil
}

// ScorePushVelocity converts raw end-to-end timing into SLO credit. The ratio
// remains visible even within budget; the score is capped at 100 and degrades as
// budget/elapsed once the SLO is exceeded. It never qualifies a non-publication.
func ScorePushVelocity(res PushResult, elapsed, budget time.Duration, runErr error) PushVelocity {
	if elapsed < 0 {
		elapsed = 0
	}
	if budget < time.Millisecond {
		budget = time.Millisecond
	}
	ratioRaw := float64(elapsed) / float64(budget)
	v := PushVelocity{
		ElapsedMS:   elapsed.Milliseconds(),
		BudgetMS:    budget.Milliseconds(),
		BudgetRatio: scorecard.Round3(ratioRaw),
		Grade:       "UNSCORED",
	}
	if runErr != nil {
		v.Notes = []string{"unscored: safe push ended with INTERNAL_ERROR"}
		return v
	}
	if !res.Pushed || res.Reason != "" {
		reason := strings.TrimSpace(res.Reason)
		if reason == "" {
			reason = "NO_PUBLICATION"
		}
		v.Notes = []string{"unscored: safe push did not publish (" + reason + ")"}
		return v
	}

	v.Qualified = true
	credit := 1.0
	if elapsed > budget {
		credit = float64(budget) / float64(elapsed)
	}
	score := int(math.Round(100 * credit))
	if score < 0 {
		score = 0
	} else if score > 100 {
		score = 100
	}
	v.Score = &score
	v.Grade = scorecard.GradeStd(float64(score))
	v.Notes = []string{"qualified: safe push published the requested ref"}
	return v
}

func safePushArgs(remote, branch, sourceRef, targetRef string) []string {
	sourceRef = strings.TrimSpace(sourceRef)
	if sourceRef == "" {
		return []string{"push", remote, branch}
	}
	targetRef = strings.TrimSpace(targetRef)
	if targetRef == "" {
		targetRef = "refs/heads/" + branch
	}
	return []string{"push", remote, sourceRef + ":" + targetRef}
}

func safePushCompareRef(sourceRef string) string {
	sourceRef = strings.TrimSpace(sourceRef)
	if sourceRef == "" {
		return "HEAD"
	}
	return sourceRef
}

// transientPushNetworkNeedles are the (lowercased) signatures of push/fetch
// failures that clear on their own: the remote dropped the connection, the
// network blipped, DNS wobbled, or the forge answered a transient 5xx/429. The
// set is deliberately conservative — an auth failure ("permission denied",
// "authentication failed", a 403) or a remote-side hook rejection must surface
// immediately as PUSH_ERROR, never spin in a retry loop.
var transientPushNetworkNeedles = []string{
	"could not resolve host",     // DNS wobble
	"connection timed out",       // TCP connect/read timeout
	"operation timed out",        // curl's phrasing of the same
	"connection reset",           // mid-transfer reset
	"connection was reset",       // Windows/curl: "Recv failure: Connection was reset"
	"connection refused",         // remote/proxy momentarily not accepting
	"the remote end hung up",     // git transport dropped mid-conversation
	"early eof",                  // truncated transfer
	"unexpected disconnect",      // pack transfer dropped
	"network is unreachable",     // route flap
	"failed to connect",          // curl connect failure
	"couldn't connect to server", // curl connect failure (alt phrasing)
	"returned error: 429",        // forge rate limit — comes back after the window
	"returned error: 500",        // forge transient 5xx family
	"returned error: 502",
	"returned error: 503",
	"returned error: 504",
	"rpc failed; http 5",               // git's smart-http phrasing of a 5xx
	"rpc failed; curl",                 // git's smart-http phrasing of a transport error
	"connection closed by remote host", // GitHub SSH throttle (kex/ssh_exchange_identification)
	"kex_exchange_identification",      // OpenSSH: the connection died during key exchange
	"ssh_exchange_identification",      // older OpenSSH phrasing of the same throttle
	"empty reply from server",          // curl 52: the server dropped before answering
}

// permanentPushMarkers force a failure OUT of the transient class even when a
// transport needle also matched. A permanent rejection routinely DRAGS transport
// trailer lines behind it — a real HTTP 403 push emits
//
//	error: RPC failed; HTTP 403 curl 22 The requested URL returned error: 403
//	send-pack: unexpected disconnect while reading sideband packet
//	fatal: the remote end hung up unexpectedly
//
// where the trailers match "unexpected disconnect" / "the remote end hung up"
// though the CAUSE is authorization. When any of these markers is present the
// whole blob is permanent (PUSH_ERROR) — retrying an auth/permission/rule
// rejection just spins against the same wall.
var permanentPushMarkers = []string{
	"authentication failed",
	"permission denied",
	"permission to",     // remote: Permission to <repo> denied to <user>
	"gh013",             // GitHub push protection / repository rules
	"push declined",     // ... (push declined due to repository rule violations)
	"[remote rejected]", // the remote-side refusal bracket (hooks, protections)
	"returned error: 401",
	"returned error: 403",
	"returned error: 404",
	"http 401",
	"http 403",
	"http 404",
}

// isTransientPushNetwork reports whether push/fetch output describes a transient
// network/forge failure worth retrying — the class auditreason files under
// REMOTE_UNREACHABLE (retry-eligible), as opposed to a permanent rejection.
// Permanent markers win over transient needles: an auth/permission/rule
// rejection stays PUSH_ERROR even when it drags transport trailer lines.
func isTransientPushNetwork(out string) bool {
	low := strings.ToLower(out)
	for _, marker := range permanentPushMarkers {
		if strings.Contains(low, marker) {
			return false
		}
	}
	for _, needle := range transientPushNetworkNeedles {
		if strings.Contains(low, needle) {
			return true
		}
	}
	return false
}

// pushWait sleeps d cancellably under ctx, returning ctx.Err() when the wait
// was cut short. Injectable so tests exercise the retry loop without real
// waits. Plain time.Sleep here would strand a cancelled caller for up to a
// full backoff and then misreport the killed git run as PUSH_ERROR /
// GIT_UNAVAILABLE instead of the honest CANCELLED.
var pushWait = func(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// pushBackoff waits the pre-retry backoff after failed attempt `attempt`:
// attempt²×250ms capped at 3s, equal-jittered to [base/2, base] — the same
// shape as internal/agent's upstream schedule, scaled to git-push latencies.
// The jitter is the point under high concurrency: peers that lost the same
// race/blip at the same instant fan out instead of re-colliding in lockstep.
// A non-nil return means ctx was cancelled mid-wait.
func pushBackoff(ctx context.Context, attempt int) error {
	base := time.Duration(attempt*attempt) * 250 * time.Millisecond
	if base > 3*time.Second {
		base = 3 * time.Second
	}
	half := int64(base / 2)
	return pushWait(ctx, time.Duration(half)+time.Duration(rand.Int63n(half+1)))
}

// cancelledResult stamps res with the honest cancellation outcome: the caller's
// ctx died mid-backoff, so no further attempt was (or will be) made.
func cancelledResult(res PushResult, err error) PushResult {
	res.Reason = PushReasonCancelled
	res.Detail = "cancelled during retry backoff: " + err.Error()
	return res
}

// classifyPushDivergence compares the pushed local ref to the (already fetched) remote ref.
func classifyPushDivergence(ctx context.Context, run Runner, repo, localRef, remoteRef string) (PushDivergence, error) {
	remoteInHead, err := isAncestor(ctx, run, repo, remoteRef, localRef)
	if err != nil {
		return "", err
	}
	if remoteInHead {
		return PushAhead, nil // remote is an ancestor of localRef (or equal): the rejection was a race
	}
	headInRemote, err := isAncestor(ctx, run, repo, localRef, remoteRef)
	if err != nil {
		return "", err
	}
	if headInRemote {
		return PushBehind, nil
	}
	return PushDiverged, nil
}

// DivergenceReason maps a PushDivergence to its corresponding closed typed sync reason.
func DivergenceReason(div PushDivergence) string {
	switch div {
	case PushBehind:
		return ReasonBehindFastForwardable
	case PushDiverged:
		return ReasonDivergedOverlap
	default:
		return ""
	}
}

// classifyDivergedPaths inspects changes between localRef and remoteRef to determine
// whether their modified file sets are disjoint (DIVERGED_DISJOINT) or overlap (DIVERGED_OVERLAP).
func classifyDivergedPaths(ctx context.Context, run Runner, repo, localRef, remoteRef string) string {
	mbRes := run(ctx, repo, "merge-base", localRef, remoteRef)
	if mbRes.Err != nil || mbRes.Code != 0 {
		return ReasonDivergedOverlap
	}
	mb := strings.TrimSpace(string(mbRes.Stdout))
	if mb == "" {
		return ReasonDivergedOverlap
	}
	localDiff := run(ctx, repo, "diff", "--name-only", mb, localRef)
	if localDiff.Err != nil || localDiff.Code != 0 {
		return ReasonDivergedOverlap
	}
	remoteDiff := run(ctx, repo, "diff", "--name-only", mb, remoteRef)
	if remoteDiff.Err != nil || remoteDiff.Code != 0 {
		return ReasonDivergedOverlap
	}
	localFiles := make(map[string]bool)
	for _, f := range strings.Split(strings.TrimSpace(string(localDiff.Stdout)), "\n") {
		f = strings.TrimSpace(f)
		if f != "" {
			localFiles[f] = true
		}
	}
	for _, f := range strings.Split(strings.TrimSpace(string(remoteDiff.Stdout)), "\n") {
		f = strings.TrimSpace(f)
		if f != "" && localFiles[f] {
			return ReasonDivergedOverlap
		}
	}
	return ReasonDivergedDisjoint
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

// pushBlockedMarker is how every pre-push gate spells its REFUSING form: `GATE (blocked): ...`.
// The same gates also emit `(advisory)`, `(warn)` and `(cured)` lines, which do NOT stop a push.
const pushBlockedMarker = "(blocked)"

// pushHeadline picks the line of a hook rejection that actually explains the refusal.
//
// The naive first-non-empty-line rule misattributes it. A pre-push run prints its gates in order
// and only the LAST one can be the blocker, so a gate that merely warned — DUPLICATION is the
// common one, it is advisory by default and prints before the claim review — lands on stderr
// first and gets reported as the reason. The operator then reads `PUSH_ERROR: DUPLICATION
// (advisory)`, goes off to dedupe code that was never blocking, and never sees the
// CLAIM_UNWITNESSED (blocked) line further down that actually refused the push. Preferring the
// `(blocked)` line names the gate that has to be cleared; without one (an auth failure, a remote
// hook, any non-fak rejection) the first line is still the best headline.
func pushHeadline(s string) string {
	for _, ln := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(ln); strings.Contains(t, pushBlockedMarker) {
			return t
		}
	}
	return pushFirstLine(s)
}
