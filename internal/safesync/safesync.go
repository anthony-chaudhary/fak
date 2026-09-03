package safesync

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	StateInSync      = "in-sync"
	StateBehind      = "behind"
	StateAhead       = "ahead"
	StateDiverged    = "diverged"
	StateNoRemoteRef = "no-remote-ref"
)

// Runner executes a git subcommand in repo. Err is non-nil only when git could
// not be started; a non-zero git exit is reported through Code and Stderr.
type Runner func(ctx context.Context, repo string, args ...string) RunResult

type RunResult struct {
	Stdout []byte
	Stderr []byte
	Code   int
	Err    error
}

type Options struct {
	Repo                string
	Remote              string
	Branch              string
	Fetch               bool
	Runner              Runner           `json:"-"`
	Now                 func() time.Time `json:"-"`
	ApplyVelocityBudget time.Duration    `json:"-"`
	// LeaseOwner labels this process in the cooperative worktree-writer lease Apply holds
	// across its assess+apply window (default: a pid-derived id).
	LeaseOwner string `json:"lease_owner,omitempty"`
	// WriterLeaseTTL bounds how long the lease is honored before a peer may reclaim it as
	// crash residue (default DefaultWriterLeaseTTL).
	WriterLeaseTTL time.Duration `json:"-"`
	// QuarantineScratch enables shift-left untracked artifact isolation across fast-forward (#10913).
	QuarantineScratch bool `json:"quarantine_scratch,omitempty"`
	// barrier is a test-only seam fired while Apply holds the writer lease, so a
	// concurrency test can prove a second managed writer is refused mid-window.
	barrier func()
}

type Entry struct {
	Status string `json:"status"`
	Path   string `json:"path"`
}

type Assessment struct {
	OK            bool               `json:"ok"`
	State         string             `json:"state"`
	Head          string             `json:"head,omitempty"`
	Target        string             `json:"target,omitempty"`
	TargetRef     string             `json:"target_ref,omitempty"`
	Branch        string             `json:"branch,omitempty"`
	WriteCount    int                `json:"write_count,omitempty"`
	Identical     []Entry            `json:"identical,omitempty"`
	Divergent     []Entry            `json:"divergent,omitempty"`
	Reason        string             `json:"reason,omitempty"`
	Applied       bool               `json:"applied,omitempty"`
	NewHead       string             `json:"new_head,omitempty"`
	PushAudit     *PushAudit         `json:"push_audit,omitempty"`
	Worktree      *Worktree          `json:"worktree,omitempty"`
	Quarantine    *QuarantineReceipt `json:"quarantine,omitempty"`
	ApplyVelocity PushVelocity       `json:"apply_velocity"`
	// Indeterminate is set when a fast-forward failed partway (a partial checkout, an
	// in-progress MERGE_HEAD, or HEAD moved despite a non-zero exit): the worktree may be
	// partially updated, so Apply neither claims a clean refusal nor swallows a plain
	// error. Recover (`git status`, `git merge --abort`) before re-syncing.
	Indeterminate bool `json:"indeterminate,omitempty"`
	// Lease names the peer worktree-writer lease holder when Apply refused to enter its
	// assess/apply window because a cooperative fak-managed writer already held the lease.
	Lease *WriterLeaseInfo `json:"lease,omitempty"`
}

// PushAudit is optional, read-only evidence attached by higher-level callers when an
// ahead branch is not just "needs push" but would be refused by the pre-push commit
// honesty audit. The safesync core stays git-only; cmd/fak fills this when DOS is
// available in a fak workspace.
type PushAudit struct {
	OK        bool                `json:"ok"`
	Range     string              `json:"range,omitempty"`
	Residuals []PushAuditResidual `json:"residuals,omitempty"`
}

type PushAuditResidual struct {
	SHA       string `json:"sha,omitempty"`
	Subject   string `json:"subject,omitempty"`
	Verdict   string `json:"verdict,omitempty"`
	ClaimKind string `json:"claim_kind,omitempty"`
	Witness   string `json:"witness,omitempty"`
	Reason    string `json:"reason,omitempty"`
}

// Worktree is optional dirty-tree guidance attached by the CLI. It deliberately does not affect
// the branch sync verdict: a checkout can be remote-in-sync while still carrying local work.
type Worktree struct {
	Dirty        bool     `json:"dirty"`
	TotalDirty   int      `json:"total_dirty,omitempty"`
	Stampable    int      `json:"stampable,omitempty"`
	Lanes        int      `json:"lanes,omitempty"`
	NoLane       int      `json:"no_lane,omitempty"`
	Junk         int      `json:"junk,omitempty"`
	JunkPaths    []string `json:"junk_paths,omitempty"`
	OldestPath   string   `json:"oldest_path,omitempty"`
	OldestAgeSec int64    `json:"oldest_age_seconds,omitempty"`
	NextAction   string   `json:"next_action,omitempty"`
}

type GitError struct {
	Args   []string
	Code   int
	Detail string
}

func (e *GitError) Error() string {
	if e == nil {
		return ""
	}
	detail := strings.TrimSpace(e.Detail)
	if detail == "" {
		detail = "git command failed"
	}
	return fmt.Sprintf("git %s -> %d: %s", strings.Join(e.Args, " "), e.Code, detail)
}

// Assess reports whether repo can safely fast-forward to remote/branch without
// clobbering dirty shared-worktree content. It is read-only except for the
// optional fetch.
func Assess(ctx context.Context, opts Options) (Assessment, error) {
	opts = normalizeOptions(opts)
	run := opts.Runner

	branch := strings.TrimSpace(opts.Branch)
	if branch == "" {
		var err error
		branch, err = currentBranch(ctx, run, opts.Repo)
		if err != nil {
			return Assessment{}, err
		}
	}
	targetRef := opts.Remote + "/" + branch

	if opts.Fetch {
		if _, err := checked(ctx, run, opts.Repo, "fetch", opts.Remote, branch); err != nil {
			return Assessment{}, err
		}
	}

	head, err := rev(ctx, run, opts.Repo, "HEAD")
	if err != nil {
		return Assessment{}, err
	}
	target, err := rev(ctx, run, opts.Repo, targetRef)
	if err != nil {
		var ge *GitError
		if errors.As(err, &ge) {
			return Assessment{
				OK:        false,
				State:     StateNoRemoteRef,
				TargetRef: targetRef,
				Branch:    branch,
				Reason:    "remote-tracking ref " + targetRef + " not found; fetch first",
			}, nil
		}
		return Assessment{}, err
	}

	base := Assessment{Head: head, Target: target, TargetRef: targetRef, Branch: branch}
	if head == target {
		base.OK = true
		base.State = StateInSync
		return base, nil
	}
	targetIsAncestor, err := isAncestor(ctx, run, opts.Repo, target, head)
	if err != nil {
		return Assessment{}, err
	}
	if targetIsAncestor {
		base.State = StateAhead
		base.Reason = fmt.Sprintf("local branch is ahead of remote; nothing to fast-forward; run `fak sync push --remote %s --branch %s` to publish through the safe push path", opts.Remote, branch)
		return base, nil
	}
	headIsAncestor, err := isAncestor(ctx, run, opts.Repo, head, target)
	if err != nil {
		return Assessment{}, err
	}
	if !headIsAncestor {
		base.State = StateDiverged
		base.Reason = "local and remote have diverged; not a fast-forward"
		return base, nil
	}

	entries, err := ffWriteSet(ctx, run, opts.Repo, head, target)
	if err != nil {
		return Assessment{}, err
	}
	identical, divergent := classify(opts.Repo, run, ctx, head, target, entries, opts.QuarantineScratch)
	base.State = StateBehind
	base.WriteCount = len(entries)
	base.Identical = identical
	base.Divergent = divergent
	base.OK = len(divergent) == 0
	if base.OK {
		base.Reason = "every fast-forward path is clean at HEAD or already byte-identical to the remote; safe to fast-forward"
	} else {
		base.Reason = fmt.Sprintf("%d path(s) diverge locally; refusing; sync at a quiescent moment or resolve by hand", len(divergent))
	}
	return base, nil
}

// Apply performs the same assessment and runs the fast-forward only when Assess
// says the behind state is safe. Refused states leave the tree untouched.
func Apply(ctx context.Context, opts Options) (info Assessment, err error) {
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	budget := opts.ApplyVelocityBudget
	if budget == 0 {
		budget = DefaultPushVelocityBudget
	}
	started := now()
	defer func() { info.ApplyVelocity = ScoreApplyVelocity(info, now().Sub(started), budget, err) }()
	opts = normalizeOptions(opts)
	run := opts.Runner

	// Hold the cooperative worktree writer lease for the WHOLE assess+apply window, so a
	// fak-managed peer writer cannot edit a classified path between Assess and the
	// checkout (#4240). A live peer holding it means we must not enter the window at all:
	// refuse without touching the tree. Released on every return path.
	lease, lerr := AcquireWriterLease(opts.Repo, opts.LeaseOwner, now, opts.WriterLeaseTTL)
	if lerr != nil {
		var held *WriterLeaseHeldError
		if errors.As(lerr, &held) {
			holder := held.Info
			info.OK = false
			info.Applied = false
			info.Lease = &holder
			info.Reason = fmt.Sprintf("worktree writer lease held by %s; refusing to enter the assess/apply window so a peer writer's bytes are never overwritten", holder.Owner)
			return info, nil
		}
		return info, lerr // an I/O failure taking the lease is an infrastructure error
	}
	defer func() { _ = lease.Release() }()
	// Heartbeat the lease for the whole window (#4612): an apply that outlives the TTL
	// (a pathologically slow fast-forward) renews itself instead of being reclaimed
	// mid-window as crash residue. Stopped (and joined) before the deferred Release.
	stopHeartbeat := lease.keepAlive(now)
	defer stopHeartbeat()
	if opts.barrier != nil {
		opts.barrier()
	}
	if lease.Lost() {
		return Assessment{OK: false, State: "refused", Reason: "writer lease lost before apply; displaced result suppressed", Lease: func() *WriterLeaseInfo { info := lease.Info(); return &info }()}, nil
	}

	info, err = Assess(ctx, opts)
	if err != nil {
		return info, err
	}
	if info.State == StateInSync {
		return info, nil
	}
	if info.State != StateBehind || !info.OK {
		info.Applied = false
		return info, nil
	}

	var qTx *QuarantineTransaction
	if opts.QuarantineScratch && info.Target != "" {
		entries, ffErr := ffWriteSet(ctx, run, opts.Repo, info.Head, info.Target)
		if ffErr == nil {
			var qPaths []string
			identMap := make(map[string]bool)
			for _, e := range entries {
				if e.Status == "A" {
					if _, exists := worktreeBytes(opts.Repo, e.Path); exists {
						qPaths = append(qPaths, e.Path)
						identMap[e.Path] = cleanEquivalentTo(ctx, run, opts.Repo, info.Target, e.Path)
					}
				}
			}
			if len(qPaths) > 0 {
				var qErr error
				qTx, qErr = PrepareQuarantine(opts.Repo, qPaths, identMap)
				if qErr != nil {
					return info, qErr
				}
				defer func() {
					if qTx != nil {
						_ = qTx.Rollback()
					}
				}()
			}
		}
	}

	applied, detail, indeterminate, err := applyFastForward(ctx, run, opts.Repo, info, lease)
	if err != nil {
		return info, err
	}
	if indeterminate {
		info.OK = false
		info.Applied = false
		info.Indeterminate = true
		info.Reason = "fast-forward failed in an indeterminate state; the worktree may be partially updated — inspect with `git status` and recover (`git merge --abort`) before re-syncing"
		if detail != "" {
			info.Reason += ": " + detail
		}
		return info, nil
	}
	if !applied {
		info.OK = false
		info.Applied = false
		info.Reason = "fast-forward refused after assessment; the worktree or repository state changed, so no sync was applied"
		if detail != "" {
			info.Reason += ": " + detail
		}
		return info, nil
	}
	if qTx != nil {
		receipt, commitErr := qTx.Commit()
		if commitErr != nil {
			return info, commitErr
		}
		qTx = nil
		info.Quarantine = &receipt
	}
	newHead, err := rev(ctx, run, opts.Repo, "HEAD")
	if err != nil {
		return info, err
	}
	info.Applied = true
	info.NewHead = newHead
	return info, nil
}

func normalizeOptions(opts Options) Options {
	if strings.TrimSpace(opts.Repo) == "" {
		opts.Repo = "."
	}
	if strings.TrimSpace(opts.Remote) == "" {
		opts.Remote = "origin"
	}
	if opts.Runner == nil {
		opts.Runner = RealRunner
	}
	return opts
}

func RealRunner(ctx context.Context, repo string, args ...string) RunResult {
	cmdArgs := append([]string{"-C", repo}, args...)
	cmd := exec.CommandContext(ctx, "git", cmdArgs...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			code = exit.ExitCode()
			err = nil
		} else {
			code = -1
		}
	}
	return RunResult{Stdout: stdout.Bytes(), Stderr: stderr.Bytes(), Code: code, Err: err}
}

func currentBranch(ctx context.Context, run Runner, repo string) (string, error) {
	out, err := checked(ctx, run, repo, "rev-parse", "--abbrev-ref", "HEAD")
	if err != nil {
		return "", err
	}
	branch := strings.TrimSpace(string(out))
	if branch == "" || branch == "HEAD" {
		return "", fmt.Errorf("detached HEAD; no branch to sync")
	}
	return branch, nil
}

func rev(ctx context.Context, run Runner, repo, ref string) (string, error) {
	out, err := checked(ctx, run, repo, "rev-parse", "--verify", ref)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func checked(ctx context.Context, run Runner, repo string, args ...string) ([]byte, error) {
	res := run(ctx, repo, args...)
	if res.Err != nil {
		return nil, res.Err
	}
	if res.Code != 0 {
		detail := strings.TrimSpace(string(res.Stderr))
		if detail == "" {
			detail = strings.TrimSpace(string(res.Stdout))
		}
		return nil, &GitError{Args: append([]string(nil), args...), Code: res.Code, Detail: detail}
	}
	return res.Stdout, nil
}

func isAncestor(ctx context.Context, run Runner, repo, a, b string) (bool, error) {
	res := run(ctx, repo, "merge-base", "--is-ancestor", a, b)
	if res.Err != nil {
		return false, res.Err
	}
	switch res.Code {
	case 0:
		return true, nil
	case 1:
		return false, nil
	default:
		detail := strings.TrimSpace(string(res.Stderr))
		if detail == "" {
			detail = strings.TrimSpace(string(res.Stdout))
		}
		return false, &GitError{Args: []string{"merge-base", "--is-ancestor", a, b}, Code: res.Code, Detail: detail}
	}
}

func ffWriteSet(ctx context.Context, run Runner, repo, head, target string) ([]Entry, error) {
	out, err := checked(ctx, run, repo, "diff", "--name-status", "-z", head, target)
	if err != nil {
		return nil, err
	}
	return parseNameStatusZ(out), nil
}

func parseNameStatusZ(out []byte) []Entry {
	fields := strings.Split(string(out), "\x00")
	entries := make([]Entry, 0, len(fields)/2)
	for i := 0; i < len(fields); {
		status := fields[i]
		if status == "" {
			i++
			continue
		}
		code := status[:1]
		if code == "R" || code == "C" {
			if i+2 >= len(fields) {
				break
			}
			entries = append(entries, Entry{Status: status, Path: fields[i+1]})
			entries = append(entries, Entry{Status: status, Path: fields[i+2]})
			i += 3
			continue
		}
		if i+1 >= len(fields) {
			break
		}
		entries = append(entries, Entry{Status: code, Path: fields[i+1]})
		i += 2
	}
	return entries
}

func classify(repo string, run Runner, ctx context.Context, head, target string, entries []Entry, quarantineScratch ...bool) (identical, divergent []Entry) {
	allowQuarantine := len(quarantineScratch) > 0 && quarantineScratch[0]
	for _, e := range entries {
		safe := false
		switch e.Status {
		case "M":
			_, ok := worktreeBytes(repo, e.Path)
			// A target-identical local edit is content-safe in a snapshot, but
			// `git merge --ff-only` deliberately refuses it. Classify only the
			// worktree shape Git can update without pre-cleaning so check/apply
			// agree and no helper ever needs to overwrite the path first. Ask Git
			// to compare the filtered worktree view so clean-filter checkout
			// representations (notably core.autocrlf CRLF) match their HEAD blob.
			safe = ok && cleanEquivalentTo(ctx, run, repo, head, e.Path)
		case "A":
			_, ok := worktreeBytes(repo, e.Path)
			if !ok {
				safe = true
			} else if allowQuarantine {
				// With shift-left pre-sync quarantine, colliding untracked files are
				// safely isolated across fast-forward and verified/restored (#10913).
				safe = true
			}
		case "D":
			_, ok := worktreeBytes(repo, e.Path)
			safe = !ok || cleanEquivalentTo(ctx, run, repo, head, e.Path)
		default:
			safe = false
		}
		if safe {
			identical = append(identical, e)
		} else {
			divergent = append(divergent, e)
		}
	}
	return identical, divergent
}

func cleanEquivalentTo(ctx context.Context, run Runner, repo, ref, path string) bool {
	if _, ok := safeWorktreePath(repo, path); !ok {
		return false
	}
	base := run(ctx, repo, "rev-parse", "--verify", ref+":"+path)
	if base.Err != nil || base.Code != 0 || len(bytes.TrimSpace(base.Stdout)) == 0 {
		return false
	}
	// --path selects the same clean filters Git uses when adding this tracked
	// path, while omitting -w keeps classification read-only. Comparing object
	// IDs avoids assuming checkout bytes equal canonical blob bytes.
	worktree := run(ctx, repo, "hash-object", "--path="+path, "--", path)
	if worktree.Err != nil || worktree.Code != 0 || len(bytes.TrimSpace(worktree.Stdout)) == 0 {
		return false
	}
	return bytes.Equal(bytes.TrimSpace(worktree.Stdout), bytes.TrimSpace(base.Stdout))
}

func worktreeBytes(repo, path string) ([]byte, bool) {
	full, ok := safeWorktreePath(repo, path)
	if !ok {
		return nil, false
	}
	b, err := os.ReadFile(full)
	if err != nil {
		return nil, false
	}
	return b, true
}

func applyFastForward(ctx context.Context, run Runner, repo string, info Assessment, lease *WriterLease) (applied bool, detail string, indeterminate bool, err error) {
	// The assessment is a snapshot, not a lease over arbitrary raw file writers. Do not pre-clean
	// "identical" paths here: a peer can edit or create one after classify reads it,
	// and checkout/remove would then destroy that newer work. Git's merge worktree
	// checks close that explicit assess->preclean window for ordinary tracked and
	// untracked paths; --no-overwrite-ignore extends the check to ignored paths.
	//
	// Merge the immutable object that Assess inspected, never the mutable remote
	// tracking ref. A concurrent fetch may advance TargetRef between these calls,
	// but it cannot change the tree identified by Target.
	if strings.TrimSpace(info.Target) == "" {
		return false, "", false, fmt.Errorf("assessed target SHA is empty")
	}
	branch, err := currentBranch(ctx, run, repo)
	if err != nil {
		return false, "", false, err
	}
	if info.Branch != "" && branch != info.Branch {
		return false, "branch changed after assessment", false, nil
	}
	head, err := rev(ctx, run, repo, "HEAD")
	if err != nil {
		return false, "", false, err
	}
	if info.Head != "" && head != info.Head {
		return false, "HEAD changed after assessment", false, nil
	}
	if lease.Lost() {
		return false, "writer lease lost before fast-forward; displaced result suppressed", false, nil
	}
	args := []string{"merge", "--ff-only", "--no-autostash", "--no-overwrite-ignore", info.Target}
	res := run(ctx, repo, args...)
	if res.Err != nil {
		return false, "", false, res.Err
	}
	if res.Code != 0 {
		detail := runDetail(res)
		if safeFastForwardRefusal(detail) {
			return false, pushFirstLine(detail), false, nil
		}
		// Not a recognized clean pre-merge refusal. Probe whether the tree is provably
		// untouched (a pre-mutation infra failure, e.g. index.lock) or the fast-forward
		// died partway (a partial checkout, an in-progress MERGE_HEAD, or HEAD moved).
		// The latter is INDETERMINATE: the worktree may be half-updated, so Apply must
		// neither claim a clean refusal nor swallow it as a plain error.
		if partialFastForward(ctx, run, repo, info.Head) {
			return false, pushFirstLine(detail), true, nil
		}
		return false, "", false, &GitError{Args: args, Code: res.Code, Detail: detail}
	}
	return true, "", false, nil
}

// partialFastForward reports whether a failed fast-forward left the worktree/index in a
// partial state — an in-progress MERGE_HEAD, an unmerged index entry, or HEAD advanced
// despite the non-zero exit. Each is positive evidence the merge started mutating and
// did not cleanly finish; the absence of all three (HEAD unmoved, no MERGE_HEAD, clean
// index — the index.lock class) means the tree is provably untouched and the caller
// keeps the honest infrastructure error.
func partialFastForward(ctx context.Context, run Runner, repo, assessedHead string) bool {
	if res := run(ctx, repo, "rev-parse", "--verify", "-q", "MERGE_HEAD"); res.Err == nil && res.Code == 0 {
		return true
	}
	if res := run(ctx, repo, "ls-files", "-u"); res.Err == nil && res.Code == 0 && len(bytes.TrimSpace(res.Stdout)) > 0 {
		return true
	}
	if strings.TrimSpace(assessedHead) != "" {
		if res := run(ctx, repo, "rev-parse", "--verify", "HEAD"); res.Err == nil && res.Code == 0 {
			if strings.TrimSpace(string(res.Stdout)) != assessedHead {
				return true
			}
		}
	}
	return false
}

// safeFastForwardRefusal recognizes only merge failures that establish a
// non-mutating worktree/ref race. Unknown failures (index corruption/locks,
// permissions, invalid configuration, missing objects) remain infrastructure
// errors; Apply must not claim "no sync was applied" without positive evidence.
func safeFastForwardRefusal(detail string) bool {
	low := strings.ToLower(detail)
	needles := []string{
		"local changes to the following files would be overwritten by merge",
		"untracked working tree files would be overwritten by merge",
		"not possible to fast-forward",
		"refusing to merge unrelated histories",
		"not uptodate. cannot merge",
	}
	for _, needle := range needles {
		if strings.Contains(low, needle) {
			return true
		}
	}
	return false
}

func safeWorktreePath(repo, path string) (string, bool) {
	if filepath.IsAbs(path) || path == "" {
		return "", false
	}
	clean := filepath.Clean(filepath.FromSlash(path))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", false
	}
	return filepath.Join(repo, clean), true
}
