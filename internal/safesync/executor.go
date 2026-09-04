package safesync

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/mergepreview"
)

// ExecuteReceiptSchema is the schema identifier for typed reconciliation execution receipts.
const (
	ExecuteReceiptSchema = "fak.sync.execute.v1"
	ExecuteSchema        = ExecuteReceiptSchema
)

// Execution status constants for ExecutionReceipt.
const (
	ExecuteStatusExecuted = "EXECUTED"
	ExecuteStatusFailed   = "FAILED"
	ExecuteStatusRefused  = "REFUSED"
)

// ExecuteError is a typed refusal or execution error from packet execution.
type ExecuteError struct {
	Reason string
	Detail string
}

func (e *ExecuteError) Error() string {
	if e.Detail != "" {
		return fmt.Sprintf("%s: %s", e.Reason, e.Detail)
	}
	return e.Reason
}

// ExecutionReceipt is the typed evidence-backed receipt of executing a reconciliation packet.
type ExecutionReceipt struct {
	Schema                string `json:"schema"`
	Status                string `json:"status"`
	LocalCommitsContained bool   `json:"local_commits_contained"`
	PeerBytesPreserved    bool   `json:"peer_bytes_preserved"`
	Pushed                bool   `json:"pushed"`
	TargetSHA             string `json:"target_sha"`
	NewHEAD               string `json:"new_head"`
	Reason                string `json:"reason,omitempty"`
	Detail                string `json:"detail,omitempty"`
}

// ExecuteOptions configures PacketExecutor.
type ExecuteOptions struct {
	Repo               string           `json:"repo"`
	Remote             string           `json:"remote"`
	Branch             string           `json:"branch,omitempty"`
	LeaseOwner         string           `json:"lease_owner,omitempty"`
	WriterLeaseTTL     time.Duration    `json:"-"`
	PushVelocityBudget time.Duration    `json:"-"`
	MaxPushRetries     int              `json:"max_push_retries,omitempty"`
	SuspendPaths       []string         `json:"suspend_paths,omitempty"`
	Session            string           `json:"session,omitempty"`
	Runner             Runner           `json:"-"`
	EnvRunner          EnvRunner        `json:"-"`
	Now                func() time.Time `json:"-"`
}

// PacketExecutor executes leased reconciliation packets with independent graph readback.
type PacketExecutor struct {
	opts ExecuteOptions
}

// NewPacketExecutor constructs a PacketExecutor with normalized defaults.
func NewPacketExecutor(opts ExecuteOptions) *PacketExecutor {
	if strings.TrimSpace(opts.Repo) == "" {
		opts.Repo = "."
	}
	if strings.TrimSpace(opts.Remote) == "" {
		opts.Remote = "origin"
	}
	if opts.Runner == nil {
		opts.Runner = RealRunner
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.WriterLeaseTTL <= 0 {
		opts.WriterLeaseTTL = DefaultWriterLeaseTTL
	}
	return &PacketExecutor{opts: opts}
}

// Execute executes a reconciliation packet with independent graph readback.
func (e *PacketExecutor) Execute(ctx context.Context, packet *ReconciliationPacket) (*ExecutionReceipt, error) {
	if packet == nil {
		return nil, errors.New("reconciliation packet is nil")
	}

	repo := e.opts.Repo
	run := e.opts.Runner
	now := e.opts.Now

	branch := strings.TrimSpace(e.opts.Branch)
	if branch == "" {
		b, err := currentBranch(ctx, run, repo)
		if err == nil && b != "" {
			branch = b
		} else {
			branch = "main"
		}
	}

	targetRef := strings.TrimSpace(packet.TargetRef)
	if targetRef == "" {
		targetRef = e.opts.Remote + "/" + branch
	}

	// 1. Verify packet integrity: current local HEAD == packet.LocalHead, target ref SHA == packet.TargetSHA, dirty paths match.
	// If target moved -> error with reason TARGET_MOVED.
	// If local HEAD changed -> error with reason PATHSPEC_RACE.
	// If dirty paths drifted -> error with reason DIRTY_WRITE_OVERLAP.
	curTarget, err := rev(ctx, run, repo, targetRef)
	if err != nil {
		return nil, fmt.Errorf("resolve target ref %s: %w", targetRef, err)
	}
	if curTarget != packet.TargetSHA {
		receipt := &ExecutionReceipt{
			Schema:    ExecuteReceiptSchema,
			Status:    ExecuteStatusRefused,
			TargetSHA: packet.TargetSHA,
			Reason:    ReasonTargetMoved,
			Detail:    fmt.Sprintf("target ref %s moved from %s to %s", targetRef, packet.TargetSHA, curTarget),
		}
		return receipt, &ExecuteError{Reason: ReasonTargetMoved, Detail: receipt.Detail}
	}

	curHead, err := rev(ctx, run, repo, "HEAD")
	if err != nil {
		return nil, fmt.Errorf("resolve HEAD: %w", err)
	}
	if curHead != packet.LocalHead {
		receipt := &ExecutionReceipt{
			Schema:    ExecuteReceiptSchema,
			Status:    ExecuteStatusRefused,
			TargetSHA: packet.TargetSHA,
			Reason:    ReasonPathspecRace,
			Detail:    fmt.Sprintf("local HEAD changed from %s to %s", packet.LocalHead, curHead),
		}
		return receipt, &ExecuteError{Reason: ReasonPathspecRace, Detail: receipt.Detail}
	}

	curDirty, err := workingTreeDirtyPaths(ctx, run, repo)
	if err != nil {
		return nil, fmt.Errorf("inspect dirty paths: %w", err)
	}
	if !pathsEqual(curDirty, packet.DirtyPaths) {
		receipt := &ExecutionReceipt{
			Schema:    ExecuteReceiptSchema,
			Status:    ExecuteStatusRefused,
			TargetSHA: packet.TargetSHA,
			Reason:    ReasonDirtyWriteOverlap,
			Detail:    "working tree dirty paths drifted since packet generation",
		}
		return receipt, &ExecuteError{Reason: ReasonDirtyWriteOverlap, Detail: receipt.Detail}
	}

	// 2. Check packet.Dispatchable == true and disposition in allowed set (safe-disjoint, trivial-superset, owner-authorized-path-suspend).
	// If semantic-conflict-review -> error with reason DIVERGED_OVERLAP.
	// If wait-for-owner -> error with reason LEASE_OWNER_UNAVAILABLE.
	if !packet.Dispatchable || !isAllowedDisposition(packet.Disposition) {
		var reason, detail string
		switch packet.Disposition {
		case DispositionSemanticConflictReview:
			reason = ReasonDivergedOverlap
			detail = "packet requires semantic conflict review"
		case DispositionWaitForOwner:
			reason = ReasonLeaseOwnerUnavailable
			detail = "packet is waiting for lease owner"
		case DispositionOwnerHandoffRequired:
			reason = ReasonDivergedOverlap
			detail = "packet requires owner handoff"
		default:
			reason = ReasonDivergedOverlap
			detail = fmt.Sprintf("packet is not dispatchable (disposition: %s)", packet.Disposition)
		}
		receipt := &ExecutionReceipt{
			Schema:    ExecuteReceiptSchema,
			Status:    ExecuteStatusRefused,
			TargetSHA: packet.TargetSHA,
			Reason:    reason,
			Detail:    detail,
		}
		return receipt, &ExecuteError{Reason: reason, Detail: detail}
	}

	// 3. Acquire writer lease for the repo.
	leaseOwner := strings.TrimSpace(e.opts.LeaseOwner)
	if leaseOwner == "" {
		leaseOwner = fmt.Sprintf("executor-%d", os.Getpid())
	}
	lease, err := AcquireWriterLease(repo, leaseOwner, now, e.opts.WriterLeaseTTL)
	if err != nil {
		var held *WriterLeaseHeldError
		if errors.As(err, &held) {
			receipt := &ExecutionReceipt{
				Schema:    ExecuteReceiptSchema,
				Status:    ExecuteStatusRefused,
				TargetSHA: packet.TargetSHA,
				Reason:    ReasonLeaseOwnerUnavailable,
				Detail:    held.Error(),
			}
			return receipt, &ExecuteError{Reason: ReasonLeaseOwnerUnavailable, Detail: held.Error()}
		}
		return nil, fmt.Errorf("acquire writer lease: %w", err)
	}
	defer func() { _ = lease.Release() }()
	stopHeartbeat := lease.keepAlive(now)
	defer stopHeartbeat()

	// Snapshot peer dirty bytes before executing graph operation
	peerPaths := identifyPeerPaths(curDirty, packet.Disposition, e.opts.SuspendPaths)
	peerSnapshotsBefore, err := snapshotFileBytes(repo, peerPaths)
	if err != nil {
		return nil, fmt.Errorf("snapshot peer bytes: %w", err)
	}

	// 4. Execute graph operation:
	// - If trivial-superset: in-place fast-forward or textless merge.
	// - If safe-disjoint: git merge --no-ff target.
	// - If owner-authorized-path-suspend: ParkAndReapply.
	var newHEAD string

	switch packet.Disposition {
	case DispositionTrivialSuperset:
		isAnc, _ := isAncestor(ctx, run, repo, curHead, packet.TargetSHA)
		if isAnc {
			ffRes := run(ctx, repo, "merge", "--ff-only", "--no-autostash", "--no-overwrite-ignore", packet.TargetSHA)
			if ffRes.Err != nil || ffRes.Code != 0 {
				return nil, fmt.Errorf("trivial-superset fast-forward failed: %v %s", ffRes.Err, string(ffRes.Stderr))
			}
			newHEAD, err = rev(ctx, run, repo, "HEAD")
			if err != nil {
				return nil, fmt.Errorf("resolve HEAD after fast-forward: %w", err)
			}
		} else if curHead == packet.TargetSHA {
			newHEAD = curHead
		} else {
			// textless merge
			mpRunner := func(c context.Context, dir string, args ...string) (mergepreview.RunResult, error) {
				r := run(c, dir, args...)
				return mergepreview.RunResult{Stdout: r.Stdout, Stderr: r.Stderr, Code: r.Code}, r.Err
			}
			msg := fmt.Sprintf("Merge %s (trivial superset)", targetRef)
			mpRes, mpErr := mergepreview.Apply(ctx, repo, packet.TargetSHA, msg, mpRunner)
			if mpErr != nil {
				return nil, fmt.Errorf("trivial-superset textless merge failed: %w", mpErr)
			}
			if mpRes.ApplyOutcome != mergepreview.ApplyResolvedSuperset {
				return nil, fmt.Errorf("trivial-superset textless merge deferred: %s", mpRes.Detail)
			}
			if mpRes.MergeCommit != "" {
				newHEAD = mpRes.MergeCommit
			} else {
				newHEAD, err = rev(ctx, run, repo, "HEAD")
				if err != nil {
					return nil, fmt.Errorf("resolve HEAD after textless merge: %w", err)
				}
			}
		}

	case DispositionSafeDisjoint:
		msg := fmt.Sprintf("Merge %s (safe-disjoint)", targetRef)
		mergeRes := run(ctx, repo, "merge", "--no-ff", "--no-edit", "--signoff", "-m", msg, packet.TargetSHA)
		if mergeRes.Err != nil || mergeRes.Code != 0 {
			return nil, fmt.Errorf("safe-disjoint merge failed: %v %s", mergeRes.Err, string(mergeRes.Stderr))
		}
		newHEAD, err = rev(ctx, run, repo, "HEAD")
		if err != nil {
			return nil, fmt.Errorf("resolve HEAD after safe-disjoint merge: %w", err)
		}

	case DispositionOwnerAuthorizedPathSuspend:
		suspendPaths := e.opts.SuspendPaths
		if len(suspendPaths) == 0 {
			remoteWriteSet := make(map[string]bool)
			for _, c := range packet.RemoteCommits {
				for _, p := range c.Paths {
					remoteWriteSet[p] = true
				}
			}
			for _, dp := range packet.DirtyPaths {
				if remoteWriteSet[dp] {
					suspendPaths = append(suspendPaths, dp)
				}
			}
		}
		session := strings.TrimSpace(e.opts.Session)
		if session == "" {
			session = "executor"
		}
		parkOpts := ParkOptions{
			Repo:           repo,
			Session:        session,
			Paths:          suspendPaths,
			TargetRef:      targetRef,
			Apply:          true,
			Runner:         run,
			EnvRunner:      e.opts.EnvRunner,
			Now:            now,
			WriterLeaseTTL: e.opts.WriterLeaseTTL,
			LeaseOwner:     leaseOwner,
			Lease:          lease,
		}
		parkRec, parkErr := ParkAndReapply(ctx, parkOpts)
		if parkErr != nil {
			return nil, fmt.Errorf("park and reapply: %w", parkErr)
		}
		if !parkRec.OK || parkRec.Status == ParkStatusConflict {
			return nil, fmt.Errorf("park and reapply conflict: %s (%s)", parkRec.Reason, parkRec.Detail)
		}
		newHEAD = parkRec.NewHEAD
		if newHEAD == "" {
			newHEAD, err = rev(ctx, run, repo, "HEAD")
			if err != nil {
				return nil, fmt.Errorf("resolve HEAD after park and reapply: %w", err)
			}
		}

	default:
		return nil, fmt.Errorf("unsupported disposition %q", packet.Disposition)
	}

	// 5. Push: execute SafePush.
	targetPushRef := "refs/heads/" + branch
	pushOpts := PushOptions{
		Repo:           repo,
		Remote:         e.opts.Remote,
		Branch:         branch,
		SourceRef:      newHEAD,
		TargetRef:      targetPushRef,
		MaxRetries:     e.opts.MaxPushRetries,
		VelocityBudget: e.opts.PushVelocityBudget,
		Runner:         run,
		Now:            now,
	}
	pushRes, pushErr := SafePush(ctx, pushOpts)
	if pushErr != nil {
		receipt := &ExecutionReceipt{
			Schema:    ExecuteReceiptSchema,
			Status:    ExecuteStatusFailed,
			Pushed:    false,
			TargetSHA: packet.TargetSHA,
			NewHEAD:   newHEAD,
			Reason:    pushRes.Reason,
			Detail:    pushErr.Error(),
		}
		return receipt, fmt.Errorf("push failed: %w", pushErr)
	}
	if !pushRes.Pushed {
		receipt := &ExecutionReceipt{
			Schema:    ExecuteReceiptSchema,
			Status:    ExecuteStatusFailed,
			Pushed:    false,
			TargetSHA: packet.TargetSHA,
			NewHEAD:   newHEAD,
			Reason:    pushRes.Reason,
			Detail:    pushRes.Detail,
		}
		return receipt, fmt.Errorf("push failed: %s", pushRes.Reason)
	}

	// 6. Independent graph readback: verify git merge-base --is-ancestor each local commit in origin/main, verify peer dirty bytes unchanged before/after.
	if e.opts.Remote != "" && branch != "" {
		_ = run(ctx, repo, "fetch", e.opts.Remote, branch)
	}

	readbackRef := targetRef

	localCommitsContained := true
	for _, c := range packet.LocalCommits {
		if c.SHA == "" {
			continue
		}
		mbRes := run(ctx, repo, "merge-base", "--is-ancestor", c.SHA, readbackRef)
		if mbRes.Err != nil || mbRes.Code != 0 {
			localCommitsContained = false
			break
		}
	}

	peerSnapshotsAfter, err := snapshotFileBytes(repo, peerPaths)
	if err != nil {
		return nil, fmt.Errorf("snapshot peer bytes after execution: %w", err)
	}
	peerBytesPreserved := verifySnapshotsEqual(peerSnapshotsBefore, peerSnapshotsAfter)

	// 7. Return ExecutionReceipt with {Schema, Status, LocalCommitsContained, PeerBytesPreserved, Pushed, TargetSHA, NewHEAD}.
	receipt := &ExecutionReceipt{
		Schema:                ExecuteReceiptSchema,
		Status:                ExecuteStatusExecuted,
		LocalCommitsContained: localCommitsContained,
		PeerBytesPreserved:    peerBytesPreserved,
		Pushed:                pushRes.Pushed,
		TargetSHA:             packet.TargetSHA,
		NewHEAD:               newHEAD,
	}

	if !localCommitsContained {
		receipt.Status = ExecuteStatusFailed
		receipt.Reason = "LOCAL_COMMITS_NOT_CONTAINED"
		receipt.Detail = fmt.Sprintf("local commits not contained in %s", readbackRef)
		return receipt, fmt.Errorf("independent graph readback failed: local commits not contained in %s", readbackRef)
	}

	if !peerBytesPreserved {
		receipt.Status = ExecuteStatusFailed
		receipt.Reason = "PEER_BYTES_CORRUPTED"
		receipt.Detail = "peer dirty bytes were modified during execution"
		return receipt, fmt.Errorf("independent graph readback failed: peer dirty bytes not preserved")
	}

	return receipt, nil
}

// ExecutePacket executes a reconciliation packet with the given options.
func ExecutePacket(ctx context.Context, packet *ReconciliationPacket, opts ExecuteOptions) (*ExecutionReceipt, error) {
	return NewPacketExecutor(opts).Execute(ctx, packet)
}

func isAllowedDisposition(d Disposition) bool {
	switch d {
	case DispositionSafeDisjoint, DispositionTrivialSuperset, DispositionOwnerAuthorizedPathSuspend:
		return true
	default:
		return false
	}
}

func pathsEqual(a, b []string) bool {
	na := normalizePathSlice(a)
	nb := normalizePathSlice(b)
	if len(na) != len(nb) {
		return false
	}
	for i := range na {
		if na[i] != nb[i] {
			return false
		}
	}
	return true
}

func normalizePathSlice(paths []string) []string {
	if len(paths) == 0 {
		return []string{}
	}
	set := make(map[string]bool, len(paths))
	for _, p := range paths {
		p = filepath.Clean(filepath.ToSlash(strings.TrimSpace(p)))
		if p != "" && p != "." {
			set[p] = true
		}
	}
	res := make([]string, 0, len(set))
	for p := range set {
		res = append(res, p)
	}
	sort.Strings(res)
	return res
}

func identifyPeerPaths(dirtyPaths []string, disposition Disposition, suspendPaths []string) []string {
	suspendSet := make(map[string]bool)
	if disposition == DispositionOwnerAuthorizedPathSuspend {
		for _, sp := range suspendPaths {
			sp = filepath.Clean(filepath.ToSlash(strings.TrimSpace(sp)))
			if sp != "" {
				suspendSet[sp] = true
			}
		}
	}
	var peerPaths []string
	for _, dp := range dirtyPaths {
		norm := filepath.Clean(filepath.ToSlash(strings.TrimSpace(dp)))
		if norm != "" && !suspendSet[norm] {
			peerPaths = append(peerPaths, norm)
		}
	}
	sort.Strings(peerPaths)
	return peerPaths
}

func snapshotFileBytes(repo string, paths []string) (map[string][]byte, error) {
	snapshots := make(map[string][]byte, len(paths))
	for _, p := range paths {
		norm := filepath.Clean(filepath.ToSlash(p))
		full := filepath.Join(repo, norm)
		b, err := os.ReadFile(full)
		if err != nil {
			if os.IsNotExist(err) {
				snapshots[norm] = nil
				continue
			}
			return nil, err
		}
		snapshots[norm] = b
	}
	return snapshots, nil
}

func verifySnapshotsEqual(before, after map[string][]byte) bool {
	if len(before) != len(after) {
		return false
	}
	for p, b1 := range before {
		b2, ok := after[p]
		if !ok {
			return false
		}
		if (b1 == nil && b2 != nil) || (b1 != nil && b2 == nil) {
			return false
		}
		if !bytes.Equal(b1, b2) {
			return false
		}
	}
	return true
}
