package workerworktree

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// PreservationGate rechecks the real admitted lease and current resources.
// It must be read-only, honor ctx, and fail closed. No lease is acquired here.
type PreservationGate func(context.Context) error

// PreparePreservingBounded creates only an absent unique target. Unlike normal
// prepare it never sweeps, adopts, resets a pool member, or cleans up failures.
// Failure may leave a partial new target for explicit preservation/recovery.
func PreparePreservingBounded(root, lane, key, baseSHA, wtRoot string, owner OwnerStamp, budget time.Duration, gate PreservationGate, intent ...PreservingIntent) Result {
	if budget <= 0 {
		budget = 2 * time.Minute
	}
	ctx, cancel := context.WithTimeout(context.Background(), budget)
	defer cancel()
	return preparePreserving(ctx, root, lane, key, baseSHA, wtRoot, owner, boundedPreservingGitRunner(ctx), gate, intent...)
}

func preparePreserving(ctx context.Context, root, lane, key, baseSHA, wtRoot string, owner OwnerStamp, git GitRunner, gate PreservationGate, intent ...PreservingIntent) Result {
	canonicalRoot, canonicalWorkers, err := CanonicalPreservingRoots(root, wtRoot)
	if err != nil {
		return Result{OK: false, Code: "PRESERVATION_TARGET_CONFLICT", Preserved: true, Reason: "canonical preparation identity unavailable: " + err.Error()}
	}
	root, wtRoot = canonicalRoot, canonicalWorkers
	target := Path(lane, key, wtRoot)
	attempted := false
	refuse := func(code, reason string) Result {
		preserved := true
		if attempted {
			_, err := os.Lstat(target)
			preserved = !os.IsNotExist(err)
		}
		return Result{OK: false, Code: code, Path: target, BaseSHA: baseSHA, Preserved: preserved, Reason: reason + " - no ready receipt emitted"}
	}
	if strings.TrimSpace(lane) == "" || strings.TrimSpace(key) == "" || strings.TrimSpace(baseSHA) == "" || owner.PID <= 0 || strings.TrimSpace(owner.LeaseID) == "" || gate == nil {
		return refuse("PRESERVATION_ADMISSION_REFUSED", "explicit lane, unique key, exact base, live owner and admission gate required")
	}
	if err := validatePreservingIntent(intent); err != nil {
		return refuse("PRESERVATION_ADMISSION_REFUSED", err.Error())
	}
	check := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !processalive.Check(owner.PID) {
			return fmt.Errorf("owner process is not live")
		}
		return gate(ctx)
	}
	if err := check(); err != nil {
		return refuse("PRESERVATION_ADMISSION_REFUSED", err.Error())
	}
	rc, resolved := run(git, root, []string{"rev-parse", "--verify", baseSHA + "^{commit}"})
	if rc != 0 || strings.TrimSpace(resolved) != baseSHA {
		return refuse("PREPARE_NOT_READY", "base must name the full exact commit object")
	}
	rc, metadata := run(git, root, []string{"ls-tree", "-r", "--name-only", "-z", baseSHA, "--", WorkerLeaseFileName, WorkerLeaseFileName + ".tmp"})
	if rc != 0 || metadata != "" {
		return refuse("PRESERVATION_TARGET_CONFLICT", "pinned source contains reserved worker metadata or cannot be inspected")
	}
	policy, err := preservingCheckoutPolicyProof(root, target, baseSHA, git, nil)
	if err != nil {
		return refuse("PRESERVATION_CONFIG_REFUSED", err.Error())
	}
	var childPolicy *preservingCheckoutProof
	recheckPolicy := func(child bool) error {
		// The source remains authoritative after add. A changed original must
		// not be hidden by an unchanged copy in the new checkout.
		current, err := preservingCheckoutPolicyProof(root, target, baseSHA, git, nil)
		if err != nil {
			return err
		}
		if current != policy {
			return fmt.Errorf("source checkout policy drifted from admitted proof")
		}
		if child {
			current, err = preservingCheckoutPolicyProof(target, target, baseSHA, git, &policy)
			if err != nil {
				return err
			}
			if current.digest != policy.digest || childPolicy != nil && current != *childPolicy {
				return fmt.Errorf("new-worktree checkout policy differs from admitted copy proof")
			}
			childPolicy = &current
		}
		return nil
	}
	wait := prepareLockWait()
	if deadline, ok := ctx.Deadline(); ok {
		wait = time.Until(deadline)
	}
	res := refuse(PrepareCodeBusy, "preserving prepare lock unavailable")
	err = withPrepareTargetLock(target, wait, func() {
		if _, err := os.Lstat(target); !os.IsNotExist(err) {
			res = refuse(PrepareCodeOrphanTargetRefused, "target already exists or cannot be inspected; unique target required")
			return
		}
		for _, sidecar := range preservingSidecars(target) {
			if _, err := os.Lstat(sidecar); !os.IsNotExist(err) {
				res = refuse("PRESERVATION_TARGET_CONFLICT", "existing or unreadable target metadata: "+sidecar)
				return
			}
		}
		rc, listing := run(git, root, []string{"worktree", "list", "--porcelain"})
		if rc != 0 {
			res = refuse("PREPARE_NOT_READY", "worktree registration inventory unavailable")
			return
		}
		paths := map[string]bool{}
		for _, entry := range parseWorktreeListing(listing) {
			if samePath(entry.Path, target) {
				res = refuse("PRESERVATION_TARGET_CONFLICT", "target already has a Git registration")
				return
			}
			if IsWorkerWorktree(entry.Path) {
				paths[canonicalComparisonPath(entry.Path)] = true
			}
		}
		// Include unregistered residue without reading/removing its contents.
		for _, dir := range []string{filepath.Join(root, "_scratch"), resolveWorktreeRoot(wtRoot)} {
			entries, err := os.ReadDir(dir)
			if os.IsNotExist(err) {
				continue
			}
			if err != nil {
				res = refuse("PRESERVATION_CAPACITY_REFUSED", "worker-root inventory unavailable: "+err.Error())
				return
			}
			for _, entry := range entries {
				if IsWorkerWorktree(entry.Name()) {
					paths[canonicalComparisonPath(filepath.Join(dir, entry.Name()))] = true
				}
			}
		}
		// Count remains advisory; fresh resource admission must provide headroom.
		if err := check(); err != nil {
			res = refuse("PRESERVATION_ADMISSION_REFUSED", err.Error())
			return
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			res = refuse("PREPARE_NOT_READY", err.Error())
			return
		}
		if err := recheckPolicy(false); err != nil {
			res = refuse("PRESERVATION_CONFIG_REFUSED", err.Error())
			return
		}
		attempted = true
		// No force, pool, index-failure retry or no-checkout fallback. Disable
		// expiration-based Git administrative pruning during add as well.
		rc, detail := run(git, root, []string{"-c", "gc.worktreePruneExpire=never", "-c", "core.longpaths=true", "worktree", "add", "--quiet", "--detach", target, baseSHA})
		if rc != 0 {
			code := "PREPARE_NOT_READY"
			if rc == ReapTimeoutExitCode || ctx.Err() != nil {
				code = "PREPARE_TIMEOUT"
			}
			res = refuse(code, "Git add failed; surviving attempt evidence left untouched (Git may roll back its own add): "+tail(detail, 500))
			return
		}
		// Requalify the actual new-worktree context before status can run helpers.
		if err := recheckPolicy(true); err != nil {
			res = refuse("PRESERVATION_CONFIG_REFUSED", err.Error())
			return
		}
		res = verifyPreparedWorktree(Result{OK: true, Path: target, BaseSHA: baseSHA}, git)
		if !res.OK {
			_, err := os.Lstat(target)
			res.Preserved = !os.IsNotExist(err)
			return
		}
		if live, known := liveWorktreeRegistration(root, target, git); !known || !live {
			res = refuse("PREPARE_NOT_READY", "new checkout registration unavailable")
			return
		}
		if err := check(); err != nil {
			res = refuse("PRESERVATION_ADMISSION_REFUSED", err.Error())
			return
		}
		if err := preservingMetadataAbsent(target); err != nil {
			res = refuse("PRESERVATION_TARGET_CONFLICT", err.Error())
			return
		}
		owner = normalizeOwnerStamp(owner)
		if err := writePreservingJSON(OwnerStampPath(target), owner); err != nil {
			res = refuse("PREPARE_NOT_READY", "owner metadata failed; partial state retained: "+err.Error())
			return
		}
		if err := writePreservingJSON(filepath.Join(target, WorkerLeaseFileName), WorkerLease{PID: owner.PID, SessionID: owner.LeaseID, CreatedAt: owner.CreatedAt, HeartbeatTS: time.Now().UTC()}); err != nil {
			res = refuse("PREPARE_NOT_READY", "lease metadata failed; partial state retained: "+err.Error())
			return
		}
		if err := writePreservingIntent(target, baseSHA, intent); err != nil {
			res = refuse("PREPARE_NOT_READY", "intent publication failed; surviving evidence untouched: "+err.Error())
			return
		}
		// Do not mutate the common Git exclude file or create/reset pool state.
		// Native dirty-status helpers already filter lease.json. This new lease
		// can appear in raw git status until separately configured by its owner.
		if err := check(); err != nil {
			res = refuse("PRESERVATION_ADMISSION_REFUSED", err.Error())
			return
		}
		stamp, err := readOwnerStamp(target)
		lease, leaseErr := ReadWorkerLease(target)
		if err != nil || leaseErr != nil || stamp.PID != owner.PID || stamp.LeaseID != owner.LeaseID || lease.PID != owner.PID || lease.SessionID != owner.LeaseID {
			res = refuse("PREPARE_NOT_READY", "owner metadata readback failed; partial state retained")
			return
		}
		if live, known := liveWorktreeRegistration(root, target, git); !known || !live {
			res = refuse("PREPARE_NOT_READY", "new registration lost after stamping; partial state retained")
			return
		}
		if err := recheckPolicy(true); err != nil {
			res = refuse("PRESERVATION_CONFIG_REFUSED", err.Error())
			return
		}
		// No Git query may follow the publication fence inside this target lock.
		if err := check(); err != nil {
			res = refuse("PRESERVATION_ADMISSION_REFUSED", err.Error())
			return
		}
		res = Result{OK: true, Path: target, BaseSHA: baseSHA}
	})
	if err != nil {
		return refuse(PrepareCodeBusy, err.Error())
	}
	return res
}
