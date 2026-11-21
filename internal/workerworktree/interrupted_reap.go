package workerworktree

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ReapInterruptedResidue recovers the exact-path interrupted-reap residue: a managed
// worker whose .git marker and Git registration are gone but whose owner and pool
// sidecars remain, with a dead owner and a released lease. It removes only that
// residue and clears its exact sidecars, emitting a closed typed receipt. Every other
// state is refused and preserved byte-for-byte.
//
// The classifier is deliberately strict: ALL of the following must hold, or the
// residue is preserved and a refusal is returned:
//   - the target is a managed worker by name (IsWorkerWorktree);
//   - the target has no .git marker (the interrupted-reap signature);
//   - the target is absent from Git registration;
//   - the target has a valid owner sidecar (readOwnerStamp);
//   - the target has a valid pool sidecar (readPoolMember);
//   - the owner and pool sidecars agree on the owner identity;
//   - the owner PID is dead (processAlive);
//   - the owner lease is released (leaseLive).
//
// This is the exact-path complement to the bulk collectors (gc, reap --all-cold):
// it touches only the named target and its two sidecars, never a neighboring worker.
func ReapInterruptedResidue(root, target string, git GitRunner, processAlive ProcessLiveFn, leaseLive LeaseIDLiveFn) Result {
	target = filepath.Clean(strings.TrimSpace(target))
	refuse := func(code, reason, detail string) Result {
		return Result{OK: false, Code: code, Path: target, Preserved: true, Reason: reason, Detail: strings.TrimSpace(detail)}
	}
	// 1. Must be a managed worker by name.
	if !IsWorkerWorktree(target) {
		return refuse("REAP_PATH_REFUSED", "refusing to reap a non-worker worktree", "")
	}
	// 2. The .git marker must be gone (the interrupted-reap signature).
	if _, err := os.Lstat(filepath.Join(target, ".git")); err == nil {
		return refuse("REAP_PATH_REFUSED", "interrupted-reap recovery requires the .git marker to be absent", "git marker present")
	}
	// 3. Must be absent from Git registration.
	rc, out := run(git, root, []string{"worktree", "list", "--porcelain"})
	if rc != 0 {
		return refuse("REAP_PROOF_REFUSED", "worktree registration unavailable", tail(out, 300))
	}
	for _, entry := range parseWorktreeListing(out) {
		if samePath(entry.Path, target) {
			return refuse("REAP_PATH_REFUSED", "target is still a registered worker", "registration present")
		}
	}
	// 4. Valid matching owner sidecar.
	stamp, err := readOwnerStamp(target)
	if err != nil {
		return refuse("REAP_PROOF_REFUSED", "owner sidecar missing or invalid", err.Error())
	}
	// 5. Valid matching pool sidecar.
	meta, err := readPoolMember(target)
	if err != nil {
		return refuse("REAP_PROOF_REFUSED", "pool sidecar missing or invalid", err.Error())
	}
	// 6. Owner and pool must agree on the owner identity.
	if meta.Owner.PID != stamp.PID || meta.Owner.LeaseID != stamp.LeaseID {
		return refuse("REAP_PROOF_REFUSED", "owner and pool sidecars disagree on owner identity", "")
	}
	// 7. Dead owner.
	if processAlive == nil {
		return refuse("REAP_PROOF_REFUSED", "owner liveness unknown", "")
	}
	if processAlive(stamp.PID) {
		return refuse("REAP_PROOF_REFUSED", "owner process is live", "")
	}
	// 8. Released lease.
	if leaseLive == nil || stamp.LeaseID == "" {
		return refuse("REAP_PROOF_REFUSED", "lease liveness unknown", "")
	}
	if leaseLive(stamp.LeaseID) {
		return refuse("REAP_PROOF_REFUSED", "owner lease is live", "")
	}
	// All conditions met: recover the residue. Remove the directory first so a
	// sidecar-clearing failure cannot strand an unowned checkout; then clear the
	// exact sidecars and verify both are gone.
	if err := safeRemoveAll(target); err != nil {
		return refuse("REAP_PROOF_REFUSED", "residue removal failed", err.Error())
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		return refuse("REAP_PROOF_REFUSED", "residue remains after removal", "")
	}
	removePoolMemberState(target)
	removeOwnerStamp(target)
	if _, err := os.Stat(OwnerStampPath(target)); !os.IsNotExist(err) {
		return refuse("REAP_PROOF_REFUSED", "owner sidecar remains after removal", "")
	}
	if _, err := os.Stat(poolMemberPath(target)); !os.IsNotExist(err) {
		return refuse("REAP_PROOF_REFUSED", "pool sidecar remains after removal", "")
	}
	return Result{
		OK:      true,
		Code:    ReapCodeInterruptedResidueReaped,
		Path:    target,
		Removed: true,
		Reason:  fmt.Sprintf("interrupted-reap residue recovered: owner pid %d dead, lease %q released, unregistered, no .git marker", stamp.PID, stamp.LeaseID),
	}
}
