package safecommit

import (
	"context"
	"errors"
	"fmt"
	"io/fs"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// writer_lease.go — the #4611 wiring that makes the fak-managed commit writer honor the
// cooperative worktree writer lease from #4240 (safesync.AcquireWriterLease).
//
// safesync.Apply holds that lease across its whole assess+apply window so no fak-managed
// peer can edit a classified path between assessment and the fast-forward checkout. This
// package is the other side of that contract: every lock-riding mutation CommitWith
// performs (the pathspec-scoped add, the commit, the auto-index note write) first takes
// the same lease, so
//
//   - a managed commit is REFUSED (WRITER_LEASE_HELD, a retryable value) while a sync
//     apply window is open, and
//   - a sync apply is refused (WriterLeaseHeldError → Assessment.Lease) while a managed
//     commit's mutation window is open,
//
// each direction witnessed by the temp-repo pairing test in
// internal/safesync/writer_lease_wiring_test.go. Because `fak commit` and `fak sweep
// --apply` both route through CommitWith, wiring the lease here covers every fak-managed
// staging/commit writer with one seam. The lease is cooperative/advisory by design
// (#4240): raw non-fak writers that ignore it are explicitly out of scope.

// ReasonWriterLeaseHeld — part of the closed refusal vocabulary in safecommit.go —
// reports that a peer fak-managed writer (a sync apply window) holds the #4240 worktree
// writer lease, so the commit refused before mutating the tree. Retryable: nothing
// landed (the exit-3 pre-commit class, like LOCK_BUSY).
const ReasonWriterLeaseHeld = "WRITER_LEASE_HELD"

// writerLeaseOwner labels the commit writer in the on-disk lease record, so a refused
// peer's error names who holds the tree.
const writerLeaseOwner = "fak-commit"

// acquireWorktreeWriterLease takes the cooperative worktree writer lease for the
// commit's lock-riding mutation window. Mirroring acquireCommitLock's contract, a held
// lease is a VALUE (a non-empty heldDetail for Result.Detail), never an error; err is
// infrastructure only. On success the returned release drops the lease.
func acquireWorktreeWriterLease(opts Options) (release func(), heldDetail string, err error) {
	var lease *safesync.WriterLease
	var lerr error
	if opts.Lock.Timeout > 0 && !opts.Lock.NoWait {
		lease, lerr = safesync.AcquireQueuedWriterLease(context.Background(), opts.Dir, writerLeaseOwner, opts.Now, 0, opts.Lock.Timeout)
	} else {
		lease, lerr = safesync.AcquireWriterLease(opts.Dir, writerLeaseOwner, opts.Now, 0)
	}
	if lerr != nil {
		if errors.Is(lerr, safesync.ErrLeaseOwnerUnavailable) {
			return nil, lerr.Error(), nil
		}
		var held *safesync.WriterLeaseHeldError
		if errors.As(lerr, &held) {
			return nil, held.Error(), nil
		}
		if errors.Is(lerr, fs.ErrNotExist) {
			// No .git exists on the real filesystem: an injected-runner harness (a fake
			// repo path), not a real work tree — the work-tree gate already vouched for
			// the repo through the runner, and a real repo always has a per-worktree git
			// dir for the lease to live in (so does every cooperative peer). Proceed
			// leaseless rather than failing a repo-less harness on infrastructure.
			return func() {}, "", nil
		}
		return nil, "", fmt.Errorf("safecommit: worktree writer lease: %w", lerr)
	}
	return func() { _ = lease.Release() }, "", nil
}
