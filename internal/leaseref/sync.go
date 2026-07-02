package leaseref

// sync.go is the CONVERGENCE verb of the cross-machine lease substrate (#825): one
// call that moves the refs/fak/locks/* namespace between this clone and a remote.
// The substrate's whole premise is that lease state rides ordinary git fetch/push —
// but until this verb, the riding was OPERATOR MUSCLE MEMORY: every consumer doc
// said "run `git fetch origin 'refs/fak/locks/*:refs/fak/locks/*'` before claiming,
// `git push` after" (cmd/fak/intent.go, docs/cli-reference.md), and nothing in the
// binary did it. On a single machine that is a tolerable wart; with multiple
// hardware nodes developing concurrently it is the difference between an arbiter
// that SEES a peer node's lease and one that is structurally blind. Sync is the
// missing wire: `fak leaseref sync` in a loop tick (or before an acquire / after a
// release) converges the namespace without anyone remembering a refspec.
//
// ORDER (push THEN fetch — load-bearing). The fetch side uses a FORCE refspec
// (+refs/fak/locks/*:refs/fak/locks/*), because lease refs point at blobs — there is
// no ancestry, so a non-forced update of an existing ref is rejected by git on both
// sides. A force-fetch OVERWRITES the local ref with the remote's value, which would
// REGRESS a just-acquired local lease the remote has not seen yet (local generation 2
// force-reset to the remote's stale generation 1 — the caller's own fencing token
// silently rolled back). Pushing first publishes local state, making the fetch a
// no-op for every ref this clone last wrote; only genuinely-newer peer refs change
// locally. For the same reason a FAILED push STOPS the sync: force-fetching over
// unpublished local state is exactly the regression the ordering exists to prevent.
//
// THE HONEST BOUNDARY (kept in lockstep with the package doc):
//   - Still DISTRIBUTION / VISIBILITY, not cross-machine arbitration. Two nodes that
//     write the same lease id in the same sync window last-writer-win on the remote;
//     sync makes the conflict VISIBLE (Fence reads the surviving record and refuses
//     the stale holder), it does not prevent it. Atomic cross-node acquisition needs
//     a single arbiter (a dev-server / gateway seam) — out of scope here, by design.
//   - DELETIONS DO NOT RIDE A REFSPEC. A glob push/fetch transports existing refs
//     only; a released/reaped lease converges on peers via TTL expiry + each clone's
//     own `fak leaseref reap`, not via sync. (A prune-style sync is deliberately NOT
//     offered: `fetch --prune` would delete this clone's not-yet-pushed acquisitions,
//     `push --prune` would delete a peer's not-yet-fetched ones — both destroy live
//     state on the losing side of a window.)
//   - SIDE REFS ONLY, like every write in this package: the refspec is confined to
//     refs/fak/locks/* on both ends. No branch, no HEAD, no tag ever moves.

import (
	"context"
	"fmt"
	"strings"
)

// syncRefspec is the one refspec sync ever uses, on both the push and the fetch side:
// the whole lock namespace, forced (lease refs point at blobs — no ancestry, so any
// update of an existing ref needs the force), confined to refs/fak/locks/* at both
// ends. A wildcard refspec matching ZERO refs is a successful no-op in git, so syncing
// an empty namespace is clean on both sides.
const syncRefspec = "+" + refPrefix + "*:" + refPrefix + "*"

// SyncResult reports what one Sync call actually did — which directions ran and the
// exact refspec used, so a caller (or a loop's ledger) can record the convergence
// action without re-deriving it.
type SyncResult struct {
	Remote  string `json:"remote"`
	Pushed  bool   `json:"pushed"`
	Fetched bool   `json:"fetched"`
	Refspec string `json:"refspec"`
}

// validRemote rejects a remote that cannot safely be one git argv token: empty, a
// leading dash (would misparse as a flag), or embedded whitespace/control bytes. A
// remote NAME (origin) and a remote URL (ssh://..., https://...) both pass; this is
// argv hygiene, not URL validation — git itself decides whether the remote exists.
func validRemote(remote string) bool {
	if remote == "" || strings.HasPrefix(remote, "-") {
		return false
	}
	for _, c := range []byte(remote) {
		if c <= ' ' || c == 0x7f {
			return false
		}
	}
	return true
}

// Sync converges the refs/fak/locks/* namespace with remote: push the local records,
// then fetch the remote's (see the file doc for why that order and why a failed push
// stops the sync). doPush/doFetch select the directions; both false is a usage error,
// not a silent no-op. Errors are INFRASTRUCTURE only (git not executable, a non-zero
// push/fetch exit — network, auth, missing remote); there is no policy verdict here,
// because moving refs is transport, not admission.
func (s *Store) Sync(ctx context.Context, remote string, doPush, doFetch bool) (SyncResult, error) {
	if !validRemote(remote) {
		return SyncResult{}, fmt.Errorf("leaseref: invalid remote %q (must be one safe git argv token)", remote)
	}
	if !doPush && !doFetch {
		return SyncResult{}, fmt.Errorf("leaseref: sync with neither push nor fetch does nothing — enable at least one direction")
	}
	res := SyncResult{Remote: remote, Refspec: syncRefspec}
	if doPush {
		if _, code, err := s.run(ctx, s.dir, "push", remote, syncRefspec); err != nil {
			return res, fmt.Errorf("leaseref: git not executable: %w", err)
		} else if code != 0 {
			// Stop here: force-fetching over unpublished local state would regress the
			// very leases this clone just wrote (the ordering rationale in the file doc).
			return res, fmt.Errorf("leaseref: push %s %s exited %d — sync stopped before fetch (never force-fetch over unpublished local leases)", remote, syncRefspec, code)
		}
		res.Pushed = true
	}
	if doFetch {
		if _, code, err := s.run(ctx, s.dir, "fetch", remote, syncRefspec); err != nil {
			return res, fmt.Errorf("leaseref: git not executable: %w", err)
		} else if code != 0 {
			return res, fmt.Errorf("leaseref: fetch %s %s exited %d", remote, syncRefspec, code)
		}
		res.Fetched = true
	}
	return res, nil
}
