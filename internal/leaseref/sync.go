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
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// syncRefspec is the one refspec sync ever uses, on both the push and the fetch side:
// the whole lock namespace, forced (lease refs point at blobs — no ancestry, so any
// update of an existing ref needs the force), confined to refs/fak/locks/* at both
// ends.
//
// ZERO MATCHES ARE NOT SYMMETRIC (#5550). On the FETCH side a wildcard refspec matching
// no remote ref really is a successful no-op — git exits 0 and nothing changes. On the
// PUSH side that is not reliably true. With nothing to send AND no ref in common with the
// remote — a bare remote that advertises nothing yet — git 2.51 answers:
//
//	fatal: the remote end hung up unexpectedly
//	No refs in common and none specified; doing nothing.
//
// with EXIT 1: a legitimate no-op reported as a transport failure. Against a remote that
// does advertise a shared ref the identical push exits 0 ("Everything up-to-date"), so
// this is narrower than "every clone that has not taken a lease" — sync_test.go measures
// all three remote shapes. It still matters, because a failed push STOPS the sync, so a
// clone in the failing shape could never reach the fetch and therefore could never learn
// a peer's lease: zero leases in, zero leases out, permanently. Sync short-circuits an
// empty local namespace rather than hand this refspec to push at all (see
// localNamespaceEmpty) — which is also simply correct, since a push with nothing to send
// is a pointless subprocess in every shape. The sibling checkpoint substrate documents
// the same asymmetry on its own push refspec (internal/wipref.PushRefspec).
const syncRefspec = "+" + refPrefix + "*:" + refPrefix + "*"

// SyncResult reports what one Sync call actually did — which directions ran and the
// exact refspec used, so a caller (or a loop's ledger) can record the convergence
// action without re-deriving it.
type SyncResult struct {
	Remote  string `json:"remote"`
	Pushed  bool   `json:"pushed"`
	Fetched bool   `json:"fetched"`
	Refspec string `json:"refspec"`
	// PushSkippedEmpty reports that the push direction was ASKED FOR and had NOTHING TO
	// SEND: this clone holds zero refs under refs/fak/locks/, so no git push ran. Pushed
	// stays FALSE, because no push happened and this struct may not claim one; the pair
	// (Pushed=false, PushSkippedEmpty=true) is the honest shape of a clean no-op and is
	// not a failure — Sync returns a nil error and goes on to fetch. Kept as its own
	// field rather than folded into Pushed precisely so a ledger can tell "published my
	// leases" from "had none to publish".
	PushSkippedEmpty bool `json:"push_skipped_empty,omitempty"`
}

func validRemote(remote string) bool { return wipref.ValidRemote(remote) }

// Sync converges the refs/fak/locks/* namespace with remote: push the local records,
// then fetch the remote's (see the file doc for why that order and why a failed push
// stops the sync). doPush/doFetch select the directions; both false is a usage error,
// not a silent no-op. Errors are INFRASTRUCTURE only (git not executable, a non-zero
// push/fetch exit — network, auth, missing remote); there is no policy verdict here,
// because moving refs is transport, not admission. A requested push with NOTHING LOCAL
// to send is not one of those errors: it is skipped and reported as PushSkippedEmpty,
// and the fetch still runs (#5550).
func (s *Store) Sync(ctx context.Context, remote string, doPush, doFetch bool) (SyncResult, error) {
	if !wipref.ValidRemote(remote) {
		return SyncResult{}, fmt.Errorf("leaseref: invalid remote %q (must be one safe git argv token)", remote)
	}
	if !doPush && !doFetch {
		return SyncResult{}, fmt.Errorf("leaseref: sync with neither push nor fetch does nothing — enable at least one direction")
	}
	res := SyncResult{Remote: remote, Refspec: syncRefspec}
	if _, err := s.QuarantineMalformedLockRefs(ctx, time.Now()); err != nil {
		return res, err
	}
	if doPush {
		empty, err := s.localNamespaceEmpty(ctx)
		if err != nil {
			return res, err
		}
		if empty {
			// NOTHING TO SEND is not a failure (#5550). A zero-match push refspec can exit 1
			// (see syncRefspec), and because a failed push STOPS the sync, a clone holding
			// no leases could never reach the fetch that would teach it a peer's — the
			// failure was self-perpetuating: zero leases in, zero leases out, forever. The
			// push is skipped, not forgiven: no subprocess runs, so no real push failure can
			// be swallowed here, and the fetch below proceeds exactly as it would after a
			// successful publication. Force-fetching is safe in this case for the ordering's
			// own reason — there is no unpublished local state to regress.
			res.PushSkippedEmpty = true
		} else {
			// Stop here: force-fetching over unpublished local state would regress the
			// very leases this clone just wrote (the ordering rationale in the file doc).
			if err := s.runSyncDirection(ctx, "push", remote, " — sync stopped before fetch (never force-fetch over unpublished local leases)"); err != nil {
				return res, err
			}
			res.Pushed = true
		}
	}
	if doFetch {
		if err := s.runSyncDirection(ctx, "fetch", remote, ""); err != nil {
			return res, err
		}
		res.Fetched = true
	}
	return res, nil
}

// localNamespaceEmpty answers ONE question, positively: does this clone hold zero refs
// under refs/fak/locks/? It is what separates "the push had nothing to send" from "the
// push failed", and it is deliberately a POSITIVE test — asking the local ref database
// what is THERE — rather than a negative one that inspects a failure after the fact.
//
// The alternative was to run the push and forgive an exit whose stderr matched git's
// zero-match wording. That is rejected: git's stderr is UI, not contract (it has been
// reworded across releases and is translatable), and exit 1 is ALSO how git reports an
// auth failure, an unknown remote, and a rejected update. A matcher that drifted even
// slightly would start swallowing those, and a swallowed push failure strands this
// clone's lease state on one machine while reporting success — strictly worse than the
// spurious failure being fixed. Enumerating BEFORE the push cannot confuse the two
// because it never examines a failure at all: the push either runs untouched, or never
// runs.
//
// Only a CLEAN enumeration authorizes the skip. If for-each-ref itself exits non-zero
// the answer is UNKNOWN, and unknown falls through to the push — a spurious sync failure
// is a much cheaper mistake than silently not publishing a lease this clone really holds.
// --count=1 stops git after the first match: the caller needs "any?", not "how many".
func (s *Store) localNamespaceEmpty(ctx context.Context) (bool, error) {
	out, code, err := s.run(ctx, s.dir, "for-each-ref", "--count=1", "--format=%(refname)", refPrefix)
	if err != nil {
		return false, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return false, nil // unknown — fall through to the push rather than skip it
	}
	return strings.TrimSpace(out) == "", nil
}

// runSyncDirection runs one sync direction (git verb remote syncRefspec) and
// translates its outcome into the error shape both Sync directions use: a
// non-executable git is an infrastructure error, and a non-zero exit is a
// verb-tagged message with an optional direction-specific extra suffix (push
// carries the stop-before-fetch rationale; fetch carries none). Returns nil
// once the direction completed (exit 0).
func (s *Store) runSyncDirection(ctx context.Context, verb, remote, extra string) error {
	_, code, err := s.run(ctx, s.dir, verb, remote, syncRefspec)
	if err != nil {
		return fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return fmt.Errorf("leaseref: %s %s %s exited %d%s", verb, remote, syncRefspec, code, extra)
	}
	return nil
}

// QuarantineMalformedLockRefs moves malformed loose files out of refs/fak/locks
// before a push or fetch asks git to enumerate the ref database. A torn loose ref
// (notably an all-NUL object id) makes otherwise-unrelated fetches fail before git
// can repair anything. Valid refs and transient *.lock files are never moved.
func (s *Store) QuarantineMalformedLockRefs(ctx context.Context, now time.Time) ([]string, error) {
	out, code, err := s.run(ctx, s.dir, "rev-parse", "--path-format=absolute", "--git-common-dir")
	if err != nil {
		return nil, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return nil, fmt.Errorf("leaseref: rev-parse --git-common-dir exited %d", code)
	}
	common := strings.TrimSpace(out)
	if common == "" {
		// Injected runners used by the pure argv tests predate this filesystem
		// maintenance seam. Real git always returns a path here.
		return nil, nil
	}
	return QuarantineMalformedLockRefsInDir(
		filepath.Join(common, "refs", "fak", "locks"),
		filepath.Join(common, "fak", "quarantine", "malformed-lock-refs", now.UTC().Format("20060102T150405.000000000Z")),
	)
}

// QuarantineMalformedLockRefsInDir is the bounded filesystem core. It examines
// regular loose-ref files only, validates their trimmed content as a SHA-1 or
// SHA-256 object ID, and atomically renames malformed entries into quarantine.
func QuarantineMalformedLockRefsInDir(refDir, quarantineDir string) ([]string, error) {
	var moved []string
	err := filepath.WalkDir(refDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			if os.IsNotExist(walkErr) {
				return nil
			}
			return walkErr
		}
		if d.IsDir() || strings.HasSuffix(d.Name(), lockSuffix) {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if validObjectID(strings.TrimSpace(string(b))) {
			return nil
		}
		rel, err := filepath.Rel(refDir, path)
		if err != nil || rel == "." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return fmt.Errorf("leaseref: malformed ref escaped lock namespace: %s", path)
		}
		dst := filepath.Join(quarantineDir, rel)
		if err := os.MkdirAll(filepath.Dir(dst), 0o700); err != nil {
			return err
		}
		if err := os.Rename(path, dst); err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return fmt.Errorf("leaseref: quarantine malformed ref %s: %w", filepath.ToSlash(rel), err)
		}
		moved = append(moved, filepath.ToSlash(rel))
		return nil
	})
	if os.IsNotExist(err) {
		return moved, nil
	}
	return moved, err
}

func validObjectID(s string) bool {
	if len(s) != 40 && len(s) != 64 {
		return false
	}
	nonzero := false
	for i := range len(s) {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
		if c != '0' {
			nonzero = true
		}
	}
	return nonzero
}
