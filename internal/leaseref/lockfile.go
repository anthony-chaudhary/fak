package leaseref

// lockfile.go reaps the ORPHANED FILESYSTEM LOCKS under the refs/fak/locks/*
// namespace — the missing third reaper #4605 names. Reap and ReapSessions delete
// expired lease RECORDS (refs, via `update-ref -d`); neither can touch the transient
// `<gitdir>/refs/fak/locks/<name>.lock` FILE git itself creates while an update-ref
// compare-and-swap is in flight. A healthy CAS holds that file for milliseconds; a
// holder killed mid-write (timeout SIGTERM, crash, power loss) leaves it behind
// forever, because git only removes it on the path it never got to run. The observed
// harm is double (#4605): gitgate's git-maint lock preflight counts each ghost as an
// in-flight transaction and defers object-DB maintenance PERMANENTLY, and the next
// heartbeat CAS on that same ref cannot take its lock, so a LIVE session can be
// unable to publish liveness.
//
// THE SAFETY ARGUMENT (why deleting someone else's lock file can be safe at all):
//   - NAMESPACE CONFINEMENT, by construction. The sweep walks ONLY
//     <git-common-dir>/refs/fak/locks/ — fak's own side-ref namespace. It can never
//     see index.lock, packed-refs.lock, a refs/heads lock, or the main-worktree
//     fak-commit.lock (that one belongs to safecommit/gitgate — see the companion
//     issues). The worst possible mistake is bounded to fak's own lease plumbing.
//   - AGE BOUND ≥ the longest legitimate hold by ~6 orders of magnitude. A real
//     update-ref lock lives milliseconds; the sweep only removes a .lock whose mtime
//     is at least maxAge old (default: the 2400 s session-heartbeat TTL — the bound
//     the issue derives: no process holds a ref lock longer than the lease it
//     guards). A git lock file records no owner pid, so age against the TTL is the
//     conservative orphan proxy, and a FRESH .lock (a possibly-live CAS) is always
//     kept and reported, never raced.
//   - IDEMPOTENT + BEST-EFFORT, like every reaper here: a file already gone lost a
//     race to a peer's sweep (fine — the desired post-state holds), and a per-file
//     remove failure (e.g. a Windows handle still open, which itself proves a live
//     holder) is joined into err and never aborts the sweep. Git recreates the lock
//     on the next real update, so even a wrong delete degrades to a retried CAS,
//     never to corruption.
//
// Wiring this into session start, the dispatch tick, and the git-maint lock
// preflight is the callers' side of #4605 and lives in their trees, not here.

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DefaultLockFileMaxAge is the sweep's default orphan bound: the 2400 s session
// heartbeat TTL (#4605). A legitimate update-ref lock lives milliseconds, so any
// .lock older than the very lease it guards is definitively orphaned; equating the
// bound to the TTL keeps "the lock outlived its lease" as the one honest rule.
const DefaultLockFileMaxAge = 2400 * time.Second

// lockSuffix marks git's transient lockfiles: <ref-path>.lock alongside the loose
// ref it guards. Only paths with this suffix are ever considered by the sweep.
const lockSuffix = ".lock"

// ReapLockFiles removes every orphaned *.lock file under the repo's own
// refs/fak/locks/ directory — the filesystem-side reaper completing the ref-side
// Reap/ReapSessions pair. It resolves the git common dir through the one Runner seam
// (`rev-parse --path-format=absolute --git-common-dir`; the COMMON dir, because a
// linked worktree shares its refs — and their locks — with the main clone), then
// sweeps <common>/refs/fak/locks with ReapLockFilesInDir. maxAge <= 0 means
// DefaultLockFileMaxAge. Returns the removed paths and the fresh .lock files kept
// (each repo-relative to the locks dir), so a maintenance preflight can distinguish
// "orphans cleared" from "a possibly-live CAS is genuinely in flight".
func (s *Store) ReapLockFiles(ctx context.Context, now time.Time, maxAge time.Duration) (reaped, kept []string, err error) {
	out, code, err := s.run(ctx, s.dir, "rev-parse", "--path-format=absolute", "--git-common-dir")
	if err != nil {
		return nil, nil, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return nil, nil, fmt.Errorf("leaseref: rev-parse --git-common-dir exited %d", code)
	}
	common := strings.TrimSpace(out)
	if common == "" {
		return nil, nil, fmt.Errorf("leaseref: rev-parse --git-common-dir produced no path")
	}
	return ReapLockFilesInDir(filepath.Join(common, "refs", "fak", "locks"), now, maxAge)
}

// ReapLockFilesInDir is the pure filesystem core of ReapLockFiles: it walks dir,
// removes every regular *.lock file whose mtime is at least maxAge before now, and
// keeps (and reports) the rest. maxAge <= 0 means DefaultLockFileMaxAge. An absent
// dir is a clean no-op (nothing was ever locked — the same "absence is a valid,
// empty view" rule as the ref readers), a file that vanished mid-sweep lost a race
// to a peer and is skipped, and a per-file remove failure is joined into err without
// aborting the sweep. A .lock with an mtime in the future (clock skew) is KEPT —
// unknown age fails closed to not-orphaned. Non-.lock entries are never touched.
// Returned paths are relative to dir, slash-separated, sorted by the walk order
// (lexical), so output is stable across platforms.
func ReapLockFilesInDir(dir string, now time.Time, maxAge time.Duration) (reaped, kept []string, err error) {
	return walkLockFiles(dir, now, maxAge, true)
}

// ScanLockFilesInDir is ReapLockFilesInDir's read-only twin: it classifies the same
// directory by the same age bound and returns which *.lock files WOULD be reaped and
// which would be kept, removing nothing. It exists so a dry-run (`fak git-daily
// --dry-run`) previews the orphan set through the SAME walk that would do the reaping
// — a caller that re-derived "which locks are orphaned" from its own copy of the rule
// is exactly how a preview drifts away from the action it previews.
func ScanLockFilesInDir(dir string, now time.Time, maxAge time.Duration) (orphans, kept []string, err error) {
	return walkLockFiles(dir, now, maxAge, false)
}

// walkLockFiles is the one classification walk behind both the reaping and the
// read-only entry points. With apply=false it stops short of os.Remove and reports the
// same partition; every other rule (age bound, future-mtime fail-closed, vanished-file
// tolerance, error joining) is shared by construction rather than by duplication.
func walkLockFiles(dir string, now time.Time, maxAge time.Duration, apply bool) (reaped, kept []string, err error) {
	if maxAge <= 0 {
		maxAge = DefaultLockFileMaxAge
	}
	var errs []error
	walkErr := filepath.WalkDir(dir, func(path string, d fs.DirEntry, werr error) error {
		if werr != nil {
			if os.IsNotExist(werr) {
				return nil // absent dir (or an entry a peer just reaped): a valid, empty view
			}
			errs = append(errs, fmt.Errorf("reap lock files: walk %s: %w", path, werr))
			return nil // keep sweeping the rest
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), lockSuffix) {
			return nil
		}
		rel, rerr := filepath.Rel(dir, path)
		if rerr != nil {
			rel = path // unreachable in practice; fall back to the absolute path
		}
		rel = filepath.ToSlash(rel)
		info, ierr := d.Info()
		if ierr != nil {
			if os.IsNotExist(ierr) {
				return nil // vanished mid-sweep: a peer reaped it first
			}
			errs = append(errs, fmt.Errorf("reap lock files: stat %s: %w", rel, ierr))
			return nil
		}
		age := now.Sub(info.ModTime())
		if age < maxAge {
			kept = append(kept, rel) // fresh (or future-dated): possibly a live CAS, never raced
			return nil
		}
		if !apply {
			reaped = append(reaped, rel) // read-only preview: the orphan set, unremoved
			return nil
		}
		if rmerr := os.Remove(path); rmerr != nil {
			if os.IsNotExist(rmerr) {
				return nil // already gone: the desired post-state holds
			}
			// An open handle here (Windows) itself proves a live holder — keep, don't abort.
			errs = append(errs, fmt.Errorf("reap lock files: remove %s: %w", rel, rmerr))
			return nil
		}
		reaped = append(reaped, rel)
		return nil
	})
	if walkErr != nil && !os.IsNotExist(walkErr) {
		errs = append(errs, fmt.Errorf("reap lock files: walk %s: %w", dir, walkErr))
	}
	return reaped, kept, errors.Join(errs...)
}
