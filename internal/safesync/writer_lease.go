package safesync

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// writer_lease.go — a cooperative, worktree-scoped advisory writer lease that closes
// the remaining assess->checkout race in Apply (#4240).
//
// #4215 removed the explicit assess->preclean data-loss window and pinned the target,
// but git's index lock is not a WORKTREE lock: a fak-managed peer writer can still edit
// a path between the moment Assess classifies it and the moment `git merge --ff-only`
// writes it. This lease is the rendezvous that closes that window WITHOUT stashing,
// force, or a journaled protocol: Apply holds it for its whole assess+apply window, and
// every *cooperative* fak-managed worktree writer calls AcquireWriterLease first and
// blocks/refuses (WriterLeaseHeldError) while it is held.
//
// It is deliberately COOPERATIVE (advisory), matching the issue's scope: it protects
// against fak-managed writers that honor it, not against a raw program that overwrites
// the path anyway (that is explicitly out of scope — only a journaled protocol could).
//
// Crash recovery is TTL-based: the on-disk record carries the acquire time, and a lease
// older than ttl is reclaimed as crash residue (a holder that died without releasing).
// The apply window is sub-second, so DefaultWriterLeaseTTL is generous enough that only
// a genuinely dead holder is ever reclaimed. Release is owner-checked, so a writer that
// was already reclaimed by a peer never deletes the peer's live lease.
//
// The lock lives in the per-worktree git dir (never the tree), so it is invisible to
// git status, never staged, and correctly scoped per worktree (a linked worktree has
// its own dir, hence its own lease).

// DefaultWriterLeaseTTL bounds how long a lease is honored before a peer may reclaim it
// as crash residue. Apply's assess+fast-forward window is sub-second; a generous default
// means only a genuinely dead holder is ever reclaimed.
const DefaultWriterLeaseTTL = 2 * time.Minute

// writerLeaseFile is the lock basename, placed in the per-worktree git dir.
const writerLeaseFile = "fak-worktree-writer.lock"

// WriterLeaseInfo is the on-disk owner record of a held worktree writer lease.
type WriterLeaseInfo struct {
	Owner        string `json:"owner"`
	PID          int    `json:"pid"`
	Host         string `json:"host,omitempty"`
	AcquiredUnix int64  `json:"acquired_unix"`
}

func (i WriterLeaseInfo) acquiredAt() time.Time { return time.Unix(i.AcquiredUnix, 0) }

// WriterLeaseHeldError reports that a live peer holds the worktree writer lease. A
// cooperative writer inspects Info and blocks/refuses rather than overwriting bytes.
type WriterLeaseHeldError struct{ Info WriterLeaseInfo }

func (e *WriterLeaseHeldError) Error() string {
	return fmt.Sprintf("worktree writer lease held by %s (pid %d) since %s",
		e.Info.Owner, e.Info.PID, e.Info.acquiredAt().Format(time.RFC3339))
}

// WriterLease is a held worktree writer lease. Release it when the write window closes.
type WriterLease struct {
	path string
	info WriterLeaseInfo
}

// Info returns the owner record of a held lease.
func (l *WriterLease) Info() WriterLeaseInfo { return l.info }

// AcquireWriterLease takes the cooperative worktree writer lease for repo. It returns
// (lease, nil) on success; (nil, *WriterLeaseHeldError) when a live peer holds it; and
// (nil, err) on an I/O failure. A lease older than ttl is reclaimed as crash residue
// (a holder that died without releasing). now/ttl default to time.Now /
// DefaultWriterLeaseTTL when zero, so a plain writer can call AcquireWriterLease(repo,
// owner, nil, 0).
func AcquireWriterLease(repo, owner string, now func() time.Time, ttl time.Duration) (*WriterLease, error) {
	if now == nil {
		now = time.Now
	}
	if ttl <= 0 {
		ttl = DefaultWriterLeaseTTL
	}
	dir, err := worktreeGitDir(repo)
	if err != nil {
		return nil, err
	}
	path := filepath.Join(dir, writerLeaseFile)
	host, _ := os.Hostname()
	rec := WriterLeaseInfo{Owner: strings.TrimSpace(owner), PID: os.Getpid(), Host: host, AcquiredUnix: now().Unix()}
	if rec.Owner == "" {
		rec.Owner = fmt.Sprintf("pid-%d", rec.PID)
	}

	// Two attempts: the second only runs after we reclaim a stale lease.
	for attempt := 0; attempt < 2; attempt++ {
		ok, err := writeLeaseExclusive(path, rec)
		if err != nil {
			return nil, err
		}
		if ok {
			return &WriterLease{path: path, info: rec}, nil
		}
		// Someone holds it: read the record and decide live-vs-stale.
		cur, rerr := readLease(path)
		if rerr != nil {
			if os.IsNotExist(rerr) {
				continue // the holder released between our create and read; retry
			}
			return nil, rerr
		}
		if !leaseStale(cur, now(), ttl) {
			return nil, &WriterLeaseHeldError{Info: cur}
		}
		// Stale (crash residue / expired TTL): reclaim, then retry the exclusive create.
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return nil, err
		}
	}

	// Lost the reclaim race to another writer that recreated it first: treat as held.
	cur, rerr := readLease(path)
	if rerr != nil {
		if os.IsNotExist(rerr) {
			// It vanished again; one more exclusive create decides the outcome.
			if ok, werr := writeLeaseExclusive(path, rec); werr != nil {
				return nil, werr
			} else if ok {
				return &WriterLease{path: path, info: rec}, nil
			}
			cur, rerr = readLease(path)
			if rerr != nil {
				return nil, rerr
			}
		} else {
			return nil, rerr
		}
	}
	return nil, &WriterLeaseHeldError{Info: cur}
}

// Release drops the lease if we still own it. It is a no-op when a peer already
// reclaimed the lease (our record was overwritten), so a reclaimed writer never deletes
// the peer's live lease. Safe to call on a nil lease.
func (l *WriterLease) Release() error {
	if l == nil {
		return nil
	}
	cur, err := readLease(l.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if cur != l.info {
		return nil // reclaimed by a peer; leave theirs intact
	}
	if err := os.Remove(l.path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// leaseStale reports whether info is old enough (>= ttl) to reclaim as crash residue.
func leaseStale(info WriterLeaseInfo, now time.Time, ttl time.Duration) bool {
	return now.Sub(info.acquiredAt()) >= ttl
}

// writeLeaseExclusive atomically creates path with rec, reporting ok=false (no error)
// when the file already exists. The O_EXCL create is the mutual-exclusion primitive.
func writeLeaseExclusive(path string, rec WriterLeaseInfo) (bool, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return false, nil
		}
		return false, err
	}
	defer f.Close()
	enc, err := json.Marshal(rec)
	if err != nil {
		return false, err
	}
	if _, err := f.Write(enc); err != nil {
		return false, err
	}
	return true, nil
}

// readLease reads and parses the on-disk record. A corrupt/partial record parses to the
// zero value, whose acquiredAt() is the Unix epoch, so leaseStale() treats it as
// reclaimable crash residue rather than a permanent wedge.
func readLease(path string) (WriterLeaseInfo, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return WriterLeaseInfo{}, err
	}
	var info WriterLeaseInfo
	if err := json.Unmarshal(b, &info); err != nil {
		return WriterLeaseInfo{}, nil
	}
	return info, nil
}

// worktreeGitDir resolves the per-worktree git dir that holds the lease lock: <repo>/.git
// when it is a directory, or the "gitdir: <path>" target when <repo>/.git is the pointer
// file of a linked worktree.
func worktreeGitDir(repo string) (string, error) {
	repo = strings.TrimSpace(repo)
	if repo == "" {
		repo = "."
	}
	dot := filepath.Join(repo, ".git")
	fi, err := os.Stat(dot)
	if err != nil {
		return "", err
	}
	if fi.IsDir() {
		return dot, nil
	}
	b, err := os.ReadFile(dot)
	if err != nil {
		return "", err
	}
	line := strings.TrimSpace(string(b))
	const prefix = "gitdir:"
	if strings.HasPrefix(line, prefix) {
		gd := strings.TrimSpace(strings.TrimPrefix(line, prefix))
		if !filepath.IsAbs(gd) {
			gd = filepath.Join(repo, gd)
		}
		return filepath.Clean(gd), nil
	}
	return "", fmt.Errorf("unrecognized .git pointer file in %s", repo)
}
