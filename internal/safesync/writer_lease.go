package safesync

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
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
// LONG WINDOWS HEARTBEAT (#4612, closing #4240's "apply fits inside the TTL"
// assumption): a window that outlives the TTL — a pathologically slow apply, a huge
// fast-forward on a slow FS — must not have its live lease reclaimed mid-window as
// crash residue. Refresh renews the record's staleness clock (RenewedUnix, the same
// renew-aware window leaseref.Record uses), and Apply runs a keepAlive heartbeat for
// its whole window, so staleness measures a holder's SILENCE, not the window's length.
//
// CROSS-HOST SCOPE (#4612, closing #4240's "single host" assumption): file-only scope
// is SUFFICIENT for this race, because the lock is exactly as visible as the worktree
// it protects. The race is two writers mutating ONE worktree's bytes; a writer on
// another machine only reaches those bytes through a shared mount of the same checkout,
// and that mount carries the per-worktree git dir — hence this lock file — with it.
// Reclaim is deliberately TTL-only, never a local pid/hostname liveness probe (which
// would misjudge a live foreign-host holder as dead: its pid means nothing here), so a
// peer host's record is honored identically to a local one (see
// TestWriterLeaseCrossHostPeerIsHonored). Publishing the lease through the
// refs/fak/locks/* leaseref namespace (internal/leaseref, tier 1 — a direct import
// would be layering-legal, no injected seam needed) was considered and REJECTED as
// same-scope-redundant: on a shared mount the ref would live in the SAME git dir as
// this file with strictly weaker atomicity (update-ref is last-writer-wins; the O_EXCL
// create is the exclusion primitive), and across distinct clones there is no shared
// worktree — each host fast-forwards its own bytes — so a fetched ref would guard
// nothing this lock does not. Honest residuals, out of scope: externally mirrored
// checkouts (rsync-style sync propagates neither locks nor refs atomically — a raw-
// writer topology, out of scope per #4240), and cross-host clock skew (a reader more
// than ttl ahead of the holder's clock can mis-reclaim; the heartbeat bounds the
// exposure to one renew interval, and NTP-order skew is far below the 2m TTL). A
// linked worktree shared without its main repo is not a functioning checkout on the
// peer host (git cannot resolve its gitdir pointer), so no fak-managed writer arises
// there at all.
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
	// RenewedUnix is the instant (unix seconds) of the last same-holder heartbeat renew
	// (#4612) — the same renew-aware staleness clock leaseref.Record.RenewedAt keeps. A
	// renew moves the TTL window forward without changing the holder; 0 means never
	// renewed, in which case staleness measures from AcquiredUnix exactly as a pre-#4612
	// record did (omitempty keeps such records byte-identical).
	RenewedUnix int64 `json:"renewed_unix,omitempty"`
}

func (i WriterLeaseInfo) acquiredAt() time.Time { return time.Unix(i.AcquiredUnix, 0) }

// effectiveActiveAt is the later of the acquire and last-renew instants — the moment the
// TTL staleness window is measured from, so a heartbeated lease stays live while only a
// genuinely silent (crashed) holder ages out.
func (i WriterLeaseInfo) effectiveActiveAt() time.Time {
	if i.RenewedUnix > i.AcquiredUnix {
		return time.Unix(i.RenewedUnix, 0)
	}
	return i.acquiredAt()
}

// ErrWriterLeaseLost reports a Refresh on a lease this holder no longer owns: a peer
// reclaimed it (the record was overwritten or removed) after the TTL lapsed. The holder's
// window is no longer protected — it must stop writing and must NOT clobber the peer's
// live record.
var ErrWriterLeaseLost = errors.New("worktree writer lease lost (reclaimed by a peer)")

// WriterLeaseHeldError reports that a live peer holds the worktree writer lease. A
// cooperative writer inspects Info and blocks/refuses rather than overwriting bytes.
type WriterLeaseHeldError struct{ Info WriterLeaseInfo }

func (e *WriterLeaseHeldError) Error() string {
	on := ""
	if e.Info.Host != "" {
		on = " on " + e.Info.Host // names the peer HOST, so a shared-mount refusal is diagnosable cross-machine
	}
	return fmt.Sprintf("worktree writer lease held by %s (pid %d%s) since %s",
		e.Info.Owner, e.Info.PID, on, e.Info.acquiredAt().Format(time.RFC3339))
}

// WriterLease is a held worktree writer lease. Release it when the write window closes.
type WriterLease struct {
	path string
	ttl  time.Duration

	mu       sync.Mutex // guards info: the keepAlive heartbeat renews it concurrently with Release
	info     WriterLeaseInfo
	lost     chan struct{}
	lostOnce sync.Once
}

// Info returns the owner record of a held lease.
func (l *WriterLease) Info() WriterLeaseInfo {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.info
}

// Lost reports whether this holder has observed that a peer displaced its lease. Writers
// must consult it immediately before publishing work produced inside the leased window.
func (l *WriterLease) Lost() bool {
	select {
	case <-l.lost:
		return true
	default:
		return false
	}
}

// LostSignal closes when a refresh proves that this holder no longer owns the lease.
// It never closes for transient I/O failures.
func (l *WriterLease) LostSignal() <-chan struct{} { return l.lost }

func (l *WriterLease) markLost() { l.lostOnce.Do(func() { close(l.lost) }) }

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
			return &WriterLease{path: path, ttl: ttl, info: rec, lost: make(chan struct{})}, nil
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
				return &WriterLease{path: path, ttl: ttl, info: rec, lost: make(chan struct{})}, nil
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

// ActiveWriterLease checks whether repo currently has an active, non-stale writer lease.
func ActiveWriterLease(repo string, now func() time.Time, ttl time.Duration) (*WriterLeaseInfo, bool) {
	if now == nil {
		now = time.Now
	}
	if ttl <= 0 {
		ttl = DefaultWriterLeaseTTL
	}
	gd, err := worktreeGitDir(repo)
	if err != nil {
		return nil, false
	}
	path := filepath.Join(gd, writerLeaseFile)
	info, err := readLease(path)
	if err != nil {
		return nil, false
	}
	if !leaseStale(info, now(), ttl) {
		return &info, true
	}
	return nil, false
}

// Release drops the lease if we still own it. It is a no-op when a peer already
// reclaimed the lease (our record was overwritten), so a reclaimed writer never deletes
// the peer's live lease. Safe to call on a nil lease.
func (l *WriterLease) Release() error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
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

// Refresh renews the lease's staleness window (a heartbeat): when the on-disk record is
// still ours, it is atomically rewritten with RenewedUnix = now, so a window that
// outlives the TTL is measured from the renew, not the acquire (#4612). Owner identity,
// not freshness, decides — a stale-but-still-ours record is renewable (that is the point
// of the heartbeat). When a peer has already reclaimed the lease (our record was
// overwritten or removed) it returns ErrWriterLeaseLost and leaves the peer's record
// untouched: the window is no longer protected and the holder must stop writing.
func (l *WriterLease) Refresh(now func() time.Time) error {
	if now == nil {
		now = time.Now
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	cur, err := readLease(l.path)
	if err != nil {
		if os.IsNotExist(err) {
			l.markLost()
			return ErrWriterLeaseLost
		}
		return err
	}
	if cur != l.info {
		l.markLost()
		return ErrWriterLeaseLost
	}
	next := l.info
	next.RenewedUnix = now().Unix()
	if err := writeLeaseReplace(l.path, next); err != nil {
		return err
	}
	l.info = next // Release's owner-check must keep matching the on-disk record
	return nil
}

// keepAlive starts the heartbeat that Refreshes the lease every quarter-TTL until stop
// is called, so a window that outlives the TTL renews itself instead of being reclaimed
// as crash residue (#4612). stop blocks until the goroutine exits, so no refresh can
// race the caller's Release. A lost lease (ErrWriterLeaseLost) ends the heartbeat — the
// reclaiming peer must not be fought; a transient I/O failure leaves the previous renew
// in place and the next tick retries.
func (l *WriterLease) keepAlive(now func() time.Time) (stop func()) {
	interval := l.ttl / 4
	if interval <= 0 {
		interval = DefaultWriterLeaseTTL / 4
	}
	done := make(chan struct{})
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				if err := l.Refresh(now); errors.Is(err, ErrWriterLeaseLost) {
					return
				}
			}
		}
	}()
	return func() {
		close(done)
		<-finished
	}
}

// leaseStale reports whether info is old enough (>= ttl) to reclaim as crash residue.
// The window is renew-aware: it measures from effectiveActiveAt, so a heartbeated lease
// stays live and only a genuinely silent holder ages out.
func leaseStale(info WriterLeaseInfo, now time.Time, ttl time.Duration) bool {
	return now.Sub(info.effectiveActiveAt()) >= ttl
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

// writeLeaseReplace atomically replaces path with rec via a same-dir temp file + rename,
// so a concurrent reader sees either the previous record or the renewed one, never a torn
// write (a torn record parses to the zero value, which reads as reclaimable crash residue
// — exactly the wrong verdict for a live renew).
func writeLeaseReplace(path string, rec WriterLeaseInfo) error {
	enc, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), writerLeaseFile+".renew-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	if _, err := f.Write(enc); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
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
