package safesync

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// ErrLeaseOwnerUnavailable is returned when a writer lease cannot be acquired
// under contention before the wait deadline expires.
var ErrLeaseOwnerUnavailable = errors.New("lease owner unavailable (" + ReasonLeaseOwnerUnavailable + ")")

// ticketQueue provides fair FIFO queuing for acquiring exclusive write leases.
type ticketQueue struct {
	mu      sync.Mutex
	tail    uint64
	head    uint64
	serving uint64
}

var (
	repoQueuesMu sync.Mutex
	repoQueues   = make(map[string]*ticketQueue)
)

func getRepoQueue(repo string) *ticketQueue {
	abs, err := filepath.Abs(repo)
	if err == nil {
		repo = abs
	}
	repoQueuesMu.Lock()
	defer repoQueuesMu.Unlock()
	q, ok := repoQueues[repo]
	if !ok {
		q = &ticketQueue{serving: 1}
		repoQueues[repo] = q
	}
	return q
}

func (q *ticketQueue) takeTicket() uint64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.tail++
	return q.tail
}

func (q *ticketQueue) isServing(ticket uint64) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.serving >= ticket
}

func (q *ticketQueue) advance(ticket uint64) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if ticket >= q.serving {
		q.serving = ticket + 1
	}
}

// isProcessAlive checks whether a process with the given pid is running.
func isProcessAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	return processalive.Check(pid)
}

// acquireWriterLease attempts to acquire the exclusive worktree writer lease for repo,
// verifying process liveness via OS PID check before declaring LEASE_OWNER_UNAVAILABLE
// or WriterLeaseHeldError, and reaping stale leases from dead processes (#11234).
func acquireWriterLease(repo, owner string, now func() time.Time, ttl time.Duration) (*WriterLease, error) {
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
	rec := WriterLeaseInfo{
		Owner:        strings.TrimSpace(owner),
		PID:          os.Getpid(),
		Host:         host,
		AcquiredUnix: now().Unix(),
		Mode:         "exclusive-write",
	}
	if rec.Owner == "" {
		rec.Owner = fmt.Sprintf("pid-%d", rec.PID)
	}

	// Clean up any stale shared readers first
	cleanDeadSharedReaders(dir, now(), ttl)

	// Check if any live shared readers exist
	if readerInfo, hasLiveReaders := activeSharedReader(dir, now(), ttl); hasLiveReaders {
		return nil, &WriterLeaseHeldError{Info: readerInfo}
	}

	for attempt := 0; attempt < 2; attempt++ {
		ok, err := writeLeaseExclusive(path, rec)
		if err != nil {
			return nil, err
		}
		if ok {
			return &WriterLease{path: path, ttl: ttl, info: rec, lost: make(chan struct{})}, nil
		}

		// Someone holds the lease: read record and verify liveness
		cur, rerr := readLease(path)
		if rerr != nil {
			if os.IsNotExist(rerr) {
				continue
			}
			return nil, rerr
		}

		// OS PID check for processes on the same host: reap dead processes immediately
		isLocal := cur.Host == "" || cur.Host == host
		if isLocal && cur.PID > 0 && !isProcessAlive(cur.PID) {
			// Owner process is dead! Reap stale lease residue immediately without waiting for TTL
			_ = os.Remove(path)
			continue
		}

		// If not local or PID alive, verify TTL staleness
		if !leaseStale(cur, now(), ttl) {
			return nil, &WriterLeaseHeldError{Info: cur}
		}

		// Stale TTL: reap and retry
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return nil, err
		}
	}

	// Final check on held lease
	cur, rerr := readLease(path)
	if rerr != nil {
		if os.IsNotExist(rerr) {
			if ok, werr := writeLeaseExclusive(path, rec); werr == nil && ok {
				return &WriterLease{path: path, ttl: ttl, info: rec, lost: make(chan struct{})}, nil
			}
			cur, _ = readLease(path)
		} else {
			return nil, rerr
		}
	}

	// Final PID liveness check before refusing
	isLocal := cur.Host == "" || cur.Host == host
	if isLocal && cur.PID > 0 && !isProcessAlive(cur.PID) {
		_ = os.Remove(path)
		if ok, err := writeLeaseExclusive(path, rec); err == nil && ok {
			return &WriterLease{path: path, ttl: ttl, info: rec, lost: make(chan struct{})}, nil
		}
	}

	return nil, &WriterLeaseHeldError{Info: cur}
}

// AcquireQueuedWriterLease acquires an exclusive write lease using ticket-based fair
// queuing with exponential backoff and jitter (#11234).
func AcquireQueuedWriterLease(ctx context.Context, repo, owner string, now func() time.Time, ttl time.Duration, maxWait time.Duration) (*WriterLease, error) {
	if maxWait <= 0 {
		maxWait = 500 * time.Millisecond
	}
	if now == nil {
		now = time.Now
	}

	q := getRepoQueue(repo)
	ticket := q.takeTicket()
	defer func() {
		// Ensure ticket is advanced if we exit without acquiring
		// (if acquired, release will advance it)
	}()

	start := now()
	deadline := start.Add(maxWait)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}

	backoff := 5 * time.Millisecond
	maxBackoff := 50 * time.Millisecond

	for {
		if q.isServing(ticket) {
			lease, err := acquireWriterLease(repo, owner, now, ttl)
			if err == nil && lease != nil {
				// Wrap release to advance the queue
				var released int32
				lease.releaseHook = func() error {
					if atomic.CompareAndSwapInt32(&released, 0, 1) {
						q.advance(ticket)
					}
					return nil
				}
				return lease, nil
			}

			// If error is not a held error, return immediately
			var held *WriterLeaseHeldError
			if !errors.As(err, &held) {
				q.advance(ticket)
				return nil, err
			}
		}

		// Check deadline before sleeping
		currentTime := now()
		if !currentTime.Before(deadline) || ctx.Err() != nil {
			// Before declaring LEASE_OWNER_UNAVAILABLE, verify process liveness via OS PID check
			dir, dirErr := worktreeGitDir(repo)
			if dirErr == nil {
				path := filepath.Join(dir, writerLeaseFile)
				cur, rerr := readLease(path)
				host, _ := os.Hostname()
				isLocal := cur.Host == "" || cur.Host == host
				if rerr == nil && isLocal && cur.PID > 0 && !isProcessAlive(cur.PID) {
					// Stale dead process lease reaped at deadline!
					_ = os.Remove(path)
					if lease, lerr := acquireWriterLease(repo, owner, now, ttl); lerr == nil && lease != nil {
						var released int32
						lease.releaseHook = func() error {
							if atomic.CompareAndSwapInt32(&released, 0, 1) {
								q.advance(ticket)
							}
							return nil
						}
						return lease, nil
					}
				}
			}

			q.advance(ticket)
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, ErrLeaseOwnerUnavailable
		}

		// Exponential backoff with jitter
		jitter := time.Duration(cryptoRandInt(int64(backoff) / 2))
		sleepDuration := backoff + jitter
		if remaining := deadline.Sub(currentTime); sleepDuration > remaining {
			sleepDuration = remaining
		}

		select {
		case <-ctx.Done():
			q.advance(ticket)
			return nil, ctx.Err()
		case <-time.After(sleepDuration):
		}

		backoff = backoff * 3 / 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

// AcquireQueuedSharedReadLease acquires a shared read lease (Phase 1), waiting with
// exponential backoff and jitter if an exclusive write lease is currently held, until
// maxWait expires (#11234/#11616).
func AcquireQueuedSharedReadLease(ctx context.Context, repo, owner string, now func() time.Time, ttl time.Duration, maxWait time.Duration) (*SharedReadLease, error) {
	if now == nil {
		now = time.Now
	}
	if maxWait <= 0 {
		return AcquireSharedReadLease(repo, owner, now, ttl)
	}

	start := now()
	deadline := start.Add(maxWait)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}

	backoff := 5 * time.Millisecond
	maxBackoff := 50 * time.Millisecond
	var lastHeld *WriterLeaseHeldError

	for {
		lease, err := AcquireSharedReadLease(repo, owner, now, ttl)
		if err == nil && lease != nil {
			return lease, nil
		}

		var held *WriterLeaseHeldError
		if !errors.As(err, &held) {
			return nil, err
		}
		lastHeld = held

		// Verify if the exclusive holder process died on the local host before sleeping
		dir, dirErr := worktreeGitDir(repo)
		if dirErr == nil {
			path := filepath.Join(dir, writerLeaseFile)
			cur, rerr := readLease(path)
			host, _ := os.Hostname()
			isLocal := cur.Host == "" || cur.Host == host
			if rerr == nil && isLocal && cur.PID > 0 && !isProcessAlive(cur.PID) {
				_ = os.Remove(path)
				if lease, lerr := AcquireSharedReadLease(repo, owner, now, ttl); lerr == nil && lease != nil {
					return lease, nil
				}
			}
		}

		currentTime := now()
		if !currentTime.Before(deadline) || ctx.Err() != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			if lastHeld != nil {
				return nil, lastHeld
			}
			return nil, ErrLeaseOwnerUnavailable
		}

		jitter := time.Duration(cryptoRandInt(int64(backoff) / 2))
		sleepDuration := backoff + jitter
		if remaining := deadline.Sub(currentTime); sleepDuration > remaining {
			sleepDuration = remaining
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(sleepDuration):
		}

		backoff = backoff * 3 / 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

func cryptoRandInt(max int64) int64 {
	if max <= 0 {
		return 0
	}
	n, err := rand.Int(rand.Reader, big.NewInt(max))
	if err != nil {
		return 0
	}
	return n.Int64()
}

func cleanDeadSharedReaders(gitDir string, now time.Time, ttl time.Duration) {
	readersDir := filepath.Join(gitDir, "fak-worktree-readers")
	entries, err := os.ReadDir(readersDir)
	if err != nil {
		return
	}
	host, _ := os.Hostname()
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		path := filepath.Join(readersDir, entry.Name())
		info, err := readLease(path)
		if err != nil {
			_ = os.Remove(path)
			continue
		}
		isLocal := info.Host == "" || info.Host == host
		if isLocal && info.PID > 0 && !isProcessAlive(info.PID) {
			_ = os.Remove(path)
			continue
		}
		if leaseStale(info, now, ttl) {
			_ = os.Remove(path)
		}
	}
}

func activeSharedReader(gitDir string, now time.Time, ttl time.Duration) (WriterLeaseInfo, bool) {
	readersDir := filepath.Join(gitDir, "fak-worktree-readers")
	entries, err := os.ReadDir(readersDir)
	if err != nil {
		return WriterLeaseInfo{}, false
	}
	host, _ := os.Hostname()
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		path := filepath.Join(readersDir, entry.Name())
		info, err := readLease(path)
		if err != nil {
			continue
		}
		isLocal := info.Host == "" || info.Host == host
		if isLocal && info.PID > 0 && !isProcessAlive(info.PID) {
			_ = os.Remove(path)
			continue
		}
		if !leaseStale(info, now, ttl) {
			return info, true
		}
	}
	return WriterLeaseInfo{}, false
}
