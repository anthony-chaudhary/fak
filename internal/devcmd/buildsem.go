package devcmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// DefaultBuildConcurrency is the default maximum number of concurrent builds/tests
// allowed host-wide to prevent compiler storms and system commit exhaustion.
const DefaultBuildConcurrency = 2

// ErrBuildSlotTimeout is returned when all build slots remain busy past the timeout.
var ErrBuildSlotTimeout = errors.New("build slot acquisition timed out: all slots busy")

var (
	buildSemMu              sync.RWMutex
	defaultBuildConcurrency = DefaultBuildConcurrency
	buildSemBaseDir         = defaultBuildSemaphoreDir
)

// defaultBuildSemaphoreDir resolves the host-wide build semaphore location:
// %LOCALAPPDATA%/Fleet/build.lock on Windows when available, or
// os.TempDir()/fak-build-semaphore elsewhere.
func defaultBuildSemaphoreDir() string {
	if localAppData := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); localAppData != "" {
		return filepath.Join(localAppData, "Fleet", "build.lock")
	}
	return filepath.Join(os.TempDir(), "fak-build-semaphore")
}

// SetDefaultBuildConcurrency sets the host-wide concurrency limit.
// A limit <= 0 resets to DefaultBuildConcurrency.
func SetDefaultBuildConcurrency(n int) {
	buildSemMu.Lock()
	defer buildSemMu.Unlock()
	if n <= 0 {
		n = DefaultBuildConcurrency
	}
	defaultBuildConcurrency = n
}

// GetDefaultBuildConcurrency returns the currently configured default concurrency limit.
func GetDefaultBuildConcurrency() int {
	buildSemMu.RLock()
	defer buildSemMu.RUnlock()
	return defaultBuildConcurrency
}

// SetDefaultBuildSemaphoreDir overrides the directory where semaphore locks live.
// Passing "" resets to defaultBuildSemaphoreDir.
func SetDefaultBuildSemaphoreDir(dir string) {
	buildSemMu.Lock()
	defer buildSemMu.Unlock()
	if dir == "" {
		buildSemBaseDir = defaultBuildSemaphoreDir
	} else {
		buildSemBaseDir = func() string { return dir }
	}
}

// BuildSemaphore manages host-wide build and test concurrency using inter-process
// advisory file locks over a pool of slot files.
type BuildSemaphore struct {
	dir   string
	limit int
	// label names this holder in the shared wait ledger (waits.jsonl).
	label string
}

// NewBuildSemaphore creates a semaphore with the specified base directory and limit.
// If dir is empty, the default semaphore dir is used.
// If limit <= 0, the current default concurrency limit is used.
func NewBuildSemaphore(dir string, limit int) *BuildSemaphore {
	if dir == "" {
		buildSemMu.RLock()
		dir = buildSemBaseDir()
		buildSemMu.RUnlock()
	}
	if limit <= 0 {
		limit = GetDefaultBuildConcurrency()
	}
	return &BuildSemaphore{
		dir:   dir,
		limit: limit,
		label: "fak buildsem",
	}
}

// Dir returns the directory path backing this semaphore.
func (s *BuildSemaphore) Dir() string {
	return s.dir
}

// Limit returns the concurrency limit of this semaphore.
func (s *BuildSemaphore) Limit() int {
	return s.limit
}

func (s *BuildSemaphore) slotPath(i int) string {
	fi, err := os.Stat(s.dir)
	if err == nil && !fi.IsDir() {
		return fmt.Sprintf("%s.slot-%d.lock", s.dir, i)
	}
	return filepath.Join(s.dir, fmt.Sprintf("slot-%d.lock", i))
}

// buildSemLedgerName is the shared host build-lease wait ledger. These
// ticket-less rows are counted by the private fold's legacy path, so public
// holders become visible to true utilization without a reader change.
const buildSemLedgerName = "waits.jsonl"

// appendBuildSemLedger is best effort: the ledger is observability, never
// admission. A failure here must not fail the build.
func appendBuildSemLedger(dir string, row map[string]any) {
	if dir == "" {
		return
	}
	row["at"] = time.Now().UTC().Format(time.RFC3339Nano)
	b, err := json.Marshal(row)
	if err != nil {
		return
	}
	f, err := os.OpenFile(filepath.Join(dir, buildSemLedgerName), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o666)
	if err != nil {
		return
	}
	_, _ = f.Write(append(b, '\n'))
	_ = f.Close()
}

// buildSemToolKey folds a label to its tool verb for per-tool folds, mirroring
// the private buildlease.ToolKey shape.
func buildSemToolKey(label string) string {
	var out []string
	for _, tok := range strings.Fields(label) {
		if strings.HasPrefix(tok, "-") || strings.ContainsAny(tok, `/\:`) {
			break
		}
		out = append(out, tok)
		if len(out) == 3 {
			break
		}
	}
	if len(out) == 0 {
		return "?"
	}
	return strings.Join(out, " ")
}

func (s *BuildSemaphore) appendLedger(row map[string]any) {
	row["label"] = s.label
	row["tool"] = buildSemToolKey(s.label)
	row["pid"] = os.Getpid()
	row["slots"] = s.limit
	appendBuildSemLedger(s.dir, row)
}

// Acquire attempts to acquire a build slot within timeout.
// If a slot is acquired, returns a cleanup release function and nil error.
// If slots are busy, waits/queues up to timeout before failing closed.
// Every outcome appends a best-effort row to the shared wait ledger so the
// host pool's occupancy accounting sees public holders too.
func (s *BuildSemaphore) Acquire(ctx context.Context, timeout time.Duration) (func(), error) {
	start := time.Now()
	refused := func(err error) (func(), error) {
		s.appendLedger(map[string]any{"outcome": "timeout", "wait_ms": time.Since(start).Milliseconds()})
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return refused(err)
	}

	fi, err := os.Stat(s.dir)
	if err != nil || fi.IsDir() {
		if err := os.MkdirAll(s.dir, 0o755); err != nil {
			return nil, fmt.Errorf("create build semaphore dir: %w", err)
		}
	}

	tryAcquire := func() (func(), bool, error) {
		for i := 0; i < s.limit; i++ {
			p := s.slotPath(i)
			f, err := os.OpenFile(p, os.O_CREATE|os.O_RDWR, 0o666)
			if err != nil {
				continue
			}
			if err := flock.TryLock(f); err == nil {
				acquiredAt := time.Now()
				_ = f.Truncate(0)
				_, _ = f.Seek(0, io.SeekStart)
				_, _ = fmt.Fprintf(f, "pid=%d slot=%d acquired=%s\n", os.Getpid(), i, acquiredAt.UTC().Format(time.RFC3339Nano))
				s.appendLedger(map[string]any{"outcome": "acquired", "wait_ms": acquiredAt.Sub(start).Milliseconds()})

				var once sync.Once
				release := func() {
					once.Do(func() {
						s.appendLedger(map[string]any{"outcome": "released", "held_ms": time.Since(acquiredAt).Milliseconds()})
						_ = flock.Unlock(f)
						_ = f.Close()
					})
				}
				return release, true, nil
			}
			_ = f.Close()
		}
		return nil, false, nil
	}

	if release, ok, err := tryAcquire(); ok {
		return release, nil
	} else if err != nil {
		return nil, err
	}

	if timeout == 0 {
		return refused(ErrBuildSlotTimeout)
	}

	var deadline time.Time
	hasDeadline := false
	if timeout > 0 {
		deadline = time.Now().Add(timeout)
		hasDeadline = true
	}
	if ctxDeadline, ok := ctx.Deadline(); ok {
		if !hasDeadline || ctxDeadline.Before(deadline) {
			deadline = ctxDeadline
			hasDeadline = true
		}
	}

	const (
		minPollInterval = 20 * time.Millisecond
		maxPollInterval = 100 * time.Millisecond
	)
	pollInterval := minPollInterval

	timer := time.NewTimer(pollInterval)
	defer timer.Stop()

	for {
		if err := ctx.Err(); err != nil {
			return refused(err)
		}
		if hasDeadline && time.Now().After(deadline) {
			return refused(ErrBuildSlotTimeout)
		}

		sleepDur := pollInterval
		if hasDeadline {
			rem := time.Until(deadline)
			if rem <= 0 {
				return refused(ErrBuildSlotTimeout)
			}
			if rem < sleepDur {
				sleepDur = rem
			}
		}

		timer.Reset(sleepDur)
		select {
		case <-ctx.Done():
			return refused(ctx.Err())
		case <-timer.C:
		}

		if release, ok, err := tryAcquire(); ok {
			return release, nil
		} else if err != nil {
			return nil, err
		}

		if pollInterval < maxPollInterval {
			pollInterval = pollInterval * 3 / 2
			if pollInterval > maxPollInterval {
				pollInterval = maxPollInterval
			}
		}
	}
}

// AcquireBuildSlot acquires an execution slot using the host-wide default concurrency limit
// and default semaphore directory.
func AcquireBuildSlot(ctx context.Context, timeout time.Duration) (release func(), err error) {
	buildSemMu.RLock()
	dir := buildSemBaseDir()
	limit := defaultBuildConcurrency
	buildSemMu.RUnlock()

	sem := NewBuildSemaphore(dir, limit)
	return sem.Acquire(ctx, timeout)
}

// AcquireBuildSlotLabel is AcquireBuildSlot with a caller-named label recorded
// in the shared wait ledger (e.g. "fak buildcheck").
func AcquireBuildSlotLabel(ctx context.Context, timeout time.Duration, label string) (release func(), err error) {
	buildSemMu.RLock()
	dir := buildSemBaseDir()
	limit := defaultBuildConcurrency
	buildSemMu.RUnlock()

	sem := NewBuildSemaphore(dir, limit)
	if strings.TrimSpace(label) != "" {
		sem.label = label
	}
	return sem.Acquire(ctx, timeout)
}
