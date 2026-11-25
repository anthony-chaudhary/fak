// Package committedtree materializes committed git trees without reading a
// shared checkout's dirty worktree or index.
package committedtree

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	defaultTimeout     = 2 * time.Minute
	gitPipeBufferSize  = 64 * 1024
	maxGitRecordSize   = gitPipeBufferSize
	maxGitStderrBytes  = 64 * 1024
	regularFileMode    = "100644"
	executableFileMode = "100755"
	symlinkMode        = "120000"
	gitlinkMode        = "160000"
)

// extractionBudgetEnv lets a loaded host raise the extraction budget per
// invocation without moving the bounded default: a Go duration (e.g. 20m).
// Measured on a saturated Windows host, extracting the committed tree of this
// repo can exceed the 2m default, turning every commit's build gate into a
// retryable BUILD_CHECK_TIMEOUT. The default stays tight so a wedged
// extraction still fails closed fast as a typed timeout, never a stall.
const extractionBudgetEnv = "FAK_COMMITTEDTREE_EXTRACT_BUDGET"

// ExtractionBudgetEnv is the knob callers name when this budget, not their own
// deadline, stopped an extraction.
const ExtractionBudgetEnv = extractionBudgetEnv

// extractionBudget resolves the per-invocation extraction budget: the env
// override when it parses to a positive duration, defaultTimeout otherwise.
func extractionBudget() time.Duration {
	if v := strings.TrimSpace(os.Getenv(extractionBudgetEnv)); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			return d
		}
	}
	return defaultTimeout
}

// BudgetError reports that extraction ran out of its own budget. That budget
// is independent of any caller deadline, so the message names the one knob that
// extends it; it unwraps to context.DeadlineExceeded.
type BudgetError struct {
	Budget time.Duration
}

func (e *BudgetError) Error() string {
	return fmt.Sprintf("committed tree extraction timed out after %s (raise %s to extend this budget): %v",
		e.Budget, extractionBudgetEnv, context.DeadlineExceeded)
}

func (e *BudgetError) Unwrap() error { return context.DeadlineExceeded }

// budgetKey carries the resolved extraction budget on the extraction context so
// timeout diagnostics report the budget that actually bounded the run.
type budgetKey struct{}

type commandFactory func(context.Context, string, ...string) *exec.Cmd

// ownedGitCommand binds one Git subprocess and its pipes to a cancellation
// watcher. On Windows the job owns the launcher and every descendant; closing
// stdout as well as the job makes a blocked reader return even when a child
// inherited the launcher's pipe handle.
type ownedGitCommand struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	job    *windowgate.JobObject

	started   bool
	waited    bool
	abortOnce sync.Once
	stop      chan struct{}
	done      chan struct{}
}

// Resolve returns the full object ID for ref in repo.
func Resolve(repo, ref string) (string, error) {
	cmd := gitCommand(context.Background(), repo, "rev-parse", ref)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// DefaultStaleTreeAge is the default quiet-period floor under which an extracted
// committed tree is considered still in use and never reaped.
const DefaultStaleTreeAge = 2 * time.Hour

var (
	reapMu     sync.Mutex
	lastReapAt time.Time
)

// StaleTree is one directory selected by the existing stale committed-tree reap.
type StaleTree struct {
	Path      string `json:"path"`
	Bytes     int64  `json:"bytes"`
	Reaped    bool   `json:"reaped,omitempty"`
	RemoveErr string `json:"remove_err,omitempty"`
}

// StaleReport separates candidates from successful removals, including in previews.
type StaleReport struct {
	Root          string      `json:"root"`
	MinAgeSeconds int64       `json:"min_age_seconds"`
	Apply         bool        `json:"apply"`
	Candidates    []StaleTree `json:"candidates,omitempty"`
	Reaped        int         `json:"reaped"`
	ReapedBytes   int64       `json:"reaped_bytes"`
	Err           string      `json:"err,omitempty"`
}

// ReapStale removes stale committed trees and returns successful removal counts.
// It retains the same selection and error behavior as SweepStale under apply.
func ReapStale(tempParent string, maxAge time.Duration) (int, int64, error) {
	rep, err := SweepStale(tempParent, maxAge, true)
	return rep.Reaped, rep.ReapedBytes, err
}

// SweepStale reports temporary directories matching "fak-committed-tree-*" under
// tempParent whose newest entry is older than maxAge. Only apply removes them.
// Empty tempParent uses os.TempDir(); non-positive maxAge uses DefaultStaleTreeAge.
// Preview and apply use the same selection loop; reported bytes are logical sizes.
func SweepStale(tempParent string, maxAge time.Duration, apply bool) (StaleReport, error) {
	if tempParent == "" {
		tempParent = os.TempDir()
	}
	if maxAge <= 0 {
		maxAge = DefaultStaleTreeAge
	}
	rep := StaleReport{Root: tempParent, MinAgeSeconds: int64(maxAge / time.Second), Apply: apply}
	entries, err := os.ReadDir(tempParent)
	if err != nil {
		rep.Err = err.Error()
		return rep, err
	}

	now := time.Now()

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, "fak-committed-tree-") {
			continue
		}
		dirPath := filepath.Join(tempParent, name)
		fi, err := entry.Info()
		if err != nil {
			continue
		}
		newest := fi.ModTime()
		var dirBytes int64

		_ = filepath.WalkDir(dirPath, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			info, err := d.Info()
			if err != nil {
				return nil
			}
			if info.ModTime().After(newest) {
				newest = info.ModTime()
			}
			if !d.IsDir() {
				dirBytes += info.Size()
			}
			return nil
		})

		if now.Sub(newest) >= maxAge {
			candidate := StaleTree{Path: dirPath, Bytes: dirBytes}
			if apply {
				if removeErr := os.RemoveAll(dirPath); removeErr == nil {
					candidate.Reaped = true
					rep.Reaped++
					rep.ReapedBytes += dirBytes
				} else {
					candidate.RemoveErr = removeErr.Error()
				}
			}
			rep.Candidates = append(rep.Candidates, candidate)
		}
	}
	return rep, nil
}

func maybeOpportunisticReap(tempParent string) {
	reapMu.Lock()
	now := time.Now()
	if now.Sub(lastReapAt) < 10*time.Minute {
		reapMu.Unlock()
		return
	}
	lastReapAt = now
	reapMu.Unlock()

	go func() {
		_, _, _ = ReapStale(tempParent, DefaultStaleTreeAge)
	}()
}

// Extract resolves no refs: object must already identify a commit or tree. The
// caller owns the returned temporary directory.
func Extract(repo, object string) (string, error) {
	return extractTemp(repo, object, "")
}

// ExtractPaths materializes only the given path prefixes of object into a new
// temporary directory, with the same budget, cleanup, and export-ignore
// behavior as Extract. Each prefix names a file or a directory; a directory
// prefix selects everything beneath it. At least one prefix is required.
func ExtractPaths(repo, object string, paths ...string) (string, error) {
	cleaned, err := normalizePathPrefixes(paths)
	if err != nil {
		return "", err
	}
	return extractTempPaths(repo, object, "", cleaned)
}

// normalizePathPrefixes validates caller prefixes and returns them as clean
// slash-separated tree paths. Prefixes that could escape the tree or select
// the root are rejected before any git process starts.
func normalizePathPrefixes(paths []string) ([]string, error) {
	if len(paths) == 0 {
		return nil, errors.New("committed tree path selection needs at least one path prefix")
	}
	cleaned := make([]string, 0, len(paths))
	for _, p := range paths {
		rel := filepath.ToSlash(filepath.Clean(filepath.FromSlash(p)))
		if rel == "." {
			return nil, fmt.Errorf("committed tree path prefix %q selects the whole tree; use Extract", p)
		}
		if strings.HasPrefix(rel, "/") {
			return nil, fmt.Errorf("committed tree path prefix %q must be relative", p)
		}
		if _, err := safeJoin(string(filepath.Separator), rel); err != nil {
			return nil, err
		}
		cleaned = append(cleaned, rel)
	}
	return cleaned, nil
}

func extractTemp(repo, object, tempParent string) (string, error) {
	return extractTempPaths(repo, object, tempParent, nil)
}

func extractTempPaths(repo, object, tempParent string, paths []string) (string, error) {
	maybeOpportunisticReap(tempParent)
	dir, err := os.MkdirTemp(tempParent, "fak-committed-tree-*")
	if err != nil {
		return "", err
	}
	ctx, cancel := operationContext(context.Background())
	defer cancel()
	if err := extractPaths(ctx, repo, object, dir, paths); err != nil {
		_ = os.RemoveAll(dir)
		return "", err
	}
	return dir, nil
}

// operationContext bounds one snapshot. A deadline already on parent is the
// bound, so callers keep control of their own time; otherwise the package
// budget applies and is recorded so timeout errors can name it.
func operationContext(parent context.Context) (context.Context, context.CancelFunc) {
	if _, ok := parent.Deadline(); ok {
		return context.WithCancel(parent)
	}
	budget := extractionBudget()
	ctx, cancel := context.WithTimeout(parent, budget)
	return context.WithValue(ctx, budgetKey{}, budget), cancel
}

func gitCommand(ctx context.Context, repo string, args ...string) *exec.Cmd {
	gitArgs := make([]string, 0, len(args)+2)
	gitArgs = append(gitArgs, "-C", repo)
	gitArgs = append(gitArgs, args...)
	return windowgate.CommandContext(ctx, "git", gitArgs...)
}

func (p *ownedGitCommand) start(ctx context.Context) error {
	job, err := windowgate.StartInNewJob(p.cmd)
	if err != nil {
		return err
	}
	p.job = job
	p.started = true
	p.stop = make(chan struct{})
	p.done = make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			p.abortNow()
		case <-p.stop:
		}
		close(p.done)
	}()
	return nil
}

func (p *ownedGitCommand) abortNow() {
	if p == nil {
		return
	}
	p.abortOnce.Do(func() {
		// Closing both ends owned by the parent is what unblocks an in-flight
		// WriteString, ReadSlice, or CopyN. Closing the Windows job then reaps
		// launcher descendants that still hold their inherited pipe handles.
		if p.stdin != nil {
			_ = p.stdin.Close()
		}
		if p.stdout != nil {
			_ = p.stdout.Close()
		}
		_ = p.job.Close()
		if p.cmd != nil && p.cmd.Process != nil {
			_ = p.cmd.Process.Kill()
		}
	})
}

func (p *ownedGitCommand) stopWatcher() {
	if p == nil || p.stop == nil {
		return
	}
	select {
	case <-p.done:
		return
	default:
	}
	close(p.stop)
	<-p.done
}

func (p *ownedGitCommand) wait() error {
	if p == nil || !p.started || p.waited {
		return nil
	}
	err := p.cmd.Wait()
	p.waited = true
	p.stopWatcher()
	_ = p.job.Close()
	return err
}

func (p *ownedGitCommand) finish(failed bool) {
	if p == nil {
		return
	}
	if failed {
		p.abortNow()
	}
	if p.started && !p.waited {
		_ = p.cmd.Wait()
		p.waited = true
	}
	p.stopWatcher()
	_ = p.job.Close()
}

func safeJoin(root, name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("git tree entry has an empty path")
	}
	rel := filepath.Clean(filepath.FromSlash(name))
	if rel == "." || filepath.IsAbs(rel) || filepath.VolumeName(rel) != "" || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("git tree entry %q escapes extraction root", name)
	}
	return filepath.Join(root, rel), nil
}

func validObjectID(oid string) bool {
	if len(oid) != 40 && len(oid) != 64 {
		return false
	}
	for _, c := range oid {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

func commandError(ctx context.Context, stage string, err error) error {
	if ctx.Err() != nil {
		return extractionStopError(ctx)
	}
	return fmt.Errorf("%s: %w", stage, err)
}

// extractionStopError names the package budget only when that budget expired.
// A caller's own deadline or cancellation stays a plain context error, which
// still satisfies errors.Is(err, context.DeadlineExceeded) or context.Canceled.
func extractionStopError(ctx context.Context) error {
	if budget, ok := ctx.Value(budgetKey{}).(time.Duration); ok && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return &BudgetError{Budget: budget}
	}
	return fmt.Errorf("committed tree extraction canceled: %w", ctx.Err())
}

// cappedBuffer keeps subprocess diagnostics useful without making stderr an
// unbounded memory side channel during a bounded extraction.
type cappedBuffer struct {
	b []byte
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	remaining := maxGitStderrBytes - len(b.b)
	if remaining > 0 {
		if len(p) > remaining {
			p = p[:remaining]
		}
		b.b = append(b.b, p...)
	}
	return n, nil
}

func (b *cappedBuffer) String() string {
	return strings.TrimSpace(string(b.b))
}
