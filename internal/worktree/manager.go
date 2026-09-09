package worktree

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/pkg/sysproc"
)

var (
	ErrEmptyTicketID    = errors.New("worktree: empty ticket ID")
	ErrWorktreeExists   = errors.New("worktree: directory already exists")
	ErrWorktreeNotFound = errors.New("worktree: worktree not found")
)

// Runner executes a git command in dir with optional extra env and returns stdout, stderr, err.
type Runner func(ctx context.Context, dir string, env []string, args ...string) (string, string, error)

// DefaultRunner executes git using sysproc.
func DefaultRunner(ctx context.Context, dir string, env []string, args ...string) (string, string, error) {
	cmd := sysproc.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	if len(env) > 0 {
		cmd.Env = append(os.Environ(), env...)
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), string(out), fmt.Errorf("git %s failed: %w (output: %s)", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return string(out), "", nil
}

// WorktreeContext holds paths and environment variables for an ephemeral ticket worktree.
type WorktreeContext struct {
	TicketID    string            `json:"ticket_id"`
	Path        string            `json:"path"`
	Branch      string            `json:"branch"`
	BaseCommit  string            `json:"base_commit"`
	Env         map[string]string `json:"env"`
	AllocatedAt time.Time         `json:"allocated_at"`
}

// EnvList returns the isolated git environment variables as KEY=VALUE slice.
func (w *WorktreeContext) EnvList() []string {
	if len(w.Env) == 0 {
		return nil
	}
	res := make([]string, 0, len(w.Env))
	for k, v := range w.Env {
		res = append(res, k+"="+v)
	}
	return res
}

// ContractProvider supplies active contracts for janitor sweeps.
type ContractProvider func(ctx context.Context) ([]leaseref.ContractRecord, error)

// AutoSweepConfig configures automatic janitor sweeps on allocation threshold.
type AutoSweepConfig struct {
	Threshold        int              // Max allocations before triggering auto-sweep (e.g. 10)
	DebounceWindow   time.Duration    // Coalesce window (e.g. 50ms)
	ContractProvider ContractProvider // Supplies active contracts
}

// Option configures Manager.
type Option func(*Manager)

// WithRunner injects a custom git runner for testing or sandboxing.
func WithRunner(r Runner) Option {
	return func(m *Manager) {
		m.runner = r
	}
}

// WithWorktreesDir sets the relative or absolute directory where ephemeral worktrees live.
func WithWorktreesDir(dir string) Option {
	return func(m *Manager) {
		m.worktreesDir = dir
	}
}

// WithAutoSweep configures automatic janitor sweeps on allocation threshold.
func WithAutoSweep(cfg *AutoSweepConfig) Option {
	return func(m *Manager) {
		m.autoSweepCfg = cfg
	}
}

// Manager allocates, deallocates, and sanitizes ephemeral ticket worktrees.
type Manager struct {
	mu           sync.Mutex
	repoRoot     string
	worktreesDir string
	runner       Runner

	autoSweepCfg   *AutoSweepConfig
	allocCounter   uint64
	sweepRunning   int32
	sweepTriggerCh chan struct{}
	stopCh         chan struct{}
	loopDone       chan struct{}
	cancelLoop     context.CancelFunc
	stopOnce       sync.Once
	sweepWg        sync.WaitGroup
}

// NewManager creates a worktree manager for repoRoot.
func NewManager(repoRoot string, opts ...Option) *Manager {
	m := &Manager{
		repoRoot:     repoRoot,
		worktreesDir: ".worktrees",
		runner:       DefaultRunner,
	}
	for _, opt := range opts {
		opt(m)
	}
	if m.autoSweepCfg != nil {
		m.sweepTriggerCh = make(chan struct{}, 1)
		m.stopCh = make(chan struct{})
		m.loopDone = make(chan struct{})
		ctx, cancel := context.WithCancel(context.Background())
		m.cancelLoop = cancel
		go m.runJanitorLoop(ctx)
	}
	return m
}

// CleanTicketID normalizes ticketID into a safe leaf name.
func CleanTicketID(ticketID string) string {
	id := strings.TrimSpace(ticketID)
	id = strings.TrimPrefix(id, "refs/fak/locks/")
	id = strings.TrimPrefix(id, "contract-")
	id = strings.TrimPrefix(id, "ticket-")
	id = strings.TrimPrefix(id, "issue-")
	id = strings.ReplaceAll(id, "/", "-")
	id = strings.ReplaceAll(id, "\\", "-")
	return id
}

// WorktreePath returns the filesystem path for a ticket's worktree.
func (m *Manager) WorktreePath(ticketID string) string {
	clean := CleanTicketID(ticketID)
	if filepath.IsAbs(m.worktreesDir) {
		return filepath.Join(m.worktreesDir, "ticket-"+clean)
	}
	return filepath.Join(m.repoRoot, m.worktreesDir, "ticket-"+clean)
}

// BranchName returns the git branch name for a ticket.
func (m *Manager) BranchName(ticketID string) string {
	return "fak/ticket-" + CleanTicketID(ticketID)
}

// Allocate creates an ephemeral git worktree anchored at baseCommit on branch fak/ticket-<id>.
func (m *Manager) Allocate(ctx context.Context, ticketID, baseCommit string) (*WorktreeContext, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	cleanID := CleanTicketID(ticketID)
	if cleanID == "" {
		return nil, ErrEmptyTicketID
	}

	wtPath := m.WorktreePath(cleanID)
	branch := m.BranchName(cleanID)

	// Ensure parent worktrees dir exists
	parentDir := filepath.Dir(wtPath)
	if err := os.MkdirAll(parentDir, 0755); err != nil {
		return nil, fmt.Errorf("worktree: failed to create worktrees dir: %w", err)
	}

	// Resolve base commit if not provided
	commit := strings.TrimSpace(baseCommit)
	if commit == "" {
		out, _, err := m.runner(ctx, m.repoRoot, nil, "rev-parse", "HEAD")
		if err != nil {
			return nil, fmt.Errorf("worktree: failed to resolve HEAD: %w", err)
		}
		commit = strings.TrimSpace(out)
	}

	// Clean up stale branch if it already exists from a prior abandoned run
	_, _, _ = m.runner(ctx, m.repoRoot, nil, "branch", "-D", branch)

	// Clean up stale worktree dir if it was abandoned
	if _, err := os.Stat(wtPath); err == nil {
		_, _, _ = m.runner(ctx, m.repoRoot, nil, "worktree", "remove", "--force", wtPath)
		_ = os.RemoveAll(wtPath)
	}

	// Create worktree with new branch anchored at base commit:
	// git worktree add -b fak/ticket-<id> <wtPath> <commit>
	_, _, err := m.runner(ctx, m.repoRoot, nil, "worktree", "add", "-b", branch, wtPath, commit)
	if err != nil {
		return nil, fmt.Errorf("worktree: git worktree add failed: %w", err)
	}

	// Configure isolated git environment variables
	gitDir := filepath.Join(m.repoRoot, ".git", "worktrees", filepath.Base(wtPath))
	env := map[string]string{
		"GIT_DIR":       gitDir,
		"GIT_WORK_TREE": wtPath,
	}

	if m.autoSweepCfg != nil {
		count := atomic.AddUint64(&m.allocCounter, 1)
		if m.autoSweepCfg.Threshold > 0 && count%uint64(m.autoSweepCfg.Threshold) == 0 {
			select {
			case m.sweepTriggerCh <- struct{}{}:
			default:
			}
		}
	}

	return &WorktreeContext{
		TicketID:    cleanID,
		Path:        wtPath,
		Branch:      branch,
		BaseCommit:  commit,
		Env:         env,
		AllocatedAt: time.Now(),
	}, nil
}

// Deallocate forces worktree removal, runs git worktree prune, and optionally deletes the branch.
func (m *Manager) Deallocate(ctx context.Context, ticketID string, keepBranch bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	cleanID := CleanTicketID(ticketID)
	if cleanID == "" {
		return ErrEmptyTicketID
	}

	wtPath := m.WorktreePath(cleanID)
	branch := m.BranchName(cleanID)

	var errs []string

	// 1. Remove git worktree
	_, _, err := m.runner(ctx, m.repoRoot, nil, "worktree", "remove", "--force", wtPath)
	if err != nil {
		// Non-fatal if already removed or directory gone
		errs = append(errs, fmt.Sprintf("worktree remove: %v", err))
	}

	// 2. Fallback filesystem cleanup if untracked files remained
	if err := os.RemoveAll(wtPath); err != nil && !os.IsNotExist(err) {
		errs = append(errs, fmt.Sprintf("fs remove: %v", err))
	}

	// 3. Prune git worktree administrative records
	_, _, _ = m.runner(ctx, m.repoRoot, nil, "worktree", "prune")

	// 4. Delete branch if requested
	if !keepBranch {
		_, _, err := m.runner(ctx, m.repoRoot, nil, "branch", "-D", branch)
		if err != nil {
			errs = append(errs, fmt.Sprintf("branch delete: %v", err))
		}
	}

	if len(errs) > 0 {
		// If worktree dir is truly gone, consider it successfully deallocated
		if _, statErr := os.Stat(wtPath); os.IsNotExist(statErr) {
			return nil
		}
		return fmt.Errorf("worktree deallocate partial errors: %s", strings.Join(errs, "; "))
	}

	return nil
}

// Release deallocates the ephemeral worktree and removes its branch.
// It is equivalent to Deallocate(ctx, ticketID, false).
func (m *Manager) Release(ctx context.Context, ticketID string) error {
	return m.Deallocate(ctx, ticketID, false)
}

// runJanitorLoop runs a debounced background loop executing sweeps when triggered.
func (m *Manager) runJanitorLoop(ctx context.Context) {
	defer close(m.loopDone)

	debounce := 50 * time.Millisecond
	if m.autoSweepCfg != nil && m.autoSweepCfg.DebounceWindow > 0 {
		debounce = m.autoSweepCfg.DebounceWindow
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-m.stopCh:
			return
		case <-m.sweepTriggerCh:
		}

		// Debounce: coalesce rapid allocation triggers
		timer := time.NewTimer(debounce)
		coalescing := true
		for coalescing {
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-m.stopCh:
				timer.Stop()
				return
			case <-m.sweepTriggerCh:
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				timer.Reset(debounce)
			case <-timer.C:
				coalescing = false
			}
		}

		// Asynchronously execute sweep if no sweep is currently running
		if atomic.CompareAndSwapInt32(&m.sweepRunning, 0, 1) {
			m.sweepWg.Add(1)
			go func() {
				defer m.sweepWg.Done()
				defer atomic.StoreInt32(&m.sweepRunning, 0)

				var contracts []leaseref.ContractRecord
				if m.autoSweepCfg != nil && m.autoSweepCfg.ContractProvider != nil {
					sweepCtx, sweepCancel := context.WithTimeout(ctx, 15*time.Second)
					defer sweepCancel()
					c, err := m.autoSweepCfg.ContractProvider(sweepCtx)
					if err != nil {
						// Fail-closed: do not sweep if contract status cannot be proven
						return
					}
					contracts = c
					_, _ = m.Sweep(sweepCtx, contracts)
				} else {
					sweepCtx, sweepCancel := context.WithTimeout(ctx, 15*time.Second)
					defer sweepCancel()
					_, _ = m.Sweep(sweepCtx, contracts)
				}
			}()
		}
	}
}

// Stop cleanly terminates the background janitor loop if started.
func (m *Manager) Stop() {
	m.stopOnce.Do(func() {
		if m.cancelLoop != nil {
			m.cancelLoop()
		}
		if m.stopCh != nil {
			close(m.stopCh)
		}
		if m.loopDone != nil {
			<-m.loopDone
		}
		m.sweepWg.Wait()
	})
}

// Close terminates the background janitor loop and satisfies io.Closer.
func (m *Manager) Close() error {
	m.Stop()
	return nil
}
