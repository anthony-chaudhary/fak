package ops

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Engine coordinates the autonomous operations background tasks.
type Engine struct {
	mu        sync.Mutex
	RepoRoot  string
	Config    Config
	Ledger    *Ledger
	Storage   *StorageManager
	Process   *ProcessManager
	Workspace *WorkspaceManager

	lastStorageTick time.Time
	lastQuickTick   time.Time
}

// NewEngine constructs a unified operations Engine.
func NewEngine(repoRoot string, cfg Config) (*Engine, error) {
	ledPath := DefaultLedgerPath(repoRoot)
	led, err := OpenLedger(ledPath)
	if err != nil {
		return nil, err
	}

	return &Engine{
		RepoRoot:  repoRoot,
		Config:    cfg,
		Ledger:    led,
		Storage:   NewStorageManager(repoRoot, cfg),
		Process:   NewProcessManager(cfg),
		Workspace: NewWorkspaceManager(repoRoot),
	}, nil
}

// Tick executes one operations pass. If forceAll is true, all tiers run.
// Otherwise, quick checks (locks, processes) run every 60s and storage sweeps run every 15m.
func (e *Engine) Tick(ctx context.Context, forceAll bool) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	now := time.Now()

	// 1. Quick checks: Locks & Worktrees (every 60s or forceAll)
	if forceAll || now.Sub(e.lastQuickTick) >= 60*time.Second {
		start := time.Now()
		healRes, err := e.Workspace.SweepLocksAndWorktrees(ctx, false)
		dur := time.Since(start).Milliseconds()

		if len(healRes.LocksEvicted) > 0 {
			_ = e.Ledger.Record(Event{
				ActionType: ActionLockEvict,
				Details:    fmt.Sprintf("evicted dead locks: %s", strings.Join(healRes.LocksEvicted, ", ")),
				DurationMS: dur,
			})
		}
		if len(healRes.WorktreesPruned) > 0 {
			_ = e.Ledger.Record(Event{
				ActionType: ActionWorktreePrune,
				Details:    fmt.Sprintf("pruned cold worktrees: %s", strings.Join(healRes.WorktreesPruned, ", ")),
				DurationMS: dur,
			})
		}

		// Process runaway & orphan checks
		startProc := time.Now()
		procRes, _ := e.Process.SweepProcessRunaways(ctx, false)
		durProc := time.Since(startProc).Milliseconds()

		if len(procRes.PIDsReaped) > 0 {
			_ = e.Ledger.Record(Event{
				ActionType:   ActionProcessReap,
				Details:      fmt.Sprintf("reaped %d processes: %s", len(procRes.PIDsReaped), strings.Join(procRes.Names, ", ")),
				PIDsAffected: procRes.PIDsReaped,
				DurationMS:   durProc,
			})
		}

		e.lastQuickTick = now
		_ = err
	}

	// 2. Storage & Cache Lifecycle (every 15m or forceAll)
	if forceAll || now.Sub(e.lastStorageTick) >= 15*time.Minute {
		start := time.Now()
		reclaimRes, err := e.Storage.ReclaimCascade(ctx, false)
		dur := time.Since(start).Milliseconds()

		if reclaimRes.TotalBytes > 0 {
			_ = e.Ledger.Record(Event{
				ActionType:     ActionStorageReclaim,
				Details:        fmt.Sprintf("reclaimed %d bytes across %d files (%s)", reclaimRes.TotalBytes, reclaimRes.FilesCount, strings.Join(reclaimRes.Actions, "; ")),
				BytesReclaimed: reclaimRes.TotalBytes,
				DurationMS:     dur,
			})
		}

		e.lastStorageTick = now
		_ = err
	}

	return nil
}
