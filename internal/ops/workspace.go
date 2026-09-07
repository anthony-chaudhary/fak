package ops

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// WorkspaceHealingResult records the outcome of a self-healing and lock eviction sweep.
type WorkspaceHealingResult struct {
	LocksEvicted    []string `json:"locks_evicted"`
	WorktreesPruned []string `json:"worktrees_pruned"`
	GitMaintDone    bool     `json:"git_maint_done"`
	Errors          []string `json:"errors"`
}

// WorkspaceManager oversees dead-PID lock clearance, cold worktree pruning, and repo maintenance.
type WorkspaceManager struct {
	RepoRoot             string
	ProcessAlive         func(int) bool
	WorkerWorktreeRunner func(string, []string) (int, string)
}

// NewWorkspaceManager creates a WorkspaceManager.
func NewWorkspaceManager(repoRoot string) *WorkspaceManager {
	return &WorkspaceManager{
		RepoRoot:     repoRoot,
		ProcessAlive: defaultProcessAlive,
	}
}

func defaultProcessAlive(pid int) bool {
	return processalive.Check(pid)
}

func defaultTreeDoctorRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	c := exec.CommandContext(ctx, "git", args...)
	c.Dir = dir
	configureDispatchHelperCommand(c)
	var out bytes.Buffer
	c.Stdout = &out
	c.Stderr = &out
	err := c.Run()
	if err == nil {
		return out.String(), 0, nil
	}
	if exitErr, ok := err.(*exec.ExitError); ok {
		return out.String(), exitErr.ExitCode(), nil
	}
	return out.String(), 1, err
}

func defaultWorkerWorktreeRunner(root string, args []string) (int, string) {
	c := exec.Command("git", args...)
	c.Dir = root
	configureDispatchHelperCommand(c)
	var out bytes.Buffer
	c.Stdout = &out
	c.Stderr = &out
	err := c.Run()
	if err == nil {
		return 0, out.String()
	}
	if exitErr, ok := err.(*exec.ExitError); ok {
		return exitErr.ExitCode(), out.String()
	}
	return 1, out.String()
}

// SweepLocksAndWorktrees evicts dead-PID locks, prunes abandoned worker worktrees, and runs git maint.
func (wm *WorkspaceManager) SweepLocksAndWorktrees(ctx context.Context, dryRun bool) (WorkspaceHealingResult, error) {
	var res WorkspaceHealingResult
	if wm.RepoRoot == "" {
		return res, nil
	}

	processAlive := wm.ProcessAlive
	if processAlive == nil {
		processAlive = defaultProcessAlive
	}

	gitDir := filepath.Join(wm.RepoRoot, ".git")
	if _, err := os.Stat(gitDir); os.IsNotExist(err) {
		return res, nil
	}

	// 1. Dead-PID lock eviction (.git/fak-commit.lock, .git/index.lock, .git/*.lock.orphan*)
	lockFiles := []string{
		filepath.Join(gitDir, "fak-commit.lock"),
		filepath.Join(gitDir, "index.lock"),
	}

	// Also find any .lock.orphan files
	entries, err := os.ReadDir(gitDir)
	if err == nil {
		for _, e := range entries {
			if strings.Contains(e.Name(), ".lock.orphan") || strings.HasSuffix(e.Name(), ".lock") {
				lockFiles = append(lockFiles, filepath.Join(gitDir, e.Name()))
			}
		}
	}

	// Deduplicate lockFiles
	seen := make(map[string]bool)
	for _, lf := range lockFiles {
		if seen[lf] {
			continue
		}
		seen[lf] = true

		info, err := os.Stat(lf)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			continue
		}

		// Read lock file content to see if it records a PID
		data, _ := os.ReadFile(lf)
		content := strings.TrimSpace(string(data))
		pid := 0
		if content != "" {
			if parsed, err := strconv.Atoi(content); err == nil {
				pid = parsed
			}
		}

		shouldEvict := false
		if pid > 0 {
			if !processAlive(pid) {
				shouldEvict = true
			}
		} else {
			// If no PID recorded, evict if older than 15 minutes and frozen
			if time.Since(info.ModTime()) > 15*time.Minute {
				shouldEvict = true
			}
		}

		if shouldEvict {
			if !dryRun {
				if err := os.Remove(lf); err == nil {
					res.LocksEvicted = append(res.LocksEvicted, filepath.Base(lf))
				} else {
					res.Errors = append(res.Errors, fmt.Sprintf("evict %s: %v", filepath.Base(lf), err))
				}
			} else {
				res.LocksEvicted = append(res.LocksEvicted, fmt.Sprintf("dry-run: would evict %s", filepath.Base(lf)))
			}
		}
	}

	// 2. Ref-locks & general doctor sweep via treedoctor
	_, _ = treedoctor.Sweep(ctx, defaultTreeDoctorRunner, treedoctor.Options{
		RepoRoot:     wm.RepoRoot,
		ProcessAlive: processAlive,
		LocksOnly:    true,
	}, !dryRun)

	// 3. Cold worker worktrees pruning
	now := time.Now()
	liveLeases, _, leaserefErr := leaseref.NewInDir(wm.RepoRoot).Live(ctx, now)

	oracle := func(wtPath string) bool {
		lane := workerworktree.LaneOf(wtPath)
		if lane == "" {
			return true // unclassifiable worktree -> fail toward keep
		}

		hasDeadProof := false

		// Check worker lease (lease.json)
		lease, leaseErr := workerworktree.ReadWorkerLease(wtPath)
		if leaseErr == nil {
			if lease.PID > 0 && processAlive(lease.PID) {
				return true // live process ownership
			}
			if !lease.HeartbeatTS.IsZero() && time.Since(lease.HeartbeatTS) < workerworktree.DefaultHeartbeatStaleThreshold {
				return true // live heartbeat
			}
			if lease.PID > 0 && !processAlive(lease.PID) {
				hasDeadProof = true
			}
		} else if !os.IsNotExist(leaseErr) {
			// Metadata cannot be read -> fail toward keep
			return true
		}

		// Check owner stamp sidecar
		stampPath := workerworktree.OwnerStampPath(wtPath)
		if stampBytes, stampErr := os.ReadFile(stampPath); stampErr == nil {
			var stamp workerworktree.OwnerStamp
			if err := json.Unmarshal(stampBytes, &stamp); err != nil {
				// Metadata cannot be read -> fail toward keep
				return true
			}
			if stamp.PID > 0 {
				if processAlive(stamp.PID) {
					return true // live process ownership
				}
				hasDeadProof = true
			}
		} else if !os.IsNotExist(stampErr) {
			// Metadata cannot be read -> fail toward keep
			return true
		}

		// Fail toward keep if no dead process ownership was proven
		if !hasDeadProof {
			return true
		}

		// Check leaseref for active leases on this lane
		if leaserefErr != nil {
			return true // fail toward keep on query failure
		}
		for _, rec := range liveLeases {
			if leaseMatchesLane(rec.ID, lane) {
				return true // active lease exists on this lane
			}
		}

		// Only return false (reapable) when proven to have dead process ownership and no live lease
		return false
	}

	gitRunner := wm.WorkerWorktreeRunner
	if gitRunner == nil {
		gitRunner = defaultWorkerWorktreeRunner
	}
	cwList := workerworktree.ColdReapList(wm.RepoRoot, gitRunner, now, 30*time.Minute, oracle)
	for _, cw := range cwList {
		if cw.Eligible && !cw.HeldByWork {
			if !dryRun {
				if err := os.RemoveAll(cw.Path); err == nil {
					_ = os.Remove(workerworktree.OwnerStampPath(cw.Path))
					res.WorktreesPruned = append(res.WorktreesPruned, filepath.Base(cw.Path))
				}
			} else {
				res.WorktreesPruned = append(res.WorktreesPruned, fmt.Sprintf("dry-run: would prune %s", filepath.Base(cw.Path)))
			}
		}
	}

	// 4. Lightweight git maintenance during quiet period
	if !dryRun {
		cmd := exec.CommandContext(ctx, "git", "pack-refs", "--all")
		cmd.Dir = wm.RepoRoot
		configureDispatchHelperCommand(cmd)
		_ = cmd.Run()
		res.GitMaintDone = true
	}

	return res, nil
}

func isDigitsOnly(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func leaseMatchesLane(leaseID, lane string) bool {
	if lane == "" || leaseID == "" {
		return false
	}
	if strings.EqualFold(leaseID, lane) {
		return true
	}
	rest := strings.TrimPrefix(strings.ToLower(leaseID), "resolve-")
	if strings.EqualFold(rest, lane) {
		return true
	}
	if i := strings.LastIndex(rest, "-"); i > 0 && isDigitsOnly(rest[i+1:]) {
		if strings.EqualFold(rest[:i], lane) {
			return true
		}
	}
	return false
}
