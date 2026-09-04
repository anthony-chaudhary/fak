package ops

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

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
	RepoRoot     string
	ProcessAlive func(int) bool
}

// NewWorkspaceManager creates a WorkspaceManager.
func NewWorkspaceManager(repoRoot string) *WorkspaceManager {
	return &WorkspaceManager{
		RepoRoot:     repoRoot,
		ProcessAlive: defaultProcessAlive,
	}
}

func defaultProcessAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
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
			if !wm.ProcessAlive(pid) {
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
		ProcessAlive: wm.ProcessAlive,
		LocksOnly:    true,
	}, !dryRun)

	// 3. Cold worker worktrees pruning
	cwList := workerworktree.ColdReapList(wm.RepoRoot, defaultWorkerWorktreeRunner, time.Now(), 30*time.Minute, func(wtPath string) bool { return false })
	for _, cw := range cwList {
		if cw.Eligible && !cw.HeldByWork {
			if !dryRun {
				if err := os.RemoveAll(cw.Path); err == nil {
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
