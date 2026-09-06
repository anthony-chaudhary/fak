package safecommit

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// TurnCommitResult records the outcome of an automated per-turn commit.
type TurnCommitResult struct {
	Committed  bool
	RolledBack bool
	CommitSHA  string
	Error      error
}

// PerTurnCommitter coordinates automated commits per turn with test execution and automatic rollback.
type PerTurnCommitter struct {
	RepoDir string
	TestCmd string
	Timeout time.Duration
}

// NewPerTurnCommitter initializes a committer for the given repository and verification test command.
func NewPerTurnCommitter(repoDir, testCmd string) *PerTurnCommitter {
	return &PerTurnCommitter{
		RepoDir: repoDir,
		TestCmd: testCmd,
	}
}

// Commit executes an automated commit for the turn.
func (c *PerTurnCommitter) Commit(paths []string, message string) (*TurnCommitResult, error) {
	return c.CommitTurn(paths, message)
}

// CommitTurn executes an automated commit for the turn using configured parameters.
func (c *PerTurnCommitter) CommitTurn(paths []string, message string) (*TurnCommitResult, error) {
	if c == nil {
		return nil, errors.New("safecommit: nil PerTurnCommitter")
	}
	ctx := context.Background()
	if c.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.Timeout)
		defer cancel()
	}
	return CommitTurnWithContext(ctx, c.RepoDir, paths, message, c.TestCmd)
}

// CommitTurnWithContext executes an automated commit for the turn with explicit context.
func (c *PerTurnCommitter) CommitTurnWithContext(ctx context.Context, paths []string, message string) (*TurnCommitResult, error) {
	if c == nil {
		return nil, errors.New("safecommit: nil PerTurnCommitter")
	}
	if c.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.Timeout)
		defer cancel()
	}
	return CommitTurnWithContext(ctx, c.RepoDir, paths, message, c.TestCmd)
}

// CommitTurn stages paths, runs testCmd (if provided), commits on test success,
// and automatically rolls back changes to the pre-turn commit if testCmd fails.
func CommitTurn(repoDir string, paths []string, message string, testCmd string) (*TurnCommitResult, error) {
	return CommitTurnWithContext(context.Background(), repoDir, paths, message, testCmd)
}

// CommitTurnWithContext stages paths, runs testCmd (if provided) under ctx, commits on test success,
// and automatically rolls back changes to the pre-turn commit if testCmd fails.
func CommitTurnWithContext(ctx context.Context, repoDir string, paths []string, message string, testCmd string) (*TurnCommitResult, error) {
	if repoDir == "" {
		repoDir = "."
	}

	// Ensure the directory is inside a valid git repository.
	checkCmd := newGitCmd(ctx, repoDir, "rev-parse", "--is-inside-work-tree")
	if out, err := checkCmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("safecommit: %s is not a git repository: %w (%s)", repoDir, err, strings.TrimSpace(string(out)))
	}

	// Capture pre-turn HEAD commit SHA if present (fails gracefully on unborn branch).
	headCmd := newGitCmd(ctx, repoDir, "rev-parse", "HEAD")
	headOut, headErr := headCmd.CombinedOutput()
	preSHA := ""
	if headErr == nil {
		preSHA = strings.TrimSpace(string(headOut))
	}

	// Stage requested paths if non-empty.
	var stagePaths []string
	for _, p := range paths {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			stagePaths = append(stagePaths, trimmed)
		}
	}
	if len(stagePaths) > 0 {
		args := append([]string{"add", "--"}, stagePaths...)
		addCmd := newGitCmd(ctx, repoDir, args...)
		if out, err := addCmd.CombinedOutput(); err != nil {
			return nil, fmt.Errorf("safecommit: staging paths failed: %w: %s", err, strings.TrimSpace(string(out)))
		}
	}

	// Run testCmd when specified.
	testCmd = strings.TrimSpace(testCmd)
	if testCmd != "" {
		testExec := buildShellCmd(ctx, testCmd)
		testExec.Dir = repoDir
		testExec.Env = os.Environ()

		testOut, testErr := testExec.CombinedOutput()
		if testErr != nil {
			// Roll back uncommitted changes / restore working tree to pre-turn commit.
			rollbackErr := rollbackTurn(ctx, repoDir, preSHA, stagePaths)
			resErr := testErr
			if len(testOut) > 0 {
				resErr = fmt.Errorf("test failed (%w): %s", testErr, strings.TrimSpace(string(testOut)))
			}
			if rollbackErr != nil {
				return &TurnCommitResult{
					Committed:  false,
					RolledBack: false,
					Error:      fmt.Errorf("test failed and rollback failed (%w): %v", rollbackErr, resErr),
				}, rollbackErr
			}
			return &TurnCommitResult{
				Committed:  false,
				RolledBack: true,
				Error:      resErr,
			}, nil
		}
	}

	// Test succeeded or was empty: commit staged changes.
	if strings.TrimSpace(message) == "" {
		message = "per-turn automated commit"
	}

	commitCmd := newGitCmd(ctx, repoDir, "commit", "-m", message)
	commitOut, commitErr := commitCmd.CombinedOutput()
	if commitErr != nil {
		return &TurnCommitResult{
			Committed:  false,
			RolledBack: false,
			Error:      fmt.Errorf("safecommit: git commit failed: %w: %s", commitErr, strings.TrimSpace(string(commitOut))),
		}, fmt.Errorf("safecommit: git commit failed: %w: %s", commitErr, strings.TrimSpace(string(commitOut)))
	}

	// Query resulting HEAD commit SHA.
	shaCmd := newGitCmd(ctx, repoDir, "rev-parse", "HEAD")
	shaOut, shaErr := shaCmd.CombinedOutput()
	if shaErr != nil {
		return nil, fmt.Errorf("safecommit: failed to resolve commit SHA: %w", shaErr)
	}
	newSHA := strings.TrimSpace(string(shaOut))

	return &TurnCommitResult{
		Committed:  true,
		RolledBack: false,
		CommitSHA:  newSHA,
		Error:      nil,
	}, nil
}

func buildShellCmd(ctx context.Context, command string) *exec.Cmd {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd.exe", "/d", "/s", "/c", command)
	} else {
		cmd = exec.CommandContext(ctx, "/bin/sh", "-c", command)
	}
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd
}

func rollbackTurn(ctx context.Context, repoDir string, preSHA string, targetPaths []string) error {
	if len(targetPaths) > 0 {
		if preSHA != "" {
			// 1. Unstage target paths so new files become untracked and modified files are unstaged
			for _, p := range targetPaths {
				resetCmd := newGitCmd(ctx, repoDir, "reset", "HEAD", "--", p)
				_, _ = resetCmd.CombinedOutput()
			}

			// 2. Restore tracked files back to HEAD
			for _, p := range targetPaths {
				checkoutCmd := newGitCmd(ctx, repoDir, "checkout", "HEAD", "--", p)
				_, _ = checkoutCmd.CombinedOutput()
			}
		} else {
			// Unborn branch: unstage target paths
			for _, p := range targetPaths {
				rmCmd := newGitCmd(ctx, repoDir, "rm", "-rf", "--cached", "--", p)
				_, _ = rmCmd.CombinedOutput()
			}
		}

		// 3. Clean untracked files in targetPaths
		for _, p := range targetPaths {
			cleanCmd := newGitCmd(ctx, repoDir, "clean", "-fd", "--", p)
			_, _ = cleanCmd.CombinedOutput()

			// In case git clean didn't remove newly added untracked file
			full := filepath.Join(repoDir, p)
			if preSHA == "" {
				_ = os.RemoveAll(full)
			} else {
				// Only remove if it was not in pre-turn commit
				catCmd := newGitCmd(ctx, repoDir, "cat-file", "-e", "HEAD:"+p)
				if err := catCmd.Run(); err != nil {
					_ = os.RemoveAll(full)
				}
			}
		}
		return nil
	}

	if preSHA != "" {
		// Reset working tree and index to pre-turn commit when no specific paths were declared.
		resetCmd := newGitCmd(ctx, repoDir, "reset", "--hard", "HEAD")
		if out, err := resetCmd.CombinedOutput(); err != nil {
			return fmt.Errorf("git reset --hard failed: %w: %s", err, strings.TrimSpace(string(out)))
		}
		cleanCmd := newGitCmd(ctx, repoDir, "clean", "-fd")
		if out, err := cleanCmd.CombinedOutput(); err != nil {
			return fmt.Errorf("git clean -fd failed: %w: %s", err, strings.TrimSpace(string(out)))
		}
		return nil
	}

	// No pre-turn commit (unborn branch): unstage all files and clean working tree.
	rmCmd := newGitCmd(ctx, repoDir, "rm", "-rf", "--cached", ".")
	_, _ = rmCmd.CombinedOutput()

	cleanCmd := newGitCmd(ctx, repoDir, "clean", "-fd")
	if out, err := cleanCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git clean -fd failed: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}
