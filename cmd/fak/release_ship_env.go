package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func releaseShipCmd(result *releaseShipResult, cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
	result.ExecutedCommands = append(result.ExecutedCommands, releaseShipCommand{
		CWD:  cwd,
		Name: name,
		Args: append([]string(nil), args...),
	})
	return releaseShipRunCommand(cwd, name, args, env, timeout)
}

func runReleaseShipCommand(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
	cmd := exec.Command(name, args...)
	cmd.Dir = cwd
	if env != nil {
		cmd.Env = env
	}
	var combined strings.Builder
	cmd.Stdout = &combined
	cmd.Stderr = &combined
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	done := make(chan error, 1)
	if err := cmd.Start(); err != nil {
		return 127, err.Error()
	}
	go func() { done <- cmd.Wait() }()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case err := <-done:
		if err != nil {
			if ee, ok := err.(*exec.ExitError); ok {
				return ee.ExitCode(), combined.String()
			}
			return 127, combined.String() + err.Error()
		}
		return 0, combined.String()
	case <-timer.C:
		// Tree-kill: the child may be a Python cut/publish orchestrator that spawned
		// in-flight `git`/`npm` children; a single-PID kill orphans them (#3103).
		_, _ = procguard.KillPID(cmd.Process.Pid)
		err := <-done
		if err != nil {
			return 124, combined.String() + fmt.Sprintf("\n(timed out after %s)", timeout)
		}
		return 124, combined.String() + fmt.Sprintf("\n(timed out after %s)", timeout)
	}
}

func releaseShipWorktreeDir(root string, opts releaseShipOptions) (string, error) {
	if opts.worktreeDir != "" {
		if filepath.IsAbs(opts.worktreeDir) {
			return filepath.Clean(opts.worktreeDir), nil
		}
		return filepath.Join(root, opts.worktreeDir), nil
	}
	parent := os.TempDir()
	dir, err := releaseShipMkdirTemp(parent, "fak-release-ship-*")
	if err != nil {
		return "", err
	}
	return dir, nil
}

func releaseShipPython() string {
	return releasePython()
}

func releaseShipScript(root, script string) string {
	return filepath.Join(root, "tools", script)
}

func releaseShipOwner() string {
	for _, key := range []string{"FAK_RELEASE_OWNER", "CLAUDE_CODE_SESSION_ID"} {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			return value
		}
	}
	return fmt.Sprintf("fak-release-ship-%d-%d", os.Getpid(), time.Now().UnixNano())
}

func releaseShipEnv(lockRoot, owner string) []string {
	env := os.Environ()
	env = setEnv(env, "FAK_RELEASE_LOCK_ROOT", lockRoot)
	env = setEnv(env, "FAK_RELEASE_OWNER", owner)
	return env
}

func releaseShipPromotionEnv(env []string, result releaseShipResult) []string {
	env = setEnv(env, "FAK_RELEASE_SOURCE_BRANCH", result.SourceBranch)
	env = setEnv(env, "FAK_RELEASE_SOURCE_SHA", result.SourceSHA)
	env = setEnv(env, "FAK_RELEASE_TARGET_BRANCH", result.TargetBranch)
	env = setEnv(env, "FAK_RELEASE_TARGET_SHA", result.TargetSHA)
	env = setEnv(env, "FAK_RELEASE_SOURCE_RANGE", releaseShipSourceRange(result))
	if result.SourceCI != nil {
		env = setEnv(env, "FAK_RELEASE_SOURCE_CI", stringFromAny(result.SourceCI["status"]))
	}
	return env
}

func releaseShipSourceRange(result releaseShipResult) string {
	if result.SourceSHA == "" {
		return ""
	}
	if result.TargetSHA == "" || sameSHA(result.TargetSHA, result.SourceSHA) {
		return result.SourceSHA
	}
	return result.TargetSHA + ".." + result.SourceSHA
}

func setEnv(env []string, key, value string) []string {
	prefix := key + "="
	out := make([]string, 0, len(env)+1)
	found := false
	for _, item := range env {
		if strings.HasPrefix(item, prefix) {
			out = append(out, prefix+value)
			found = true
		} else {
			out = append(out, item)
		}
	}
	if !found {
		out = append(out, prefix+value)
	}
	return out
}

func (r *releaseShipResult) fail(kind, detail string) {
	r.Errors = append(r.Errors, kind+": "+tail(detail))
	r.OK = false
}

func finishReleaseShip(result releaseShipResult) releaseShipResult {
	if len(result.Errors) > 0 {
		result.OK = false
	}
	if len(result.CommandTail) == 0 {
		result.CommandTail = nil
	}
	if len(result.RemoteBranch) == 0 {
		result.RemoteBranch = nil
	}
	if len(result.ExecutedCommands) == 0 {
		result.ExecutedCommands = nil
	}
	return result
}

func finishReleaseShipWithCleanup(result *releaseShipResult, root string, opts releaseShipOptions, worktreeAdded bool) releaseShipResult {
	if worktreeAdded && !opts.keepWorktree {
		cleanup := cleanupReleaseShipWorktree(root, result.Worktree)
		result.Cleanup = cleanup
		if ok, _ := cleanup["ok"].(bool); !ok {
			result.Warnings = append(result.Warnings, "detached worktree cleanup failed")
		}
	} else if opts.keepWorktree && worktreeAdded {
		result.Cleanup = map[string]any{"ok": true, "kept": true, "path": result.Worktree}
	}
	return finishReleaseShip(*result)
}
