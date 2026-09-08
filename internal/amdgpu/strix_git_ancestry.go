package amdgpu

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	strixGitSemanticEpoch       = "c3d5ac66e4bfdfa7eee783bd7b06abb8dbaec584"
	strixGitAncestryGitPath     = "/usr/bin/git"
	strixGitAncestryTimeout     = 30 * time.Second
	strixGitAncestryOutputLimit = 4 << 10
)

// strixGitAncestry is an authenticated relationship to the fixed semantic
// epoch. Unattested is never authority; callers must also require a nil error.
type strixGitAncestry uint8

const (
	strixGitAncestryUnattested strixGitAncestry = iota
	strixGitAncestryEpochEqual
	strixGitAncestryDescendant
	strixGitAncestryPreEpoch
	strixGitAncestryUnrelated
)

type strixGitAncestryRefusal struct {
	token string
}

func (e *strixGitAncestryRefusal) Error() string {
	return e.token + ": sterile Git ancestry refused"
}

func refuseStrixGitAncestry() error {
	return &strixGitAncestryRefusal{token: strixGitSnapshotUnattestedToken}
}

type strixGitCommand struct {
	path string
	args []string
	env  []string
	dir  string
}

type strixGitCommandResult struct {
	stdout   []byte
	exitCode int
	err      error
}

type strixGitAncestryOptions struct {
	epoch   string
	timeout time.Duration
	run     func(context.Context, strixGitCommand) strixGitCommandResult
}

func (o strixGitAncestryOptions) withDefaults() (strixGitAncestryOptions, bool) {
	if o.timeout < 0 {
		return strixGitAncestryOptions{}, false
	}
	if o.epoch == "" {
		o.epoch = strixGitSemanticEpoch
	}
	if o.timeout == 0 {
		o.timeout = strixGitAncestryTimeout
	}
	if o.run == nil {
		o.run = runStrixGitCommand
	}
	return o, isLowerHex(o.epoch, 40) && o.timeout > 0
}

// classifyStrixGitSnapshotAncestry is the production same-package seam. It
// consumes the snapshot: every return attempts destruction, and a cleanup
// failure revokes any classification while retaining ownership for retry.
func classifyStrixGitSnapshotAncestry(ctx context.Context, snapshot *strixGitObjectSnapshot, observedRevision string) (strixGitAncestry, error) {
	return classifyStrixGitSnapshotAncestryWithOptions(ctx, snapshot, observedRevision, strixGitAncestryOptions{})
}

func classifyStrixGitSnapshotAncestryWithOptions(ctx context.Context, snapshot *strixGitObjectSnapshot, observedRevision string, rawOpts strixGitAncestryOptions) (result strixGitAncestry, resultErr error) {
	result = strixGitAncestryUnattested
	if snapshot != nil {
		defer func() {
			if cleanupErr := snapshot.close(); cleanupErr != nil {
				result = strixGitAncestryUnattested
				if resultErr == nil {
					resultErr = refuseStrixGitAncestry()
				}
				resultErr = errors.Join(resultErr, cleanupErr)
			}
		}()
	}
	if ctx == nil || snapshot == nil || !isLowerHex(observedRevision, 40) {
		return result, refuseStrixGitAncestry()
	}
	opts, ok := rawOpts.withDefaults()
	if !ok || ctx.Err() != nil {
		return result, refuseStrixGitAncestry()
	}
	root := snapshot.gitDir()
	if root == "" || !filepath.IsAbs(root) || filepath.Clean(root) != root {
		return result, refuseStrixGitAncestry()
	}

	boundedCtx, cancel := context.WithTimeout(ctx, opts.timeout)
	defer cancel()
	if err := validateStrixGitAncestrySnapshotTree(boundedCtx, root); err != nil {
		return result, refuseStrixGitAncestry()
	}
	if err := prepareStrixGitConsumerRoot(root); err != nil {
		return result, refuseStrixGitAncestry()
	}

	run := func(args ...string) strixGitCommandResult {
		commandArgs := append([]string{"--no-replace-objects", "--git-dir=" + root}, args...)
		return opts.run(boundedCtx, strixGitCommand{
			path: strixGitAncestryGitPath,
			args: commandArgs,
			env:  strixGitAncestryEnvironment(),
			dir:  "/",
		})
	}
	if !commandSucceeded(run("fsck", "--full", "--strict", "--no-reflogs", "--no-dangling", observedRevision, opts.epoch)) {
		return result, refuseStrixGitAncestry()
	}
	if !commandOutputIsCommit(run("cat-file", "-t", observedRevision)) || !commandOutputIsCommit(run("cat-file", "-t", opts.epoch)) {
		return result, refuseStrixGitAncestry()
	}

	epochAncestor := run("merge-base", "--is-ancestor", opts.epoch, observedRevision)
	switch mergeBaseVerdict(epochAncestor) {
	case 0:
		if observedRevision == opts.epoch {
			return strixGitAncestryEpochEqual, nil
		}
		return strixGitAncestryDescendant, nil
	case 1:
		observedAncestor := run("merge-base", "--is-ancestor", observedRevision, opts.epoch)
		switch mergeBaseVerdict(observedAncestor) {
		case 0:
			return strixGitAncestryPreEpoch, nil
		case 1:
			return strixGitAncestryUnrelated, nil
		default:
			return result, refuseStrixGitAncestry()
		}
	default:
		return result, refuseStrixGitAncestry()
	}
}

// prepareStrixGitConsumerRoot adds only the fixed control skeleton Git needs
// to recognize the already-validated object snapshot as a bare repository.
// Exclusive creation makes any preexisting or raced control state a refusal.
func prepareStrixGitConsumerRoot(root string) error {
	destination, err := os.OpenRoot(root)
	if err != nil {
		return err
	}
	defer destination.Close()
	if err := destination.Mkdir("refs", 0o700); err != nil {
		return err
	}
	if err := destination.Mkdir("refs/heads", 0o700); err != nil {
		return err
	}
	head, err := destination.OpenFile("HEAD", os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	keep := false
	defer func() {
		_ = head.Close()
		if !keep {
			_ = destination.Remove("HEAD")
		}
	}()
	if _, err := io.WriteString(head, "ref: refs/heads/fak-sterile\n"); err != nil {
		return err
	}
	if err := head.Sync(); err != nil {
		return err
	}
	if err := head.Close(); err != nil {
		return err
	}
	keep = true
	return nil
}

func strixGitAncestryEnvironment() []string {
	return []string{
		"HOME=/nonexistent",
		"XDG_CONFIG_HOME=/nonexistent",
		"PATH=/usr/bin:/bin",
		"LANG=C",
		"LC_ALL=C",
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_CONFIG_GLOBAL=/dev/null",
		"GIT_CONFIG_SYSTEM=/dev/null",
		"GIT_TERMINAL_PROMPT=0",
		"GIT_ASKPASS=/bin/false",
		"SSH_ASKPASS=/bin/false",
		"GIT_OPTIONAL_LOCKS=0",
		"GIT_NO_LAZY_FETCH=1",
		"GIT_PROTOCOL_FROM_USER=0",
		"GIT_ALLOW_PROTOCOL=",
		"GIT_CEILING_DIRECTORIES=/",
		"NO_COLOR=1",
	}
}

func commandSucceeded(result strixGitCommandResult) bool {
	return result.err == nil && result.exitCode == 0
}

func commandOutputIsCommit(result strixGitCommandResult) bool {
	return commandSucceeded(result) && bytes.Equal(result.stdout, []byte("commit\n"))
}

func mergeBaseVerdict(result strixGitCommandResult) int {
	if result.exitCode == 0 && result.err == nil && len(result.stdout) == 0 {
		return 0
	}
	var exitErr *exec.ExitError
	if result.exitCode == 1 && errors.As(result.err, &exitErr) && len(result.stdout) == 0 {
		return 1
	}
	return -1
}

func runStrixGitCommand(ctx context.Context, command strixGitCommand) strixGitCommandResult {
	result := strixGitCommandResult{exitCode: -1}
	if ctx == nil || command.path != strixGitAncestryGitPath || command.dir != "/" {
		result.err = fmt.Errorf("invalid fixed Git command")
		return result
	}
	cmd := exec.CommandContext(ctx, command.path, command.args...)
	cmd.Dir = command.dir
	cmd.Env = append([]string(nil), command.env...)
	stdout := &strixGitLimitedBuffer{remaining: strixGitAncestryOutputLimit}
	cmd.Stdout = stdout
	cmd.Stderr = io.Discard
	err := cmd.Run()
	result.stdout = append([]byte(nil), stdout.body.Bytes()...)
	result.err = err
	if stdout.exceeded {
		result.err = fmt.Errorf("Git output limit exceeded")
		return result
	}
	if err == nil {
		result.exitCode = 0
		return result
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		result.exitCode = exitErr.ExitCode()
	}
	return result
}

type strixGitLimitedBuffer struct {
	body      bytes.Buffer
	remaining int
	exceeded  bool
}

func (w *strixGitLimitedBuffer) Write(p []byte) (int, error) {
	if len(p) > w.remaining {
		if w.remaining > 0 {
			_, _ = w.body.Write(p[:w.remaining])
		}
		w.remaining = 0
		w.exceeded = true
		return len(p), nil
	}
	_, _ = w.body.Write(p)
	w.remaining -= len(p)
	return len(p), nil
}

func validateStrixGitAncestrySnapshotTree(ctx context.Context, root string) error {
	rootInfo, err := os.Lstat(root)
	if err != nil || rootInfo.Mode()&os.ModeSymlink != 0 || !rootInfo.IsDir() {
		return fmt.Errorf("invalid snapshot root")
	}
	if runtime.GOOS != "windows" && rootInfo.Mode().Perm() != 0o700 {
		return fmt.Errorf("invalid snapshot root mode")
	}
	resolved, err := filepath.EvalSymlinks(root)
	if err != nil || !sameStrixFilesystemPath(root, resolved) {
		return fmt.Errorf("redirected snapshot root")
	}
	seenConfig := false
	packKinds := make(map[string]uint8)
	entries := 0
	err = filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil || ctx.Err() != nil {
			if walkErr != nil {
				return walkErr
			}
			return ctx.Err()
		}
		if path == root {
			return nil
		}
		entries++
		if entries > strixGitSnapshotDefaultMaxEntries+260 {
			return fmt.Errorf("snapshot tree cap exceeded")
		}
		rel, err := filepath.Rel(root, path)
		if err != nil || !filepath.IsLocal(rel) {
			return fmt.Errorf("invalid snapshot path")
		}
		rel = filepath.ToSlash(rel)
		info, err := entry.Info()
		if err != nil || info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("unsupported snapshot entry")
		}
		if entry.IsDir() {
			if !validStrixGitSnapshotDirectory(rel) || (runtime.GOOS != "windows" && info.Mode().Perm() != 0o700) {
				return fmt.Errorf("unsupported snapshot directory")
			}
			return nil
		}
		if !info.Mode().IsRegular() || (runtime.GOOS != "windows" && info.Mode().Perm() != 0o600) {
			return fmt.Errorf("unsupported snapshot file")
		}
		switch {
		case rel == "config":
			if seenConfig {
				return fmt.Errorf("duplicate snapshot config")
			}
			if info.Size() != int64(len(strixGitSnapshotConfig)) {
				return fmt.Errorf("unsupported snapshot config")
			}
			body, readErr := os.ReadFile(path)
			if readErr != nil || string(body) != strixGitSnapshotConfig {
				return fmt.Errorf("unsupported snapshot config")
			}
			seenConfig = true
			return nil
		case validStrixGitLooseSnapshotPath(rel):
			return nil
		default:
			id, kind, ok := validStrixGitPackSnapshotPath(rel)
			if !ok {
				return fmt.Errorf("unsupported snapshot path")
			}
			packKinds[id] |= kind
			return nil
		}
	})
	if err != nil || !seenConfig {
		return fmt.Errorf("invalid snapshot tree")
	}
	for _, kinds := range packKinds {
		if kinds != 3 {
			return fmt.Errorf("incomplete snapshot pack pair")
		}
	}
	return nil
}

func validStrixGitSnapshotDirectory(rel string) bool {
	if rel == "objects" || rel == "objects/pack" {
		return true
	}
	parts := strings.Split(rel, "/")
	return len(parts) == 2 && parts[0] == "objects" && isLowerHex(parts[1], 2)
}

func validStrixGitLooseSnapshotPath(rel string) bool {
	parts := strings.Split(rel, "/")
	return len(parts) == 3 && parts[0] == "objects" && isLowerHex(parts[1], 2) && isLowerHex(parts[2], 38)
}

func validStrixGitPackSnapshotPath(rel string) (string, uint8, bool) {
	parts := strings.Split(rel, "/")
	if len(parts) != 3 || parts[0] != "objects" || parts[1] != "pack" {
		return "", 0, false
	}
	return canonicalStrixPackObjectName(parts[2])
}
