// Package worktreereapcli owns the bounded, single-worktree retirement command.
// It does not grant owner completion or replace the separate bulk GC policy.
package worktreereapcli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
	"github.com/anthony-chaudhary/fak/internal/wiplifecycle"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// Run exposes only the existing single-target, inherently applying operation.
func Run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak-worktree-reap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "explicit canonical repository root")
	target := fs.String("worktree", "", "registered, independently quiescent managed worker")
	superseded := fs.String("superseded-by", "", "accepted trunk commit proving exact dirty source")
	wait := fs.Duration("max-wait", 10*time.Second, "total inspection, proof and removal deadline")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*root) == "" || strings.TrimSpace(*target) == "" {
		fmt.Fprintln(stderr, "fak-worktree-reap: explicit --root and --worktree required; no bulk or force modes")
		return 2
	}
	return RunSingle(stdout, stderr, *root, *target, *superseded, *wait)
}

// RunSingle is shared by the full CLI and the inference-free standalone command.
// The caller must establish owner completion and recoverable backups first.
func RunSingle(stdout, stderr io.Writer, root, target, superseded string, maxWait time.Duration) int {
	emit := func(result workerworktree.Result, code int) int {
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(result); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return code
	}
	refuse := func(code, reason, detail string) int {
		return emit(workerworktree.Result{OK: false, Code: code, Path: target, Preserved: true, Reason: reason, Detail: detail}, 1)
	}
	// Fail closed rather than mutate process-global Git configuration. Every Git
	// child must observe the explicit root/target, not a hook's inherited selectors.
	for _, key := range []string{"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_PREFIX", "GIT_NAMESPACE", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT"} {
		if os.Getenv(key) != "" {
			return refuse("REAP_PATH_REFUSED", "inherited Git repository selector is not admitted", key)
		}
	}
	if maxWait <= 0 {
		return emit(workerworktree.Result{Code: workerworktree.ReapCodeTimeout, Path: target, Preserved: true, Reason: "--max-wait must be positive"}, 2)
	}
	var err error
	root, err = filepath.Abs(strings.TrimSpace(root))
	if err != nil {
		return refuse("REAP_PATH_REFUSED", "cannot resolve root", err.Error())
	}
	target, err = filepath.Abs(strings.TrimSpace(target))
	if err != nil {
		return refuse("REAP_PATH_REFUSED", "cannot resolve target", err.Error())
	}
	ctx, cancel := context.WithTimeout(context.Background(), maxWait)
	defer cancel()
	git := workerworktree.BoundedGitRunner(ctx)
	if err := registeredTarget(root, target, git); err != nil {
		return refuse("REAP_PATH_REFUSED", "target is not a registered worker of the canonical repository", err.Error())
	}
	fmt.Fprintf(stderr, "REAP_PROGRESS code=REAP_STARTED max_wait=%s\n", maxWait)
	beforeCtx, beforeCancel := context.WithTimeout(ctx, maxWait/2)
	receipt, err := wiplifecycle.BeginWithRunner(root, "worker-reap", "", time.Now(), wipinventory.DeadlineRunner{Context: beforeCtx}, wipinventory.Options{Focus: target})
	beforeCancel()
	if receipt.OperationID != "" {
		defer func() {
			finished, finishErr := wiplifecycle.FinishWithRunner(root, receipt.OperationID, time.Now(), wipinventory.DeadlineRunner{Context: ctx})
			if finishErr != nil {
				fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=after kind=worker-reap operation=%s error=%v\n", receipt.OperationID, finishErr)
				return
			}
			fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURED phase=after kind=worker-reap operation=%s artifact=%s receipt=%s\n", finished.OperationID, finished.After.Artifact, finished.ReceiptPath)
		}()
	}
	if err != nil || !receipt.Before.Known {
		detail := receipt.Before.Error
		if err != nil {
			detail = err.Error()
		}
		return refuse("REAP_PRESTATE_UNKNOWN", "target before lifecycle inventory could not be captured completely", detail)
	}
	fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURED phase=before kind=worker-reap operation=%s artifact=%s\n", receipt.OperationID, receipt.Before.Artifact)
	for _, advisory := range receipt.Before.Advisory {
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_ADVISORY phase=before kind=worker-reap operation=%s error=%q\n", receipt.OperationID, advisory)
	}
	if err := registeredTarget(root, target, git); err != nil {
		return refuse("REAP_PATH_REFUSED", "registration changed during lifecycle capture", err.Error())
	}
	result := workerworktree.ReapChecked(root, target, strings.TrimSpace(superseded), git)
	if !result.OK {
		return emit(result, 1)
	}
	return emit(result, 0)
}

// Focused inventory is not registration authority. Require a current exact
// same-repository registration independently, without accepting symlink aliases.
func registeredTarget(root, target string, git workerworktree.GitRunner) error {
	for _, path := range []string{root, target} {
		real, err := filepath.EvalSymlinks(path)
		if err != nil || filepath.Clean(real) != path {
			return fmt.Errorf("missing or aliased path: %s", path)
		}
	}
	if root == target || !workerworktree.IsWorkerWorktree(target) {
		return fmt.Errorf("not a distinct managed worker")
	}
	rc, listing := git(root, []string{"worktree", "list", "--porcelain", "-z"})
	if rc != 0 {
		return fmt.Errorf("worktree registration unavailable (exit %d)", rc)
	}
	canonical, found := "", false
	for _, block := range strings.Split(listing, "\x00\x00") {
		path, blocked := "", false
		for _, field := range strings.Split(block, "\x00") {
			if strings.HasPrefix(field, "worktree ") {
				path = filepath.Clean(strings.TrimPrefix(field, "worktree "))
			}
			if field == "bare" || field == "locked" || strings.HasPrefix(field, "locked ") || field == "prunable" || strings.HasPrefix(field, "prunable ") {
				blocked = true
			}
		}
		if canonical == "" && path != "" {
			canonical = path
		}
		if path == target {
			if blocked {
				return fmt.Errorf("worker registration is locked, bare or prunable")
			}
			found = true
		}
	}
	if canonical != root || !found {
		return fmt.Errorf("root or target registration mismatch")
	}
	common := ""
	for _, path := range []string{root, target} {
		rc, top := git(path, []string{"rev-parse", "--show-toplevel"})
		if rc != 0 || filepath.Clean(strings.TrimSpace(top)) != path {
			return fmt.Errorf("effective Git working tree differs from explicit path")
		}
		rc, out := git(path, []string{"rev-parse", "--path-format=absolute", "--git-common-dir"})
		if rc != 0 {
			return fmt.Errorf("common Git directory unavailable (exit %d)", rc)
		}
		actual, err := filepath.EvalSymlinks(strings.TrimSpace(out))
		if err != nil {
			return err
		}
		if common == "" {
			common = actual
		} else if actual != common {
			return fmt.Errorf("foreign Git object store")
		}
	}
	// A regular .git pointer can be swapped between siblings sharing the same
	// object store. Bind its administrative backlink to this exact target too.
	rc, adminOut := git(target, []string{"rev-parse", "--absolute-git-dir"})
	if rc != 0 {
		return fmt.Errorf("worker administrative directory unavailable (exit %d)", rc)
	}
	admin, err := filepath.EvalSymlinks(strings.TrimSpace(adminOut))
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(filepath.Join(common, "worktrees"), admin)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return fmt.Errorf("worker admin directory is outside canonical registrations")
	}
	backlinkPath := filepath.Join(admin, "gitdir")
	info, err := os.Lstat(backlinkPath)
	if err != nil || !info.Mode().IsRegular() {
		return fmt.Errorf("worker registration backlink missing or unsafe")
	}
	backlink, err := os.ReadFile(backlinkPath)
	if err != nil || filepath.Clean(strings.TrimSpace(string(backlink))) != filepath.Join(target, ".git") {
		return fmt.Errorf("worker registration backlink differs from target")
	}
	// A registered path replaced by an ordinary directory must not reach ForceReap.
	if info, err := os.Lstat(filepath.Join(target, ".git")); err != nil || !info.Mode().IsRegular() {
		return fmt.Errorf("registered worker Git pointer missing or unsafe")
	}
	return nil
}
