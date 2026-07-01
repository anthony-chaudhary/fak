// Package edittx applies a batch of full-file edits as one working-tree
// transaction: every target is snapshotted first, checks run against the applied
// set, and any failure restores the touched files before returning.
package edittx

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

const Schema = "fak-edit-tx/1"

const (
	ReasonNoEdits        = "NO_EDITS"
	ReasonInvalidPath    = "INVALID_PATH"
	ReasonDuplicatePath  = "DUPLICATE_PATH"
	ReasonReadFailed     = "READ_FAILED"
	ReasonApplyFailed    = "APPLY_FAILED"
	ReasonCheckFailed    = "CHECK_FAILED"
	ReasonRollbackFailed = "ROLLBACK_FAILED"
)

type Spec struct {
	Edits  []Edit   `json:"edits"`
	Checks []string `json:"checks,omitempty"`
}

type Edit struct {
	Path    string  `json:"path"`
	Content *string `json:"content,omitempty"`
	Delete  bool    `json:"delete,omitempty"`
}

type Options struct {
	Root   string
	Spec   Spec
	Checks []string
	DryRun bool
	Run    Runner
}

type Runner func(context.Context, string, string) CheckResult

type CheckResult struct {
	Command  string `json:"command"`
	ExitCode int    `json:"exit_code"`
	Output   string `json:"output,omitempty"`
	Error    string `json:"error,omitempty"`
}

type Result struct {
	Schema     string        `json:"schema"`
	OK         bool          `json:"ok"`
	Applied    bool          `json:"applied"`
	RolledBack bool          `json:"rolled_back"`
	DryRun     bool          `json:"dry_run,omitempty"`
	Edits      int           `json:"edits"`
	Paths      []string      `json:"paths"`
	Checks     []CheckResult `json:"checks,omitempty"`
	Reason     string        `json:"reason,omitempty"`
	Detail     string        `json:"detail,omitempty"`
}

type snapshot struct {
	path    string
	abs     string
	existed bool
	data    []byte
	mode    os.FileMode
}

func Apply(ctx context.Context, opts Options) Result {
	res := Result{Schema: Schema}
	root := opts.Root
	if root == "" {
		root = "."
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return refuse(res, ReasonInvalidPath, err.Error())
	}
	root = filepath.Clean(absRoot)

	edits, paths, err := normalizeEdits(opts.Spec.Edits, root)
	res.Edits = len(edits)
	res.Paths = paths
	if err != nil {
		var dup duplicatePathError
		if errors.As(err, &dup) {
			return refuse(res, ReasonDuplicatePath, err.Error())
		}
		return refuse(res, ReasonInvalidPath, err.Error())
	}
	if len(edits) == 0 {
		return refuse(res, ReasonNoEdits, "spec.edits is empty")
	}
	if opts.DryRun {
		res.OK = true
		res.DryRun = true
		return res
	}

	snaps, err := snapshotTargets(edits)
	if err != nil {
		return refuse(res, ReasonReadFailed, err.Error())
	}
	if err := applyEdits(edits); err != nil {
		res = refuse(res, ReasonApplyFailed, err.Error())
		return rollbackInto(res, snaps, root)
	}

	checks := append([]string{}, opts.Spec.Checks...)
	checks = append(checks, opts.Checks...)
	run := opts.Run
	if run == nil {
		run = DefaultRunner
	}
	for _, cmd := range checks {
		cmd = strings.TrimSpace(cmd)
		if cmd == "" {
			continue
		}
		cr := run(ctx, root, cmd)
		res.Checks = append(res.Checks, cr)
		if cr.ExitCode != 0 || cr.Error != "" {
			res = refuse(res, ReasonCheckFailed, fmt.Sprintf("%s exited %d: %s%s", cmd, cr.ExitCode, cr.Output, cr.Error))
			return rollbackInto(res, snaps, root)
		}
	}
	res.OK = true
	res.Applied = true
	return res
}

type normalizedEdit struct {
	path    string
	abs     string
	content *string
	delete  bool
}

type duplicatePathError string

func (e duplicatePathError) Error() string { return string(e) }

func normalizeEdits(in []Edit, root string) ([]normalizedEdit, []string, error) {
	out := make([]normalizedEdit, 0, len(in))
	seen := map[string]bool{}
	for i, e := range in {
		if e.Delete && e.Content != nil {
			return nil, nil, fmt.Errorf("edit %d %q sets both content and delete", i, e.Path)
		}
		if !e.Delete && e.Content == nil {
			return nil, nil, fmt.Errorf("edit %d %q has neither content nor delete", i, e.Path)
		}
		clean, abs, err := cleanTarget(root, e.Path)
		if err != nil {
			return nil, nil, fmt.Errorf("edit %d: %w", i, err)
		}
		if seen[clean] {
			return nil, nil, duplicatePathError(fmt.Sprintf("duplicate edit path %q", clean))
		}
		seen[clean] = true
		out = append(out, normalizedEdit{path: clean, abs: abs, content: e.Content, delete: e.Delete})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].path < out[j].path })
	paths := make([]string, len(out))
	for i, e := range out {
		paths[i] = e.path
	}
	return out, paths, nil
}

func cleanTarget(root, p string) (string, string, error) {
	p = strings.TrimSpace(p)
	if p == "" {
		return "", "", errors.New("empty path")
	}
	p = strings.ReplaceAll(p, "\\", "/")
	if filepath.IsAbs(p) {
		return "", "", fmt.Errorf("absolute path %q is outside the edit transaction", p)
	}
	clean := filepath.Clean(filepath.FromSlash(p))
	if clean == "." || strings.HasPrefix(clean, ".."+string(os.PathSeparator)) || clean == ".." {
		return "", "", fmt.Errorf("path %q escapes the workspace", p)
	}
	abs := filepath.Join(root, clean)
	rel, err := filepath.Rel(root, abs)
	if err != nil {
		return "", "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) || filepath.IsAbs(rel) {
		return "", "", fmt.Errorf("path %q escapes the workspace", p)
	}
	if err := ensureSymlinkResolvedUnderRoot(root, abs); err != nil {
		return "", "", err
	}
	return filepath.ToSlash(rel), abs, nil
}

func ensureSymlinkResolvedUnderRoot(root, abs string) error {
	probe := abs
	for {
		if _, err := os.Lstat(probe); err == nil {
			resolved, err := filepath.EvalSymlinks(probe)
			if err != nil {
				return err
			}
			return ensureUnderRoot(root, resolved)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			return nil
		}
		probe = parent
	}
}

func ensureUnderRoot(root, path string) error {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) || filepath.IsAbs(rel) {
		return fmt.Errorf("path resolves outside workspace: %s", path)
	}
	return nil
}

func snapshotTargets(edits []normalizedEdit) ([]snapshot, error) {
	snaps := make([]snapshot, 0, len(edits))
	for _, e := range edits {
		info, err := os.Lstat(e.abs)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				snaps = append(snaps, snapshot{path: e.path, abs: e.abs})
				continue
			}
			return nil, fmt.Errorf("snapshot %s: %w", e.path, err)
		}
		if info.IsDir() {
			return nil, fmt.Errorf("snapshot %s: target is a directory", e.path)
		}
		b, err := os.ReadFile(e.abs)
		if err != nil {
			return nil, fmt.Errorf("snapshot %s: %w", e.path, err)
		}
		snaps = append(snaps, snapshot{path: e.path, abs: e.abs, existed: true, data: b, mode: info.Mode().Perm()})
	}
	return snaps, nil
}

func applyEdits(edits []normalizedEdit) error {
	for _, e := range edits {
		if e.delete {
			if err := os.Remove(e.abs); err != nil && !errors.Is(err, os.ErrNotExist) {
				return fmt.Errorf("delete %s: %w", e.path, err)
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(e.abs), 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", e.path, err)
		}
		mode := os.FileMode(0o644)
		if info, err := os.Lstat(e.abs); err == nil {
			mode = info.Mode().Perm()
		}
		if err := os.WriteFile(e.abs, []byte(*e.content), mode); err != nil {
			return fmt.Errorf("write %s: %w", e.path, err)
		}
	}
	return nil
}

func rollbackInto(res Result, snaps []snapshot, root string) Result {
	if err := rollback(snaps, root); err != nil {
		res.RolledBack = false
		res.Reason = ReasonRollbackFailed
		if res.Detail != "" {
			res.Detail += "; "
		}
		res.Detail += err.Error()
		return res
	}
	res.RolledBack = true
	return res
}

func rollback(snaps []snapshot, root string) error {
	var errs []string
	for i := len(snaps) - 1; i >= 0; i-- {
		s := snaps[i]
		if s.existed {
			if err := os.MkdirAll(filepath.Dir(s.abs), 0o755); err != nil {
				errs = append(errs, fmt.Sprintf("mkdir %s: %v", s.path, err))
				continue
			}
			if err := os.WriteFile(s.abs, s.data, s.mode); err != nil {
				errs = append(errs, fmt.Sprintf("restore %s: %v", s.path, err))
			}
			continue
		}
		if err := os.Remove(s.abs); err != nil && !errors.Is(err, os.ErrNotExist) {
			errs = append(errs, fmt.Sprintf("remove new %s: %v", s.path, err))
			continue
		}
		pruneEmptyParents(filepath.Dir(s.abs), root)
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func pruneEmptyParents(dir, root string) {
	root = filepath.Clean(root)
	for {
		dir = filepath.Clean(dir)
		if dir == root || dir == "." || dir == string(os.PathSeparator) {
			return
		}
		if rel, err := filepath.Rel(root, dir); err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			return
		}
		if err := os.Remove(dir); err != nil {
			return
		}
		dir = filepath.Dir(dir)
	}
}

func DefaultRunner(ctx context.Context, root, command string) CheckResult {
	var c *exec.Cmd
	if runtime.GOOS == "windows" {
		c = exec.CommandContext(ctx, "cmd", "/C", command)
	} else {
		c = exec.CommandContext(ctx, "sh", "-c", command)
	}
	c.Dir = root
	var buf bytes.Buffer
	c.Stdout = &buf
	c.Stderr = &buf
	err := c.Run()
	code := 0
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		code = exit.ExitCode()
		err = nil
	} else if err != nil {
		code = 127
	}
	cr := CheckResult{Command: command, ExitCode: code, Output: strings.TrimSpace(buf.String())}
	if err != nil {
		cr.Error = err.Error()
	}
	return cr
}

func refuse(res Result, reason, detail string) Result {
	res.OK = false
	res.Reason = reason
	res.Detail = strings.TrimSpace(detail)
	return res
}

func DigestTree(root string, paths []string) (map[string]string, error) {
	out := make(map[string]string, len(paths))
	for _, p := range paths {
		_, abs, err := cleanTarget(root, p)
		if err != nil {
			return nil, err
		}
		b, err := os.ReadFile(abs)
		if errors.Is(err, os.ErrNotExist) {
			out[p] = "<missing>"
			continue
		}
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(b)
		out[p] = hex.EncodeToString(sum[:])
	}
	return out, nil
}
