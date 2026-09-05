package codetools

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// mutation.go owns the side-effecting filesystem engines. They do not invoke a shell.
type writeEngine struct{ t *Toolset }

func (writeEngine) Caps() []abi.Capability { return nil }
func (writeEngine) WeightBearing() bool    { return false }
func (e writeEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, bad := e.t.write(ctx, in)
	return result(ctx, c, in, out, bad, EngineWrite), nil
}

type editEngine struct{ t *Toolset }

func (editEngine) Caps() []abi.Capability { return nil }
func (editEngine) WeightBearing() bool    { return false }
func (e editEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, bad := e.t.edit(ctx, in)
	return result(ctx, c, in, out, bad, EngineEdit), nil
}

type mutationResult struct {
	Path         string `json:"path"`
	Bytes        int    `json:"bytes"`
	Replacements int    `json:"replacements,omitempty"`
	Version      string `json:"version"`
}

func (t *Toolset) write(ctx context.Context, body []byte) ([]byte, bool) {
	RecordSubprocessAvoided()
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	var a WriteArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if int64(len(a.Content)) > t.limits.MaxWriteBytes {
		return refuse(CodeTooLarge, "Write content exceeds byte bound").JSON(), true
	}
	initial, r := t.resolveMutation(a.FilePath)
	if r != nil {
		return r.JSON(), true
	}
	return t.withMutationLock(initial.Key, func() ([]byte, bool) {
		target, r := t.resolveMutation(a.FilePath)
		if r != nil {
			return r.JSON(), true
		}
		if target.Key != initial.Key {
			return staleVersion("Write target identity changed before mutation")
		}
		return t.writeLocked(ctx, a, target)
	})
}

func (t *Toolset) writeLocked(ctx context.Context, a WriteArgs, target mutationTarget) ([]byte, bool) {
	info, statErr := os.Lstat(target.Abs)
	exists := statErr == nil
	if statErr != nil && !errors.Is(statErr, fs.ErrNotExist) {
		return refuse(CodeIO, statErr.Error()).JSON(), true
	}
	if exists && info.IsDir() {
		return refuse(CodeIsDir, "Write target is a directory").JSON(), true
	}
	if exists && info.Mode()&os.ModeSymlink != 0 {
		return refuse(CodeSymlinkEscape, "Write refuses a symlink target").JSON(), true
	}
	if a.Mode == "create" && exists {
		return refuse(CodeExists, "Write create target already exists").JSON(), true
	}
	if a.Mode == "overwrite" && !exists {
		return staleVersion("Write target no longer exists")
	}
	if a.Mode == "upsert" && exists && a.ExpectedVersion == "" {
		return staleVersion("Write upsert must present expected_version for an existing target")
	}
	if a.Mode == "upsert" && !exists && a.ExpectedVersion != "" {
		return staleVersion("Write target no longer exists")
	}

	var observed fileObservation
	if exists {
		var r *Refusal
		observed, r = observeFile(ctx, target.Abs, 0)
		if r != nil {
			return r.JSON(), true
		}
		if observed.Version != a.ExpectedVersion {
			return staleVersion("Write target changed since it was read")
		}
	}
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	fresh, r := t.resolveMutation(a.FilePath)
	if r != nil {
		return r.JSON(), true
	}
	if fresh.Key != target.Key {
		return staleVersion("Write target identity changed before publication")
	}
	if exists {
		current, r := observeFile(ctx, fresh.Abs, 0)
		if r != nil {
			return r.JSON(), true
		}
		if current.Version != observed.Version {
			return staleVersion("Write target changed before publication")
		}
	} else if _, err := os.Lstat(fresh.Abs); err == nil {
		if a.Mode == "create" {
			return refuse(CodeExists, "Write create target already exists").JSON(), true
		}
		return staleVersion("Write target appeared before publication")
	} else if !errors.Is(err, fs.ErrNotExist) {
		return refuse(CodeIO, err.Error()).JSON(), true
	}
	if err := atomicReplace(fresh.Abs, []byte(a.Content), exists, 0); err != nil {
		if !exists && errors.Is(err, fs.ErrExist) {
			if a.Mode == "create" {
				return refuse(CodeExists, "Write create target already exists").JSON(), true
			}
			return staleVersion("Write target appeared before publication")
		}
		return refuse(CodeIO, err.Error()).JSON(), true
	}
	after, r := observeFile(context.WithoutCancel(ctx), fresh.Abs, 0)
	if r != nil {
		return r.JSON(), true
	}
	return okJSON(mutationResult{Path: fresh.Rel, Bytes: len(a.Content), Version: after.Version}), false
}

func (t *Toolset) edit(ctx context.Context, body []byte) ([]byte, bool) {
	RecordSubprocessAvoided()
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	var a EditArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	initial, r := t.resolveMutation(a.FilePath)
	if r != nil {
		return r.JSON(), true
	}
	return t.withMutationLock(initial.Key, func() ([]byte, bool) {
		target, r := t.resolveMutation(a.FilePath)
		if r != nil {
			return r.JSON(), true
		}
		if target.Key != initial.Key {
			return staleVersion("Edit target identity changed before mutation")
		}
		return t.editLocked(ctx, a, target)
	})
}

func (t *Toolset) editLocked(ctx context.Context, a EditArgs, target mutationTarget) ([]byte, bool) {
	info, err := os.Lstat(target.Abs)
	if errors.Is(err, fs.ErrNotExist) {
		return staleVersion("Edit target no longer exists")
	}
	if err != nil {
		return refuse(CodeIO, err.Error()).JSON(), true
	}
	if info.IsDir() {
		return refuse(CodeIsDir, "Edit target is a directory").JSON(), true
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return refuse(CodeSymlinkEscape, "Edit refuses a symlink target").JSON(), true
	}
	observed, r := observeFile(ctx, target.Abs, t.limits.MaxWriteBytes)
	if r != nil {
		return r.JSON(), true
	}
	if observed.Truncated {
		return refuse(CodeTooLarge, "Edit target exceeds byte bound").JSON(), true
	}
	if observed.Version != a.ExpectedVersion {
		return staleVersion("Edit target changed since it was read")
	}
	b := observed.Content
	oldBytes := []byte(a.OldString)
	newBytes := []byte(a.NewString)
	n := bytes.Count(b, oldBytes)
	if n == 0 {
		return refuse(CodeEditConflict, "Edit old_string matched 0 occurrences; file not changed. Read the same authorized file_path with bounded offset and limit around the intended edit; use the returned version as expected_version and current exact text with unique surrounding context for one explicit Edit retry. If unresolved, stop; do not guess or retry automatically.").JSON(), true
	}
	if !a.ReplaceAll && n != 1 {
		return refuse(CodeEditConflict, fmt.Sprintf("Edit old_string matched %d occurrences; want exactly 1; file not changed. Read the same authorized file_path with bounded offset and limit around the intended edit; use the returned version as expected_version and extend old_string with exact unique surrounding context for one explicit Edit retry. If unresolved, stop; do not guess or retry automatically.", n)).JSON(), true
	}
	limit := 1
	repCount := 1
	if a.ReplaceAll {
		limit = -1
		repCount = n
	}
	newLen := len(b) + repCount*(len(newBytes)-len(oldBytes))
	if int64(newLen) > t.limits.MaxWriteBytes {
		return refuse(CodeTooLarge, "Edit result exceeds byte bound").JSON(), true
	}
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	fresh, r := t.resolveMutation(a.FilePath)
	if r != nil {
		return r.JSON(), true
	}
	if fresh.Key != target.Key {
		return staleVersion("Edit target identity changed before publication")
	}
	current, r := observeFile(ctx, fresh.Abs, 0)
	if r != nil {
		return r.JSON(), true
	}
	if current.Version != observed.Version {
		return staleVersion("Edit target changed before publication")
	}

	buf := AcquireBuffer(newLen)
	defer ReleaseBuffer(buf)

	w := 0
	rem := b
	for i := 0; limit < 0 || i < limit; i++ {
		idx := bytes.Index(rem, oldBytes)
		if idx < 0 {
			break
		}
		copy(buf[w:], rem[:idx])
		w += idx
		copy(buf[w:], newBytes)
		w += len(newBytes)
		rem = rem[idx+len(oldBytes):]
	}
	copy(buf[w:], rem)
	w += len(rem)
	next := buf[:w]

	if err := atomicReplace(fresh.Abs, next, true, info.Mode().Perm()); err != nil {
		return refuse(CodeIO, err.Error()).JSON(), true
	}
	after, r := observeFile(context.WithoutCancel(ctx), fresh.Abs, 0)
	if r != nil {
		return r.JSON(), true
	}
	return okJSON(mutationResult{Path: fresh.Rel, Bytes: len(next), Replacements: repCount, Version: after.Version}), false
}

func staleVersion(detail string) ([]byte, bool) {
	return refuse(CodeStaleVersion, detail).JSON(), true
}

func atomicReplace(path string, body []byte, existed bool, perm fs.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".fak-write-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	ok := false
	defer func() {
		if !ok {
			_ = f.Close()
		}
	}()
	if existed && perm != 0 {
		_ = f.Chmod(perm)
	}
	if _, err = f.Write(body); err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if existed {
		if err = os.Rename(tmp, path); err != nil {
			return err
		}
	} else {
		// Linking an already-synced temp file publishes a create atomically without the
		// overwrite-on-Rename race that would violate create-if-absent semantics.
		if err = os.Link(tmp, path); err != nil {
			return err
		}
		_ = os.Remove(tmp)
	}
	ok = true
	return nil
}
