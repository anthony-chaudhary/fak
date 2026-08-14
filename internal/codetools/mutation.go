package codetools

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

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
}

func (t *Toolset) write(ctx context.Context, body []byte) ([]byte, bool) {
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
	target, r := t.resolve(a.FilePath)
	if r != nil {
		return r.JSON(), true
	}
	info, statErr := os.Lstat(target.Abs)
	exists := statErr == nil
	if statErr != nil && !errors.Is(statErr, fs.ErrNotExist) {
		return refuse(CodeIO, statErr.Error()).JSON(), true
	}
	if exists && info.IsDir() {
		return refuse(CodeIsDir, "Write target is a directory").JSON(), true
	}
	if a.Mode == "create" && exists {
		return refuse(CodeExists, "Write create target already exists").JSON(), true
	}
	if a.Mode == "overwrite" && !exists {
		return refuse(CodeNotFound, "Write overwrite target does not exist").JSON(), true
	}
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	if err := atomicReplace(target.Abs, []byte(a.Content), exists, 0); err != nil {
		return refuse(CodeIO, err.Error()).JSON(), true
	}
	return okJSON(mutationResult{Path: target.Rel, Bytes: len(a.Content)}), false
}

func (t *Toolset) edit(ctx context.Context, body []byte) ([]byte, bool) {
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
	target, r := t.resolve(a.FilePath)
	if r != nil {
		return r.JSON(), true
	}
	info, err := os.Lstat(target.Abs)
	if errors.Is(err, fs.ErrNotExist) {
		return refuse(CodeNotFound, "Edit target does not exist").JSON(), true
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
	b, err := os.ReadFile(target.Abs)
	if err != nil {
		return refuse(CodeIO, err.Error()).JSON(), true
	}
	if int64(len(b)) > t.limits.MaxWriteBytes {
		return refuse(CodeTooLarge, "Edit target exceeds byte bound").JSON(), true
	}
	n := strings.Count(string(b), a.OldString)
	if n == 0 {
		return refuse(CodeEditConflict, "Edit old_string matched 0 occurrences").JSON(), true
	}
	if !a.ReplaceAll && n != 1 {
		return refuse(CodeEditConflict, fmt.Sprintf("Edit old_string matched %d occurrences; want exactly 1", n)).JSON(), true
	}
	limit := 1
	if a.ReplaceAll {
		limit = -1
	}
	next := strings.Replace(string(b), a.OldString, a.NewString, limit)
	if int64(len(next)) > t.limits.MaxWriteBytes {
		return refuse(CodeTooLarge, "Edit result exceeds byte bound").JSON(), true
	}
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	if err := atomicReplace(target.Abs, []byte(next), true, info.Mode().Perm()); err != nil {
		return refuse(CodeIO, err.Error()).JSON(), true
	}
	return okJSON(mutationResult{Path: target.Rel, Bytes: len(next), Replacements: func() int {
		if a.ReplaceAll {
			return n
		}
		return 1
	}()}), false
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
	if err = os.Rename(tmp, path); err != nil {
		return err
	}
	ok = true
	return nil
}
