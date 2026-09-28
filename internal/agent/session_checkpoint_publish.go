// MIT License (upstream publish routine)
// Copyright (c) 2026 Unreal Labs
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package agent

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
)

// Crash-safe checkpoint publish, adapted from unreallabsai/unreal-agent@1b9f778453f4
// harness/sessionstore/localfile/store.go (publishFile, persistTemporary,
// syncDirectory; MIT, copyright 2026 Unreal Labs). Adaptations: the temp file lives in
// the target's own directory under a hidden name that never ends in ".json" (so
// LoadSessionCheckpoint cannot resolve it), the write and rename go through test seams,
// and the directory fsync is skipped on Windows, where a directory cannot be opened for
// FlushFileBuffers and the file's own flush is the portable boundary (the
// internal/serviceledger idiom).

// checkpointWriteTemp and checkpointRename are fault-injection seams for the
// crash-safety witness (session_checkpoint_test.go). Production never reassigns them.
var (
	checkpointWriteTemp = func(f *os.File, data []byte) error {
		_, err := f.Write(data)
		return err
	}
	checkpointRename = os.Rename
)

// publishSessionCheckpoint replaces target with data so that, at every instant, target
// holds either the previous complete checkpoint or the new one, never a torn mix. The
// bytes go to a temp file in target's directory (rename is only atomic within one
// filesystem), are fsynced, renamed over target, and the directory is fsynced so the
// rename itself survives power loss. Any failure before the rename removes the temp file
// and leaves target untouched.
func publishSessionCheckpoint(target string, data []byte) (err error) {
	dir := filepath.Dir(target)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(target)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temporary file: %w", err)
	}
	tmpPath := tmp.Name()
	defer func() {
		if tmpPath == "" {
			return
		}
		if rmErr := os.Remove(tmpPath); rmErr != nil && !errors.Is(rmErr, fs.ErrNotExist) {
			err = errors.Join(err, fmt.Errorf("remove temporary file: %w", rmErr))
		}
	}()

	if err := persistCheckpointTemp(tmp, data); err != nil {
		return err
	}
	if err := checkpointRename(tmpPath, target); err != nil {
		return fmt.Errorf("publish: %w", err)
	}
	tmpPath = ""
	return syncCheckpointDir(dir)
}

// persistCheckpointTemp writes data to the temp file and fsyncs it before the rename can
// make it visible under the target name.
func persistCheckpointTemp(f *os.File, data []byte) (err error) {
	defer func() {
		if closeErr := f.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close temporary file: %w", closeErr))
		}
	}()
	if err := checkpointWriteTemp(f, data); err != nil {
		return fmt.Errorf("write temporary file: %w", err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("sync temporary file: %w", err)
	}
	return nil
}

// syncCheckpointDir fsyncs dir so a completed rename is durable, not just visible.
func syncCheckpointDir(dir string) (err error) {
	if runtime.GOOS == "windows" {
		return nil
	}
	d, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("open directory for sync: %w", err)
	}
	defer func() {
		if closeErr := d.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close directory: %w", closeErr))
		}
	}()
	if err := d.Sync(); err != nil {
		return fmt.Errorf("sync directory: %w", err)
	}
	return nil
}
