package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type guardRelaunchFiles struct{ files []guardRelaunchFile }
type guardRelaunchFile struct {
	path string
	data []byte
	mode os.FileMode
}

var guardRelaunchFileFlags = map[string]bool{"--settings": true, "--mcp-config": true}

// captureGuardRelaunchFiles keeps an in-process copy of generated child config.
// A concurrent stale-dir sweep must not turn a later budget restart into
// "Settings file not found"; ensure restores only a missing referenced file.
func captureGuardRelaunchFiles(command []string) (guardRelaunchFiles, error) {
	var out guardRelaunchFiles
	seen := make(map[string]bool)
	for i := 0; i < len(command); i++ {
		arg, path := command[i], ""
		if guardRelaunchFileFlags[arg] && i+1 < len(command) {
			i++
			path = command[i]
		} else {
			for flag := range guardRelaunchFileFlags {
				if value, ok := strings.CutPrefix(arg, flag+"="); ok {
					path = value
					break
				}
			}
		}
		path = strings.TrimSpace(path)
		if path == "" {
			continue
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return guardRelaunchFiles{}, fmt.Errorf("resolve generated child config %q: %w", path, err)
		}
		// Only fak's PID-owned hook directories are lifecycle-managed here. A
		// caller-supplied settings file remains the caller's responsibility and
		// is never retained in memory or recreated after deletion.
		if _, _, ok := guardTempDirOwner(filepath.Base(filepath.Dir(abs))); !ok {
			continue
		}
		key := strings.ToLower(filepath.Clean(abs))
		if seen[key] {
			continue
		}
		data, err := os.ReadFile(abs)
		if err != nil {
			return guardRelaunchFiles{}, fmt.Errorf("snapshot generated child config %q: %w", abs, err)
		}
		mode := os.FileMode(0o600)
		if info, statErr := os.Stat(abs); statErr == nil {
			mode = info.Mode().Perm()
		}
		out.files = append(out.files, guardRelaunchFile{path: abs, data: append([]byte(nil), data...), mode: mode})
		seen[key] = true
	}
	return out, nil
}

func (s guardRelaunchFiles) ensure() error {
	for _, file := range s.files {
		if _, err := os.Stat(file.path); err == nil {
			continue
		} else if !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("check generated child config %q: %w", file.path, err)
		}
		if err := os.MkdirAll(filepath.Dir(file.path), 0o700); err != nil {
			return fmt.Errorf("restore generated child config directory %q: %w", filepath.Dir(file.path), err)
		}
		tmp, err := os.CreateTemp(filepath.Dir(file.path), ".fak-restore-*")
		if err != nil {
			return fmt.Errorf("restore generated child config %q: %w", file.path, err)
		}
		tmpPath := tmp.Name()
		// Any failure while the temp file is still open must close AND unlink it before
		// reporting, so a half-restored config never survives to be picked up as real. stage
		// names the step that failed, which is all that differs between the two exits below.
		abortRestore := func(stage string, err error) error {
			_ = tmp.Close()
			_ = os.Remove(tmpPath)
			return fmt.Errorf("restore generated child config %s %q: %w", stage, file.path, err)
		}
		if err := tmp.Chmod(file.mode); err != nil {
			return abortRestore("mode", err)
		}
		if _, err := tmp.Write(file.data); err != nil {
			return abortRestore("bytes", err)
		}
		if err := tmp.Close(); err != nil {
			_ = os.Remove(tmpPath)
			return fmt.Errorf("restore generated child config close %q: %w", file.path, err)
		}
		if err := os.Rename(tmpPath, file.path); err != nil {
			_ = os.Remove(tmpPath)
			return fmt.Errorf("restore generated child config publish %q: %w", file.path, err)
		}
	}
	return nil
}
