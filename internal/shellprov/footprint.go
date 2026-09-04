// Package shellprov provides execution footprint recording and provenance tracking.
package shellprov

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// ExecutionFootprint captures post-execution corroboration of file mutations
// observed across a monitored directory tree.
type ExecutionFootprint struct {
	SessionID      string    `json:"session_id"`
	AddedFiles     []string  `json:"added_files"`
	ModifiedFiles  []string  `json:"modified_files"`
	DeletedFiles   []string  `json:"deleted_files"`
	TotalMutations int       `json:"total_mutations"`
	BytesModified  int64     `json:"bytes_modified"`
	Timestamp      time.Time `json:"timestamp"`
}

// FileState stores bounded file metadata and content hash captured at snapshot time.
type FileState struct {
	Path    string    `json:"path"`
	Size    int64     `json:"size"`
	ModTime time.Time `json:"mod_time"`
	Hash    string    `json:"hash"`
}

// FilesystemSnapshot records directory file states at a point in time.
type FilesystemSnapshot struct {
	Dir       string               `json:"dir"`
	Timestamp time.Time            `json:"timestamp"`
	Files     map[string]FileState `json:"files"`
}

// FootprintRecorder captures and diffs filesystem snapshots across process executions.
type FootprintRecorder struct {
	now func() time.Time
}

// NewFootprintRecorder constructs an initialized FootprintRecorder.
func NewFootprintRecorder() *FootprintRecorder {
	return &FootprintRecorder{
		now: time.Now,
	}
}

var defaultRecorder = NewFootprintRecorder()

// Snapshot walks dir and captures the current file state map.
func (r *FootprintRecorder) Snapshot(dir string) (FilesystemSnapshot, error) {
	if dir == "" {
		return FilesystemSnapshot{}, errors.New("shellprov: directory is required")
	}
	cleanDir := filepath.Clean(dir)
	info, err := os.Stat(cleanDir)
	if err != nil {
		return FilesystemSnapshot{}, fmt.Errorf("shellprov: stat snapshot directory: %w", err)
	}
	if !info.IsDir() {
		return FilesystemSnapshot{}, fmt.Errorf("shellprov: snapshot target %s is not a directory", cleanDir)
	}

	now := time.Now
	if r != nil && r.now != nil {
		now = r.now
	}

	snap := FilesystemSnapshot{
		Dir:       cleanDir,
		Timestamp: now().UTC(),
		Files:     make(map[string]FileState),
	}

	walkErr := filepath.WalkDir(cleanDir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}

		rel, err := filepath.Rel(cleanDir, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)

		fileInfo, err := d.Info()
		if err != nil {
			return err
		}
		if !fileInfo.Mode().IsRegular() {
			statInfo, statErr := os.Stat(path)
			if statErr != nil || statInfo.IsDir() || !statInfo.Mode().IsRegular() {
				return nil
			}
			fileInfo = statInfo
		}

		hashStr, size, err := hashFile(path)
		if err != nil {
			return err
		}

		snap.Files[rel] = FileState{
			Path:    rel,
			Size:    size,
			ModTime: fileInfo.ModTime().UTC(),
			Hash:    hashStr,
		}
		return nil
	})
	if walkErr != nil {
		return FilesystemSnapshot{}, fmt.Errorf("shellprov: walk directory %s: %w", cleanDir, walkErr)
	}

	return snap, nil
}

// Diff computes the execution footprint diff between two snapshots.
func (r *FootprintRecorder) Diff(before, after FilesystemSnapshot) ExecutionFootprint {
	added := make([]string, 0)
	modified := make([]string, 0)
	deleted := make([]string, 0)
	var bytesModified int64

	// Detect added and modified files.
	for path, afterFile := range after.Files {
		beforeFile, exists := before.Files[path]
		if !exists {
			added = append(added, path)
		} else if beforeFile.Hash != afterFile.Hash || beforeFile.Size != afterFile.Size {
			modified = append(modified, path)
			bytesModified += afterFile.Size
		}
	}

	// Detect deleted files.
	for path := range before.Files {
		if _, exists := after.Files[path]; !exists {
			deleted = append(deleted, path)
		}
	}

	sort.Strings(added)
	sort.Strings(modified)
	sort.Strings(deleted)

	now := time.Now
	if r != nil && r.now != nil {
		now = r.now
	}

	ts := after.Timestamp
	if ts.IsZero() {
		ts = now().UTC()
	}

	return ExecutionFootprint{
		AddedFiles:     added,
		ModifiedFiles:  modified,
		DeletedFiles:   deleted,
		TotalMutations: len(added) + len(modified) + len(deleted),
		BytesModified:  bytesModified,
		Timestamp:      ts,
	}
}

// RecordExecution snapshots dir before and after fn execution, computes the diff,
// and stamps SessionID and Timestamp. If fn returns an error, the footprint is still
// returned alongside the error so callers can inspect partial mutations.
func (r *FootprintRecorder) RecordExecution(dir string, sessionID string, fn func() error) (*ExecutionFootprint, error) {
	if dir == "" {
		return nil, errors.New("shellprov: directory is required")
	}

	before, err := r.Snapshot(dir)
	if err != nil {
		return nil, fmt.Errorf("shellprov: snapshot before execution: %w", err)
	}

	var fnErr error
	if fn != nil {
		fnErr = fn()
	}

	after, err := r.Snapshot(dir)
	if err != nil {
		if fnErr != nil {
			return nil, errors.Join(fnErr, fmt.Errorf("shellprov: snapshot after execution: %w", err))
		}
		return nil, fmt.Errorf("shellprov: snapshot after execution: %w", err)
	}

	fp := r.Diff(before, after)
	fp.SessionID = sessionID

	now := time.Now
	if r != nil && r.now != nil {
		now = r.now
	}
	fp.Timestamp = now().UTC()

	return &fp, fnErr
}

// WriteJSONL serializes fp as a single JSON line followed by a newline into w.
func (r *FootprintRecorder) WriteJSONL(w io.Writer, fp ExecutionFootprint) error {
	if w == nil {
		return errors.New("shellprov: writer is required")
	}
	if fp.AddedFiles == nil {
		fp.AddedFiles = []string{}
	}
	if fp.ModifiedFiles == nil {
		fp.ModifiedFiles = []string{}
	}
	if fp.DeletedFiles == nil {
		fp.DeletedFiles = []string{}
	}
	encoded, err := json.Marshal(fp)
	if err != nil {
		return fmt.Errorf("shellprov: marshal execution footprint: %w", err)
	}
	if _, err := w.Write(append(encoded, '\n')); err != nil {
		return fmt.Errorf("shellprov: write execution footprint line: %w", err)
	}
	return nil
}

// Snapshot takes a filesystem snapshot using the default FootprintRecorder.
func Snapshot(dir string) (FilesystemSnapshot, error) {
	return defaultRecorder.Snapshot(dir)
}

// Diff computes an execution footprint diff using the default FootprintRecorder.
func Diff(before, after FilesystemSnapshot) ExecutionFootprint {
	return defaultRecorder.Diff(before, after)
}

// RecordExecution executes fn and captures its footprint using the default FootprintRecorder.
func RecordExecution(dir string, sessionID string, fn func() error) (*ExecutionFootprint, error) {
	return defaultRecorder.RecordExecution(dir, sessionID, fn)
}

// WriteJSONL writes fp as a JSONL row using the default FootprintRecorder.
func WriteJSONL(w io.Writer, fp ExecutionFootprint) error {
	return defaultRecorder.WriteJSONL(w, fp)
}

func hashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()

	hasher := sha256.New()
	size, err := io.Copy(hasher, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(hasher.Sum(nil)), size, nil
}
