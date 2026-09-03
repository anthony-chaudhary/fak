package dataslot

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

var (
	ErrSnapshotNotFound = errors.New("dataslot: snapshot not found")
	ErrRollbackFailed   = errors.New("dataslot: snapshot rollback failed")
	ErrCorruptBackup    = errors.New("dataslot: snapshot backup file is corrupted or missing")
)

// FileItem represents a backed-up database or journal file.
type FileItem struct {
	OriginalPath string `json:"original_path"`
	BackupPath   string `json:"backup_path"`
	SHA256       string `json:"sha256"`
	SizeBytes    int64  `json:"size_bytes"`
	Existed      bool   `json:"existed"`
}

// FileDBSnapshot records the pre-turn file states for a database.
type FileDBSnapshot struct {
	ID        string     `json:"id"`
	DBPath    string     `json:"db_path"`
	ScopeID   string     `json:"scope_id"`
	TurnIndex int        `json:"turn_index"`
	CreatedAt time.Time  `json:"created_at"`
	Files     []FileItem `json:"files"`
}

// FileDBSnapshotManager manages pre-turn database snapshots and rollbacks.
type FileDBSnapshotManager struct {
	BaseDir string
}

// NewSnapshotManager creates a snapshot manager rooted under baseDir.
func NewSnapshotManager(baseDir string) *FileDBSnapshotManager {
	if baseDir == "" {
		baseDir = filepath.Join(".fak", "scratch", "dbsnapshots")
	}
	return &FileDBSnapshotManager{BaseDir: baseDir}
}

// Snapshot captures an atomic snapshot of a database and its WAL/SHM companion files.
func (m *FileDBSnapshotManager) Snapshot(ctx context.Context, dbPath, scopeID string, turnIndex int) (*FileDBSnapshot, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	cleanDB := filepath.Clean(dbPath)
	if _, err := os.Stat(cleanDB); err != nil {
		return nil, fmt.Errorf("dataslot: target database %q: %w", cleanDB, err)
	}

	snapID := fmt.Sprintf("%s-turn%d-%d", sanitizeScope(scopeID), turnIndex, time.Now().UnixNano())
	snapDir := filepath.Join(m.BaseDir, sanitizeScope(scopeID), snapID)
	if err := os.MkdirAll(snapDir, 0755); err != nil {
		return nil, fmt.Errorf("dataslot: failed to create snapshot directory: %w", err)
	}

	targets := []string{
		cleanDB,
		cleanDB + "-wal",
		cleanDB + "-shm",
	}

	var items []FileItem
	for _, target := range targets {
		info, err := os.Stat(target)
		if err != nil {
			// Sibling does not exist prior to turn
			items = append(items, FileItem{
				OriginalPath: target,
				Existed:      false,
			})
			continue
		}

		if info.IsDir() {
			continue
		}

		backupFile := filepath.Join(snapDir, filepath.Base(target))
		hash, size, err := copyWithChecksum(target, backupFile)
		if err != nil {
			_ = os.RemoveAll(snapDir)
			return nil, fmt.Errorf("dataslot: failed to backup %q: %w", target, err)
		}

		items = append(items, FileItem{
			OriginalPath: target,
			BackupPath:   backupFile,
			SHA256:       hash,
			SizeBytes:    size,
			Existed:      true,
		})
	}

	return &FileDBSnapshot{
		ID:        snapID,
		DBPath:    cleanDB,
		ScopeID:   scopeID,
		TurnIndex: turnIndex,
		CreatedAt: time.Now().UTC(),
		Files:     items,
	}, nil
}

// Rollback restores target files to their exact pre-snapshot state.
func (m *FileDBSnapshotManager) Rollback(ctx context.Context, snap *FileDBSnapshot) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if snap == nil || len(snap.Files) == 0 {
		return ErrSnapshotNotFound
	}

	for _, item := range snap.Files {
		if !item.Existed {
			// Sibling did not exist pre-turn: remove if created during turn
			if _, err := os.Stat(item.OriginalPath); err == nil {
				_ = os.Remove(item.OriginalPath)
			}
			continue
		}

		// Verify backup file exists and matches recorded checksum
		if _, err := os.Stat(item.BackupPath); err != nil {
			return fmt.Errorf("%w: backup file missing %q", ErrCorruptBackup, item.BackupPath)
		}

		backupHash, _, err := computeFileHash(item.BackupPath)
		if err != nil || backupHash != item.SHA256 {
			return fmt.Errorf("%w: checksum mismatch on backup %q", ErrCorruptBackup, item.BackupPath)
		}

		// Restore atomically
		if err := restoreFile(item.BackupPath, item.OriginalPath); err != nil {
			return fmt.Errorf("%w: failed restoring %q from %q: %v", ErrRollbackFailed, item.OriginalPath, item.BackupPath, err)
		}

		// Verify restored file checksum
		restoredHash, _, err := computeFileHash(item.OriginalPath)
		if err != nil || restoredHash != item.SHA256 {
			return fmt.Errorf("%w: restored file %q checksum mismatch", ErrRollbackFailed, item.OriginalPath)
		}
	}

	return nil
}

// Reap removes stored snapshots for the specified scope ID.
func (m *FileDBSnapshotManager) Reap(ctx context.Context, scopeID string) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	scopeDir := filepath.Join(m.BaseDir, sanitizeScope(scopeID))
	return os.RemoveAll(scopeDir)
}

func sanitizeScope(s string) string {
	if s == "" {
		return "default"
	}
	var b []byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' {
			b = append(b, c)
		} else {
			b = append(b, '_')
		}
	}
	return string(b)
}

func copyWithChecksum(src, dst string) (string, int64, error) {
	in, err := os.Open(src)
	if err != nil {
		return "", 0, err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return "", 0, err
	}
	defer out.Close()

	h := sha256.New()
	mw := io.MultiWriter(out, h)

	n, err := io.Copy(mw, in)
	if err != nil {
		return "", 0, err
	}

	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func computeFileHash(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()

	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func restoreFile(src, dst string) error {
	// Attempt safe rename via temporary file in the target directory
	dstDir := filepath.Dir(dst)
	tmpFile, err := os.CreateTemp(dstDir, ".dbrestore-*")
	if err != nil {
		// Fallback: direct copy
		return directCopy(src, dst)
	}
	tmpName := tmpFile.Name()
	defer os.Remove(tmpName)

	in, err := os.Open(src)
	if err != nil {
		tmpFile.Close()
		return err
	}
	defer in.Close()

	if _, err := io.Copy(tmpFile, in); err != nil {
		tmpFile.Close()
		return err
	}
	if err := tmpFile.Sync(); err != nil {
		tmpFile.Close()
		return err
	}
	tmpFile.Close()

	// Atomic replace: remove existing and rename
	_ = os.Remove(dst)
	if err := os.Rename(tmpName, dst); err == nil {
		return nil
	}

	// Secondary fallback: direct copy overwrite
	return directCopy(src, dst)
}

func directCopy(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Sync()
}
