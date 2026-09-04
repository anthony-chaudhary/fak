package safesync

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// QuarantineReceipt records the witnessed outcome of a pre-sync quarantine
// transaction across a fast-forward sync (#10913, #11233).
type QuarantineReceipt struct {
	QuarantinedCount int               `json:"quarantined_count"`
	RestoredCount    int               `json:"restored_count"`
	IdenticalCount   int               `json:"identical_count"`
	RelocatedCount   int               `json:"relocated_count,omitempty"`
	Paths            []string          `json:"paths,omitempty"`
	Files            []QuarantinedFile `json:"files,omitempty"`
	Preserved        map[string]string `json:"preserved,omitempty"`
	SHA256           string            `json:"sha256,omitempty"`
	ReceiptPath      string            `json:"receipt_path,omitempty"`
}

// QuarantinedFile records one preserved untracked artifact and its cryptographic digest.
type QuarantinedFile struct {
	RelPath         string `json:"rel_path"`
	StashPath       string `json:"stash_path"`
	SHA256          string `json:"sha256"`
	Size            int64  `json:"size"`
	TargetIdentical bool   `json:"target_identical"`
}

// QuarantineTransaction manages the atomic pre-sync quarantine lifecycle:
// 1. Snapshot and move colliding untracked files into gc-safe temporary stash.
// 2. Allow fast-forward checkout to proceed without "untracked file would be overwritten" error.
// 3. Post-sync, verify identical files or restore/relocate preserved scratch with identical hashes.
type QuarantineTransaction struct {
	Repo     string
	StashDir string
	Files    []QuarantinedFile
}

// FileSHA256 computes the SHA256 hex digest of a local file.
func FileSHA256(path string) (string, int64, error) {
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

// PrepareQuarantine isolates declared untracked files before fast-forward sync.
func PrepareQuarantine(repo string, paths []string, targetIdentical map[string]bool, sessionOrTS ...string) (*QuarantineTransaction, error) {
	if len(paths) == 0 {
		return &QuarantineTransaction{Repo: repo}, nil
	}

	gitDir, err := worktreeGitDir(repo)
	if err != nil {
		gitDir = filepath.Join(repo, ".git")
	}

	tag := fmt.Sprintf("%d", time.Now().UnixNano())
	if len(sessionOrTS) > 0 && strings.TrimSpace(sessionOrTS[0]) != "" {
		tag = strings.TrimSpace(sessionOrTS[0])
	}
	stashDir := filepath.Join(gitDir, "fak-quarantine", tag)
	if err := os.MkdirAll(stashDir, 0o755); err != nil {
		return nil, fmt.Errorf("create quarantine stash dir: %w", err)
	}

	tx := &QuarantineTransaction{
		Repo:     repo,
		StashDir: stashDir,
	}

	for _, rel := range paths {
		fullPath := filepath.Join(repo, filepath.FromSlash(rel))
		if _, err := os.Stat(fullPath); os.IsNotExist(err) {
			continue
		}

		hash, size, err := FileSHA256(fullPath)
		if err != nil {
			_ = tx.Rollback()
			return nil, fmt.Errorf("hash untracked file %s: %w", rel, err)
		}

		stashPath := filepath.Join(stashDir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(stashPath), 0o755); err != nil {
			_ = tx.Rollback()
			return nil, fmt.Errorf("mkdir for stashed file %s: %w", rel, err)
		}

		// Move untracked file to quarantine stash
		if err := os.Rename(fullPath, stashPath); err != nil {
			// Fallback to copy + remove across filesystem boundaries
			if copyErr := copyFile(fullPath, stashPath); copyErr != nil {
				_ = tx.Rollback()
				return nil, fmt.Errorf("quarantine file %s: %w", rel, copyErr)
			}
			_ = os.Remove(fullPath)
		}

		tx.Files = append(tx.Files, QuarantinedFile{
			RelPath:         rel,
			StashPath:       stashPath,
			SHA256:          hash,
			Size:            size,
			TargetIdentical: targetIdentical[rel],
		})
	}

	return tx, nil
}

// Commit finalizes the quarantine transaction post fast-forward:
// - Verifies byte-identical files were populated by git with matching hashes.
// - Restores non-colliding scratch files.
// - Relocates genuinely colliding scratch to .git/fak-quarantine/<session_or_ts>/ and .fak/quarantine/
//   to guarantee zero data loss, and writes a quarantine receipt to disk (#11233).
func (tx *QuarantineTransaction) Commit() (QuarantineReceipt, error) {
	receipt := QuarantineReceipt{
		QuarantinedCount: len(tx.Files),
		Preserved:        make(map[string]string),
	}
	if len(tx.Files) == 0 {
		return receipt, nil
	}

	hasDivergentPreserved := false
	for _, f := range tx.Files {
		receipt.Paths = append(receipt.Paths, f.RelPath)
		receipt.Files = append(receipt.Files, f)
		destPath := filepath.Join(tx.Repo, filepath.FromSlash(f.RelPath))

		if f.TargetIdentical {
			// File should have been placed by fast-forward merge. Verify hash.
			if hash, _, err := FileSHA256(destPath); err == nil && hash == f.SHA256 {
				receipt.IdenticalCount++
				_ = os.Remove(f.StashPath)
				continue
			}
		}

		// If destination does not exist, restore original untracked file
		if _, err := os.Stat(destPath); os.IsNotExist(err) {
			if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err == nil {
				if renameErr := os.Rename(f.StashPath, destPath); renameErr == nil {
					receipt.RestoredCount++
					continue
				}
			}
		}

		// If destination exists and differs from stashed file, preserve in quarantine stash
		hasDivergentPreserved = true
		receipt.RelocatedCount++
		receipt.Preserved[f.RelPath] = f.SHA256
		if receipt.SHA256 == "" {
			receipt.SHA256 = f.SHA256
		}

		// Also preserve copy in .fak/quarantine for backward compatibility
		relocDir := filepath.Join(tx.Repo, ".fak", "quarantine")
		_ = os.MkdirAll(relocDir, 0o755)
		relocPath := filepath.Join(relocDir, filepath.Base(f.RelPath))
		_ = copyFile(f.StashPath, relocPath)
	}

	if !hasDivergentPreserved {
		if tx.StashDir != "" {
			_ = os.RemoveAll(tx.StashDir)
		}
	} else {
		receiptData, err := json.MarshalIndent(receipt, "", "  ")
		if err == nil {
			receiptFile := filepath.Join(tx.StashDir, "receipt.json")
			_ = os.WriteFile(receiptFile, receiptData, 0o644)
			_ = os.WriteFile(filepath.Join(tx.StashDir, "quarantine-receipt.json"), receiptData, 0o644)
			receipt.ReceiptPath = receiptFile

			gitDir, err := worktreeGitDir(tx.Repo)
			if err == nil {
				_ = os.WriteFile(filepath.Join(gitDir, "fak-quarantine", "receipt.json"), receiptData, 0o644)
				_ = os.WriteFile(filepath.Join(gitDir, "fak-quarantine", "latest-receipt.json"), receiptData, 0o644)
			}
		}
	}

	return receipt, nil
}

// Rollback restores all stashed files to their original working tree locations.
func (tx *QuarantineTransaction) Rollback() error {
	if tx == nil || len(tx.Files) == 0 {
		return nil
	}
	var errs []string
	for _, f := range tx.Files {
		destPath := filepath.Join(tx.Repo, filepath.FromSlash(f.RelPath))
		if _, err := os.Stat(destPath); os.IsNotExist(err) {
			_ = os.MkdirAll(filepath.Dir(destPath), 0o755)
			if err := os.Rename(f.StashPath, destPath); err != nil {
				if copyErr := copyFile(f.StashPath, destPath); copyErr != nil {
					errs = append(errs, fmt.Sprintf("restore %s: %v", f.RelPath, copyErr))
				}
			}
		}
	}
	if tx.StashDir != "" {
		_ = os.RemoveAll(tx.StashDir)
	}
	if len(errs) > 0 {
		return fmt.Errorf("quarantine rollback errors: %s", strings.Join(errs, "; "))
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err = io.Copy(out, in); err != nil {
		return err
	}
	return out.Sync()
}
