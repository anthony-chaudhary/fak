package sessionmine

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

var (
	ErrDowngradeRefused = errors.New("sessionmine: index schema downgrade is refused to prevent data loss")
	ErrAlreadyCurrent   = errors.New("sessionmine: index is already on the current schema version")
	ErrInvalidBackup    = errors.New("sessionmine: backup file is invalid or missing")
)

// IndexMigrationResult records the verifiable facts of a schema migration.
type IndexMigrationResult struct {
	Path          string `json:"path"`
	FromSchema    string `json:"from_schema"`
	ToSchema      string `json:"to_schema"`
	BackupPath    string `json:"backup_path,omitempty"`
	ItemsMigrated int    `json:"items_migrated"`
	DryRun        bool   `json:"dry_run"`
	AppliedAt     string `json:"applied_at"`
	SHA256Pre     string `json:"sha256_pre"`
	SHA256Post    string `json:"sha256_post,omitempty"`
}

// MigrateIndex upgrades an on-disk index file to the current schema.
// When dryRun is true, it verifies compatibility and returns the plan without mutating the file.
func MigrateIndex(path string, dryRun bool) (*IndexMigrationResult, error) {
	cleanPath := filepath.Clean(path)
	data, err := os.ReadFile(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("sessionmine: read index for migration: %w", err)
	}

	preHash := sha256Hex(data)

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("sessionmine: parse index for migration: %w", err)
	}

	foundSchema, _ := raw["schema"].(string)
	if foundSchema == "" {
		return nil, errors.New("sessionmine: index missing schema field")
	}

	if foundSchema == SchemaV2 {
		return nil, ErrAlreadyCurrent
	}

	if foundSchema != SchemaV1 {
		return nil, fmt.Errorf("sessionmine: unsupported migration source schema %q", foundSchema)
	}

	// Migrate V1 -> V2
	var v1State IndexState
	if err := json.Unmarshal(data, &v1State); err != nil {
		return nil, fmt.Errorf("sessionmine: decode v1 index: %w", err)
	}

	v2State := v1State
	v2State.Schema = SchemaV2
	if v2State.Lineage == nil {
		v2State.Lineage = make(map[string]string)
	}
	if v2State.OutcomeStats == nil {
		v2State.OutcomeStats = make(map[string]int)
	}

	// Tally initial outcome stats from existing sessions
	for _, f := range v2State.Files {
		if f.Session.ToolErrors > 0 {
			v2State.OutcomeStats["errors_observed"]++
		} else {
			v2State.OutcomeStats["clean_runs"]++
		}
	}

	res := &IndexMigrationResult{
		Path:          cleanPath,
		FromSchema:    foundSchema,
		ToSchema:      SchemaV2,
		ItemsMigrated: len(v2State.Files),
		DryRun:        dryRun,
		AppliedAt:     time.Now().UTC().Format(time.RFC3339),
		SHA256Pre:     preHash,
	}

	if dryRun {
		return res, nil
	}

	// 1. Automatic pre-migration backup
	backupPath := cleanPath + ".backup-" + time.Now().UTC().Format("20060102150405")
	if err := copyFileExact(cleanPath, backupPath); err != nil {
		return nil, fmt.Errorf("sessionmine: pre-migration backup failed: %w", err)
	}
	res.BackupPath = backupPath

	// 2. Atomic publish
	if err := writeIndexAtomic(cleanPath, v2State); err != nil {
		return nil, fmt.Errorf("sessionmine: write migrated index: %w", err)
	}

	// 3. Post-hash
	postData, err := os.ReadFile(cleanPath)
	if err == nil {
		res.SHA256Post = sha256Hex(postData)
	}

	return res, nil
}

// RollbackIndexMigration restores an index file from its pre-migration backup.
func RollbackIndexMigration(targetPath, backupPath string) error {
	cleanTarget := filepath.Clean(targetPath)
	cleanBackup := filepath.Clean(backupPath)

	backupData, err := os.ReadFile(cleanBackup)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidBackup, err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(backupData, &parsed); err != nil {
		return fmt.Errorf("%w: corrupt backup data: %v", ErrInvalidBackup, err)
	}

	// Write backup data atomically over target
	tmpName := cleanTarget + ".rollback-tmp"
	if err := os.WriteFile(tmpName, backupData, 0600); err != nil {
		return err
	}
	defer os.Remove(tmpName)

	_ = os.Remove(cleanTarget)
	return os.Rename(tmpName, cleanTarget)
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func copyFileExact(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Sync()
}
