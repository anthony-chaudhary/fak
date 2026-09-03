package sessionmine

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func createGoldenV1Index(t *testing.T, path string) []byte {
	t.Helper()
	v1Data := []byte(`{
  "schema": "fak-sessionmine-index/1",
  "files": {
    "fp-001": {
      "provider": "codex",
      "size": 1024,
      "mod_unix_nano": 1725350000000000000,
      "session": {
        "id": "sess-1",
        "provider": "codex",
        "user_turns": 5,
        "assistant_turns": 5,
        "tool_calls": 8,
        "failed_turns": 0
      }
    },
    "fp-002": {
      "provider": "claude",
      "size": 2048,
      "mod_unix_nano": 1725350001000000000,
      "session": {
        "id": "sess-2",
        "provider": "claude",
        "user_turns": 3,
        "assistant_turns": 3,
        "tool_calls": 4,
        "tool_errors": 1
      }
    }
  },
  "seen_candidates": {
    "cand-001": true
  },
  "updated_at": "2026-09-03T10:00:00Z"
}`)
	if err := os.WriteFile(path, v1Data, 0600); err != nil {
		t.Fatal(err)
	}
	return v1Data
}

func TestMigrateV1ToV2Golden(t *testing.T) {
	dir := t.TempDir()
	indexPath := filepath.Join(dir, "index.json")
	originalV1 := createGoldenV1Index(t, indexPath)

	// 1. Dry run
	plan, err := MigrateIndex(indexPath, true)
	if err != nil {
		t.Fatalf("dry run failed: %v", err)
	}
	if !plan.DryRun {
		t.Errorf("expected DryRun=true")
	}
	if plan.FromSchema != SchemaV1 || plan.ToSchema != SchemaV2 {
		t.Errorf("unexpected schema transition: %s -> %s", plan.FromSchema, plan.ToSchema)
	}
	if plan.ItemsMigrated != 2 {
		t.Errorf("expected 2 items migrated, got %d", plan.ItemsMigrated)
	}

	// Verify file was untouched by dry run
	currentBytes, _ := os.ReadFile(indexPath)
	if !bytes.Equal(currentBytes, originalV1) {
		t.Fatalf("dry run mutated index file!")
	}

	// 2. Real migration
	receipt, err := MigrateIndex(indexPath, false)
	if err != nil {
		t.Fatalf("real migration failed: %v", err)
	}
	if receipt.DryRun {
		t.Errorf("expected DryRun=false")
	}
	if receipt.BackupPath == "" {
		t.Errorf("expected non-empty backup path")
	}

	// Verify backup is byte-identical to original V1
	backupBytes, err := os.ReadFile(receipt.BackupPath)
	if err != nil {
		t.Fatalf("failed reading backup: %v", err)
	}
	if !bytes.Equal(backupBytes, originalV1) {
		t.Errorf("backup bytes mismatch with original V1")
	}

	// 3. Load migrated V2 index
	migrated, err := LoadIndex(indexPath)
	if err != nil {
		t.Fatalf("LoadIndex failed on migrated V2: %v", err)
	}
	if migrated.Schema != SchemaV2 {
		t.Errorf("expected schema %s, got %s", SchemaV2, migrated.Schema)
	}
	if len(migrated.Files) != 2 {
		t.Errorf("expected 2 files, got %d", len(migrated.Files))
	}
	if migrated.OutcomeStats["clean_runs"] != 1 || migrated.OutcomeStats["errors_observed"] != 1 {
		t.Errorf("unexpected outcome stats: %+v", migrated.OutcomeStats)
	}

	// 4. Rollback
	if err := RollbackIndexMigration(indexPath, receipt.BackupPath); err != nil {
		t.Fatalf("Rollback failed: %v", err)
	}

	// Verify restored index is byte-identical to original V1
	restoredBytes, _ := os.ReadFile(indexPath)
	if !bytes.Equal(restoredBytes, originalV1) {
		t.Fatalf("rollback did not restore byte-identical V1!")
	}
}

func TestBackwardCompatibleLoadV1(t *testing.T) {
	dir := t.TempDir()
	indexPath := filepath.Join(dir, "v1_index.json")
	createGoldenV1Index(t, indexPath)

	// Load directly without migration command
	state, err := LoadIndex(indexPath)
	if err != nil {
		t.Fatalf("LoadIndex failed on unmigrated V1: %v", err)
	}
	if state.Schema != SchemaV2 {
		t.Errorf("expected in-memory upgrade to SchemaV2, got %s", state.Schema)
	}
	if len(state.Files) != 2 {
		t.Errorf("expected 2 files, got %d", len(state.Files))
	}
}

func TestUnsupportedSchema(t *testing.T) {
	dir := t.TempDir()
	indexPath := filepath.Join(dir, "future.json")
	if err := os.WriteFile(indexPath, []byte(`{"schema":"fak-sessionmine-index/999","files":{}}`), 0600); err != nil {
		t.Fatal(err)
	}

	_, err := LoadIndex(indexPath)
	if err == nil {
		t.Fatal("expected error on unsupported future schema, got nil")
	}

	_, err = MigrateIndex(indexPath, false)
	if err == nil {
		t.Fatal("expected migration error on unsupported future schema, got nil")
	}
}

func TestAlreadyCurrentRefused(t *testing.T) {
	dir := t.TempDir()
	indexPath := filepath.Join(dir, "v2.json")
	if err := os.WriteFile(indexPath, []byte(`{"schema":"fak-sessionmine-index/2","files":{}}`), 0600); err != nil {
		t.Fatal(err)
	}

	_, err := MigrateIndex(indexPath, false)
	if !errors.Is(err, ErrAlreadyCurrent) {
		t.Fatalf("expected ErrAlreadyCurrent, got %v", err)
	}
}
