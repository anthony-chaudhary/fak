package agentopt

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestProvenanceMemoryInvalidation(t *testing.T) {
	t.Run("basic_store_and_active_retrieval", func(t *testing.T) {
		store := NewProvenanceMemoryStore()

		filePath := "internal/auth/tokens.go"
		fileContent := "package auth\nconst TokenExpiry = 15\n"

		cell := store.StoreMemory("Token expiry is configured to 15 minutes", filePath, fileContent)
		if cell == nil {
			t.Fatalf("expected non-nil cell")
		}
		if cell.ID == "" {
			t.Errorf("expected generated ID, got empty")
		}
		if cell.Statement != "Token expiry is configured to 15 minutes" {
			t.Errorf("unexpected statement: %s", cell.Statement)
		}
		if cell.FilePath != filePath {
			t.Errorf("expected filePath %s, got %s", filePath, cell.FilePath)
		}
		expectedDigest := ComputeContentDigest([]byte(fileContent))
		if cell.FileDigest != expectedDigest {
			t.Errorf("expected digest %s, got %s", expectedDigest, cell.FileDigest)
		}
		if cell.Stale {
			t.Errorf("expected cell not to be stale initially")
		}

		active := store.GetActiveMemories()
		if len(active) != 1 {
			t.Fatalf("expected 1 active memory, got %d", len(active))
		}
		if active[0].ID != cell.ID {
			t.Errorf("expected active memory ID %s, got %s", cell.ID, active[0].ID)
		}

		// QueryMemories matching statement
		matches := store.QueryMemories("expiry")
		if len(matches) != 1 {
			t.Errorf("expected 1 match for query 'expiry', got %d", len(matches))
		}

		// QueryMemories matching path
		matches = store.QueryMemories("tokens.go")
		if len(matches) != 1 {
			t.Errorf("expected 1 match for query 'tokens.go', got %d", len(matches))
		}

		// QueryMemories non-matching
		matches = store.QueryMemories("nonexistent")
		if len(matches) != 0 {
			t.Errorf("expected 0 matches for 'nonexistent', got %d", len(matches))
		}
	})

	t.Run("invalidate_on_file_change_mutation", func(t *testing.T) {
		store := NewProvenanceMemoryStore()

		filePath1 := "pkg/database/conn.go"
		content1 := "package database\nvar MaxOpen = 50\n"

		filePath2 := "pkg/database/pool.go"
		content2 := "package database\nvar IdleTimeout = 300\n"

		cell1 := store.StoreMemory("Max open connections limit is 50", filePath1, content1)
		cell2 := store.StoreMemory("Idle timeout duration is 300s", filePath2, content2)

		// File 1 is unmodified: InvalidateOnFileChange with same content should not invalidate
		staleIDs := store.InvalidateOnFileChange(filePath1, content1)
		if len(staleIDs) != 0 {
			t.Errorf("expected no invalidation on identical content, got %v", staleIDs)
		}

		c1, ok := store.GetMemory(cell1.ID)
		if !ok || c1.Stale {
			t.Errorf("expected cell1 to remain active after unchanged check")
		}

		// File 1 is modified: InvalidateOnFileChange with new content
		mutatedContent1 := "package database\nvar MaxOpen = 100\n"
		staleIDs = store.InvalidateOnFileChange(filePath1, mutatedContent1)
		if len(staleIDs) != 1 || staleIDs[0] != cell1.ID {
			t.Fatalf("expected cell1 %s to be invalidated, got %v", cell1.ID, staleIDs)
		}

		c1After, _ := store.GetMemory(cell1.ID)
		if !c1After.Stale {
			t.Errorf("expected cell1 to be marked stale after mutation")
		}

		// Cell 2 should remain active
		c2After, _ := store.GetMemory(cell2.ID)
		if c2After.Stale {
			t.Errorf("expected cell2 to remain active")
		}

		// GetActiveMemories should now only return cell2
		active := store.GetActiveMemories()
		if len(active) != 1 || active[0].ID != cell2.ID {
			t.Fatalf("expected only cell2 active, got %d memories", len(active))
		}

		// QueryMemories should exclude stale cell1
		queries := store.QueryMemories("connections")
		if len(queries) != 0 {
			t.Errorf("expected stale cell1 to be excluded from QueryMemories, got %d", len(queries))
		}
	})

	t.Run("invalidate_on_mutation_alias", func(t *testing.T) {
		store := NewProvenanceMemoryStore()
		path := "config/app.json"
		initContent := `{"port": 8080}`
		cell := store.StoreMemory("HTTP service port is 8080", path, initContent)

		modifiedContent := `{"port": 9090}`
		invalidated := store.InvalidateOnMutation(path, modifiedContent)
		if len(invalidated) != 1 || invalidated[0] != cell.ID {
			t.Fatalf("expected cell %s to be invalidated by InvalidateOnMutation, got %v", cell.ID, invalidated)
		}

		retrieved, found := store.GetMemory(cell.ID)
		if !found || !retrieved.Stale {
			t.Errorf("expected cell to be stale after mutation")
		}
	})

	t.Run("check_staleness_with_file_reader", func(t *testing.T) {
		store := NewProvenanceMemoryStore()

		// Virtual working tree representation
		workingTree := map[string][]byte{
			"src/main.go":    []byte("package main\nfunc main() {}\n"),
			"src/version.go": []byte("package main\nconst Version = \"1.0.0\"\n"),
			"src/helper.go":  []byte("package main\nfunc Help() {}\n"),
		}

		cellMain := store.StoreMemory("Entrypoint defined in src/main.go", "src/main.go", string(workingTree["src/main.go"]))
		cellVersion := store.StoreMemory("Release version is 1.0.0", "src/version.go", string(workingTree["src/version.go"]))
		cellHelper := store.StoreMemory("Helper function Help exported", "src/helper.go", string(workingTree["src/helper.go"]))

		fileReader := func(path string) ([]byte, error) {
			content, exists := workingTree[path]
			if !exists {
				return nil, os.ErrNotExist
			}
			return content, nil
		}

		// First check: nothing changed
		invalidated := store.CheckStaleness(fileReader)
		if len(invalidated) != 0 {
			t.Errorf("expected 0 invalidated on initial check, got %d", len(invalidated))
		}

		// Mutate src/version.go and delete src/helper.go
		workingTree["src/version.go"] = []byte("package main\nconst Version = \"1.1.0\"\n")
		delete(workingTree, "src/helper.go")

		// Run check staleness
		invalidated = store.CheckStaleness(fileReader)
		if len(invalidated) != 2 {
			t.Fatalf("expected 2 cells to be invalidated (version mutated + helper deleted), got %d: %v", len(invalidated), invalidated)
		}

		// Verify individual statuses
		mMain, _ := store.GetMemory(cellMain.ID)
		if mMain.Stale {
			t.Errorf("expected cellMain to remain active")
		}

		mVersion, _ := store.GetMemory(cellVersion.ID)
		if !mVersion.Stale {
			t.Errorf("expected cellVersion to be flagged stale due to digest mismatch")
		}

		mHelper, _ := store.GetMemory(cellHelper.ID)
		if !mHelper.Stale {
			t.Errorf("expected cellHelper to be flagged stale due to file deletion")
		}

		active := store.GetActiveMemories()
		if len(active) != 1 || active[0].ID != cellMain.ID {
			t.Fatalf("expected only cellMain active, got %d", len(active))
		}
	})

	t.Run("invalidate_on_file_deletion", func(t *testing.T) {
		store := NewProvenanceMemoryStore()
		path := "docs/architecture.md"
		content := "# Architecture Overview"

		cell := store.StoreMemory("Architecture docs outline system components", path, content)
		if cell.Stale {
			t.Fatalf("expected non-stale initial cell")
		}

		invalidated := store.InvalidateOnFileDeletion(path)
		if len(invalidated) != 1 || invalidated[0] != cell.ID {
			t.Fatalf("expected cell %s to be invalidated upon deletion, got %v", cell.ID, invalidated)
		}

		retrieved, found := store.GetMemory(cell.ID)
		if !found || !retrieved.Stale {
			t.Errorf("expected cell to be marked stale after deletion")
		}
	})

	t.Run("real_filesystem_integration", func(t *testing.T) {
		tempDir := t.TempDir()
		realFile := filepath.Join(tempDir, "config.txt")

		initialData := []byte("setting=alpha\n")
		if err := os.WriteFile(realFile, initialData, 0600); err != nil {
			t.Fatalf("failed to write temp file: %v", err)
		}

		store := NewProvenanceMemoryStore()
		cell := store.StoreMemory("Config setting is alpha", realFile, string(initialData))

		// Check staleness using os.ReadFile
		invalidated := store.CheckStaleness(os.ReadFile)
		if len(invalidated) != 0 {
			t.Errorf("expected 0 invalidated before change, got %d", len(invalidated))
		}

		// Mutate file on disk
		if err := os.WriteFile(realFile, []byte("setting=beta\n"), 0600); err != nil {
			t.Fatalf("failed to mutate temp file: %v", err)
		}

		invalidated = store.CheckStaleness(os.ReadFile)
		if len(invalidated) != 1 || invalidated[0] != cell.ID {
			t.Errorf("expected cell %s invalidated after disk write, got %v", cell.ID, invalidated)
		}

		// Now remove file from disk
		if err := os.Remove(realFile); err != nil {
			t.Fatalf("failed to remove temp file: %v", err)
		}

		// Add another memory for the now removed file
		cell2 := store.StoreMemory("Another note about config", realFile, "setting=beta\n")
		invalidated = store.CheckStaleness(os.ReadFile)
		if len(invalidated) != 1 || invalidated[0] != cell2.ID {
			t.Errorf("expected cell2 %s to be marked stale because file is gone, got %v", cell2.ID, invalidated)
		}
	})

	t.Run("concurrent_operations", func(t *testing.T) {
		store := NewProvenanceMemoryStore()
		var wg sync.WaitGroup

		numWorkers := 10
		iterations := 50

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				filePath := fmt.Sprintf("pkg/mod%d/file.go", workerID)
				content := fmt.Sprintf("package mod%d\n", workerID)

				for i := 0; i < iterations; i++ {
					stmt := fmt.Sprintf("Worker %d statement %d", workerID, i)
					cell := store.StoreMemory(stmt, filePath, content)

					if i%2 == 0 {
						_ = store.QueryMemories(fmt.Sprintf("Worker %d", workerID))
					} else {
						_ = store.GetActiveMemories()
					}

					if i%5 == 0 {
						newContent := fmt.Sprintf("package mod%d\n// updated %d\n", workerID, i)
						_ = store.InvalidateOnFileChange(filePath, newContent)
					}

					_, _ = store.GetMemory(cell.ID)
				}
			}(w)
		}

		wg.Wait()

		all := store.GetAllMemories()
		if len(all) != numWorkers*iterations {
			t.Errorf("expected %d total memories, got %d", numWorkers*iterations, len(all))
		}
	})
}
