package tb4bench

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWorkspaceDiffSnapshot(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "tb4-workspace-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	mockEngine := NewMockContainerEngine()
	defer mockEngine.Close()

	ctx := context.Background()
	config := ContainerConfig{
		ImageDigest: "ghcr.io/fak/tb4@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		Name:        "tb4-ws-test",
		NetworkMode: NetworkModeNone,
		WorkingDir:  "/workspace",
	}
	inst, err := mockEngine.CreateContainer(ctx, config)
	if err != nil {
		t.Fatalf("failed to create container: %v", err)
	}

	wsDir := mockEngine.workspaces[inst.ID]
	wsMgr := NewWorkspaceManager(mockEngine, inst.ID, "task-diff-01", wsDir)

	// 1. Seed workspace
	initialFiles := map[string][]byte{
		"main.py": []byte("print('broken'\n"),
		"test.py": []byte("assert True\n"),
	}
	initialDigest, err := wsMgr.SeedWorkspace(ctx, "Fix the syntax error in main.py", initialFiles)
	if err != nil {
		t.Fatalf("failed to seed workspace: %v", err)
	}
	if len(initialDigest) != 64 {
		t.Fatalf("expected 64-char initial digest, got %q", initialDigest)
	}

	// 2. Perform agent modification (edit main.py)
	fixCode := []byte("print('fixed')\n")
	if err := os.WriteFile(filepath.Join(wsDir, "main.py"), fixCode, 0644); err != nil {
		t.Fatalf("failed to edit main.py: %v", err)
	}

	// 3. Capture diff snapshot
	snapshot, err := wsMgr.SnapshotDiff(ctx, initialDigest)
	if err != nil {
		t.Fatalf("failed to snapshot diff: %v", err)
	}

	if snapshot.TaskID != "task-diff-01" {
		t.Errorf("expected task-diff-01, got %s", snapshot.TaskID)
	}
	if snapshot.InitialDigest == snapshot.FinalDigest {
		t.Errorf("expected initial and final digests to differ after modification")
	}
	if !strings.Contains(snapshot.UnifiedDiff, "print('fixed')") {
		t.Errorf("unified diff does not contain fix: %s", snapshot.UnifiedDiff)
	}

	// 4. Test deterministic directory hashing
	digest1, err := HashDirectoryTree(wsDir)
	if err != nil {
		t.Fatalf("failed to hash tree: %v", err)
	}
	digest2, err := HashDirectoryTree(wsDir)
	if err != nil {
		t.Fatalf("failed to re-hash tree: %v", err)
	}
	if digest1 != digest2 {
		t.Errorf("expected deterministic tree digest, got %s and %s", digest1, digest2)
	}
}
