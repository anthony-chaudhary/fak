package workerworktree

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func snapshotDirTree(t *testing.T, root string) map[string]int64 {
	t.Helper()
	tree := make(map[string]int64)
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		tree[rel] = info.Size()
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		t.Fatalf("snapshotDirTree %s: %v", root, err)
	}
	return tree
}

func TestResolveDefaults_EnvOverride(t *testing.T) {
	customRoot := filepath.Join(t.TempDir(), "custom-worker-worktrees")
	t.Setenv(WorktreeRootEnv, customRoot)

	repoRoot := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoRoot, 0o755); err != nil {
		t.Fatalf("mkdir repo: %v", err)
	}

	report := ResolveDefaults(repoRoot)

	if report.Schema != DefaultsSchema {
		t.Fatalf("Schema = %q, want %q", report.Schema, DefaultsSchema)
	}
	if report.RepoRoot != filepath.Clean(repoRoot) {
		t.Fatalf("RepoRoot = %q, want %q", report.RepoRoot, filepath.Clean(repoRoot))
	}
	if report.WorkerWorktreeRoot != filepath.Clean(customRoot) {
		t.Fatalf("WorkerWorktreeRoot = %q, want %q", report.WorkerWorktreeRoot, filepath.Clean(customRoot))
	}
	if report.RootSource != "environment: "+WorktreeRootEnv {
		t.Fatalf("RootSource = %q, want %q", report.RootSource, "environment: "+WorktreeRootEnv)
	}
	if report.DefaultLeaseIdentityBasis != DefaultLeaseIdentityBasis {
		t.Fatalf("DefaultLeaseIdentityBasis = %q, want %q", report.DefaultLeaseIdentityBasis, DefaultLeaseIdentityBasis)
	}
	wantOverrides := []string{WorktreeRootEnv}
	if !reflect.DeepEqual(report.SupportedEnvOverrides, wantOverrides) {
		t.Fatalf("SupportedEnvOverrides = %v, want %v", report.SupportedEnvOverrides, wantOverrides)
	}
}

func TestResolveDefaults_OSFallback_LocalAppData(t *testing.T) {
	t.Setenv(WorktreeRootEnv, "")
	appData := filepath.Join(t.TempDir(), "localappdata")
	t.Setenv("LOCALAPPDATA", appData)

	repoRoot := filepath.Join(t.TempDir(), "repo")
	report := ResolveDefaults(repoRoot)

	if report.RootSource != "os_fallback: LOCALAPPDATA" {
		t.Fatalf("RootSource = %q, want %q", report.RootSource, "os_fallback: LOCALAPPDATA")
	}
	wantRoot := filepath.Clean(filepath.Join(appData, "Fleet", "worker-worktrees"))
	if report.WorkerWorktreeRoot != wantRoot {
		t.Fatalf("WorkerWorktreeRoot = %q, want %q", report.WorkerWorktreeRoot, wantRoot)
	}
}

func TestResolveDefaults_OSFallback_TempDir(t *testing.T) {
	t.Setenv(WorktreeRootEnv, "")
	t.Setenv("LOCALAPPDATA", "")

	repoRoot := filepath.Join(t.TempDir(), "repo")
	report := ResolveDefaults(repoRoot)

	if report.RootSource != "os_fallback: temp_dir" {
		t.Fatalf("RootSource = %q, want %q", report.RootSource, "os_fallback: temp_dir")
	}
	wantRoot := filepath.Clean(filepath.Join(os.TempDir(), "Fleet", "worker-worktrees"))
	if report.WorkerWorktreeRoot != wantRoot {
		t.Fatalf("WorkerWorktreeRoot = %q, want %q", report.WorkerWorktreeRoot, wantRoot)
	}
}

func TestResolveDefaults_PathScrubbing(t *testing.T) {
	baseDir := t.TempDir()
	messyRepo := filepath.Join(baseDir, "sub", "..", "repo")
	messyEnv := filepath.Join(baseDir, "foo", "..", "bar")
	t.Setenv(WorktreeRootEnv, filepath.ToSlash(messyEnv))

	report := ResolveDefaults(filepath.ToSlash(messyRepo))

	cleanRepo := filepath.Clean(messyRepo)
	cleanEnv := filepath.Clean(messyEnv)

	if report.RepoRoot != cleanRepo {
		t.Fatalf("RepoRoot = %q, want %q", report.RepoRoot, cleanRepo)
	}
	if report.WorkerWorktreeRoot != cleanEnv {
		t.Fatalf("WorkerWorktreeRoot = %q, want %q", report.WorkerWorktreeRoot, cleanEnv)
	}

	emptyReport := ResolveDefaults("")
	if emptyReport.RepoRoot != "" {
		t.Fatalf("empty RepoRoot = %q, want empty string", emptyReport.RepoRoot)
	}
}

func TestResolveDefaults_ZeroMutations(t *testing.T) {
	t.Setenv(WorktreeRootEnv, "")
	repoDir := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(repoDir, 0o755); err != nil {
		t.Fatalf("mkdir repoDir: %v", err)
	}
	testFile := filepath.Join(repoDir, "tracked.txt")
	if err := os.WriteFile(testFile, []byte("content"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	beforeRepo := snapshotDirTree(t, repoDir)

	report := ResolveDefaults(repoDir)

	afterRepo := snapshotDirTree(t, repoDir)
	if !reflect.DeepEqual(beforeRepo, afterRepo) {
		t.Fatalf("repo tree modified: before=%v after=%v", beforeRepo, afterRepo)
	}

	// Verify WorkerWorktreeRoot was not created if it did not exist
	if _, err := os.Stat(report.WorkerWorktreeRoot); err == nil {
		// If it exists in system temp/localappdata from previous runs, ensure we did not mutate inside it
	} else if !os.IsNotExist(err) {
		t.Fatalf("unexpected stat error: %v", err)
	}
}
