package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func snapshotFiles(t *testing.T, dir string) map[string]int64 {
	t.Helper()
	m := make(map[string]int64)
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		m[rel] = info.Size()
		return nil
	})
	if err != nil {
		t.Fatalf("snapshotFiles %s: %v", dir, err)
	}
	return m
}

func TestWorktreeWorkerDefaults_Text(t *testing.T) {
	tempRepo := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(tempRepo, 0o755); err != nil {
		t.Fatalf("mkdir tempRepo: %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := runWorktreeWorkerDefaults([]string{"--root", tempRepo}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("runWorktreeWorkerDefaults failed: %v (stderr: %s)", err, stderr.String())
	}

	out := stdout.String()
	requiredLines := []string{
		"schema: " + workerworktree.DefaultsSchema,
		"repo_root: " + filepath.Clean(tempRepo),
		"worker_worktree_root: ",
		"root_source: ",
		"default_lease_identity_basis: " + workerworktree.DefaultLeaseIdentityBasis,
		"supported_env_overrides: " + workerworktree.WorktreeRootEnv,
	}

	for _, line := range requiredLines {
		if !strings.Contains(out, line) {
			t.Errorf("output missing expected line/prefix %q in:\n%s", line, out)
		}
	}
}

func TestWorktreeWorkerDefaults_JSON(t *testing.T) {
	tempRepo := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(tempRepo, 0o755); err != nil {
		t.Fatalf("mkdir tempRepo: %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := runWorktreeWorkerDefaults([]string{"--json", "--root", tempRepo}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("runWorktreeWorkerDefaults failed: %v (stderr: %s)", err, stderr.String())
	}

	var rep workerworktree.DefaultsReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json: %v; raw=%q", err, stdout.String())
	}

	if rep.Schema != workerworktree.DefaultsSchema {
		t.Fatalf("Schema = %q, want %q", rep.Schema, workerworktree.DefaultsSchema)
	}
	if rep.RepoRoot != filepath.Clean(tempRepo) {
		t.Fatalf("RepoRoot = %q, want %q", rep.RepoRoot, filepath.Clean(tempRepo))
	}
	if rep.WorkerWorktreeRoot == "" {
		t.Fatalf("WorkerWorktreeRoot is empty")
	}
	if rep.RootSource == "" {
		t.Fatalf("RootSource is empty")
	}
	if rep.DefaultLeaseIdentityBasis != workerworktree.DefaultLeaseIdentityBasis {
		t.Fatalf("DefaultLeaseIdentityBasis = %q, want %q", rep.DefaultLeaseIdentityBasis, workerworktree.DefaultLeaseIdentityBasis)
	}
	wantOverrides := []string{workerworktree.WorktreeRootEnv}
	if !reflect.DeepEqual(rep.SupportedEnvOverrides, wantOverrides) {
		t.Fatalf("SupportedEnvOverrides = %v, want %v", rep.SupportedEnvOverrides, wantOverrides)
	}
}

func TestWorktreeWorkerDefaults_EnvOverride(t *testing.T) {
	customRoot := filepath.Join(t.TempDir(), "custom-worktrees")
	t.Setenv(workerworktree.WorktreeRootEnv, customRoot)

	tempRepo := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(tempRepo, 0o755); err != nil {
		t.Fatalf("mkdir tempRepo: %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := runWorktreeWorkerDefaults([]string{"--json", "--root", tempRepo}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("runWorktreeWorkerDefaults failed: %v", err)
	}

	var rep workerworktree.DefaultsReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}

	if rep.WorkerWorktreeRoot != filepath.Clean(customRoot) {
		t.Fatalf("WorkerWorktreeRoot = %q, want %q", rep.WorkerWorktreeRoot, filepath.Clean(customRoot))
	}
	if rep.RootSource != "environment: "+workerworktree.WorktreeRootEnv {
		t.Fatalf("RootSource = %q, want %q", rep.RootSource, "environment: "+workerworktree.WorktreeRootEnv)
	}
}

func TestWorktreeWorkerDefaults_OSFallback(t *testing.T) {
	t.Setenv(workerworktree.WorktreeRootEnv, "")

	tempRepo := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(tempRepo, 0o755); err != nil {
		t.Fatalf("mkdir tempRepo: %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := runWorktreeWorkerDefaults([]string{"--json", "--root", tempRepo}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("runWorktreeWorkerDefaults failed: %v", err)
	}

	var rep workerworktree.DefaultsReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}

	if !strings.HasPrefix(rep.RootSource, "os_fallback:") {
		t.Fatalf("RootSource = %q, want prefix 'os_fallback:'", rep.RootSource)
	}
}

func TestWorktreeWorkerDefaults_ZeroMutations(t *testing.T) {
	tempRepo := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(tempRepo, 0o755); err != nil {
		t.Fatalf("mkdir tempRepo: %v", err)
	}
	testFile := filepath.Join(tempRepo, "tracked_file.txt")
	if err := os.WriteFile(testFile, []byte("alpha beta"), 0o644); err != nil {
		t.Fatalf("write testFile: %v", err)
	}

	beforeSnapshot := snapshotFiles(t, tempRepo)

	var stdout, stderr bytes.Buffer
	if err := runWorktreeWorkerDefaults([]string{"--root", tempRepo}, &stdout, &stderr); err != nil {
		t.Fatalf("text run failed: %v", err)
	}
	stdout.Reset()
	stderr.Reset()
	if err := runWorktreeWorkerDefaults([]string{"--json", "--root", tempRepo}, &stdout, &stderr); err != nil {
		t.Fatalf("json run failed: %v", err)
	}

	afterSnapshot := snapshotFiles(t, tempRepo)
	if !reflect.DeepEqual(beforeSnapshot, afterSnapshot) {
		t.Fatalf("repository files mutated: before=%v, after=%v", beforeSnapshot, afterSnapshot)
	}
}
