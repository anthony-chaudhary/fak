package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

func setupDrillCLIRepo(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	ctx := context.Background()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "drill@cli.local"},
		{"config", "user.name", "drill cli"},
		{"config", "commit.gpgSign", "false"},
	} {
		if _, err := gitWipOut(ctx, dir, nil, args...); err != nil {
			t.Fatalf("git %v: %v", args, err)
		}
	}

	baseFile := filepath.Join(dir, "root.txt")
	if err := os.WriteFile(baseFile, []byte("root file content\n"), 0644); err != nil {
		t.Fatalf("write root.txt: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "add", "root.txt"); err != nil {
		t.Fatalf("add: %v", err)
	}
	baseTree, err := gitWipOut(ctx, dir, nil, "write-tree")
	if err != nil {
		t.Fatalf("write-tree: %v", err)
	}
	baseCommit, err := gitWipOut(ctx, dir, nil, "commit-tree", baseTree, "-m", "base")
	if err != nil {
		t.Fatalf("commit-tree: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", "HEAD", baseCommit); err != nil {
		t.Fatalf("update-ref: %v", err)
	}
	return dir, baseCommit
}

func addCLICheckpoint(t *testing.T, dir, session, baseCommit string, files map[string]string) string {
	t.Helper()
	ctx := context.Background()
	tmpIdx := filepath.Join(t.TempDir(), "idx")
	env := []string{"GIT_INDEX_FILE=" + tmpIdx}
	if _, err := gitWipOut(ctx, dir, env, "read-tree", baseCommit); err != nil {
		t.Fatalf("read-tree: %v", err)
	}

	for rel, data := range files {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		if err := os.WriteFile(p, []byte(data), 0644); err != nil {
			t.Fatalf("write: %v", err)
		}
		if _, err := gitWipOut(ctx, dir, env, "add", rel); err != nil {
			t.Fatalf("add %s: %v", rel, err)
		}
	}

	tree, err := gitWipOut(ctx, dir, env, "write-tree")
	if err != nil {
		t.Fatalf("write-tree: %v", err)
	}

	stamp := wipref.Stamp{
		SessionID:      session,
		StartSHA:       baseCommit,
		Buildable:      true,
		CheckpointedAt: time.Now().Unix(),
	}
	msg, err := wipref.EncodeStamp(stamp)
	if err != nil {
		t.Fatalf("encode stamp: %v", err)
	}

	commit, err := gitWipOut(ctx, dir, nil, "commit-tree", tree, "-p", baseCommit, "-m", msg)
	if err != nil {
		t.Fatalf("commit-tree: %v", err)
	}
	ref := wipref.SessionRef(session)
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", ref, commit); err != nil {
		t.Fatalf("update-ref: %v", err)
	}
	return commit
}

func TestWipDrillCLI_JSON(t *testing.T) {
	dir, base := setupDrillCLIRepo(t)
	addCLICheckpoint(t, dir, "cli-session-1", base, map[string]string{
		"src/alpha.txt": "alpha source\n",
	})

	var stdout, stderr bytes.Buffer
	code := runWip(&stdout, &stderr, []string{"drill", "--repo", dir, "--json"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr:\n%s", code, stderr.String())
	}

	var rep wipref.RecoveryDrillReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json report: %v\noutput:\n%s", err, stdout.String())
	}

	if rep.Schema != wipref.RecoveryDrillSchema {
		t.Errorf("schema mismatch: got %q, want %q", rep.Schema, wipref.RecoveryDrillSchema)
	}
	if rep.TotalDrilled != 1 || rep.SuccessCount != 1 || rep.FailureCount != 0 {
		t.Errorf("unexpected counts: total=%d success=%d fail=%d", rep.TotalDrilled, rep.SuccessCount, rep.FailureCount)
	}
	if !rep.MainTreePreserved {
		t.Errorf("expected main_tree_preserved to be true")
	}
	if len(rep.Results) != 1 || rep.Results[0].Status != "PASS" {
		t.Errorf("unexpected results: %+v", rep.Results)
	}
}

func TestWipDrillCLI_HumanOutput(t *testing.T) {
	dir, base := setupDrillCLIRepo(t)
	addCLICheckpoint(t, dir, "cli-session-human", base, map[string]string{
		"notes.md": "# human readable notes\n",
	})

	var stdout, stderr bytes.Buffer
	code := runWip(&stdout, &stderr, []string{"drill", "-C", dir})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr:\n%s", code, stderr.String())
	}

	outStr := stdout.String()
	if !strings.Contains(outStr, "WIP RECOVERY DRILL — 1 drilled: 1 PASS, 0 FAIL") {
		t.Errorf("expected summary line in human output, got:\n%s", outStr)
	}
	if !strings.Contains(outStr, "main tree preserved: true") {
		t.Errorf("expected 'main tree preserved: true', got:\n%s", outStr)
	}
	if !strings.Contains(outStr, "PASS  refs/fak/wip/cli-session-human") {
		t.Errorf("expected PASS result row, got:\n%s", outStr)
	}
}

func TestWipDrillCLI_SessionNotFound(t *testing.T) {
	dir, _ := setupDrillCLIRepo(t)

	var stdout, stderr bytes.Buffer
	code := runWip(&stdout, &stderr, []string{"drill", "--repo", dir, "--session", "nonexistent-sess", "--json"})
	if code != 1 {
		t.Fatalf("expected exit code 1 for missing session, got %d", code)
	}

	var rep wipref.RecoveryDrillReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}
	if rep.FailureCount != 1 || len(rep.Results) != 1 {
		t.Fatalf("expected 1 failure result, got %+v", rep)
	}
	if rep.Results[0].Status != "MISSING_OBJECT" {
		t.Errorf("expected MISSING_OBJECT, got %q", rep.Results[0].Status)
	}
}

func TestWipDrillCLI_Limit(t *testing.T) {
	dir, base := setupDrillCLIRepo(t)
	addCLICheckpoint(t, dir, "sess-1", base, map[string]string{"1.txt": "one\n"})
	addCLICheckpoint(t, dir, "sess-2", base, map[string]string{"2.txt": "two\n"})
	addCLICheckpoint(t, dir, "sess-3", base, map[string]string{"3.txt": "three\n"})

	var stdout, stderr bytes.Buffer
	code := runWip(&stdout, &stderr, []string{"drill", "--repo", dir, "--limit", "2", "--json"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr:\n%s", code, stderr.String())
	}

	var rep wipref.RecoveryDrillReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}
	if rep.TotalDrilled != 2 {
		t.Errorf("expected 2 drilled with limit=2, got %d", rep.TotalDrilled)
	}
}

func TestWipDrillCLI_UsageError(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runWip(&stdout, &stderr, []string{"drill", "--bad-flag"})
	if code != 2 {
		t.Errorf("expected exit code 2 for invalid flag, got %d", code)
	}
}
