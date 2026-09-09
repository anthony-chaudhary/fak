package safesync

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSyncHeal(t *testing.T) {
	t.Run("CleanRepo", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init", "-b", "main")
		git(t, repo, "config", "user.name", "test")
		git(t, repo, "config", "user.email", "test@example.com")

		writeFile(t, filepath.Join(repo, "a.txt"), "hello\n")
		git(t, repo, "add", "a.txt")
		git(t, repo, "commit", "-m", "initial")

		res, err := Heal(context.Background(), HealOptions{Repo: repo})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !res.OK {
			t.Fatalf("res.OK = false, want true")
		}
		if res.HealedCount != 0 {
			t.Fatalf("res.HealedCount = %d, want 0", res.HealedCount)
		}
		if len(res.MissingFiles) != 0 {
			t.Fatalf("res.MissingFiles = %v, want empty", res.MissingFiles)
		}
	})

	t.Run("UnstagedPhantomDeletion", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init", "-b", "main")
		git(t, repo, "config", "user.name", "test")
		git(t, repo, "config", "user.email", "test@example.com")

		writeFile(t, filepath.Join(repo, "foo.txt"), "head content\n")
		git(t, repo, "add", "foo.txt")
		git(t, repo, "commit", "-m", "initial")

		// Simulate unstaged deletion on disk
		if err := os.Remove(filepath.Join(repo, "foo.txt")); err != nil {
			t.Fatal(err)
		}

		res, err := Heal(context.Background(), HealOptions{Repo: repo})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !res.OK {
			t.Fatalf("res.OK = false, want true")
		}
		if res.HealedCount != 1 {
			t.Fatalf("res.HealedCount = %d, want 1", res.HealedCount)
		}
		if len(res.UnstagedDeleted) != 1 || res.UnstagedDeleted[0] != "foo.txt" {
			t.Fatalf("res.UnstagedDeleted = %v, want [foo.txt]", res.UnstagedDeleted)
		}
		if len(res.StagedDeleted) != 0 {
			t.Fatalf("res.StagedDeleted = %v, want empty", res.StagedDeleted)
		}
		if got := readFile(t, filepath.Join(repo, "foo.txt")); got != "head content\n" {
			t.Fatalf("foo.txt content = %q, want %q", got, "head content\n")
		}

		status := gitOutput(t, repo, "status", "--porcelain")
		if strings.TrimSpace(status) != "" {
			t.Fatalf("status after heal = %q, want clean", status)
		}
	})

	t.Run("StagedPhantomDeletion", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init", "-b", "main")
		git(t, repo, "config", "user.name", "test")
		git(t, repo, "config", "user.email", "test@example.com")

		writeFile(t, filepath.Join(repo, "bar.txt"), "bar head content\n")
		git(t, repo, "add", "bar.txt")
		git(t, repo, "commit", "-m", "initial")

		// Simulate staged deletion
		git(t, repo, "rm", "bar.txt")

		res, err := Heal(context.Background(), HealOptions{Repo: repo})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !res.OK {
			t.Fatalf("res.OK = false, want true")
		}
		if res.HealedCount != 1 {
			t.Fatalf("res.HealedCount = %d, want 1", res.HealedCount)
		}
		if len(res.StagedDeleted) != 1 || res.StagedDeleted[0] != "bar.txt" {
			t.Fatalf("res.StagedDeleted = %v, want [bar.txt]", res.StagedDeleted)
		}
		if got := readFile(t, filepath.Join(repo, "bar.txt")); got != "bar head content\n" {
			t.Fatalf("bar.txt content = %q, want %q", got, "bar head content\n")
		}

		status := gitOutput(t, repo, "status", "--porcelain")
		if strings.TrimSpace(status) != "" {
			t.Fatalf("status after heal = %q, want clean", status)
		}
	})

	t.Run("DeepNestedSubdirectory", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init", "-b", "main")
		git(t, repo, "config", "user.name", "test")
		git(t, repo, "config", "user.email", "test@example.com")

		nestedPath := filepath.Join(repo, "a", "b", "c", "nested.txt")
		if err := os.MkdirAll(filepath.Dir(nestedPath), 0o755); err != nil {
			t.Fatal(err)
		}
		writeFile(t, nestedPath, "nested content\n")
		git(t, repo, "add", ".")
		git(t, repo, "commit", "-m", "initial")

		// Remove the entire directory tree
		if err := os.RemoveAll(filepath.Join(repo, "a")); err != nil {
			t.Fatal(err)
		}

		res, err := Heal(context.Background(), HealOptions{Repo: repo})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !res.OK || res.HealedCount != 1 {
			t.Fatalf("Heal result = %+v, want OK=true and count=1", res)
		}
		if got := readFile(t, nestedPath); got != "nested content\n" {
			t.Fatalf("nested content = %q, want %q", got, "nested content\n")
		}
	})

	t.Run("PreservesDirtyWIP", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init", "-b", "main")
		git(t, repo, "config", "user.name", "test")
		git(t, repo, "config", "user.email", "test@example.com")

		writeFile(t, filepath.Join(repo, "clean.txt"), "clean head\n")
		writeFile(t, filepath.Join(repo, "dirty_worktree.txt"), "dirty worktree orig\n")
		writeFile(t, filepath.Join(repo, "dirty_staged.txt"), "dirty staged orig\n")
		writeFile(t, filepath.Join(repo, "recreated.txt"), "recreated orig\n")
		writeFile(t, filepath.Join(repo, "missing_unstaged.txt"), "missing unstaged head\n")
		writeFile(t, filepath.Join(repo, "missing_staged.txt"), "missing staged head\n")
		git(t, repo, "add", ".")
		git(t, repo, "commit", "-m", "initial")

		// Apply modifications and phantom deletions:
		writeFile(t, filepath.Join(repo, "dirty_worktree.txt"), "dirty worktree uncommitted edit\n")

		writeFile(t, filepath.Join(repo, "dirty_staged.txt"), "dirty staged uncommitted edit\n")
		git(t, repo, "add", "dirty_staged.txt")

		writeFile(t, filepath.Join(repo, "untracked.txt"), "untracked file\n")

		// Staged deleted but recreated with new content on disk
		git(t, repo, "rm", "recreated.txt")
		writeFile(t, filepath.Join(repo, "recreated.txt"), "recreated uncommitted edit\n")

		// Phantom unstaged deletion
		if err := os.Remove(filepath.Join(repo, "missing_unstaged.txt")); err != nil {
			t.Fatal(err)
		}

		// Phantom staged deletion
		git(t, repo, "rm", "missing_staged.txt")

		res, err := Heal(context.Background(), HealOptions{Repo: repo})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !res.OK {
			t.Fatalf("res.OK = false, want true")
		}
		if res.HealedCount != 2 {
			t.Fatalf("res.HealedCount = %d, want 2", res.HealedCount)
		}

		// Check restored files
		if got := readFile(t, filepath.Join(repo, "missing_unstaged.txt")); got != "missing unstaged head\n" {
			t.Fatalf("missing_unstaged.txt = %q", got)
		}
		if got := readFile(t, filepath.Join(repo, "missing_staged.txt")); got != "missing staged head\n" {
			t.Fatalf("missing_staged.txt = %q", got)
		}

		// Check preserved dirty files
		if got := readFile(t, filepath.Join(repo, "dirty_worktree.txt")); got != "dirty worktree uncommitted edit\n" {
			t.Fatalf("dirty_worktree.txt edit not preserved: %q", got)
		}
		if got := readFile(t, filepath.Join(repo, "dirty_staged.txt")); got != "dirty staged uncommitted edit\n" {
			t.Fatalf("dirty_staged.txt edit not preserved: %q", got)
		}
		if got := readFile(t, filepath.Join(repo, "untracked.txt")); got != "untracked file\n" {
			t.Fatalf("untracked.txt not preserved: %q", got)
		}
		if got := readFile(t, filepath.Join(repo, "recreated.txt")); got != "recreated uncommitted edit\n" {
			t.Fatalf("recreated.txt edit not preserved: %q", got)
		}

		// Ensure preserved dirty files are reported
		if len(res.PreservedDirty) == 0 {
			t.Fatalf("res.PreservedDirty is empty, want dirty files reported")
		}
	})

	t.Run("DryRun", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init", "-b", "main")
		git(t, repo, "config", "user.name", "test")
		git(t, repo, "config", "user.email", "test@example.com")

		writeFile(t, filepath.Join(repo, "file1.txt"), "file1 head\n")
		writeFile(t, filepath.Join(repo, "file2.txt"), "file2 head\n")
		git(t, repo, "add", ".")
		git(t, repo, "commit", "-m", "initial")

		if err := os.Remove(filepath.Join(repo, "file1.txt")); err != nil {
			t.Fatal(err)
		}
		git(t, repo, "rm", "file2.txt")

		statusBefore := gitOutput(t, repo, "status", "--porcelain")

		res, err := Heal(context.Background(), HealOptions{Repo: repo, DryRun: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !res.OK || !res.DryRun {
			t.Fatalf("res = %+v, want OK=true, DryRun=true", res)
		}
		if res.HealedCount != 2 {
			t.Fatalf("res.HealedCount = %d, want 2", res.HealedCount)
		}

		// Files should STILL be missing
		if _, err := os.Stat(filepath.Join(repo, "file1.txt")); !os.IsNotExist(err) {
			t.Fatalf("file1.txt should still be missing in dry run")
		}
		if _, err := os.Stat(filepath.Join(repo, "file2.txt")); !os.IsNotExist(err) {
			t.Fatalf("file2.txt should still be missing in dry run")
		}

		statusAfter := gitOutput(t, repo, "status", "--porcelain")
		if statusBefore != statusAfter {
			t.Fatalf("git status changed during dry run: before=%q after=%q", statusBefore, statusAfter)
		}
	})

	t.Run("NoHEAD", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init")

		res, err := Heal(context.Background(), HealOptions{Repo: repo})
		if err == nil && res.OK {
			t.Fatalf("expected failure or refusal on empty repo without HEAD, got %+v", res)
		}
	})
}
