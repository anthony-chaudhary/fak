package main

import (
	"archive/tar"
	"bytes"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// The committed-tip snapshot feeds a regenerate check and reads the tracked
// artifact from the snapshot, so an export-ignore row must not strip either.
func TestDisambiguationCommittedTipKeepsExportIgnoredFiles(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	writeDisambiguationSnapshotFile(t, repo, ".gitattributes", "secret.txt export-ignore\ndocs/generated/** export-ignore\n")
	writeDisambiguationSnapshotFile(t, repo, "secret.txt", "kept\n")
	writeDisambiguationSnapshotFile(t, repo, "docs/generated/disambiguation-index.json", "{}\n")
	runDisambiguationSnapshotGit(t, repo, "init", "-q")
	runDisambiguationSnapshotGit(t, repo, "add", "-A")
	runDisambiguationSnapshotGit(t, repo, "-c", "user.name=fak-test", "-c", "user.email=fak-test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "snapshot fixture")

	// Control: git archive drops export-ignored entries, so the fixture really exercises the drop.
	if tarHasEntry(t, repo, "secret.txt") {
		t.Fatal("control: git archive kept export-ignored secret.txt")
	}

	root := filepath.Join(t.TempDir(), "tip")
	archivePath := filepath.Join(t.TempDir(), "tip.tar")
	if err := extractDisambiguationCommittedTip(repo, archivePath, root); err != nil {
		t.Fatalf("extractDisambiguationCommittedTip: %v", err)
	}
	want := map[string]string{
		"secret.txt": "kept\n",
		"docs/generated/disambiguation-index.json": "{}\n",
	}
	for rel, content := range want {
		got, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatalf("snapshot lost %s: %v", rel, err)
		}
		if !bytes.Equal(got, []byte(content)) {
			t.Fatalf("snapshot %s = %q, want %q", rel, got, content)
		}
	}
}

func tarHasEntry(t *testing.T, repo, name string) bool {
	t.Helper()
	out, err := exec.Command("git", "-C", repo, "archive", "--format=tar", "HEAD").Output()
	if err != nil {
		t.Fatalf("git archive: %v", err)
	}
	reader := tar.NewReader(bytes.NewReader(out))
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			return false
		}
		if err != nil {
			t.Fatalf("read git archive tar: %v", err)
		}
		if header.Name == name {
			return true
		}
	}
}

func writeDisambiguationSnapshotFile(t *testing.T, repo, rel, content string) {
	t.Helper()
	target := filepath.Join(repo, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func runDisambiguationSnapshotGit(t *testing.T, repo string, args ...string) {
	t.Helper()
	out, err := exec.Command("git", append([]string{"-C", repo}, args...)...).CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
}
