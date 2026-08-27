package committedtree

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestExtractPreservesRawBlobBytes(t *testing.T) {
	for _, autocrlf := range []string{"true", "false"} {
		t.Run("autocrlf="+autocrlf, func(t *testing.T) {
			repo := initRepo(t)
			runGit(t, repo, "config", "core.autocrlf", autocrlf)

			body := []byte("line one\nline two\n")
			if err := os.WriteFile(filepath.Join(repo, "sample.txt"), body, 0o644); err != nil {
				t.Fatal(err)
			}
			runGit(t, repo, "add", "sample.txt")
			runGit(t, repo, "commit", "--quiet", "-m", "fixture")

			want := runGit(t, repo, "cat-file", "blob", "HEAD:sample.txt")
			if !bytes.Equal(want, body) {
				t.Fatalf("fixture blob = %q, want %q", want, body)
			}
			dir, err := Extract(repo, "HEAD")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = os.RemoveAll(dir) })
			got, err := os.ReadFile(filepath.Join(dir, "sample.txt"))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("extracted bytes = %q, raw blob bytes = %q", got, want)
			}
		})
	}
}

func TestExtractUsesBoundedGitProcessesAtScale(t *testing.T) {
	repo := initRepo(t)
	filesDir := filepath.Join(repo, "files")
	if err := os.MkdirAll(filesDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := range 256 {
		name := filepath.Join(filesDir, fmt.Sprintf("file-%03d.txt", i))
		if err := os.WriteFile(name, []byte(fmt.Sprintf("payload-%03d\n", i)), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGit(t, repo, "add", "files")
	symlinkBlob := strings.TrimSpace(string(runGitInput(t, repo, []byte("../escape"), "hash-object", "-w", "--stdin")))
	runGit(t, repo, "update-index", "--add", "--cacheinfo", "120000,"+symlinkBlob+",ignored-link")
	runGit(t, repo, "commit", "--quiet", "-m", "scale fixture")

	var operations []string
	command := func(ctx context.Context, name string, args ...string) *exec.Cmd {
		for _, arg := range args {
			switch arg {
			case "archive", "ls-tree", "cat-file":
				operations = append(operations, arg)
			}
		}
		return windowgate.CommandContext(ctx, name, args...)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dir := t.TempDir()
	if err := extractWithCommand(ctx, repo, "HEAD", dir, command); err != nil {
		t.Fatal(err)
	}
	if want := []string{"ls-tree", "cat-file"}; !slices.Equal(operations, want) {
		t.Fatalf("git operations = %v, want %v", operations, want)
	}
	if _, err := os.Lstat(filepath.Join(dir, "ignored-link")); !os.IsNotExist(err) {
		t.Fatalf("symlink was materialized: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(dir, "files", "file-255.txt"))
	if err != nil || string(got) != "payload-255\n" {
		t.Fatalf("last file = %q, err=%v", got, err)
	}
}

func TestParseTreeEntriesPreservesModesAndTypes(t *testing.T) {
	data := []byte("100755 blob aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\tbin/tool\x00" +
		"120000 blob bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\tlink\x00" +
		"160000 commit cccccccccccccccccccccccccccccccccccccccc\tsubmodule\x00")
	entries, err := parseTreeEntries(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 {
		t.Fatalf("entry count = %d, want 3", len(entries))
	}
	if !entries[0].regular || entries[0].mode.Perm() != 0o755 {
		t.Fatalf("executable entry = %+v", entries[0])
	}
	if entries[1].regular || entries[2].regular {
		t.Fatalf("symlink/gitlink classified as regular: %+v", entries[1:])
	}
}

func TestValidateTreeEntriesRejectsTraversal(t *testing.T) {
	entries := []treeEntry{{path: "safe"}, {path: "../escape"}}
	if err := validateTreeEntries(t.TempDir(), entries); err == nil {
		t.Fatal("expected traversal rejection")
	}
}

func TestExtractCleansTemporaryDirectoryOnError(t *testing.T) {
	repo := initRepo(t)
	tempRoot := t.TempDir()
	t.Setenv("TMPDIR", tempRoot)
	t.Setenv("TMP", tempRoot)
	t.Setenv("TEMP", tempRoot)
	if _, err := Extract(repo, "missing-object"); err == nil {
		t.Fatal("expected extraction failure")
	}
	entries, err := os.ReadDir(tempRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("temporary extraction directory leaked: %v", entries)
	}
}

func initRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	runGit(t, repo, "init", "--quiet")
	runGit(t, repo, "config", "user.name", "Committed Tree Test")
	runGit(t, repo, "config", "user.email", "committed-tree@example.invalid")
	return repo
}

func runGit(t *testing.T, repo string, args ...string) []byte {
	t.Helper()
	return runGitInput(t, repo, nil, args...)
}

func runGitInput(t *testing.T, repo string, input []byte, args ...string) []byte {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	cmd.Stdin = bytes.NewReader(input)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return out
}
