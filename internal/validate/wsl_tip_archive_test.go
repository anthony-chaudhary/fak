package validate

import (
	"archive/tar"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// wslTipArchiveTestRepo commits files, keyed by repo-relative path, in a fresh
// repository and returns its path and the tip SHA.
func wslTipArchiveTestRepo(t *testing.T, files map[string]string) (string, string) {
	t.Helper()
	repo := t.TempDir()
	identityTestGit(t, repo, "init", "--quiet")
	for rel, body := range files {
		path := filepath.Join(repo, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	identityTestGit(t, repo, "add", "-A")
	identityTestGit(t, repo, "-c", "user.name=fak", "-c", "user.email=fak@example.invalid", "commit", "--quiet", "-m", "fixture")
	return repo, strings.TrimSpace(identityTestGit(t, repo, "rev-parse", "HEAD"))
}

// TestWSLTipArchiveKeepsExportIgnoredPaths guards the WSL snapshot: the WSL side
// runs the tests from this tree, so a path export-ignored in .gitattributes must
// still be present, while git archive on the same commit drops it.
func TestWSLTipArchiveKeepsExportIgnoredPaths(t *testing.T) {
	repo, tip := wslTipArchiveTestRepo(t, map[string]string{
		".gitattributes":    "secret.txt export-ignore\nvendor export-ignore\n",
		"secret.txt":        "kept\n",
		"vendor/lib.h":      "#define LIB 1\n",
		"pkg/thing.go":      "package pkg\n",
		"pkg/thing_test.go": "package pkg\n",
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	archivePath, err := writeWSLTipArchive(ctx, repo, tip)
	if err != nil {
		t.Fatalf("writeWSLTipArchive: %v", err)
	}
	defer os.RemoveAll(filepath.Dir(archivePath))
	data, err := os.ReadFile(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := tarEntryNames(t, strings.NewReader(string(data)))
	for _, name := range []string{"secret.txt", "vendor/lib.h", "pkg/thing_test.go", "pkg/thing.go"} {
		if !snapshot[name] {
			t.Fatalf("WSL snapshot omitted %s; tests need every committed file", name)
		}
	}

	gitArchive := identityTestGit(t, repo, "archive", "--format=tar", tip)
	stripped := tarEntryNames(t, strings.NewReader(gitArchive))
	if stripped["secret.txt"] || stripped["vendor/lib.h"] {
		t.Fatal("git archive kept export-ignored paths; the fixture does not exercise export-ignore")
	}
}

// TestWSLTipArchiveWritesBlobBytesForTextAuto commits a text=auto file under
// Windows-style settings. Checkout would convert it to CRLF, but the WSL snapshot
// must carry the committed blob bytes, so the host configuration cannot change
// what WSL compiles and tests.
func TestWSLTipArchiveWritesBlobBytesForTextAuto(t *testing.T) {
	repo, tip := wslTipArchiveTestRepo(t, map[string]string{
		".gitattributes": "*.txt text=auto\n",
		"notes.txt":      "alpha\r\nbeta\r\n",
	})
	identityTestGit(t, repo, "config", "core.autocrlf", "true")
	identityTestGit(t, repo, "config", "core.eol", "crlf")

	blob := identityTestGit(t, repo, "cat-file", "blob", tip+":notes.txt")
	if blob != "alpha\nbeta\n" {
		t.Fatalf("committed blob = %q, want the LF bytes text=auto stores; fixture does not exercise conversion", blob)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	archivePath, err := writeWSLTipArchive(ctx, repo, tip)
	if err != nil {
		t.Fatalf("writeWSLTipArchive: %v", err)
	}
	defer os.RemoveAll(filepath.Dir(archivePath))
	data, err := os.ReadFile(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	got := tarEntryBodies(t, strings.NewReader(string(data)))["notes.txt"]
	if got != blob {
		t.Fatalf("WSL snapshot notes.txt = %q, want committed blob %q", got, blob)
	}
}

// TestWSLTipArchiveStageIsReapedAfterDeadOwner covers a validator killed before
// its deferred removal: the next staging reaps the dead owner's directory and
// keeps a live owner's.
func TestWSLTipArchiveStageIsReapedAfterDeadOwner(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)
	t.Setenv("TMP", tmp)
	t.Setenv("TEMP", tmp)

	const deadPID = 2147483632
	dead := filepath.Join(tmp, fmt.Sprintf("%spid%d-abandoned", wslTipArchiveDirPrefix, deadPID))
	live := filepath.Join(tmp, fmt.Sprintf("%spid%d-running", wslTipArchiveDirPrefix, os.Getpid()))
	for _, dir := range []string{dead, live} {
		if err := os.Mkdir(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "tip.tar"), []byte("stale"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	repo, tip := wslTipArchiveTestRepo(t, map[string]string{"pkg/thing.go": "package pkg\n"})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	archivePath, err := writeWSLTipArchive(ctx, repo, tip)
	if err != nil {
		t.Fatalf("writeWSLTipArchive: %v", err)
	}
	defer os.RemoveAll(filepath.Dir(archivePath))

	if filepath.Dir(filepath.Dir(archivePath)) != tmp {
		t.Fatalf("staged archive %s is not in an owned directory under %s", archivePath, tmp)
	}
	if _, err := os.Stat(dead); !os.IsNotExist(err) {
		t.Fatalf("abandoned stage %s survived; stat err = %v", dead, err)
	}
	if _, err := os.Stat(live); err != nil {
		t.Fatalf("live owner's stage was reaped: %v", err)
	}
}

func tarEntryNames(t *testing.T, r io.Reader) map[string]bool {
	t.Helper()
	names := make(map[string]bool)
	for name := range tarEntryBodies(t, r) {
		names[name] = true
	}
	return names
}

func tarEntryBodies(t *testing.T, r io.Reader) map[string]string {
	t.Helper()
	bodies := make(map[string]string)
	tr := tar.NewReader(r)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return bodies
		}
		if err != nil {
			t.Fatalf("read tar: %v", err)
		}
		data, err := io.ReadAll(tr)
		if err != nil {
			t.Fatalf("read tar entry %s: %v", hdr.Name, err)
		}
		bodies[hdr.Name] = string(data)
	}
}
