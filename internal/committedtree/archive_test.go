package committedtree

import (
	"archive/tar"
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type archiveFixtureEntry struct {
	typeflag byte
	mode     int64
	linkname string
	body     []byte
}

// importExportIgnoreFixture builds one commit whose .gitattributes export-ignores
// secret.txt and the vendor directory, and gives one file the text eol=lf
// attribute. Other attributed files exercise the platform default.
func importExportIgnoreFixture(t *testing.T, repo string) string {
	t.Helper()
	attributes := []byte("*.txt text\nlf.txt text eol=lf\nimplied.dat eol=crlf\nsecret.txt export-ignore\nvendor export-ignore\n")
	blobs := []struct {
		mark int
		body []byte
	}{
		{1, attributes},
		{2, []byte("a\nb\n")},
		{3, []byte("a\nb\n")},
		{4, []byte("a\r\nb\nc\r\n")},
		{5, []byte("p\nq\n")},
		{6, []byte("secret\n")},
		{7, []byte("int x;\n")},
		{8, []byte("package main\n")},
		{9, []byte("raw\r\n\x00\xff\n")},
		{10, []byte("#!/bin/sh\necho hi\n")},
		{11, []byte("regular.bin")},
	}
	var stream bytes.Buffer
	for _, blob := range blobs {
		writeFastImportBlob(&stream, blob.mark, blob.body)
	}
	stream.WriteString("commit refs/heads/child\nmark :100\ncommitter Fixture <fixture@example.invalid> 0 +0000\ndata 0\n\n")
	stream.WriteString("commit refs/heads/main\ncommitter Fixture <fixture@example.invalid> 0 +0000\ndata 0\n")
	stream.WriteString("M 100644 :1 .gitattributes\n")
	stream.WriteString("M 100644 :2 lf.txt\n")
	stream.WriteString("M 100644 :3 native.txt\n")
	stream.WriteString("M 100644 :4 mixed.txt\n")
	stream.WriteString("M 100644 :5 implied.dat\n")
	stream.WriteString("M 100644 :6 secret.txt\n")
	stream.WriteString("M 100644 :7 vendor/lib/h.h\n")
	stream.WriteString("M 100644 :8 src/main.go\n")
	stream.WriteString("M 100644 :9 regular.bin\n")
	stream.WriteString("M 100755 :10 exec.sh\n")
	stream.WriteString("M 120000 :11 link\n")
	stream.WriteString("M 160000 :100 module\n\n")
	stream.WriteString("done\n")
	runGit(t, repo, &stream, "fast-import", "--quiet")
	return strings.TrimSpace(runGit(t, repo, nil, "rev-parse", "refs/heads/main"))
}

// readTarFixture maps each non-directory, non-header entry name to its header
// and body. Directory entries are skipped because the two producers may differ
// in whether they emit them, and only file content is compared.
func readTarFixture(t *testing.T, data []byte) map[string]archiveFixtureEntry {
	t.Helper()
	entries := map[string]archiveFixtureEntry{}
	tr := tar.NewReader(bytes.NewReader(data))
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return entries
		}
		if err != nil {
			t.Fatalf("read tar: %v", err)
		}
		if hdr.Typeflag == tar.TypeDir || hdr.Typeflag == tar.TypeXGlobalHeader {
			continue
		}
		body, err := io.ReadAll(tr)
		if err != nil {
			t.Fatalf("read tar body %s: %v", hdr.Name, err)
		}
		entries[hdr.Name] = archiveFixtureEntry{
			typeflag: hdr.Typeflag,
			mode:     hdr.Mode,
			linkname: hdr.Linkname,
			body:     body,
		}
	}
}

func gitArchiveTar(t *testing.T, repo, commit string) []byte {
	t.Helper()
	cmd := exec.Command("git", "-C", repo, "archive", "--format=tar", commit)
	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("git archive: %v\n%s", err, stderr.String())
	}
	return out.Bytes()
}

// committedBlob returns the object bytes at rev:path exactly as stored, which is
// the reference every snapshot must reproduce.
func committedBlob(t *testing.T, repo, rev, path string) []byte {
	t.Helper()
	cmd := exec.Command("git", "-C", repo, "cat-file", "blob", rev+":"+path)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git cat-file blob %s:%s: %v", rev, path, err)
	}
	return out
}

func writeTarToBytes(t *testing.T, repo, commit string) []byte {
	t.Helper()
	var out bytes.Buffer
	if err := WriteTar(context.Background(), repo, commit, &out); err != nil {
		t.Fatalf("WriteTar: %v", err)
	}
	return out.Bytes()
}

// WriteTar keeps export-ignored paths that git archive drops; this is the
// deliberate difference the snapshot exists for.
func TestWriteTarKeepsExportIgnoredPathsThatGitArchiveDrops(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)

	ours := readTarFixture(t, writeTarToBytes(t, repo, commit))
	archive := readTarFixture(t, gitArchiveTar(t, repo, commit))

	for _, name := range []string{"secret.txt", "vendor/lib/h.h"} {
		if _, ok := archive[name]; ok {
			t.Fatalf("git archive contains %s; the fixture must export-ignore it", name)
		}
		if _, ok := ours[name]; !ok {
			t.Fatalf("WriteTar omitted %s; export-ignored paths must be present", name)
		}
	}
	for _, name := range []string{"src/main.go", "link", "regular.bin", "exec.sh", "lf.txt"} {
		if _, ok := ours[name]; !ok {
			t.Fatalf("WriteTar omitted %s; ordinary paths must be present", name)
		}
	}
	if _, ok := ours["module"]; ok {
		t.Fatal("WriteTar emitted gitlink module; gitlinks must be skipped")
	}
}

// Entry names, types, modes and symlink targets follow git archive; file bytes
// follow the committed blob, which git archive would have converted.
func TestWriteTarEntriesMatchGitArchiveAndCommittedBlobs(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)

	ours := readTarFixture(t, writeTarToBytes(t, repo, commit))
	archive := readTarFixture(t, gitArchiveTar(t, repo, commit))
	for name, want := range archive {
		got, ok := ours[name]
		if !ok {
			t.Fatalf("WriteTar missing %s, which git archive includes", name)
		}
		if got.typeflag != want.typeflag {
			t.Fatalf("%s typeflag = %q, git archive %q", name, got.typeflag, want.typeflag)
		}
		if got.typeflag == tar.TypeSymlink {
			if got.linkname != want.linkname {
				t.Fatalf("%s linkname = %q, git archive %q", name, got.linkname, want.linkname)
			}
			continue
		}
		if got.mode != want.mode {
			t.Fatalf("%s mode = %#o, git archive %#o", name, got.mode, want.mode)
		}
		if want := committedBlob(t, repo, commit, name); !bytes.Equal(got.body, want) {
			t.Fatalf("%s bytes = %q, committed blob %q", name, got.body, want)
		}
	}
}

func TestWriteTarPathsSelectsOnlyRequestedPrefixes(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)

	var out bytes.Buffer
	if err := WriteTarPaths(context.Background(), repo, commit, &out, "vendor", "secret.txt"); err != nil {
		t.Fatalf("WriteTarPaths: %v", err)
	}
	got := readTarFixture(t, out.Bytes())
	want := []string{"secret.txt", "vendor/lib/h.h"}
	if len(got) != len(want) {
		t.Fatalf("WriteTarPaths entries = %d, want exactly %v", len(got), want)
	}
	for _, name := range want {
		entry, ok := got[name]
		if !ok {
			t.Fatalf("WriteTarPaths missing %s", name)
		}
		if expected := committedBlob(t, repo, commit, name); !bytes.Equal(entry.body, expected) {
			t.Fatalf("%s bytes = %q, committed blob %q", name, entry.body, expected)
		}
	}
}

func TestWriteTarPathsRejectsEscapingAndWholeTreePrefixes(t *testing.T) {
	for _, paths := range [][]string{nil, {"."}, {"../outside"}, {"/abs"}} {
		var out bytes.Buffer
		if err := WriteTarPaths(context.Background(), "unused", "unused", &out, paths...); err == nil {
			t.Fatalf("WriteTarPaths(%q) returned nil error, want rejection before any git process", paths)
		}
	}
}

func TestWriteTarIsDeterministic(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)
	first := writeTarToBytes(t, repo, commit)
	second := writeTarToBytes(t, repo, commit)
	if !bytes.Equal(first, second) {
		t.Fatal("WriteTar produced different bytes for the same commit")
	}
}

// The snapshot must not depend on the host's line-ending configuration or on
// the tree's attributes: every file, extracted or archived, equals the blob.
func TestSnapshotBytesEqualCommittedBlobsUnderEOLAttributes(t *testing.T) {
	repo := newRepository(t)
	crlfBody := []byte("one\r\ntwo\nthree\r\n")
	lfBody := []byte("one\ntwo\n")
	var stream bytes.Buffer
	writeFastImportBlob(&stream, 1, []byte("* text eol=crlf\nauto.txt text=auto\n"))
	writeFastImportBlob(&stream, 2, crlfBody)
	writeFastImportBlob(&stream, 3, lfBody)
	stream.WriteString("commit refs/heads/main\ncommitter Fixture <fixture@example.invalid> 0 +0000\ndata 0\n")
	stream.WriteString("M 100644 :1 .gitattributes\n")
	stream.WriteString("M 100644 :2 eol-crlf.txt\n")
	stream.WriteString("M 100644 :3 auto.txt\n")
	stream.WriteString("M 100644 :2 auto-crlf-source.txt\n")
	stream.WriteString("\ndone\n")
	runGit(t, repo, &stream, "fast-import", "--quiet")
	commit := strings.TrimSpace(runGit(t, repo, nil, "rev-parse", "refs/heads/main"))

	// Host settings that would convert on a checkout, plus repository-local
	// attributes that would convert too. None may reach the snapshot.
	runGit(t, repo, nil, "config", "core.autocrlf", "true")
	runGit(t, repo, nil, "config", "core.eol", "crlf")
	if err := os.WriteFile(filepath.Join(repo, ".git", "info", "attributes"), []byte("* text eol=crlf\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	paths := []string{"eol-crlf.txt", "auto.txt", "auto-crlf-source.txt"}
	dir, err := Extract(repo, commit)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	selected, err := ExtractPaths(repo, commit, paths...)
	if err != nil {
		t.Fatalf("ExtractPaths: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(selected) })

	tarEntries := readTarFixture(t, writeTarToBytes(t, repo, commit))
	for _, path := range paths {
		want := committedBlob(t, repo, commit, path)
		for label, root := range map[string]string{"Extract": dir, "ExtractPaths": selected} {
			got, err := os.ReadFile(filepath.Join(root, path))
			if err != nil {
				t.Fatalf("%s read %s: %v", label, path, err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("%s %s = %q, committed blob %q", label, path, got, want)
			}
		}
		if got := tarEntries[path].body; !bytes.Equal(got, want) {
			t.Fatalf("WriteTar %s = %q, committed blob %q", path, got, want)
		}
	}
	if !bytes.Equal(committedBlob(t, repo, commit, "eol-crlf.txt"), crlfBody) {
		t.Fatal("fixture blob is not the raw mixed-ending body it was built from")
	}
}

func TestExtractIncludesExportIgnoredPathsWhileGitArchiveOmitsThem(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)

	dir, err := Extract(repo, commit)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	for _, path := range []string{"secret.txt", filepath.Join("vendor", "lib", "h.h"), "src/main.go"} {
		if _, err := os.Stat(filepath.Join(dir, path)); err != nil {
			t.Fatalf("Extract missing %s: %v", path, err)
		}
	}
	archive := readTarFixture(t, gitArchiveTar(t, repo, commit))
	if _, ok := archive["secret.txt"]; ok {
		t.Fatal("git archive included secret.txt; the fixture is not exercising export-ignore")
	}
}

func TestExtractPathsSelectsPrefixesIncludingExportIgnored(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)

	dir, err := ExtractPaths(repo, commit, "vendor", "secret.txt")
	if err != nil {
		t.Fatalf("ExtractPaths: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	got := map[string]bool{}
	walkErr := filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() {
			rel, relErr := filepath.Rel(dir, path)
			if relErr != nil {
				return relErr
			}
			got[filepath.ToSlash(rel)] = true
		}
		return nil
	})
	if walkErr != nil {
		t.Fatal(walkErr)
	}
	want := map[string]bool{"vendor/lib/h.h": true, "secret.txt": true}
	if len(got) != len(want) {
		t.Fatalf("ExtractPaths materialized %v, want exactly %v", got, want)
	}
	for path := range want {
		if !got[path] {
			t.Fatalf("ExtractPaths missing %s; got %v", path, got)
		}
	}
}

func TestExtractPathsRejectsEscapingAndWholeTreePrefixes(t *testing.T) {
	for _, paths := range [][]string{nil, {}, {"."}, {""}, {"../outside"}, {"/abs"}, {"vendor", "a/../../b"}} {
		if _, err := ExtractPaths("unused", "unused", paths...); err == nil {
			t.Fatalf("ExtractPaths(%q) returned nil error, want rejection before any git process", paths)
		}
	}
}

func TestWriteTarHonoursCancelledContext(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	start := time.Now()
	var out bytes.Buffer
	err := WriteTar(ctx, repo, commit, &out)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("WriteTar on a cancelled context = %v, want context.Canceled", err)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Fatalf("cancelled WriteTar took %s, want prompt return", elapsed)
	}
}

// A writer that cancels mid-stream must end the run: the git process is killed
// and the call returns instead of waiting for the rest of the tree.
func TestWriteTarCancelMidStreamReturnsPromptly(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- WriteTar(ctx, repo, commit, cancelOnFirstWrite{cancel: cancel})
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("WriteTar after mid-stream cancel = %v, want context.Canceled", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("WriteTar did not return after the context was cancelled mid-stream")
	}
}

type cancelOnFirstWrite struct {
	cancel context.CancelFunc
}

func (w cancelOnFirstWrite) Write(p []byte) (int, error) {
	w.cancel()
	return len(p), nil
}

// The package budget bounds a snapshot only when the caller set no deadline.
func TestBudgetAppliesWhenCallerHasNoDeadline(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)
	t.Setenv(extractionBudgetEnv, "1ns")

	var out bytes.Buffer
	err := WriteTar(context.Background(), repo, commit, &out)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("WriteTar under a 1ns budget = %v, want context.DeadlineExceeded", err)
	}
	var budgetErr *BudgetError
	if !errors.As(err, &budgetErr) || budgetErr.Budget != time.Nanosecond {
		t.Fatalf("WriteTar budget error = %#v, want BudgetError carrying the 1ns budget", err)
	}
}

// A caller deadline replaces the package budget: a generous caller deadline
// succeeds even when the package budget would already have expired.
func TestCallerDeadlineReplacesPackageBudget(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)
	t.Setenv(extractionBudgetEnv, "1ns")

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	var out bytes.Buffer
	if err := WriteTar(ctx, repo, commit, &out); err != nil {
		t.Fatalf("WriteTar under a caller deadline = %v, want success", err)
	}
	if len(readTarFixture(t, out.Bytes())) == 0 {
		t.Fatal("WriteTar under a caller deadline wrote no entries")
	}
}

// An expired caller deadline stops the run with the caller's error, not the
// package budget error, while still matching context.DeadlineExceeded.
func TestExpiredCallerDeadlineIsReportedAsCallerDeadline(t *testing.T) {
	repo := newRepository(t)
	commit := importExportIgnoreFixture(t, repo)

	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	var out bytes.Buffer
	err := WriteTar(ctx, repo, commit, &out)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("WriteTar past the caller deadline = %v, want context.DeadlineExceeded", err)
	}
	var budgetErr *BudgetError
	if errors.As(err, &budgetErr) {
		t.Fatalf("WriteTar past the caller deadline reported the package budget: %v", err)
	}
}
