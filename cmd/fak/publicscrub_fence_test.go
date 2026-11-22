package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPublicScrubFenceProtectsPrivateFilesEndToEnd(t *testing.T) {
	const (
		private = "Rank every candidate by realized slippage before sizing the order."
		shared  = "This project is released under the terms in the LICENSE file."
	)
	repo := t.TempDir()
	publicScrubGitRun(t, repo, "init", "-q", "-b", "main")
	publicScrubGitRun(t, repo, "config", "user.email", "t@t")
	publicScrubGitRun(t, repo, "config", "user.name", "t")
	publicScrubWriteFile(t, repo, ".gitignore", ".fak/\n")
	publicScrubWriteFile(t, repo, "README.md", shared+"\n")
	publicScrubGitRun(t, repo, "add", ".gitignore", "README.md")
	publicScrubGitRun(t, repo, "commit", "-qm", "seed")

	alpha := t.TempDir()
	publicScrubWriteFile(t, alpha, "skills/sizing.md", "# sizing\n"+shared+"\n"+private+"\n")

	var stdout, stderr bytes.Buffer
	if rc := runPublicScrub(&stdout, &stderr, []string{"fence", "--root", repo, alpha}); rc != 0 {
		t.Fatalf("fence rc=%d stdout=%q stderr=%q", rc, stdout.String(), stderr.String())
	}
	fence, err := os.ReadFile(filepath.Join(repo, ".fak", "alpha-fence.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, leak := range []string{"slippage", "sizing", filepath.ToSlash(alpha)} {
		if strings.Contains(string(fence), leak) || strings.Contains(stdout.String(), "slippage") {
			t.Fatalf("fence or its build output discloses %q:\n%s\n%s", leak, fence, stdout.String())
		}
	}

	stdout.Reset()
	if rc := runPublicScrub(&stdout, &stderr, []string{"audit-tree", "--root", repo}); rc != 0 {
		t.Fatalf("clean tree refused (the shared license line must not be fenced): rc=%d %s", rc, stdout.String())
	}

	publicScrubWriteFile(t, repo, "docs/howto.md", "intro\n  "+strings.ToUpper(private)+"\n")
	publicScrubGitRun(t, repo, "add", "docs/howto.md")
	stdout.Reset()
	if rc := runPublicScrub(&stdout, &stderr, []string{"audit-staged", "--root", repo}); rc != 1 {
		t.Fatalf("staged copy of a private line not refused: rc=%d stdout=%q stderr=%q", rc, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "docs/howto.md:2") || !strings.Contains(stdout.String(), "alpha fence") {
		t.Fatalf("refusal does not locate the leak: %s", stdout.String())
	}
	if strings.Contains(strings.ToLower(stdout.String()), "slippage") {
		t.Fatalf("refusal repeats the private text: %s", stdout.String())
	}
}

func TestPublicScrubFenceRefusesWhenNothingWouldBeProtected(t *testing.T) {
	repo := t.TempDir()
	publicScrubWriteFile(t, repo, "notes.md", "short\n")
	var stdout, stderr bytes.Buffer
	if rc := runPublicScrub(&stdout, &stderr, []string{"fence", "--root", repo, filepath.Join(repo, "notes.md")}); rc != 2 {
		t.Fatalf("empty fence rc=%d, want 2 (stderr=%q)", rc, stderr.String())
	}
	if _, err := os.Stat(filepath.Join(repo, ".fak", "alpha-fence.json")); err == nil {
		t.Fatal("an empty fence was written")
	}
}

func TestPublicScrubFenceKeepsItselfAndItsInputsOutOfTheRepository(t *testing.T) {
	const private = "Rank every candidate by realized slippage before sizing the order."
	repo := t.TempDir()
	publicScrubGitRun(t, repo, "init", "-q", "-b", "main")
	publicScrubGitRun(t, repo, "config", "user.email", "t@t")
	publicScrubGitRun(t, repo, "config", "user.name", "t")
	publicScrubWriteFile(t, repo, "README.md", "hello\n")
	publicScrubWriteFile(t, repo, "skills/tracked.md", private+"\n")
	publicScrubGitRun(t, repo, "add", "README.md", "skills/tracked.md")
	publicScrubGitRun(t, repo, "commit", "-qm", "seed")

	var stdout, stderr bytes.Buffer
	tracked := filepath.Join(repo, "skills", "tracked.md")
	if rc := runPublicScrub(&stdout, &stderr, []string{"fence", "--root", repo, tracked}); rc != 2 || !strings.Contains(stderr.String(), "is tracked") {
		t.Fatalf("fencing a tracked file must be refused: rc=%d stderr=%q", rc, stderr.String())
	}

	alpha := t.TempDir()
	publicScrubWriteFile(t, alpha, "sizing.md", "A different private rule about hedging the book overnight.\n")
	publicScrubWriteFile(t, alpha, "notes.bin", "utf16\x00text that cannot be fenced as written here\n")
	stdout.Reset()
	stderr.Reset()
	if rc := runPublicScrub(&stdout, &stderr, []string{"fence", "--root", repo, alpha}); rc != 0 {
		t.Fatalf("fence rc=%d stdout=%q stderr=%q", rc, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "NOT fenced") || !strings.Contains(stderr.String(), "notes.bin") {
		t.Fatalf("an input that could not be fenced was not reported: %q", stderr.String())
	}
	// The repository has no ignore rule for the fence, so the builder must have added one.
	if out := publicScrubGitRunOutput(t, repo, "status", "--porcelain"); strings.Contains(out, ".fak") {
		t.Fatalf("fence is visible to git and would be committed by an add-all:\n%s", out)
	}
	publicScrubGitRun(t, repo, "add", "-f", ".fak/alpha-fence.json")
	stdout.Reset()
	if rc := runPublicScrub(&stdout, &stderr, []string{"audit-staged", "--root", repo}); rc != 1 || !strings.Contains(stdout.String(), "alpha fence published") {
		t.Fatalf("a force-added fence must be refused: rc=%d stdout=%q", rc, stdout.String())
	}
}
