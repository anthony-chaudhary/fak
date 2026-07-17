package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// hooks_popup_scan_test.go — exit-contract tests for `fak hooks popup-scan`, the
// push-seam desktop-popup gate (reason: DESKTOP_POPUP_REGRESSION). The contract
// mirrors the sibling push gates: 0 = clean, 1 = a popup regression is present
// (block), 2 = could-not-run (fail-open so a missing verifier never wedges the
// trunk).
//
// #5145: the gate scans the PUSH RANGE (`<base>..HEAD`), not the whole worktree, so
// an untracked peer WIP file in a shared checkout — which can never reach the trunk —
// no longer trips it. These build a real temp git repo with a base commit and a HEAD
// commit and drive the range with an explicit --base, and skip when git is absent.

// popupRangeRepo builds a temp git repo with a "base" commit (baseFiles) and a HEAD
// commit (headFiles), returning the repo path and the base commit sha so a test can
// scan exactly headFiles' push range via `--base <sha>`.
func popupRangeRepo(t *testing.T, baseFiles, headFiles map[string]string) (string, string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	gitHook(t, repo, "init", "-q", "-b", "main")
	gitHook(t, repo, "config", "user.email", "t@t")
	gitHook(t, repo, "config", "user.name", "t")
	commit := func(files map[string]string, msg string) {
		if len(files) == 0 {
			gitHook(t, repo, "commit", "-q", "--allow-empty", "-m", msg)
			return
		}
		for p, content := range files {
			full := filepath.Join(repo, filepath.FromSlash(p))
			_ = os.MkdirAll(filepath.Dir(full), 0o755)
			if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
			gitHook(t, repo, "add", "--", p)
		}
		gitHook(t, repo, "commit", "-q", "-m", msg)
	}
	commit(baseFiles, "base")
	baseSha := strings.TrimSpace(gitCapture(t, repo, "rev-parse", "HEAD"))
	commit(headFiles, "head")
	return repo, baseSha
}

func gitCapture(t *testing.T, repo string, args ...string) string {
	t.Helper()
	c := exec.Command("git", append([]string{"-C", repo}, args...)...)
	c.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	out, err := c.CombinedOutput()
	if err != nil {
		t.Skipf("git %v: %s", args, out)
	}
	return string(out)
}

// unsuppressedGoExec is a cmd/fak/dispatch* Go helper whose exec.Command reaches
// .Output() without windowgate.ConfigureBackgroundCommand — a hard UNSUPPRESSED_GO_EXEC.
const unsuppressedGoExec = "package main\n\nimport \"os/exec\"\n\nfunc peerWip() {\n\tcmd := exec.Command(\"git\", \"status\")\n\tbuf, _ := cmd.Output()\n\t_ = buf\n}\n"

func TestRunHooksPopupScan_cleanRangeExit0(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo, base := popupRangeRepo(t,
		map[string]string{"src/a.go": "package x\n"},
		map[string]string{"src/b.go": "package x\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"popup-scan", "--root", repo, "--base", base})
	if code != 0 {
		t.Fatalf("a push range with no console-tool spawn should pass (0), got %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}

func TestRunHooksPopupScan_unsuppressedSpawnInRangeExit1(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	// A bare Start-Process with no -WindowStyle Hidden / -NoNewWindow, committed IN the
	// push range, is exactly the shape that flashes a desktop window from background
	// automation, so the real gate must still block it (exit 1) — no weakening (#5145).
	repo, base := popupRangeRepo(t,
		map[string]string{"src/a.go": "package x\n"},
		map[string]string{"watch.ps1": "Start-Process notepad\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"popup-scan", "--root", repo, "--base", base})
	if code != 1 {
		t.Fatalf("an unsuppressed Start-Process in the push range should block (1), got %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "popup spawn:") {
		t.Errorf("blocking run should print the finding on stdout, got %q", out.String())
	}
}

// #5145 regression: an untracked peer sibling (NOT in the push range) with a real
// UNSUPPRESSED_GO_EXEC must NOT block a push whose range is clean. The whole-tree
// ScanTree flags the sibling; the range-scoped gate does not — that is the fix.
func TestRunHooksPopupScan_untrackedPeerPopupDoesNotBlockCleanRange(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo, base := popupRangeRepo(t,
		map[string]string{"src/a.go": "package x\n"},
		map[string]string{"src/b.go": "package x\n"})
	// Peer WIP: untracked, non-ignored, with a genuine hard popup violation on disk.
	peer := filepath.Join(repo, "cmd", "fak", "dispatch_peerwip.go")
	if err := os.MkdirAll(filepath.Dir(peer), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(peer, []byte(unsuppressedGoExec), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"popup-scan", "--root", repo, "--base", base})
	if code != 0 {
		t.Fatalf("an untracked peer popup outside the push range must not block a clean range (want 0), got %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}

func TestRunHooksPopupScan_notAGitRepoExit2(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	// resolveRoot returns an explicit --root verbatim; a path git cannot chdir into
	// makes the range resolution fail, which must fail-open (2), never block.
	missing := filepath.Join(t.TempDir(), "does-not-exist")
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"popup-scan", "--root", missing})
	if code != 2 {
		t.Fatalf("an unscannable root must fail-open (2), got %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}

// An unresolvable base (no such ref) must fail-open (2), never wedge the trunk.
func TestRunHooksPopupScan_unresolvableBaseFailsOpen(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo, _ := popupRangeRepo(t,
		map[string]string{"src/a.go": "package x\n"},
		map[string]string{"src/b.go": "package x\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"popup-scan", "--root", repo, "--base", "origin/does-not-exist"})
	if code != 2 {
		t.Fatalf("an unresolvable base must fail-open (2), got %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}
