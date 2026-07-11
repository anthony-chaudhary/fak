package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

// hooks_popup_scan_test.go — exit-contract tests for `fak hooks popup-scan`, the
// push-seam desktop-popup gate (reason: DESKTOP_POPUP_REGRESSION). The contract
// mirrors the sibling push gates: 0 = clean, 1 = a popup regression is present
// (block), 2 = could-not-run (fail-open so a missing verifier never wedges the
// trunk). Each case drives the real runHooks dispatch so the subcommand wiring in
// hooks.go is covered too. These use a real temp git repo (ScanTree reads
// `git ls-files`) and skip when git is unavailable, matching hooks_test.go.

func TestRunHooksPopupScan_cleanTreeExit0(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	repo := newRepoWith(t, map[string]string{"src/x.go": "package x\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"popup-scan", "--root", repo})
	if code != 0 {
		t.Fatalf("a tree with no console-tool spawn should pass (0), got %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}

func TestRunHooksPopupScan_unsuppressedSpawnExit1(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	// A bare Start-Process with no -WindowStyle Hidden / -NoNewWindow is exactly
	// the shape that flashes a desktop window from background automation, so the
	// gate must block it (exit 1) and name the finding on stdout.
	repo := newRepoWith(t, map[string]string{"watch.ps1": "Start-Process notepad\n"})
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"popup-scan", "--root", repo})
	if code != 1 {
		t.Fatalf("an unsuppressed Start-Process should block (1), got %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "popup spawn:") {
		t.Errorf("blocking run should print the finding on stdout, got %q", out.String())
	}
}

func TestRunHooksPopupScan_notAGitRepoExit2(t *testing.T) {
	if testing.Short() {
		t.Skip("-short")
	}
	// resolveRoot returns an explicit --root verbatim; a path git cannot chdir into
	// makes ScanTree's `git ls-files` fail, which must fail-open (2), never block.
	missing := filepath.Join(t.TempDir(), "does-not-exist")
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"popup-scan", "--root", missing})
	if code != 2 {
		t.Fatalf("an unscannable root must fail-open (2), got %d; stdout=%q stderr=%q", code, out.String(), errb.String())
	}
}
