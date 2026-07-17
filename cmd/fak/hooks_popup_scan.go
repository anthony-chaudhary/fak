package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// runHooksPopupScan is the desktop-popup push gate (reason: DESKTOP_POPUP_REGRESSION).
//
// A console-tool child (git/gh/powershell) spawned without the Windows no-window hook
// flashes a visible console window every time background automation runs it. The scan in
// internal/windowgate already asserts a clean tracked tree (TestTrackedTreeHasNoPopups),
// but that test only runs in a full `go test ./...` the fleet build-guard path skips — so
// unsuppressed spawns were reaching main and flashing on the operator's desktop. This
// subcommand runs the SAME ScanTree at the push seam, BELOW the agent layer, so the floor
// binds Claude Code, Codex, and a human alike.
//
// It blocks on BOTH hard violations and the advisory watchlist: a watchlist row is a
// console-tool spawn that has not yet opted into suppression, which is exactly the shape
// that regressed. Exit contract mirrors the sibling push gates (runHooksPrePush): 0 =
// clean, 1 = a popup regression is present (block), 2 = could-not-run (fail-open; the
// shell then allows the push — a missing verifier must never wedge the trunk).
//
// SCOPE (#5145): the gate scans ONLY the push range `origin/<trunk>..HEAD` — the files
// that can actually reach the trunk — NOT the whole worktree. On a shared multi-session
// checkout an untracked peer WIP file with an unsuppressed spawn cannot regress the trunk
// (it never reaches it), so gating a push on it is a false positive by construction. The
// old whole-tree ScanTree lives on for TestTrackedTreeHasNoPopups; here we scan the range.
func runHooksPopupScan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks popup-scan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	base := fs.String("base", "", "override the base ref the push range is computed against (default: origin/<branch>→origin/main→origin/master)")
	if !parseFlags(fs, argv) {
		return 2
	}

	r := resolveRoot(*root)
	if r == "" {
		fmt.Fprintln(stderr, "fak hooks popup-scan: not in a git repo (or git unavailable); popup gate skipped")
		return 2
	}

	baseRef := strings.TrimSpace(*base)
	if baseRef == "" {
		baseRef = resolvePrepushBase(r)
	}
	rangeFiles, err := popupPushRangeFiles(r, baseRef)
	if err != nil {
		// Unresolvable/empty range (fresh clone, detached HEAD, git hiccup) → fail-open:
		// a missing verifier must never wedge the trunk.
		fmt.Fprintf(stderr, "fak hooks popup-scan: could not resolve push range %s..HEAD: %v; popup gate skipped\n", baseRef, err)
		return 2
	}

	rep, err := windowgate.ScanFiles(r, rangeFiles)
	if err != nil {
		fmt.Fprintf(stderr, "fak hooks popup-scan: scan failed: %v; popup gate skipped\n", err)
		return 2
	}

	var findings []string
	findings = append(findings, rep.PSInstallers...)
	findings = append(findings, rep.PSStartProcesses...)
	findings = append(findings, rep.PySpawns...)
	findings = append(findings, rep.GoExecs...)
	findings = append(findings, rep.PyCandidates...)
	findings = append(findings, rep.GoCandidates...)
	if len(findings) == 0 {
		return 0
	}
	for _, v := range findings {
		fmt.Fprintf(stdout, "  popup spawn: %s\n", v)
	}
	return 1
}

// popupPushRangeFiles returns the repo-relative, slash-separated .ps1/.py/.go files the
// commits in base..HEAD add or change and that STILL EXIST on disk in repo r (deletions
// are skipped — a removed file can flash nothing). It reads git's committed range only via
// three-dot `git diff` (the merge-base range = exactly what the push adds), never listing
// untracked working-tree siblings, which is the whole point of the #5145 fix. An
// unresolvable base (fresh/detached clone) returns an error so the caller fails open.
func popupPushRangeFiles(r, base string) ([]string, error) {
	if _, err := gitOut(r, "rev-parse", "--verify", "--quiet", base+"^{commit}"); err != nil {
		return nil, fmt.Errorf("base %q unresolvable", base)
	}
	changed, err := gitChangedFilesRange(r, base, "HEAD", ".ps1", ".py", ".go")
	if err != nil {
		return nil, err
	}
	var files []string
	for _, ln := range changed {
		if _, statErr := os.Stat(filepath.Join(r, filepath.FromSlash(ln))); statErr != nil {
			continue // deleted in the range → nothing on disk to flash
		}
		files = append(files, ln)
	}
	return files, nil
}
