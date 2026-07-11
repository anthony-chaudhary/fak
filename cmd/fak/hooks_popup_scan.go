package main

import (
	"flag"
	"fmt"
	"io"

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
func runHooksPopupScan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks popup-scan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	if !parseFlags(fs, argv) {
		return 2
	}

	r := resolveRoot(*root)
	if r == "" {
		fmt.Fprintln(stderr, "fak hooks popup-scan: not in a git repo (or git unavailable); popup gate skipped")
		return 2
	}

	rep, err := windowgate.ScanTree(r)
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
