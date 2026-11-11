package main

// conceptdrift.go is the thin CLI shell over internal/conceptdrift (the port of
// the retired tools/concept_drift_watcher.py pair): it resolves the workspace,
// anchors the grandfathered scorecard script to the local repo root, and owns
// the flag surface. The exit-code contract lives in the package: 0 clean /
// recorded / simulated OK, 2 operational error, 3 coverage-debt re-opened, 4
// simulate failed to detect its own drift.

import (
	"flag"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/conceptdrift"
)

func cmdConceptDrift(argv []string) {
	os.Exit(runConceptDrift(os.Stdout, os.Stderr, argv))
}

func runConceptDrift(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("conceptdrift", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	data := fs.String("data", "", "data directory (default: "+conceptdrift.DefaultDataDirRel+")")
	baseline := fs.String("baseline", "", "baseline JSON (default: <data-dir>/"+conceptdrift.DefaultBasename+")")
	recordBaseline := fs.Bool("record-baseline", false, "run the scorecard and record the baseline")
	simulate := fs.Bool("simulate", false, "self-witness: remove the biggest rows file in a temp copy and require the drift to show")
	autoTicket := fs.Bool("auto-ticket", false, "on drift, best-effort create a positioning ticket via fak-run")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	root := resolveRoot("")
	ws := *workspace
	if ws == "" {
		if root != "" {
			ws = root
		} else {
			ws = "."
		}
	}
	opts := conceptdrift.Options{
		Workspace:      ws,
		DataDir:        *data,
		Baseline:       *baseline,
		RecordBaseline: *recordBaseline,
		Simulate:       *simulate,
		AutoTicket:     *autoTicket,
	}
	// The Python watcher anchored the scorecard script to its own install
	// location (the repo carrying tools/); the Go equivalent is the local repo
	// root. An empty answer (not in a git checkout) falls back inside Run to
	// <workspace>/tools/<scorecard>.
	if root != "" {
		opts.Scorecard = filepath.Join(root, "tools", conceptdrift.ScorecardName)
	}
	return conceptdrift.Run(stdout, stderr, opts)
}
