package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// runSessionObserve turns the compact-audit measurement engine into a default user
// surface. It chooses a bounded recent window and the current workspace; every choice
// remains explicit in the output and overrideable for automation.
func runSessionObserve(stdout, stderr io.Writer, argv []string) int {
	return runSessionObserveAt(stdout, stderr, argv, time.Now, os.Getwd)
}

func runSessionObserveAt(stdout, stderr io.Writer, argv []string, now func() time.Time, getwd func() (string, error)) int {
	fs := flag.NewFlagSet("session observe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", defaultCodexSessionsRoot(), "active Codex rollout corpus root")
	var additionalRoots stringListFlag
	fs.Var(&additionalRoots, "also-root", "additional Codex rollout root to merge (repeatable)")
	days := fs.Int("days", 4, "calendar days to show, including today")
	cwd := fs.String("cwd", "", "workspace path filter (default: current directory)")
	allWorkspaces := fs.Bool("all-workspaces", false, "include every workspace in the selected profile(s)")
	asJSON := fs.Bool("json", false, "emit deterministic machine-readable data")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 || *days < 1 || *root == "" {
		fmt.Fprintln(stderr, "fak session observe: --days must be >= 1 and --root must not be empty")
		return 2
	}
	filter := *cwd
	if !*allWorkspaces && filter == "" {
		var err error
		filter, err = getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak session observe: current workspace: %v\n", err)
			return 1
		}
	}
	if *allWorkspaces && *cwd != "" {
		fmt.Fprintln(stderr, "fak session observe: --cwd and --all-workspaces are mutually exclusive")
		return 2
	}
	roots := append([]string{filepath.Clean(*root)}, additionalRoots...)
	for _, selected := range roots {
		if _, err := os.Stat(selected); err != nil {
			fmt.Fprintf(stderr, "fak session observe: corpus root %s: %v\n", selected, err)
			return 1
		}
	}
	localNow := now()
	start := time.Date(localNow.Year(), localNow.Month(), localNow.Day(), 0, 0, 0, 0, localNow.Location()).AddDate(0, 0, 1-*days)
	res, err := session.AuditCompactCorpus(session.CompactAuditOptions{Roots: roots, Since: start, Cwd: filter})
	if err != nil {
		fmt.Fprintf(stderr, "fak session observe: %v\n", err)
		return 1
	}
	if *asJSON {
		res = session.ScrubCompactResult(res)
		res.Sessions = nil
		if err := session.WriteCompactAuditJSON(stdout, res); err != nil {
			fmt.Fprintf(stderr, "fak session observe: %v\n", err)
			return 1
		}
		return 0
	}
	session.RenderCompactOverview(stdout, res, session.CompactOverviewOptions{
		Days: *days, Since: start, Workspace: filter, Roots: len(roots),
	})
	return 0
}
