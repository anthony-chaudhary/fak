package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/goalregistry"
	"github.com/anthony-chaudhary/fak/internal/goalsync"
)

func runGoalSync(stdout, stderr io.Writer, args []string) int {
	action := "push"
	flagArgs := args

	if len(args) > 0 {
		switch args[0] {
		case "push", "pull", "status":
			action = args[0]
			flagArgs = args[1:]
		default:
			if !strings.HasPrefix(args[0], "-") {
				fmt.Fprintf(stderr, "fak goal sync: unknown action %q (expected push, pull, or status)\n", args[0])
				return 2
			}
		}
	}

	fs := flag.NewFlagSet("goal sync", flag.ContinueOnError)
	fs.SetOutput(stderr)

	target := fs.String("target", "", "target directory in fak-private for synced goal artifacts")
	registry := fs.String("registry", goalregistry.DefaultPath(), "canonical goal registry JSON path")
	goalPark := fs.String("goal-park", "", "goal-park directory (default: .fak/goal-park)")
	commit := fs.Bool("commit", false, "git commit changes in target repository")
	gitPush := fs.Bool("push", false, "git push changes in target repository")
	force := fs.Bool("force", false, "force overwrite of newer local files on pull")
	dryRun := fs.Bool("dry-run", false, "dry run without disk modifications")
	asJSON := fs.Bool("json", false, "output report or status in JSON format")

	if err := fs.Parse(flagArgs); err != nil {
		return 2
	}

	if fs.NArg() > 0 {
		sub := fs.Arg(0)
		switch sub {
		case "push", "pull", "status":
			action = sub
		default:
			fmt.Fprintf(stderr, "fak goal sync: unexpected argument %q\n", sub)
			return 2
		}
	}

	wsRoot := repoRoot()
	targetDir := *target
	if targetDir == "" {
		targetDir = goalsync.DefaultTarget(wsRoot)
	}
	registryPath := *registry
	parkDir := *goalPark
	if parkDir == "" {
		parkDir = filepath.Join(wsRoot, ".fak", "goal-park")
	}

	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")

	switch action {
	case "status":
		status, err := goalsync.Status(wsRoot, targetDir, registryPath, parkDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak goal sync status: %v\n", err)
			return 1
		}
		if *asJSON {
			_ = enc.Encode(status)
			return 0
		}
		fmt.Fprintf(stdout, "Goal Sync Status\n")
		fmt.Fprintf(stdout, "  Target: %s\n", status.TargetDir)
		fmt.Fprintf(stdout, "  Total: %d | In Sync: %d | Push: %d | Pull: %d\n",
			status.TotalCount, status.InSyncCount, status.PushCount, status.PullCount)
		for _, it := range status.Items {
			fmt.Fprintf(stdout, "  [%-8s] %-35s (%s)\n", it.Action, it.RelPath, it.Reason)
		}
		return 0

	case "push":
		report, err := goalsync.Push(wsRoot, targetDir, registryPath, parkDir, *commit, *gitPush, *dryRun)
		if err != nil {
			if *asJSON {
				_ = enc.Encode(report)
			} else {
				fmt.Fprintf(stderr, "fak goal sync push: %v\n", err)
			}
			return 1
		}
		if *asJSON {
			_ = enc.Encode(report)
			return 0
		}
		fmt.Fprintf(stdout, "Goal Sync Push\n")
		fmt.Fprintf(stdout, "  Target: %s\n", report.TargetDir)
		fmt.Fprintf(stdout, "  Transferred: %d | Skipped: %d\n", len(report.Transferred), len(report.Skipped))
		for _, p := range report.Transferred {
			fmt.Fprintf(stdout, "  + %s\n", p)
		}
		if report.Committed {
			fmt.Fprintf(stdout, "  Committed: true\n")
		}
		if report.Pushed {
			fmt.Fprintf(stdout, "  Pushed: true\n")
		}
		return 0

	case "pull":
		report, err := goalsync.Pull(wsRoot, targetDir, registryPath, parkDir, *force, *dryRun)
		if err != nil {
			if *asJSON {
				_ = enc.Encode(report)
			} else {
				fmt.Fprintf(stderr, "fak goal sync pull: %v\n", err)
			}
			return 1
		}
		if *asJSON {
			_ = enc.Encode(report)
			return 0
		}
		fmt.Fprintf(stdout, "Goal Sync Pull\n")
		fmt.Fprintf(stdout, "  Target: %s\n", report.TargetDir)
		fmt.Fprintf(stdout, "  Transferred: %d | Skipped: %d\n", len(report.Transferred), len(report.Skipped))
		for _, p := range report.Transferred {
			fmt.Fprintf(stdout, "  <- %s\n", p)
		}
		return 0

	default:
		fmt.Fprintf(stderr, "fak goal sync: unknown action %q\n", action)
		return 2
	}
}
