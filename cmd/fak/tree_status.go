package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/treestatus"
)

func cmdTree(argv []string) {
	os.Exit(runTree(os.Stdout, os.Stderr, argv))
}

func runTree(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		treeUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "status":
		return runTreeStatus(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		treeUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak tree: unknown subcommand %q (want status)\n", argv[0])
		treeUsage(stderr)
		return 2
	}
}

func treeUsage(w io.Writer) {
	fmt.Fprint(w, `fak tree — working tree status and peer WIP isolation

Usage:
  fak tree status [--mine <paths>] [--lane <name>] [--json]

Examples:
  fak tree status --lane gateway
  fak tree status --mine internal/gateway,cmd/fak
  fak tree status --json
`)
}

func runTreeStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tree status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var minePaths pathList
	fs.Var(&minePaths, "mine", "repo-relative path or directory to treat as owned (repeatable)")
	laneFlag := fs.String("lane", "", "target lane name to treat as owned")
	dirFlag := fs.String("dir", "", "repo root directory (default: discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit result as structured JSON")

	if !parseFlags(fs, argv) {
		return 2
	}

	root := resolveRoot(*dirFlag)

	rep, err := treestatus.Collect(root, treestatus.Options{
		Lane: *laneFlag,
		Mine: []string(minePaths),
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak tree status: %v\n", err)
		return 1
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak tree status: encode json: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "branch %s at %s (%d total dirty files, %dms)\n", rep.Branch, rep.Head, rep.TotalDirty, rep.ElapsedMS)
	if rep.MergeInProgress {
		fmt.Fprintln(stderr, "  WARNING: MERGE_HEAD in progress — merge conflicts must be resolved before committing")
	}
	if len(rep.LockFiles) > 0 {
		fmt.Fprintf(stderr, "  LOCKS: %s\n", strings.Join(rep.LockFiles, ", "))
	}
	if rep.HasConflicts {
		fmt.Fprintf(stderr, "  CONFLICTS (%d):\n", len(rep.ConflictPaths))
		for _, cp := range rep.ConflictPaths {
			fmt.Fprintf(stderr, "    %s\n", cp)
		}
	}

	if *laneFlag != "" || len(minePaths) > 0 {
		fmt.Fprintf(stdout, "\nOwned paths (%d):\n", rep.OwnedCount)
		for _, p := range rep.OwnedPaths {
			fmt.Fprintf(stdout, "  %s  %-14s %s\n", p.Status, "["+p.Lane+"]", p.Path)
		}
		fmt.Fprintf(stdout, "\nPeer WIP paths (%d) — DO NOT COMMIT:\n", rep.PeerWIPCount)
		for _, p := range rep.PeerWIPPaths {
			fmt.Fprintf(stdout, "  %s  %-14s %s\n", p.Status, "["+p.Lane+"]", p.Path)
		}
	} else {
		fmt.Fprintln(stdout, "\nLanes with dirty files:")
		for lane, paths := range rep.LaneGroups {
			fmt.Fprintf(stdout, "  lane %-14s (%d file(s))\n", lane, len(paths))
		}
		if rep.UnclassifiedCount > 0 {
			fmt.Fprintf(stdout, "\nDirty paths (%d):\n", rep.UnclassifiedCount)
			for _, p := range rep.UnclassifiedPaths {
				lane := p.Lane
				if lane == "" {
					lane = "no-lane"
				}
				fmt.Fprintf(stdout, "  %s  %-14s %s\n", p.Status, "["+lane+"]", p.Path)
			}
		}
	}
	return 0
}
