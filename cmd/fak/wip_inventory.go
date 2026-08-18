package main

import (
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

func runWIPInventory(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("wip inventory", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit schema-versioned JSON")
	root := fs.String("root", ".", "repository root")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	abs, err := filepath.Abs(*root)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip inventory: %v\n", err)
		return 1
	}
	rep := wipinventory.Collect(abs, time.Now(), wipinventory.GitRunner{})
	if *jsonOut {
		b, err := rep.JSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak wip inventory: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintf(stdout, "WIP INVENTORY — source WIP and generated files are separate populations\n")
		fmt.Fprintf(stdout, "repo %s @ %s\n", rep.Repository, shortSHA(rep.HEAD))
		fmt.Fprintf(stdout, "main: tracked=%d untracked=%d\n", rep.Main.Tracked.Count, rep.Main.Untracked.Count)
		fmt.Fprintf(stdout, "ignored/generated: %d (not source WIP)\n", rep.Ignored.Count)
		wtTracked, wtUntracked := 0, 0
		for _, wt := range rep.Worktrees {
			wtTracked += wt.Tracked.Count
			wtUntracked += wt.Untracked.Count
		}
		fmt.Fprintf(stdout, "registered worker worktrees: %d tracked=%d untracked=%d\n", len(rep.Worktrees), wtTracked, wtUntracked)
		fmt.Fprintf(stdout, "stale worker residue: %d\n", len(rep.StaleWorkers))
		fmt.Fprintf(stdout, "WIP checkpoints: %d\n", len(rep.Checkpoints))
		fmt.Fprintf(stdout, "visibility: gitignore=%s exclude=%s sparse=%t hidden-index=%d\n", shortSHA(rep.IgnoreInputs.GitignoreHash), shortSHA(rep.IgnoreInputs.ExcludeHash), rep.IgnoreInputs.Sparse, rep.IgnoreInputs.HiddenIndex)
		if len(rep.Errors) > 0 {
			fmt.Fprintf(stdout, "unknown/errors: %d (inspect --json; counts with errors are not proof of zero)\n", len(rep.Errors))
		}
	}
	if len(rep.Errors) > 0 {
		return 1
	}
	return 0
}

func shortSHA(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	if s == "" {
		return "unknown"
	}
	return s
}
