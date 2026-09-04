package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

func runWIPInventoryReconcile(args []string, stdout, stderr io.Writer) int {
	var filteredArgs []string
	for _, arg := range args {
		if arg == "reconcile" {
			continue
		}
		filteredArgs = append(filteredArgs, arg)
	}

	fs := flag.NewFlagSet("wip inventory --reconcile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit schema-versioned JSON")
	repoPath := fs.String("repo", "", "repository path")
	rootPath := fs.String("root", "", "repository root (alias for --repo)")
	reconcileFlag := fs.Bool("reconcile", false, "reconcile raw surfaces, logical units, and unresolved join debt")
	_ = reconcileFlag

	if err := fs.Parse(filteredArgs); err != nil {
		return 2
	}

	target := *repoPath
	if target == "" && *rootPath != "" {
		target = *rootPath
	}
	if target == "" {
		target = "."
	}

	abs, err := filepath.Abs(target)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip inventory --reconcile: %v\n", err)
		return 1
	}

	runner := wipinventory.GitRunner{}
	rep := wipinventory.Collect(abs, time.Now(), runner)

	inputs := wipinventory.InventoryInputs{
		Runner: runner,
		HEAD:   rep.HEAD,
	}

	for _, cp := range rep.Checkpoints {
		sessionID := strings.TrimPrefix(cp.Ref, "refs/fak/wip/")
		inputs.Checkpoints = append(inputs.Checkpoints, wipinventory.CheckpointWIPBinding{
			CheckpointID: sessionID,
			SessionID:    sessionID,
		})
	}

	for _, wt := range rep.Worktrees {
		inputs.Worktrees = append(inputs.Worktrees, wipinventory.ManagedWorktreeBinding{
			WorktreeID: filepath.Base(wt.Path),
		})
	}

	for _, f := range rep.Main.Untracked.Samples {
		inputs.UnlinkedFiles = append(inputs.UnlinkedFiles, f)
	}

	inputs.SourceErrors = append(inputs.SourceErrors, rep.Errors...)

	ctx := context.Background()
	reconReport, err := wipinventory.ReconcileInventory(ctx, abs, inputs)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip inventory --reconcile: %v\n", err)
		return 1
	}

	if *jsonOut {
		b, err := reconReport.JSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak wip inventory --reconcile: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprint(stdout, reconReport.SummaryText())
	}

	return 0
}
