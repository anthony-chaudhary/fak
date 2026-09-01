package main

import (
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
	"github.com/anthony-chaudhary/fak/internal/wiplifecycle"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func runWIPLifecycle(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak wip lifecycle begin --kind KIND [--root DIR] [--id ID] | end --id ID [--root DIR] | list [--root DIR] [--json]")
		return 2
	}
	switch args[0] {
	case "begin":
		fs := flag.NewFlagSet("fak wip lifecycle begin", flag.ContinueOnError)
		fs.SetOutput(stderr)
		root := fs.String("root", ".", "repository root")
		rootShort := fs.String("C", "", "repository root (shorthand)")
		kind := fs.String("kind", "", "lifecycle operation class")
		id := fs.String("id", "", "stable operation identity (generated when omitted)")
		if err := fs.Parse(args[1:]); err != nil || fs.NArg() != 0 {
			return 2
		}
		if *rootShort != "" {
			*root = *rootShort
		}
		receipt, err := wiplifecycle.Begin(*root, *kind, *id, time.Now())
		if err != nil {
			fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=before kind=%s error=%v\n", *kind, err)
			return 1
		}
		return emitWIPLifecycle(stdout, stderr, receipt)
	case "list":
		fs := flag.NewFlagSet("fak wip lifecycle list", flag.ContinueOnError)
		fs.SetOutput(stderr)
		root := fs.String("root", ".", "repository root")
		rootShort := fs.String("C", "", "repository root (shorthand)")
		jsonOut := fs.Bool("json", false, "emit JSON")
		if err := fs.Parse(args[1:]); err != nil || fs.NArg() != 0 {
			return 2
		}
		if *rootShort != "" {
			*root = *rootShort
		}
		result, err := wiplifecycle.ListWithDiagnostics(*root)
		if err != nil {
			fmt.Fprintf(stderr, "fak wip lifecycle list: %v\n", err)
			return 1
		}
		if *jsonOut {
			return encodeJSONOrFail(stdout, stderr, result, "fak wip lifecycle list")
		}
		for _, diagnostic := range result.Diagnostics {
			fmt.Fprintf(stderr, "WIP_LIFECYCLE_%s operation_id=%s path=%s error=%q\n", diagnostic.Code, diagnostic.OperationID, diagnostic.Path, diagnostic.Error)
		}
		if len(result.Receipts) == 0 {
			fmt.Fprintln(stdout, "no WIP lifecycle receipts")
			return 0
		}
		for _, receipt := range result.Receipts {
			state, when := "OPEN", receipt.StartedAt
			if receipt.FinishedAt != "" {
				state, when = "FINISHED", receipt.FinishedAt
			}
			fmt.Fprintf(stdout, "%s\t%s\t%s\t%s\t%s\n", state, receipt.Kind, receipt.OperationID, when, receipt.ReceiptPath)
		}
		return 0
	case "end":
		fs := flag.NewFlagSet("fak wip lifecycle end", flag.ContinueOnError)
		fs.SetOutput(stderr)
		root := fs.String("root", ".", "repository root")
		rootShort := fs.String("C", "", "repository root (shorthand)")
		id := fs.String("id", "", "operation identity returned by begin")
		if err := fs.Parse(args[1:]); err != nil || fs.NArg() != 0 || strings.TrimSpace(*id) == "" {
			return 2
		}
		if *rootShort != "" {
			*root = *rootShort
		}
		receipt, err := wiplifecycle.Finish(*root, *id, time.Now())
		if err != nil {
			fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=after id=%s error=%v\n", *id, err)
			return 1
		}
		return emitWIPLifecycle(stdout, stderr, receipt)
	default:
		fmt.Fprintf(stderr, "unknown wip lifecycle verb %q\n", args[0])
		return 2
	}
}

func emitWIPLifecycle(stdout, stderr io.Writer, receipt wiplifecycle.Receipt) int {
	if rc := encodeJSONOrFailPrefixed(stdout, stderr, receipt, "fak wip lifecycle"); rc != 0 {
		return rc
	}
	return 0
}

func beginAutomaticWIPLifecycle(root, kind string, stderr io.Writer) func() {
	return beginAutomaticWIPLifecycleWithRunner(root, kind, stderr, wipinventory.GitRunner{})
}

type boundedLifecycleGitRunner struct {
	run workerworktree.GitRunner
}

func (r boundedLifecycleGitRunner) Run(root string, args ...string) ([]byte, error) {
	rc, out := r.run(root, args)
	if rc != 0 {
		return nil, fmt.Errorf("git %s: %s", strings.Join(args, " "), strings.TrimSpace(out))
	}
	return []byte(out), nil
}

func beginAutomaticWIPLifecycleWithGit(root, kind string, stderr io.Writer, git workerworktree.GitRunner) func() {
	return beginAutomaticWIPLifecycleWithRunner(root, kind, stderr, boundedLifecycleGitRunner{run: git})
}

func beginAutomaticWIPLifecycleWithRunner(root, kind string, stderr io.Writer, runner wipinventory.Runner) func() {
	root, err := filepath.Abs(root)
	if err != nil {
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=before kind=%s error=%v\n", kind, err)
		return func() {}
	}
	receipt, err := wiplifecycle.BeginWithRunner(root, kind, "", time.Now(), runner)
	if err != nil {
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=before kind=%s error=%v\n", kind, err)
		return func() {}
	}
	fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURED phase=before kind=%s operation=%s artifact=%s\n", kind, receipt.OperationID, receipt.Before.Artifact)
	var once sync.Once
	return func() {
		once.Do(func() {
			finished, finishErr := wiplifecycle.FinishWithRunner(root, receipt.OperationID, time.Now(), runner)
			if finishErr != nil {
				fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=after kind=%s operation=%s error=%v\n", kind, receipt.OperationID, finishErr)
				return
			}
			fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURED phase=after kind=%s operation=%s artifact=%s receipt=%s\n", kind, finished.OperationID, finished.After.Artifact, finished.ReceiptPath)
		})
	}
}
