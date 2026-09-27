package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
	"github.com/anthony-chaudhary/fak/internal/wiplifecycle"
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
	git := wipinventory.GitRunner{}
	finish, _, _ := beginAutomaticWIPLifecycleWithRunner(root, kind, stderr, git, git, wipinventory.Options{})
	return finish
}

// boundedLifecycle ties an automatic lifecycle bracket to the deadline of the
// mutation it brackets.
type boundedLifecycle struct {
	// ctx is the mutation's shared deadline; the after capture spends what is left.
	ctx context.Context
	// beforeBudget caps the before capture so a large fleet cannot starve the
	// mutation's own probes of the shared deadline.
	beforeBudget time.Duration
	// focus is the checkout the mutation targets. The before capture is complete
	// once its own state is known; other failures are recorded as advisory.
	focus string
}

func beginAutomaticWIPLifecycleWithGit(root, kind string, stderr io.Writer, bounds boundedLifecycle) (func(), wiplifecycle.Receipt, error) {
	beforeCtx, cancel := context.WithTimeout(bounds.ctx, bounds.beforeBudget)
	defer cancel()
	return beginAutomaticWIPLifecycleWithRunner(root, kind, stderr,
		wipinventory.DeadlineRunner{Ctx: beforeCtx},
		wipinventory.DeadlineRunner{Ctx: bounds.ctx},
		wipinventory.Options{Focus: bounds.focus})
}

func beginAutomaticWIPLifecycleWithRunner(root, kind string, stderr io.Writer, before, after wipinventory.Runner, opts wipinventory.Options) (func(), wiplifecycle.Receipt, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=before kind=%s error=%v\n", kind, err)
		return func() {}, wiplifecycle.Receipt{}, err
	}
	receipt, err := wiplifecycle.BeginWithOptions(root, kind, "", time.Now(), before, opts)
	if err != nil {
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=before kind=%s error=%v\n", kind, err)
		return func() {}, receipt, err
	}
	if !receipt.Before.Known {
		errDetail := receipt.Before.Error
		if errDetail == "" {
			errDetail = "before inventory is incomplete or unknown"
		}
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=before kind=%s error=%s\n", kind, errDetail)
	} else {
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURED phase=before kind=%s operation=%s artifact=%s\n", kind, receipt.OperationID, receipt.Before.Artifact)
	}
	if receipt.Before.Advisory != "" {
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_ADVISORY phase=before kind=%s focus=%s error=%s\n", kind, receipt.Before.Focus, receipt.Before.Advisory)
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			finished, finishErr := wiplifecycle.FinishWithRunner(root, receipt.OperationID, time.Now(), after)
			if finishErr != nil {
				fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=after kind=%s operation=%s error=%v\n", kind, receipt.OperationID, finishErr)
				return
			}
			fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURED phase=after kind=%s operation=%s artifact=%s receipt=%s\n", kind, finished.OperationID, finished.After.Artifact, finished.ReceiptPath)
		})
	}, receipt, nil
}
