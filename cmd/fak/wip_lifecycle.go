package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wiplifecycle"
)

func runWIPLifecycle(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak wip lifecycle begin --kind KIND [--root DIR] [--id ID] | end --id ID [--root DIR]")
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
	b, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak wip lifecycle: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(b))
	return 0
}

func beginAutomaticWIPLifecycle(root, kind string, stderr io.Writer) func() {
	root, err := filepath.Abs(root)
	if err != nil {
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=before kind=%s error=%v\n", kind, err)
		return func() {}
	}
	receipt, err := wiplifecycle.Begin(root, kind, "", time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=before kind=%s error=%v\n", kind, err)
		return func() {}
	}
	fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURED phase=before kind=%s operation=%s artifact=%s\n", kind, receipt.OperationID, receipt.Before.Artifact)
	var once sync.Once
	return func() {
		once.Do(func() {
			finished, finishErr := wiplifecycle.Finish(root, receipt.OperationID, time.Now())
			if finishErr != nil {
				fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURE_FAILED phase=after kind=%s operation=%s error=%v\n", kind, receipt.OperationID, finishErr)
				return
			}
			fmt.Fprintf(stderr, "WIP_LIFECYCLE_CAPTURED phase=after kind=%s operation=%s artifact=%s receipt=%s\n", kind, finished.OperationID, finished.After.Artifact, finished.ReceiptPath)
		})
	}
}
