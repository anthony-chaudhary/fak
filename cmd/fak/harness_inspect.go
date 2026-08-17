package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/harnessinspect"
)

func runHarnessInspect(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lockPath := fs.String("lock", "", "resolved harness lock to verify and inspect")
	jsonView := fs.Bool("json", false, "emit machine-readable inspection JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *lockPath == "" {
		fmt.Fprintln(stderr, "fak harness inspect: --lock is required")
		return 2
	}
	lock, err := readHarnessPreviewLock(*lockPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness inspect: lock: %v\n", err)
		return 1
	}
	report := harnessinspect.Inspect(*lock, *lockPath)
	if *jsonView {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak harness inspect: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, harnessinspect.Render(report))
	return 0
}
