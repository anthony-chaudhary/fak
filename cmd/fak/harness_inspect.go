package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/harnessartifact"
	"github.com/anthony-chaudhary/fak/internal/harnessinspect"
)

func runHarnessInspect(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lockPath := fs.String("lock", "", "resolved harness lock to verify and inspect")
	jsonView := fs.Bool("json", false, "emit machine-readable inspection JSON")
	lifecyclePath := fs.String("lifecycle-receipt", "", "model lifecycle receipt to verify")
	expectedDeclaration := fs.String("expected-declaration", "", "expected lifecycle declaration identity")
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
	var lifecycle *harnessartifact.ModelLifecycleReceipt
	if *lifecyclePath != "" {
		receipt, err := harnessartifact.ReadModelLifecycleReceipt(*lifecyclePath)
		if err == nil {
			err = harnessartifact.CheckLifecycleDeclaration(receipt, *expectedDeclaration)
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak harness inspect: lifecycle: %s: %s\n", harnessartifact.LifecycleDiagnosticCode(err), harnessartifact.RedactLifecycleText(err.Error()))
			return 4
		}
		lifecycle = &receipt
	}
	if *jsonView {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		var output any = report
		if lifecycle != nil {
			output = struct {
				Harness   any                                   `json:"harness"`
				Lifecycle harnessartifact.ModelLifecycleReceipt `json:"lifecycle"`
			}{report, *lifecycle}
		}
		if err := enc.Encode(output); err != nil {
			fmt.Fprintf(stderr, "fak harness inspect: %v\n", err)
			return 1
		}
		return 0
	}
	if lifecycle != nil {
		fmt.Fprintf(stdout, "MODEL LIFECYCLE | VERIFIED | state=%s declaration=%s artifact=%s runtime=%s process=%s readiness=%s stop=%s\n", lifecycle.State, lifecycle.Declaration.ID, lifecycle.Artifact.ID, lifecycle.Runtime.ID, lifecycle.Process.ID, lifecycle.Readiness.ID, lifecycle.Stop.ID)
	}
	fmt.Fprint(stdout, harnessinspect.Render(report))
	return 0
}
