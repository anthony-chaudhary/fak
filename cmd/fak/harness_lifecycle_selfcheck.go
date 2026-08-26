package main

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/harnessartifact"
)

func runHarnessLifecycleSelfcheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness selfcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lifecyclePath := fs.String("lifecycle-receipt", "", "model lifecycle receipt to verify")
	expectedDeclaration := fs.String("expected-declaration", "", "expected lifecycle declaration identity")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *lifecyclePath == "" {
		fmt.Fprintln(stderr, "fak harness selfcheck: --lifecycle-receipt is required")
		return 2
	}
	receipt, err := harnessartifact.ReadModelLifecycleReceipt(*lifecyclePath)
	if err == nil {
		err = harnessartifact.CheckLifecycleDeclaration(receipt, *expectedDeclaration)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak harness selfcheck: lifecycle: %s: %s\n", harnessartifact.LifecycleDiagnosticCode(err), harnessartifact.RedactLifecycleText(err.Error()))
		return 4
	}
	fmt.Fprintf(stdout, "HARNESS SELFCHECK | PASS | lifecycle=%s state=%s declaration=%s artifact=%s runtime=%s admission=%s process=%s readiness=%s stop=%s\n", receipt.Schema, receipt.State, receipt.Declaration.ID, receipt.Artifact.ID, receipt.Runtime.ID, receipt.Admission.ID, receipt.Process.ID, receipt.Readiness.ID, receipt.Stop.ID)
	return 0
}
