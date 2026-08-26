package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/shellprov"
)

// cmdShellprov is the top-level shell provenance handler. The repository's
// static main routing table wires top-level handlers separately; keeping the
// handler here lets that wiring remain a one-case integration step.
func cmdShellprov(argv []string) {
	os.Exit(runShellprov(os.Stdout, os.Stderr, argv))
}

func runShellprov(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "record" {
		fmt.Fprintln(stderr, "usage: fak shellprov record --ledger FILE --parent-pid PID --child-pid PID --child-created-ms UNIX_MS --launch-class tool|hook|worker|probe --shell-image pwsh|powershell --shell-edition core|desktop --shell-version VERSION --outcome started|succeeded|failed [--error-class CLASS] [--max-rows N] [--json]")
		return 2
	}

	fs := flag.NewFlagSet("shellprov record", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", "", "receipt JSONL destination (not stored in the receipt)")
	parentPID := fs.Int("parent-pid", 0, "fak owner process id")
	childPID := fs.Int("child-pid", 0, "created PowerShell process id")
	childCreatedMS := fs.Int64("child-created-ms", 0, "child creation time as UTC Unix milliseconds")
	launchClass := fs.String("launch-class", "", "launch class: tool, hook, worker, or probe")
	shellImage := fs.String("shell-image", "", "shell image family: pwsh or powershell (never a path)")
	shellEdition := fs.String("shell-edition", "", "PowerShell edition: core or desktop")
	shellVersion := fs.String("shell-version", "", "bounded PowerShell version")
	outcome := fs.String("outcome", "", "outcome: started, succeeded, or failed")
	errorClass := fs.String("error-class", string(shellprov.ErrorNone), "error class: none, launch, exit_nonzero, timeout, console_fault, io, or unknown")
	maxRows := fs.Int("max-rows", shellprov.DefaultMaxRows, "newest complete receipt rows to retain")
	asJSON := fs.Bool("json", false, "emit the appended privacy-safe receipt as JSON")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak shellprov record: positional arguments are not accepted")
		return 2
	}
	if *ledger == "" {
		fmt.Fprintln(stderr, "fak shellprov record: --ledger is required")
		return 2
	}
	if *maxRows <= 0 || *maxRows > shellprov.MaxRows {
		fmt.Fprintf(stderr, "fak shellprov record: --max-rows must be between 1 and %d\n", shellprov.MaxRows)
		return 2
	}

	receipt, err := shellprov.New(time.Now(), shellprov.Fields{
		ParentPID:         *parentPID,
		ChildPID:          *childPID,
		ChildCreatedUTCMS: *childCreatedMS,
		LaunchClass:       shellprov.LaunchClass(*launchClass),
		ShellImage:        shellprov.ShellImage(*shellImage),
		ShellEdition:      shellprov.ShellEdition(*shellEdition),
		ShellVersion:      *shellVersion,
		Outcome:           shellprov.Outcome(*outcome),
		ErrorClass:        shellprov.ErrorClass(*errorClass),
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak shellprov record: %v\n", err)
		return 2
	}
	if err := shellprov.Append(*ledger, receipt, *maxRows); err != nil {
		fmt.Fprintf(stderr, "fak shellprov record: %v\n", err)
		return 1
	}

	if *asJSON {
		if err := json.NewEncoder(stdout).Encode(receipt); err != nil {
			fmt.Fprintf(stderr, "fak shellprov record: encode output: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "recorded schema=%s launch_id=%s outcome=%s\n", receipt.Schema, receipt.LaunchID, receipt.Outcome)
	}
	return 0
}
