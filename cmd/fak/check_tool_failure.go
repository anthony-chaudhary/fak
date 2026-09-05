package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/auditreason"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdCheckToolFailure(argv []string) {
	os.Exit(runCheckToolFailure(os.Stdout, os.Stderr, argv))
}

func runCheckToolFailure(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("check-tool-failure", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit JSON")
	list := fs.Bool("list", false, "list the closed non-guard tool-failure vocabulary")
	message := fs.String("message", "", "classify a raw tool-failure message into a structured {code,cause,evidence,fix,retryable,next_command} payload")
	command := fs.String("command", "", "the exact failing command, folded into a runnable next_command for the shell-mismatch classes (used with --message)")
	exitCode := fs.Int("exit-code", -1, "process exit code (0 indicates clean exit and ignores substring error signatures)")
	resume := fs.Bool("resume", false, "report the partial state + safe resume of a mutating op killed on timeout (see --op), instead of a bare exit-143")
	op := fs.String("op", "commit-push", "the killed mutating op to diagnose for --resume (currently: commit-push)")
	dir := fs.String("dir", ".", "repository directory to inspect for --resume")
	if !parseFlags(fs, argv) {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)

	switch {
	case *resume:
		return runToolResumePlan(stdout, stderr, *op, *dir, *asJSON)
	case *list:
		return renderToolFailureList(stdout, stderr, *asJSON)
	case strings.TrimSpace(*message) != "":
		payload, ok := auditreason.ToolFailurePayloadForCommand(*message, *command, *exitCode)
		if !ok {
			fmt.Fprintln(stderr, "fak check-tool-failure: message did not match a known tool-failure token")
			return 3
		}
		return renderToolFailurePayload(stdout, stderr, payload, *asJSON)
	case len(fs.Args()) == 1:
		spec, ok := auditreason.LookupToolFailure(fs.Args()[0])
		if !ok {
			fmt.Fprintf(stderr, "fak check-tool-failure: unknown tool-failure token %q\n", fs.Args()[0])
			return 3
		}
		return renderToolFailureSpec(stdout, stderr, spec, *asJSON)
	default:
		fmt.Fprintln(stderr, "usage: fak check-tool-failure [--json] [--list | --message TEXT [--command CMD] | --resume [--op OP] [--dir DIR] | TOKEN]")
		return 2
	}
}

func renderToolFailurePayload(stdout, stderr io.Writer, payload auditreason.ToolFailurePayload, asJSON bool) int {
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, payload, "fak check-tool-failure")
	}
	fmt.Fprintf(stdout, "%s\n", payload.Code)
	fmt.Fprintf(stdout, "  cause: %s\n", payload.Cause)
	if strings.TrimSpace(payload.Evidence) != "" {
		fmt.Fprintf(stdout, "  evidence: %s\n", payload.Evidence)
	}
	fmt.Fprintf(stdout, "  fix: %s\n", payload.Fix)
	fmt.Fprintf(stdout, "  retryable: %v\n", payload.Retryable)
	fmt.Fprintf(stdout, "  next_command: %s\n", payload.NextCommand)
	return 0
}

func renderToolFailureList(stdout, stderr io.Writer, asJSON bool) int {
	rows := auditreason.ToolFailures()
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, rows, "fak check-tool-failure")
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, row := range rows {
		fmt.Fprintf(tw, "%s\tretryable=%v\t%s\n", row.Token, row.Retryable, row.Summary)
	}
	return flushTab(tw, stderr, "fak check-tool-failure")
}

func renderToolFailureSpec(stdout, stderr io.Writer, spec auditreason.ToolFailureSpec, asJSON bool) int {
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, spec, "fak check-tool-failure")
	}
	fmt.Fprintf(stdout, "%s\n", spec.Token)
	fmt.Fprintf(stdout, "  summary: %s\n", spec.Summary)
	fmt.Fprintf(stdout, "  fix: %s\n", spec.Fix)
	fmt.Fprintf(stdout, "  retryable: %v\n", spec.Retryable)
	fmt.Fprintf(stdout, "  next_command: %s\n", spec.NextCommand)
	return 0
}
