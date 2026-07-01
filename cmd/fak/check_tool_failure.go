package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/auditreason"
)

func cmdCheckToolFailure(argv []string) {
	os.Exit(runCheckToolFailure(os.Stdout, os.Stderr, argv))
}

func runCheckToolFailure(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("check-tool-failure", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit JSON")
	list := fs.Bool("list", false, "list the closed non-guard tool-failure vocabulary")
	message := fs.String("message", "", "classify a raw tool-failure message into the closed vocabulary")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	switch {
	case *list:
		return renderToolFailureList(stdout, stderr, *asJSON)
	case strings.TrimSpace(*message) != "":
		spec, ok := auditreason.ToolFailureFromMessage(*message)
		if !ok {
			fmt.Fprintln(stderr, "fak check-tool-failure: message did not match a known tool-failure token")
			return 3
		}
		return renderToolFailureSpec(stdout, stderr, spec, *asJSON)
	case len(fs.Args()) == 1:
		spec, ok := auditreason.LookupToolFailure(fs.Args()[0])
		if !ok {
			fmt.Fprintf(stderr, "fak check-tool-failure: unknown tool-failure token %q\n", fs.Args()[0])
			return 3
		}
		return renderToolFailureSpec(stdout, stderr, spec, *asJSON)
	default:
		fmt.Fprintln(stderr, "usage: fak check-tool-failure [--json] [--list | --message TEXT | TOKEN]")
		return 2
	}
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
	return 0
}
