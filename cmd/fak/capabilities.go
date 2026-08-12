package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// cmdCapabilities is the installed product's outcome-first answer to "what can
// fak do?". Repository-development capabilities remain in fak-dev; this verb
// consumes only capindex's runtime-safe product catalog.
func cmdCapabilities(argv []string) { os.Exit(runCapabilities(os.Stdout, os.Stderr, argv)) }

func runCapabilities(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("capabilities", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "capabilities")
	limit := fs.Int("limit", 0, "cap the number of outcomes (0 = all)")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	if err := fs.Parse(normalizeCapabilitiesArgs(argv)); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *limit < 0 {
		fmt.Fprintln(stderr, "fak capabilities: limit must be non-negative")
		return 2
	}
	query := strings.Join(fs.Args(), " ")
	outcomes := capindex.QueryProductOutcomes(query, *limit)
	if *asJSON {
		payload := struct {
			Query    string                    `json:"query"`
			Outcomes []capindex.ProductOutcome `json:"outcomes"`
		}{Query: query, Outcomes: outcomes}
		if err := json.NewEncoder(stdout).Encode(payload); err != nil {
			fmt.Fprintf(stderr, "fak capabilities: encode: %v\n", err)
			return 1
		}
		return 0
	}
	if len(outcomes) == 0 {
		fmt.Fprintln(stdout, "no matching capability")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "OUTCOME\tEFFECT\tTRY\tSUMMARY")
	for _, outcome := range outcomes {
		call := "-"
		if len(outcome.Command) > 0 {
			call = outcome.Command[0]
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", outcome.Name, outcome.Effect, call, truncRunes(outcome.Summary, 88))
	}
	return flushTab(tw, stderr, "fak capabilities")
}

// normalizeCapabilitiesArgs lets operators write either conventional
// `--limit 3 "turn control"` or outcome-first `"turn control" --limit 3`.
// flag.FlagSet stops at the first positional argument, which otherwise turns a
// trailing flag into query text-the worst possible behavior for a search verb.
func normalizeCapabilitiesArgs(argv []string) []string {
	flags := make([]string, 0, len(argv))
	intent := make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "--json" || arg == "-json" || arg == "-h" || arg == "--help":
			flags = append(flags, arg)
		case arg == "--limit" || arg == "-limit":
			flags = append(flags, arg)
			if i+1 < len(argv) {
				i++
				flags = append(flags, argv[i])
			}
		case strings.HasPrefix(arg, "--limit=") || strings.HasPrefix(arg, "-limit="):
			flags = append(flags, arg)
		default:
			intent = append(intent, arg)
		}
	}
	return append(flags, intent...)
}

func writeCapabilitiesUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak capabilities [<intent>] [--json] [--limit N]

Query fak's product outcomes in operator language. Examples:
  fak capabilities "token savings"
  fak capabilities "turn control"
  fak capabilities "model routing" --json

Results lead with token/turn efficiency, cache/context reuse, routing, savings
observability, and live-session control. The security capability floor remains
indexed as a supporting outcome. Each JSON result carries an exact next command,
detail reference, and shipped witness seam.
`)
}
