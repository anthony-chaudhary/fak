package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

func cmdFeature(argv []string) { os.Exit(runFeature(os.Stdout, os.Stderr, argv)) }

func runFeature(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		writeFeatureUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "query":
		return runFeatureQuery(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		writeFeatureUsage(stderr)
		return 0
	default:
		fmt.Fprintf(stderr, "fak feature: unknown subcommand %q\n", argv[0])
		writeFeatureUsage(stderr)
		return 2
	}
}

func runFeatureQuery(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("feature query", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: search upward for dos.toml)")
	plane := fs.String("plane", "all", "catalog plane: dev, live, or all")
	detail := fs.String("detail", "", "fault detail for one selected card name/detail_ref")
	limit := fs.Int("limit", 0, "cap the number of query cards (0 = all)")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	var args []string
	for rest := argv; ; {
		if err := fs.Parse(rest); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return 0
			}
			return 2
		}
		rest = fs.Args()
		if len(rest) == 0 {
			break
		}
		args = append(args, rest[0])
		rest = rest[1:]
	}
	if len(args) == 0 {
		fmt.Fprintln(stderr, "fak feature query: needs a non-empty query")
		return 2
	}
	cat, err := selfquery.Load(*root, selfquery.Options{
		Tools: selfquery.ToolDescriptorsFromMaps(gateway.ToolDescriptorsForResolver()),
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak feature query: %v\n", err)
		return 1
	}
	resp, err := cat.Query(selfquery.Request{
		Query:  joinArgs(args),
		Plane:  selfquery.Plane(*plane),
		Detail: *detail,
		Limit:  *limit,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak feature query: %v\n", err)
		return 2
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, resp, "fak feature query")
	}
	if len(resp.Cards) == 0 {
		fmt.Fprintln(stdout, "no matching feature")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, c := range resp.Cards {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", c.Name, c.Kind, c.Effect, c.Source, truncRunes(c.Summary, 96))
	}
	if code := flushTab(tw, stderr, "fak feature query"); code != 0 {
		return code
	}
	if resp.Detail != nil {
		fmt.Fprintf(stdout, "\ndetail: %s\n", resp.Detail.Card.Name)
		if len(resp.Detail.Schema) > 0 {
			fmt.Fprintf(stdout, "schema: %s\n", string(resp.Detail.Schema))
		}
		if resp.Detail.Plan != nil {
			fmt.Print(resp.Detail.Plan.Text())
		}
		if resp.Detail.DocSnippet != "" {
			fmt.Fprintf(stdout, "doc:\n%s\n", strings.TrimSpace(resp.Detail.DocSnippet))
		}
	}
	return 0
}

func writeFeatureUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak feature query <intent> [--json] [--plane dev|live|all] [--detail NAME] [--limit N] [--root DIR]

Query fak's self-feature catalog: dev facts from devindex, live MCP tools, memory
drivers, and capability cards. Queries return lightweight cards; --detail faults
only the selected schema, doc snippet, or memory explain plan.
`)
}
