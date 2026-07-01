package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

// capabilities.go answers #1500 (C2 of the #1494 self-knowledge epic): `fak
// capabilities [<intent>]` is the memory-forward, task-scoped twin of `fak
// feature query` — narrowed to exactly the three families the issue names
// (memq memory drivers, `fak index *` self-index verbs, and the kernel
// shared-path verbs fak_changes/dos_arbitrate), each with the exact call to
// make. See internal/selfquery/capabilities.go for why this stays a distinct,
// narrower surface instead of widening fak_feature_query in place.

func cmdCapabilities(argv []string) { os.Exit(runCapabilities(os.Stdout, os.Stderr, argv)) }

func runCapabilities(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("capabilities", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { writeCapabilitiesUsage(stderr) }
	root := fs.String("root", "", "repo root (default: search upward for dos.toml)")
	limit := fs.Int("limit", 0, "cap the number of cards (0 = all)")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	intent := joinArgs(fs.Args())

	cat, err := selfquery.Load(*root, selfquery.Options{
		Tools: selfquery.ToolDescriptorsFromMaps(gateway.ToolDescriptorsForResolver()),
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak capabilities: %v\n", err)
		return 1
	}
	resp, err := cat.Capabilities(selfquery.CapabilitiesRequest{Query: intent, Limit: *limit})
	if err != nil {
		fmt.Fprintf(stderr, "fak capabilities: %v\n", err)
		return 2
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, resp, "fak capabilities")
	}
	if len(resp.Cards) == 0 {
		fmt.Fprintln(stdout, "no matching capability")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "NAME\tKIND\tEFFECT\tCALL\tSUMMARY\n")
	for _, c := range resp.Cards {
		call := c.Request.MCPTool
		if call == "" && len(c.Request.Command) > 0 {
			call = c.Request.Command[0]
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", c.Name, c.Kind, c.Effect, call, truncRunes(c.Summary, 88))
	}
	return flushTab(tw, stderr, "fak capabilities")
}

func writeCapabilitiesUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak capabilities [<intent>] [--json] [--limit N] [--root DIR]

The memory-forward toolbelt: memq memory drivers (recall/render/clean/compact/
dream), the fak index * self-index verbs, and the kernel shared-path verbs
(fak_changes, dos_arbitrate), ranked by an optional intent. An omitted intent
lists every card in stable order. Each card carries the exact call to make —
a memory-driver card's call is a ready fak_memory_run (apply=false).
`)
}
