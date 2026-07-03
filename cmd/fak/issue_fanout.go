package main

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

// runIssueFanout expands one shipped working spine into its contract-ready
// follow-on backlog (QA / dogfood / productization / observability /
// integration / docs / release) — the "3..50+ follow-ons at creation time"
// default. It only plans; file the candidates with gh, or wave-plan them with
// `fak issue cohort --from-plan`.
func runIssueFanout(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue fanout", flag.ContinueOnError)
	fs.SetOutput(stderr)
	title := fs.String("title", "", "human name of the shipped spine")
	leaf := fs.String("leaf", "", "owning leaf/lane (stamps keys, lane, default paths)")
	spine := fs.String("spine", "", "spine witness: commit SHA, demo command, or doc path")
	parent := fs.String("parent", "", "epic/issue ref the fan-out hangs off (default: --spine)")
	paths := fs.String("paths", "", "comma-separated file trees (default internal/<leaf>/)")
	areas := fs.String("areas", "", "comma-separated area filter ("+strings.Join(issuefanout.AreaNames(), ",")+")")
	maxN := fs.Int("max", 0, "cap candidates (0 = full taxonomy; floor "+fmt.Sprint(issuefanout.MinFanout)+")")
	asJSON := fs.Bool("json", false, "emit the machine-readable fan-out plan (feed to fak issue cohort --from-plan)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 || *title == "" || *leaf == "" || *spine == "" {
		fmt.Fprintln(stderr, "fak issue fanout: --title, --leaf and --spine are required (the spine witness comes first; no spine yet means the spine itself is the issue to file)")
		return 2
	}

	plan, err := issuefanout.Build(issuefanout.Input{
		Title:     *title,
		Leaf:      *leaf,
		SpineRef:  *spine,
		ParentRef: *parent,
		Paths:     issueFanoutSplit(*paths),
		Areas:     issueFanoutSplit(*areas),
		Max:       *maxN,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak issue fanout: %v\n", err)
		return 2
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, plan); err != nil {
			fmt.Fprintf(stderr, "fak issue fanout: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, issuefanout.Render(plan))
	return 0
}

func issueFanoutSplit(csv string) []string {
	var out []string
	for _, part := range strings.Split(csv, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}
