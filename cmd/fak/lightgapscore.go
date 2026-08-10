package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/lightgapscore"
)

func cmdLightgapScore(argv []string) { os.Exit(runLightgapScore(os.Stdout, os.Stderr, argv)) }

func runLightgapScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score lightgap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "machine payload")
	segment := fs.String("segment", "", "one use case, in full")
	facet := fs.String("facet", "", "one facet across all use cases")
	dents := fs.Bool("dents", false, "every cell where fak loses")
	unrun := fs.Bool("unrun", false, "comparisons that would decide it")
	ceilings := fs.Bool("ceilings", false, "the c anchors and derivations")
	check := fs.Bool("check", false, "honesty gate; exit 1 on debt")
	markdownDir := fs.String("markdown-dir", "", "regenerate the doc folder")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if !parseFlags(fs, argv) || fs.NArg() != 0 {
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	sc, err := lightgapscore.New(root)
	if err != nil {
		fmt.Fprintf(stderr, "lightgap_scorecard: %v\n", err)
		return 2
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err = enc.Encode(sc.Payload()); err != nil {
			fmt.Fprintf(stderr, "lightgap_scorecard: encode json: %v\n", err)
			return 2
		}
		return 0
	}
	if *markdownDir != "" {
		paths, e := sc.WriteMarkdown(*markdownDir)
		if e != nil {
			fmt.Fprintf(stderr, "lightgap_scorecard: markdown: %v\n", e)
			return 2
		}
		for _, p := range paths {
			rel, e := filepath.Rel(root, p)
			if e != nil {
				rel = p
			}
			fmt.Fprintf(stdout, "wrote %s\n", filepath.ToSlash(rel))
		}
		return 0
	}
	if *check {
		fmt.Fprintln(stdout, sc.Check())
		if sc.Debt() > 0 {
			return 1
		}
		return 0
	}
	shown := false
	if *ceilings {
		fmt.Fprintln(stdout, sc.RenderCeilings())
		shown = true
	}
	if *dents {
		fmt.Fprintln(stdout, sc.RenderDents())
		shown = true
	}
	if *unrun {
		fmt.Fprintln(stdout, sc.RenderUnrun())
		shown = true
	}
	if *segment != "" {
		v, e := sc.RenderSegment(*segment)
		if e != nil {
			fmt.Fprintf(stderr, "lightgap_scorecard: %v\n", e)
			return 2
		}
		fmt.Fprintln(stdout, v)
		shown = true
	}
	if *facet != "" {
		v, e := sc.RenderFacet(*facet)
		if e != nil {
			fmt.Fprintf(stderr, "lightgap_scorecard: %v\n", e)
			return 2
		}
		fmt.Fprintln(stdout, v)
		shown = true
	}
	if !shown {
		fmt.Fprintln(stdout, sc.Render())
	}
	return 0
}
