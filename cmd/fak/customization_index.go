package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/customizationindex"
)

const defaultCustomizationIndex = "docs/research/agent-customization-index.json"

func runCustomizationIndex(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("customization-index", flag.ContinueOnError)
	fs.SetOutput(stderr)
	indexPath := fs.String("index", defaultCustomizationIndex, "path to the agent customization index")
	asOfValue := fs.String("as-of", "", "report date in YYYY-MM-DD (defaults to today)")
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak customization-index [--index PATH] [--as-of YYYY-MM-DD] [--json]")
		return 2
	}
	asOf, err := customizationindex.ParseAsOf(*asOfValue, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "customization-index: %v\n", err)
		return 2
	}
	file, err := os.Open(*indexPath)
	if err != nil {
		fmt.Fprintf(stderr, "customization-index: %v\n", err)
		return 1
	}
	defer file.Close()
	index, err := customizationindex.Read(file)
	if err != nil {
		fmt.Fprintf(stderr, "customization-index: %v\n", err)
		return 1
	}
	report := customizationindex.Check(index, asOf)
	if *jsonOutput {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "customization-index: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "agent customization index: valid=%t axes=%d due_sources=%d review_days=%d as_of=%s\n", report.Valid, report.Axes, report.DueSources, report.ReviewDays, report.AsOf)
		for _, group := range report.Groups {
			fmt.Fprintf(stdout, "  %-14s %-8s %d\n", group.Layer, group.Status, group.Count)
		}
		for _, source := range report.Sources {
			if source.Due {
				fmt.Fprintf(stdout, "  DUE source=%s observed=%s age_days=%d\n", source.ID, source.ObservedAt, source.AgeDays)
			}
		}
		for _, finding := range report.Errors {
			fmt.Fprintf(stdout, "  ERROR %s\n", finding)
		}
	}
	if !report.Valid {
		return 1
	}
	return 0
}
