package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/roofline"
)

func cmdRoofline(argv []string) {
	os.Exit(runRoofline(os.Stdout, os.Stderr, argv))
}

func runRoofline(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak roofline", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "repository root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit roofline dashboard as JSON")
	markdown := fs.Bool("markdown", false, "emit rendered markdown dashboard")

	if !parseFlags(fs, argv) {
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	dash, err := roofline.Generate(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak roofline: %v\n", err)
		return 2
	}

	if *asJSON {
		out := struct {
			Schema string `json:"schema"`
			roofline.Dashboard
		}{
			Schema:    "fak.roofline-dashboard/1",
			Dashboard: dash,
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fmt.Fprintf(stderr, "fak roofline: json encode: %v\n", err)
			return 2
		}
		return 0
	}

	if *markdown {
		fmt.Fprint(stdout, dash.Markdown())
		return 0
	}

	fmt.Fprintf(stdout, "Roofline Dashboard (%d row(s)):\n", len(dash.Rows))
	for _, r := range dash.Rows {
		fmt.Fprintf(stdout, "  Lane %s (%s): %s · Status: %s\n", r.Lane.ID, r.Lane.Name, r.Lane.Metric, r.Status)
	}
	return 0
}
