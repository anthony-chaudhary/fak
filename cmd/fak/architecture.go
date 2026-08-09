package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/archreport"
)

func cmdArchitecture(argv []string) { os.Exit(runArchitecture(os.Stdout, os.Stderr, argv)) }

func runArchitecture(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("architecture", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("workspace", "", "workspace root (defaults to current directory)")
	leaf := fs.String("leaf", "", "report one internal leaf")
	jsonOut := fs.Bool("json", false, "emit fak-architecture/1 JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak architecture: pass no positional arguments")
		return 2
	}
	if *root == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak architecture: %v\n", err)
			return 1
		}
		*root = cwd
	}
	report, err := archreport.Analyze(*root, *leaf)
	if err != nil {
		fmt.Fprintf(stderr, "fak architecture: %v\n", err)
		return 1
	}
	if *jsonOut {
		raw, err := report.JSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak architecture: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
		return 0
	}
	fmt.Fprintf(stdout, "architecture: %d leaves, %d upward violation(s)\n", sumArchitectureLeaves(report), report.Violations)
	for _, t := range report.Tiers {
		fmt.Fprintf(stdout, "  tier %d %-22s %d\n", t.Level, t.Name, t.Leaves)
	}
	if *leaf == "" && len(report.Hotspots) > 0 {
		fmt.Fprintln(stdout, "  hotspots (direct fan-in):")
		for _, hotspot := range report.Hotspots {
			fmt.Fprintf(stdout, "    %-22s %d\n", hotspot.Name, hotspot.FanIn)
		}
	}
	for _, l := range report.Leaves {
		if *leaf != "" || len(l.Violations) > 0 {
			fmt.Fprintf(stdout, "  %-24s declared=%s(%d) floor=%s(%d) deps=%v dependents=%v", l.Name, l.DeclaredTierName, l.DeclaredTier, l.ImportFloorName, l.ImportFloor, l.Dependencies, l.Dependents)
			if len(l.Violations) > 0 {
				fmt.Fprintf(stdout, " violations=%v", l.Violations)
			}
			fmt.Fprintln(stdout)
		}
	}
	return 0
}
func sumArchitectureLeaves(r archreport.Report) int {
	n := 0
	for _, t := range r.Tiers {
		n += t.Leaves
	}
	return n
}
