package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/perfrsiscore"
)

func cmdPerformanceRSIScorecard(argv []string) {
	os.Exit(runPerformanceRSIScorecard(os.Stdout, os.Stderr, argv))
}

func runPerformanceRSIScorecard(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "compose" {
		return runPerformanceRSICompose(stdout, stderr, argv[1:])
	}
	fs := flag.NewFlagSet("fak performance-rsi-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "versioned performance RSI evidence JSON")
	prior := fs.String("prior", "", "prior scorecard JSON to compare")
	asJSON := fs.Bool("json", false, "render JSON")
	markdown := fs.Bool("markdown", false, "render Markdown")
	if !parseFlags(fs, argv) || fs.NArg() != 0 {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "fak performance-rsi-scorecard: --input is required")
		return 2
	}
	if *asJSON && *markdown {
		fmt.Fprintln(stderr, "fak performance-rsi-scorecard: choose only one of --json or --markdown")
		return 2
	}
	e, err := perfrsiscore.Load(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak performance-rsi-scorecard: %v\n", err)
		return 2
	}
	r := perfrsiscore.Score(e)
	if *prior != "" {
		f, e := os.Open(*prior)
		if e != nil {
			fmt.Fprintf(stderr, "fak performance-rsi-scorecard: prior: %v\n", e)
			return 2
		}
		p, e := perfrsiscore.DecodeReport(f)
		f.Close()
		if e != nil {
			fmt.Fprintf(stderr, "fak performance-rsi-scorecard: prior: %v\n", e)
			return 2
		}
		if e = perfrsiscore.Compare(&r, p); e != nil {
			fmt.Fprintf(stderr, "fak performance-rsi-scorecard: compare: %v\n", e)
			return 2
		}
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(r); err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		return 0
	}
	if *markdown {
		fmt.Fprintln(stdout, perfrsiscore.RenderMarkdown(r))
		return 0
	}
	fmt.Fprintln(stdout, perfrsiscore.RenderHuman(r))
	return 0
}

func runPerformanceRSICompose(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak performance-rsi-scorecard compose", flag.ContinueOnError)
	fs.SetOutput(stderr)
	snapshot := fs.String("snapshot", "", "snapshot name for the composed evidence")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *snapshot == "" {
		fmt.Fprintln(stderr, "fak performance-rsi-scorecard compose: --snapshot is required")
		return 2
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak performance-rsi-scorecard compose: provide one or more receipt paths")
		return 2
	}
	e, err := perfrsiscore.LoadAndComposeV1(*snapshot, fs.Args())
	if err != nil {
		fmt.Fprintf(stderr, "fak performance-rsi-scorecard compose: %v\n", err)
		return 2
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(e); err != nil {
		fmt.Fprintf(stderr, "fak performance-rsi-scorecard compose: %v\n", err)
		return 2
	}
	return 0
}
