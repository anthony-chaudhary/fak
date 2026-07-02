package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/qwen36parity"
)

func cmdQwen36ParityWitnessGate(argv []string) {
	os.Exit(runQwen36ParityWitnessGate(os.Stdout, os.Stderr, argv))
}

func runQwen36ParityWitnessGate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("qwen36-parity-witness-gate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	witness := fs.String("witness", "", "path to a witness JSON")
	require := fs.Bool("require-witness", false, "fail closed when no witness is found")
	minRatio := fs.Float64("min-ratio", 0, "if >0, gate fak speed at this fraction of the bar")
	out := fs.String("out", "", "write the machine-readable gate report JSON here")
	markdown := fs.String("markdown", "", "write the markdown report here")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	root := *workspace
	if root == "" {
		root = resolveRoot("")
		if root == "" {
			root = "."
		}
	}
	report, err := qwen36parity.LoadAndGrade(root, *witness, *require, *minRatio)
	if err != nil {
		fmt.Fprintf(stderr, "qwen36-parity-witness-gate: %v\n", err)
		return 2
	}
	if *out != "" {
		path := *out
		if !filepath.IsAbs(path) {
			path = filepath.Join(root, path)
		}
		if err := qwen36parity.WriteJSON(path, report); err != nil {
			fmt.Fprintf(stderr, "qwen36-parity-witness-gate: %v\n", err)
			return 2
		}
	}
	if *markdown != "" {
		path := *markdown
		if !filepath.IsAbs(path) {
			path = filepath.Join(root, path)
		}
		if err := qwen36parity.WriteMarkdown(path, report); err != nil {
			fmt.Fprintf(stderr, "qwen36-parity-witness-gate: %v\n", err)
			return 2
		}
	}
	fmt.Fprint(stdout, qwen36parity.RenderMarkdown(report))
	if report.Passed {
		return 0
	}
	return 1
}
