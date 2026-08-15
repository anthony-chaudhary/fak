package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/stackresolve"
)

func cmdHarnessCommand(argv []string) { os.Exit(runHarnessCommand(os.Stdout, os.Stderr, argv)) }

func runHarnessCommand(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "stack" {
		return runHarnessStack(stdout, stderr, argv[1:])
	}
	return runHarness(stdout, stderr, argv)
}

func runHarnessStack(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "resolve" {
		fmt.Fprintln(stderr, "usage: fak harness stack resolve --manifest PATH [--json]")
		return 2
	}
	fs := flag.NewFlagSet("harness stack resolve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath := fs.String("manifest", "", "fak-stack-manifest/1 JSON file")
	jsonOutput := fs.Bool("json", false, "emit the fak-stack-receipt/1 JSON receipt")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if *manifestPath == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness stack resolve --manifest PATH [--json]")
		return 2
	}
	raw, err := os.ReadFile(*manifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness stack resolve: %v\n", err)
		return 1
	}
	manifest, err := stackresolve.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness stack resolve: %v\n", err)
		return 1
	}
	receipt, err := stackresolve.Resolve(context.Background(), manifest.Workload, manifest.Roots, stackresolve.ManifestProvider{Manifest: manifest})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness stack resolve: %v\n", err)
		return 1
	}
	if *jsonOutput {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(receipt); err != nil {
			fmt.Fprintf(stderr, "fak harness stack resolve: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, stackresolve.Format(receipt))
	}
	if receipt.Status == "refuse" {
		return 3
	}
	return 0
}
