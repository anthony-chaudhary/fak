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

type contractPaths []string

func (p *contractPaths) String() string { return fmt.Sprint([]string(*p)) }
func (p *contractPaths) Set(value string) error {
	*p = append(*p, value)
	return nil
}

func cmdComponent(argv []string) { os.Exit(runComponent(os.Stdout, os.Stderr, argv)) }

func runComponent(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "check" {
		fmt.Fprintln(stderr, "usage: fak component check --contract PATH [--contract PATH...] --root ID [--workload NAME] [--json]")
		return 2
	}
	fs := flag.NewFlagSet("fak component check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var paths contractPaths
	fs.Var(&paths, "contract", "standalone fak-component-contract/1 JSON path (repeatable)")
	root := fs.String("root", "", "component ID or provided capability to resolve")
	workload := fs.String("workload", "component-compatibility-check", "workload name recorded in the receipt")
	asJSON := fs.Bool("json", false, "emit the machine-readable resolver receipt")
	if !parseFlags(fs, argv[1:]) {
		return 2
	}
	if fs.NArg() != 0 || len(paths) == 0 || *root == "" {
		fmt.Fprintln(stderr, "usage: fak component check --contract PATH [--contract PATH...] --root ID [--workload NAME] [--json]")
		return 2
	}
	contracts := make([]stackresolve.ComponentContract, 0, len(paths))
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(stderr, "fak component check: read %s: %v\n", path, err)
			return 1
		}
		contract, err := stackresolve.ParseComponentContract(raw)
		if err != nil {
			fmt.Fprintf(stderr, "fak component check: %s: %v\n", path, err)
			return 1
		}
		contracts = append(contracts, contract)
	}
	manifest, err := stackresolve.ComposeContracts(*workload, []string{*root}, contracts)
	if err != nil {
		fmt.Fprintf(stderr, "fak component check: %v\n", err)
		return 1
	}
	receipt, err := stackresolve.Resolve(context.Background(), manifest.Workload, manifest.Roots, stackresolve.ManifestProvider{Manifest: manifest})
	if err != nil {
		fmt.Fprintf(stderr, "fak component check: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(receipt); err != nil {
			fmt.Fprintf(stderr, "fak component check: encode receipt: %v\n", err)
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
