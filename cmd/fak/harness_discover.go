package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/harnessdiscover"
)

func runHarnessDiscover(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness discover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	registry := fs.String("registry", "", "scoped harness discovery registry")
	path := fs.String("path", "", "current repository or project path")
	principal := fs.String("principal", "", "authenticated engineer identity for team/person sources")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *registry == "" {
		fmt.Fprintln(stderr, "fak harness discover: --registry is required")
		return 2
	}
	result, err := harnessdiscover.Discover(harnessdiscover.Options{RegistryPath: *registry, StartPath: *path, Principal: *principal})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness discover: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak harness discover: %v\n", err)
		return 1
	}
	return 0
}
