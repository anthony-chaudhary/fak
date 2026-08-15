package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func runHarnessResolve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness resolve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath := fs.String("manifest", "", "harness product manifest")
	selectionPath := fs.String("selection", "", "harness selection result JSON")
	osName := fs.String("os", "", "target operating system")
	arch := fs.String("arch", "", "target architecture")
	contract := fs.String("contract", "", "target harness contract")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *manifestPath == "" || *selectionPath == "" || *osName == "" || *arch == "" || *contract == "" {
		fmt.Fprintln(stderr, "fak harness resolve: --manifest, --selection, --os, --arch, and --contract are required")
		return 2
	}
	raw, err := os.ReadFile(*manifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness resolve: %v\n", err)
		return 1
	}
	manifest, err := harnessresolve.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness resolve: %v\n", err)
		return 1
	}
	selectionRaw, err := os.ReadFile(*selectionPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness resolve: read selection: %v\n", err)
		return 1
	}
	var selection struct {
		Layers []string `json:"layers"`
	}
	dec := json.NewDecoder(strings.NewReader(string(selectionRaw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&selection); err != nil {
		fmt.Fprintf(stderr, "fak harness resolve: parse selection: %v\n", err)
		return 1
	}
	result, err := harnessresolve.Resolve(context.Background(), manifest, selection.Layers, harnessresolve.Environment{OS: *osName, Arch: *arch, Contract: *contract})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness resolve: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak harness resolve: %v\n", err)
		return 1
	}
	return 0
}
