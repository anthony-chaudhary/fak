package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
)

func runHarnessCompose(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness compose", flag.ContinueOnError)
	fs.SetOutput(stderr)
	assetsPath := fs.String("assets", "", "typed harness asset manifest")
	selectionPath := fs.String("selection", "", "harness selection result JSON")
	var layers harnessLayerFlag
	fs.Var(&layers, "layer", "selected layer ID in precedence order (repeatable or comma-separated)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *assetsPath == "" || ((*selectionPath == "") == (len(layers.values()) == 0)) {
		fmt.Fprintln(stderr, "fak harness compose: --assets and exactly one of --selection or --layer are required")
		return 2
	}
	raw, err := os.ReadFile(*assetsPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness compose: %v\n", err)
		return 1
	}
	manifest, err := harnesscompose.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness compose: %v\n", err)
		return 1
	}
	selected := layers.values()
	if *selectionPath != "" {
		selectionRaw, err := os.ReadFile(*selectionPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak harness compose: read selection: %v\n", err)
			return 1
		}
		var receipt struct {
			Layers []string `json:"layers"`
		}
		dec := json.NewDecoder(strings.NewReader(string(selectionRaw)))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&receipt); err != nil {
			fmt.Fprintf(stderr, "fak harness compose: parse selection: %v\n", err)
			return 1
		}
		if len(receipt.Layers) == 0 {
			fmt.Fprintln(stderr, "fak harness compose: selection has no layers")
			return 1
		}
		selected = receipt.Layers
	}
	result, err := harnesscompose.Compose(manifest, selected)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness compose: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak harness compose: %v\n", err)
		return 1
	}
	return 0
}

type harnessLayerFlag []string

func (f *harnessLayerFlag) String() string { return strings.Join(*f, ",") }
func (f *harnessLayerFlag) Set(value string) error {
	for _, layer := range strings.Split(value, ",") {
		if strings.TrimSpace(layer) != "" {
			*f = append(*f, strings.TrimSpace(layer))
		}
	}
	return nil
}
func (f harnessLayerFlag) values() []string { return []string(f) }
