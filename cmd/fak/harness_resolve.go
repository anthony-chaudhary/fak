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
	"github.com/anthony-chaudhary/fak/internal/harnessserver"
)

type harnessResolveCLIResult struct {
	harnessresolve.Result
	Server *harnessserver.Verified `json:"server,omitempty"`
}

func runHarnessResolve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness resolve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath := fs.String("manifest", "", "harness product manifest")
	selectionPath := fs.String("selection", "", "harness selection result JSON")
	osName := fs.String("os", "", "target operating system")
	arch := fs.String("arch", "", "target architecture")
	contract := fs.String("contract", "", "target harness contract")
	outputPath := fs.String("output", "", "write the verified product lock for later inspect/preview/verify-run stages")
	serverBindingPath := fs.String("server-binding", "", "immutable harness server binding created by harness init")
	example := fs.String("example", "", "print a generic valid manifest or selection template")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *example != "" {
		return writeHarnessResolveExample(stdout, stderr, *example)
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
	server, err := verifyHarnessServerBinding(*serverBindingPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness resolve: server binding: %v\n", err)
		return 1
	}
	result, err := harnessresolve.Resolve(context.Background(), manifest, selection.Layers, harnessresolve.Environment{OS: *osName, Arch: *arch, Contract: *contract})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness resolve: %v\n", err)
		return 1
	}
	if *outputPath != "" {
		if err := writeDerivedJSON(*outputPath, result.Lock); err != nil {
			fmt.Fprintf(stderr, "fak harness resolve: output: %v\n", err)
			return 1
		}
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(harnessResolveCLIResult{Result: result, Server: server}); err != nil {
		fmt.Fprintf(stderr, "fak harness resolve: %v\n", err)
		return 1
	}
	return 0
}

func writeHarnessResolveExample(stdout, stderr io.Writer, example string) int {
	var raw string
	switch example {
	case "manifest":
		raw = `{
  "schema": "fak.harness-product/v1alpha1",
  "roots": ["example-core"],
  "components": [{
    "id": "example-core",
    "version": "1.0.0",
    "digest": "sha256:replace-with-component-digest",
    "source": "operator:replace-with-provenance",
    "provides": ["runtime"],
    "compatibility": {"os": ["linux"], "arch": ["amd64"], "contract": "v1"},
    "adapters": ["native"],
    "evidence": {"authority": "operator", "source": "replace-with-evidence"}
  }],
  "assets": {
    "schema": "fak.harness-assets/v1alpha1",
    "layers": [{
      "id": "base",
      "scope": "project",
      "assets": [{"kind": "instruction", "id": "response-style", "operation": "add", "value": "neutral"}]
    }]
  }
}`
	case "selection":
		raw = `{"layers":["base"]}`
	default:
		fmt.Fprintln(stderr, "fak harness resolve: --example must be manifest or selection")
		return 2
	}
	fmt.Fprintln(stdout, raw)
	return 0
}
