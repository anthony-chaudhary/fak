// stackresolvedemo is the narrow end-to-end witness for native harness stack resolution.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/stackresolve"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	fs := flag.NewFlagSet("stackresolvedemo", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	manifestPath := fs.String("manifest", "", "path to a fak-stack-manifest/1 JSON file")
	jsonOutput := fs.Bool("json", false, "emit the machine-readable receipt")
	selfcheck := fs.Bool("selfcheck", false, "run the embedded allow and transitive-refusal witnesses")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *selfcheck {
		return runSelfcheck(*jsonOutput)
	}
	if *manifestPath == "" {
		fmt.Fprintln(os.Stderr, "usage: stackresolvedemo -manifest PATH [-json] | -selfcheck [-json]")
		return 2
	}
	raw, err := os.ReadFile(*manifestPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read manifest: %v\n", err)
		return 1
	}
	manifest, err := stackresolve.Parse(raw)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	receipt, err := stackresolve.Resolve(context.Background(), manifest.Workload, manifest.Roots, stackresolve.ManifestProvider{Manifest: manifest})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	emit(receipt, *jsonOutput)
	if receipt.Status == "refuse" {
		return 3
	}
	return 0
}

func runSelfcheck(jsonOutput bool) int {
	allow, refuse, err := stackresolve.Selfcheck(context.Background())
	if err != nil {
		fmt.Fprintf(os.Stderr, "SELFCHECK FAIL: %v\n", err)
		return 1
	}
	if jsonOutput {
		emit(struct {
			Schema string               `json:"schema"`
			Allow  stackresolve.Receipt `json:"allow"`
			Refuse stackresolve.Receipt `json:"refuse"`
		}{Schema: "fak-stack-selfcheck/1", Allow: allow, Refuse: refuse}, true)
		return 0
	}
	fmt.Println("STACK RESOLUTION SELFCHECK")
	fmt.Print(stackresolve.Format(allow))
	fmt.Print(stackresolve.Format(refuse))
	fmt.Println("SELFCHECK PASS: satisfiable stack allowed; transitive hardware dependency refused")
	return 0
}

func emit(value any, jsonOutput bool) {
	if jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(value)
		return
	}
	receipt, ok := value.(stackresolve.Receipt)
	if !ok {
		fmt.Fprintln(os.Stderr, "internal error: text output requires a receipt")
		return
	}
	fmt.Print(stackresolve.Format(receipt))
}
