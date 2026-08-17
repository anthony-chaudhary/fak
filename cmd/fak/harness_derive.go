package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessderive"
)

type deriveSetFlag []string

func (f *deriveSetFlag) String() string         { return strings.Join(*f, ",") }
func (f *deriveSetFlag) Set(value string) error { *f = append(*f, value); return nil }

func runHarnessDerive(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness derive", flag.ContinueOnError)
	fs.SetOutput(stderr)
	from := fs.String("from", "", "verified base product lock")
	output := fs.String("output", "", "write the derived product lock")
	receiptPath := fs.String("receipt", "", "write the derivation lineage receipt (default: <output>.derive.json)")
	expectBase := fs.String("expect-base", "", "require this exact imported base lock ID")
	layer := fs.String("layer", "local", "local derivation layer ID")
	jsonView := fs.Bool("json", false, "emit lock and derivation receipt as JSON")
	var sets deriveSetFlag
	var denies harnessLayerFlag
	fs.Var(&sets, "set", "replace a launch-conformant capability as kind:id=value (repeatable)")
	fs.Var(&denies, "deny", "narrow policy as policy:id=value (repeatable)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *from == "" || *output == "" || len(sets)+len(denies) == 0 {
		fmt.Fprintln(stderr, "fak harness derive: --from, --output, and at least one --set or --deny are required")
		return 2
	}
	base, err := readHarnessPreviewLock(*from)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness derive: base: %v\n", err)
		return 1
	}
	if *expectBase != "" && base.ID != *expectBase {
		fmt.Fprintf(stderr, "fak harness derive: stale base: got %s want %s\n", base.ID, *expectBase)
		return 1
	}
	if *receiptPath == "" {
		*receiptPath = *output + ".derive.json"
	}
	deltas := make([]harnessderive.Delta, 0, len(sets)+len(denies))
	for _, raw := range sets {
		capability, value, ok := strings.Cut(raw, "=")
		if !ok || strings.TrimSpace(value) == "" {
			fmt.Fprintf(stderr, "fak harness derive: --set %q must be kind:id=value\n", raw)
			return 2
		}
		deltas = append(deltas, harnessderive.Delta{Capability: capability, Operation: "replace", Value: value})
	}
	for _, raw := range denies.values() {
		capability, value, ok := strings.Cut(raw, "=")
		if !ok || strings.TrimSpace(value) == "" {
			fmt.Fprintf(stderr, "fak harness derive: --deny %q must be policy:id=value\n", raw)
			return 2
		}
		deltas = append(deltas, harnessderive.Delta{Capability: capability, Operation: "deny", Denies: []string{value}})
	}
	result, err := harnessderive.Derive(*base, harnessderive.Request{Layer: *layer, Deltas: deltas})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness derive: %v\n", err)
		return 1
	}
	if err := writeDerivedJSON(*output, result.Lock); err != nil {
		fmt.Fprintf(stderr, "fak harness derive: output: %v\n", err)
		return 1
	}
	if *receiptPath != "" {
		if err := writeDerivedJSON(*receiptPath, result.Receipt); err != nil {
			fmt.Fprintf(stderr, "fak harness derive: receipt: %v\n", err)
			return 1
		}
	}
	if *jsonView {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "HARNESS DERIVE | VERIFIED\nbase: %s\nresult: %s\ndeltas: %d | layer %s\nwritten: %s\n", result.Receipt.BaseID, result.Lock.ID, len(result.Receipt.Deltas), result.Receipt.Layer, *output)
	if *receiptPath != "" {
		fmt.Fprintf(stdout, "receipt: %s\n", *receiptPath)
	}
	fmt.Fprintf(stdout, "next: fak harness preview --current %s --candidate %s\n", *from, *output)
	fmt.Fprintf(stdout, "inspect: fak harness inspect --lock %s\n", *output)
	return 0
}

func writeDerivedJSON(path string, value any) error {
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o600)
}
