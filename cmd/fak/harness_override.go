package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessoverride"
)

func runHarnessOverride(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness override", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lockPath := fs.String("lock", "", "verified current harness lock")
	capability := fs.String("capability", "", "effective kind:id from harness inspect")
	value := fs.String("value", "", "replacement value for a changeable non-policy capability")
	layerID := fs.String("layer", "operator-override", "ID for the generated person-scope layer")
	output := fs.String("output", "", "write the generated asset manifest to this path")
	jsonView := fs.Bool("json", false, "emit the full proposal as JSON")
	var denies harnessLayerFlag
	fs.Var(&denies, "deny", "policy capability to deny (repeatable; narrowing only)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *lockPath == "" || *capability == "" {
		fmt.Fprintln(stderr, "fak harness override: --lock and --capability are required")
		return 2
	}
	lock, err := readHarnessPreviewLock(*lockPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness override: lock: %v\n", err)
		return 1
	}
	proposal, err := harnessoverride.Propose(*lock, harnessoverride.Request{Capability: *capability, Value: *value, Denies: denies.values(), LayerID: *layerID})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness override: %v\n", err)
		return 1
	}
	if *output != "" {
		raw, err := json.MarshalIndent(proposal.Manifest, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak harness override: %v\n", err)
			return 1
		}
		raw = append(raw, '\n')
		if err := os.WriteFile(*output, raw, 0o600); err != nil {
			fmt.Fprintf(stderr, "fak harness override: output: %v\n", err)
			return 1
		}
	}
	if *jsonView {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(proposal); err != nil {
			fmt.Fprintf(stderr, "fak harness override: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, harnessoverride.Render(proposal))
	if strings.TrimSpace(*output) != "" {
		fmt.Fprintf(stdout, "written: %s\n", *output)
	}
	return 0
}
