package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/runtimecap"
)

func runRuntimeCapabilities(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("runtime-capabilities", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "runtime-capabilities")
	backend := fs.String("backend", "", "require an exact registered backend name; unknown names never fall back")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak runtime-capabilities: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	report := runtimecap.Probe(runtimecap.Options{RequestedBackend: *backend})
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(report); err != nil {
		fmt.Fprintf(stderr, "fak runtime-capabilities: encode: %v\n", err)
		return 1
	}
	return 0
}
