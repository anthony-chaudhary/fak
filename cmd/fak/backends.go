package main

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// backends — the shipped-binary compute-backend registry surface (`fak backends`).
//
// The compute registry (internal/compute) dispatches inference to backends that
// self-register in init(); the cpu-ref Reference floor registers in EVERY build,
// while device backends (metal, cuda, rocm, ...) register only when the binary
// was built with their tags. Before this verb, that presence was observable only
// indirectly, from serve/bench error strings — never from the published asset
// itself. Release archives ship ONLY ./cmd/fak (scripts/build.sh), so the verb
// lives here: the release gate runs the extracted binary as `fak backends` and
// fails loud when a published asset lacks the always-present cpu-ref floor. The
// verb asserts that the registry is ALIVE (non-empty, cpu-ref present); it is
// NOT a GPU-completeness witness — the darwin metal legs and the CUDA container
// assert their richer backend sets in their own workflows.
func runBackends(stdout, stderr io.Writer, args []string) int {
	backends := compute.Registered()
	if len(backends) == 0 {
		// Unreachable in any real build (cpu-ref self-registers unconditionally),
		// so an empty registry is a loud failure; the release probe reads the
		// exit code before the text.
		fmt.Fprintln(stderr, "fak backends: no compute backends registered (cpu-ref must always self-register — this build is broken)")
		return 1
	}
	if backendsJSONRequested(args) {
		if err := writeIndentedJSON(stdout, backends); err != nil {
			fmt.Fprintf(stderr, "fak backends: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	for _, name := range backends {
		fmt.Fprintln(stdout, name)
	}
	return 0
}

// backendsJSONRequested reports whether the backends args ask for JSON output.
// It mirrors version.go's flag scan (--json or -json anywhere in the args)
// rather than a FlagSet, because the verb has no other flags to parse and unknown
// words must stay non-fatal for the release gate's smoke usage.
func backendsJSONRequested(args []string) bool {
	for _, a := range args {
		if a == "--json" || a == "-json" {
			return true
		}
	}
	return false
}
