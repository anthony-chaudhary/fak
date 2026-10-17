package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// runOpencodeFreeModels discovers the current OpenCode free-tier models via
// `opencode models`, applies the free-tier quality gate, and prints them in a
// machine-readable form. This is the discovery half of the auto-update story:
// OpenCode's free roster changes often, and hardcoding ids in opencode.json /
// the router routing table / the free-tier roster drifts the moment OpenCode
// renames or retires one.
func runOpencodeFreeModels(stdout, stderr io.Writer, argv []string) int {
	fs := newFlagSet("opencode free-models", stderr)
	jsonOut := fs.Bool("json", false, "emit discovered models as JSON")
	suffix := fs.String("suffix", "-free", "model-name suffix that marks a free-tier model")
	providerFilter := fs.String("provider", "", "only include models from this provider prefix (empty = all)")
	all := fs.Bool("all", false, "include quality-rejected candidates (default: survivors only)")
	check := fs.Bool("check", false, "freshness gate: exit non-zero if any discovered free model fails the quality gate or discovery is empty")
	writePath := fs.String("write", "", "write the discovered survivor roster as JSON to this file (for piping into apply tools)")
	opencodeBin := fs.String("opencode-bin", "opencode", "opencode CLI binary to invoke for discovery")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak opencode free-models [--json] [--suffix SUFFIX] [--provider PROVIDER] [--all]")
		fmt.Fprintln(stderr, "  e.g. fak opencode free-models --json")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}
	if !parseFlags(fs, argv) {
		return 2
	}

	cmd := exec.Command(*opencodeBin, "models")
	out, err := cmd.Output()
	if err != nil {
		fmt.Fprintf(stderr, "fak opencode free-models: failed to run %s models: %v\n", *opencodeBin, err)
		if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
			fmt.Fprintln(stderr, string(ee.Stderr))
		}
		return 1
	}
	lines := strings.Split(string(out), "\n")

	cands := modelroute.ParseOpenCodeFreeModels(lines, *suffix)

	// Optional provider filter.
	if *providerFilter != "" {
		var filtered []modelroute.OpenCodeFreeModel
		for _, c := range cands {
			if c.Provider == *providerFilter {
				filtered = append(filtered, c)
			}
		}
		cands = filtered
	}

	survivors, rejected := modelroute.SplitOpenCodeFreeModels(cands)

	// Freshness gate: fail loud on drift so operators/cron catch it before a
	// spawn dies with provider.no-route.
	if *check {
		if len(cands) == 0 {
			fmt.Fprintln(stderr, "fak opencode free-models --check: no free models discovered (opencode CLI may be unavailable or roster empty)")
			return 1
		}
		if len(rejected) > 0 {
			for _, c := range rejected {
				fmt.Fprintf(stderr, "fak opencode free-models --check: REJECTED %s: %s\n", c.ID, c.Reason)
			}
			return 1
		}
		fmt.Fprintf(stdout, "fak opencode free-models --check: OK (%d quality-gated free model(s))\n", len(survivors))
		return 0
	}

	// Optional: persist the discovered roster for downstream apply tools.
	if *writePath != "" {
		type payload struct {
			Survivors []modelroute.OpenCodeFreeModel `json:"survivors"`
		}
		f, err := os.Create(*writePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak opencode free-models: write %s: %v\n", *writePath, err)
			return 1
		}
		defer f.Close()
		enc := json.NewEncoder(f)
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload{Survivors: survivors}); err != nil {
			fmt.Fprintf(stderr, "fak opencode free-models: write %s: %v\n", *writePath, err)
			return 1
		}
	}

	if *jsonOut {
		type payload struct {
			Survivors []modelroute.OpenCodeFreeModel `json:"survivors"`
			Rejected  []modelroute.OpenCodeFreeModel `json:"rejected,omitempty"`
		}
		p := payload{Survivors: survivors}
		if *all {
			p.Rejected = rejected
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(p); err != nil {
			fmt.Fprintf(stderr, "fak opencode free-models: encode: %v\n", err)
			return 1
		}
		return 0
	}

	// Plain output: survivors (one id per line), rejected to stderr.
	for _, c := range survivors {
		fmt.Fprintln(stdout, c.ID)
	}
	if *all {
		for _, c := range rejected {
			fmt.Fprintf(stderr, "REJECTED %s: %s\n", c.ID, c.Reason)
		}
	}
	if len(survivors) == 0 {
		fmt.Fprintln(stderr, "fak opencode free-models: no quality-gated free models discovered")
	}
	return 0
}

// newFlagSet is a tiny helper kept local so this file does not depend on the
// opencode launcher's flag plumbing beyond parseFlags.
func newFlagSet(name string, stderr io.Writer) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	return fs
}
