package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/citeverify"
)

// cmd/fak/citeverify.go — `fak citeverify`: mechanically verify a cited path:line
// against source. Out-of-range lines or lines lacking claimed symbols contradict.

func cmdCiteverify(argv []string) {
	os.Exit(runCiteverify(os.Stdout, os.Stderr, argv))
}

type citeverifyJSONResult struct {
	Claim    string            `json:"claim"`
	Evidence []string          `json:"evidence"`
	Root     string            `json:"root"`
	Status   citeverify.Status `json:"status"`
}

func runCiteverify(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("citeverify", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		claimFlag = fs.String("claim", "", "the claim or statement naming symbols (e.g. \"`Target` exists\")")
		rootFlag  = fs.String("root", "", "repository or workspace root directory (default: current directory)")
		asJSON    = fs.Bool("json", false, "emit result as JSON")
	)

	fs.Usage = func() {
		fmt.Fprint(stderr, `fak citeverify — mechanically verify path:line evidence citations against source.

usage:
  fak citeverify --claim "<claim>" [evidence...] [--root <dir>] [--json]

outcomes:
  supports    - cited line exists, matches claimed symbol, no contradictory citations
  contradicts - line out of range or resolved line lacks claimed symbol
  unknown     - unresolvable path, ambiguous basename, unsafe file, or empty line
  mixed       - both supporting and contradicting evidence citations found
`)
	}

	if !parseFlags(fs, argv) {
		return 2
	}

	claim := strings.TrimSpace(*claimFlag)
	evidence := fs.Args()

	if claim == "" && len(evidence) == 0 {
		fs.Usage()
		return 2
	}

	root := *rootFlag
	if root == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak citeverify: getwd: %v\n", err)
			return 1
		}
		root = cwd
	}

	status := citeverify.Verify(claim, evidence, root)

	if *asJSON {
		res := citeverifyJSONResult{
			Claim:    claim,
			Evidence: evidence,
			Root:     root,
			Status:   status,
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak citeverify: json encode: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "%s\n", status)
	}

	switch status {
	case citeverify.Supports:
		return 0
	case citeverify.Contradicts, citeverify.Mixed:
		return 3
	default: // citeverify.Unknown
		return 1
	}
}
