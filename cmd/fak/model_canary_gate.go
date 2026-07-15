package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/modelops"
)

func runModelCanaryGate(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model canary-gate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "-", "JSON input path ('-' reads stdin)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak model canary-gate: unexpected positional arguments")
		return 2
	}
	var r io.Reader = os.Stdin
	var f *os.File
	if *input != "-" {
		var err error
		f, err = os.Open(*input)
		if err != nil {
			fmt.Fprintf(stderr, "fak model canary-gate: %v\n", err)
			return 2
		}
		defer f.Close()
		r = f
	}
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var in modelops.Input
	if err := dec.Decode(&in); err != nil {
		fmt.Fprintf(stderr, "fak model canary-gate: decode input: %v\n", err)
		return 2
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		fmt.Fprintf(stderr, "fak model canary-gate: decode input: %v\n", err)
		return 2
	}
	decision := modelops.Evaluate(in)
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(decision); err != nil {
		fmt.Fprintf(stderr, "fak model canary-gate: encode decision: %v\n", err)
		return 2
	}
	switch decision.Action {
	case modelops.Promote:
		return 0
	case modelops.Rollback:
		return 3
	default:
		return 4
	}
}
