package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

func runModelAcceptanceGate(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model acceptance-gate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "-", "JSON report path ('-' reads stdin)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak model acceptance-gate: unexpected positional arguments")
		return 2
	}
	var r io.Reader = os.Stdin
	var f *os.File
	if *input != "-" {
		var err error
		f, err = os.Open(*input)
		if err != nil {
			fmt.Fprintf(stderr, "fak model acceptance-gate: %v\n", err)
			return 2
		}
		defer f.Close()
		r = f
	}
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var in modelaccept.Input
	if err := dec.Decode(&in); err != nil {
		fmt.Fprintf(stderr, "fak model acceptance-gate: decode input: %v\n", err)
		return 2
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		fmt.Fprintf(stderr, "fak model acceptance-gate: decode input: %v\n", err)
		return 2
	}
	out := modelaccept.Evaluate(in)
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		fmt.Fprintf(stderr, "fak model acceptance-gate: encode decision: %v\n", err)
		return 2
	}
	if out.Verdict == modelaccept.Pass {
		return 0
	}
	return 4
}
