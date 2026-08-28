package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/learningmesh"
)

func runLearningMesh(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 || args[0] != "compile" {
		fmt.Fprintln(stderr, "usage: fak learning-mesh compile --file LEDGER.json")
		return 2
	}
	fs := flag.NewFlagSet("learning-mesh compile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "provider-neutral mechanism ledger")
	if err := fs.Parse(args[1:]); err != nil || *file == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak learning-mesh compile --file LEDGER.json")
		return 2
	}
	raw, err := os.ReadFile(*file)
	if err != nil {
		fmt.Fprintf(stderr, "learning-mesh compile: %v\n", err)
		return 1
	}
	var ledger learningmesh.Ledger
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&ledger); err != nil {
		fmt.Fprintf(stderr, "learning-mesh compile: %v\n", err)
		return 1
	}
	result, err := learningmesh.Compile(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "learning-mesh compile: %v\n", err)
		return 1
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		fmt.Fprintf(stderr, "learning-mesh compile: %v\n", err)
		return 1
	}
	return 0
}
