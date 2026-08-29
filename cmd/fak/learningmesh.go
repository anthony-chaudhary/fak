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

type repeatedStrings []string

func (v *repeatedStrings) String() string { return fmt.Sprint([]string(*v)) }
func (v *repeatedStrings) Set(value string) error {
	*v = append(*v, value)
	return nil
}

func runLearningMesh(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak learning-mesh <compile|from-receipts> ...")
		return 2
	}
	switch args[0] {
	case "compile":
		return runLearningMeshCompile(stdout, stderr, args[1:])
	case "from-receipts":
		return runLearningMeshReceipts(stdout, stderr, args[1:])
	default:
		fmt.Fprintln(stderr, "usage: fak learning-mesh <compile|from-receipts> ...")
		return 2
	}
}

func runLearningMeshCompile(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("learning-mesh compile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "provider-neutral mechanism ledger")
	if err := fs.Parse(args); err != nil || *file == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak learning-mesh compile --file LEDGER.json")
		return 2
	}
	var ledger learningmesh.Ledger
	if err := decodeJSONFile(*file, &ledger); err != nil {
		fmt.Fprintf(stderr, "learning-mesh compile: %v\n", err)
		return 1
	}
	return writeLearningMeshResult(stdout, stderr, ledger)
}

func runLearningMeshReceipts(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("learning-mesh from-receipts", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var receipts repeatedStrings
	fs.Var(&receipts, "receipt", "native-performance receipt JSON (repeatable)")
	targetsFile := fs.String("targets", "", "learning-mesh ledger whose targets define expansion envelopes")
	if err := fs.Parse(args); err != nil || len(receipts) == 0 || *targetsFile == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak learning-mesh from-receipts --receipt R.json [--receipt R2.json...] --targets LEDGER.json")
		return 2
	}
	var targetLedger learningmesh.Ledger
	if err := decodeJSONFile(*targetsFile, &targetLedger); err != nil {
		fmt.Fprintf(stderr, "learning-mesh from-receipts: targets: %v\n", err)
		return 1
	}
	inputs := make([]learningmesh.ReceiptInput, 0, len(receipts))
	for _, path := range receipts {
		raw, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(stderr, "learning-mesh from-receipts: %v\n", err)
			return 1
		}
		inputs = append(inputs, learningmesh.ReceiptInput{Path: path, Bytes: raw})
	}
	ledger, err := learningmesh.LedgerFromReceipts(inputs, targetLedger.Targets)
	if err != nil {
		fmt.Fprintf(stderr, "learning-mesh from-receipts: %v\n", err)
		return 1
	}
	return writeLearningMeshResult(stdout, stderr, ledger)
}

func decodeJSONFile(path string, out any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(out); err != nil {
		return err
	}
	return nil
}

func writeLearningMeshResult(stdout, stderr io.Writer, ledger learningmesh.Ledger) int {
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
