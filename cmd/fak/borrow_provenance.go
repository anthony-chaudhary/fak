package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/borrowprovenance"
)

func cmdBorrowProvenance(args []string) {
	if err := runBorrowProvenance(args, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "fak borrow-provenance: %v\n", err)
		os.Exit(1)
	}
}
func runBorrowProvenance(args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: fak borrow-provenance <pin|verify> [flags]")
	}
	switch args[0] {
	case "pin":
		return runBorrowProvenancePin(args[1:], stdout)
	case "verify":
		return runBorrowProvenanceVerify(args[1:], stdout)
	default:
		return fmt.Errorf("unknown action %q (want pin or verify)", args[0])
	}
}

func runBorrowProvenancePin(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("borrow-provenance pin", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	source := fs.String("source", "", "path to exact borrowed source bytes")
	url := fs.String("url", "", "upstream repository or source URL")
	ref := fs.String("ref", "", "immutable upstream revision")
	path := fs.String("source-path", "", "path or symbol within upstream")
	license := fs.String("license", "", "SPDX license identifier")
	transformation := fs.String("transformation", "", "how fak copied or adapted the source")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *source == "" {
		return errors.New("--source is required")
	}
	raw, err := os.ReadFile(*source)
	if err != nil {
		return err
	}
	record, err := borrowprovenance.Pin(*url, *ref, *path, *license, *transformation, raw)
	if err != nil {
		return err
	}
	return writeBorrowProvenanceJSON(stdout, record)
}

func runBorrowProvenanceVerify(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("borrow-provenance verify", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	manifest := fs.String("manifest", "", "path to a borrow-provenance record")
	source := fs.String("source", "", "path to source bytes to re-check")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *manifest == "" || *source == "" {
		return errors.New("--manifest and --source are required")
	}
	manifestRaw, err := os.ReadFile(*manifest)
	if err != nil {
		return err
	}
	var record borrowprovenance.Record
	if err := json.Unmarshal(manifestRaw, &record); err != nil {
		return fmt.Errorf("decode manifest: %w", err)
	}
	sourceRaw, err := os.ReadFile(*source)
	if err != nil {
		return err
	}
	result, err := borrowprovenance.Verify(record, sourceRaw)
	if err != nil {
		return err
	}
	if err := writeBorrowProvenanceJSON(stdout, result); err != nil {
		return err
	}
	if !result.Match {
		return errors.New("source drifted from pinned SHA-256")
	}
	return nil
}

func writeBorrowProvenanceJSON(w io.Writer, value any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(value)
}
