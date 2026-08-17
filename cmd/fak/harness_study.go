package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscreationreceipt"
	"github.com/anthony-chaudhary/fak/internal/harnesscreationstudy"
	"github.com/anthony-chaudhary/fak/internal/harnesscrossover"
)

func runHarnessStudy(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "receipt" {
		return runHarnessCreationReceipt(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "creation" {
		return runHarnessCreationStudy(stdout, stderr, argv[1:])
	}
	if len(argv) == 0 || argv[0] != "crossover" {
		fmt.Fprintln(stderr, "usage: fak harness study <creation|crossover> --input STUDY.json")
		return 2
	}
	fs := flag.NewFlagSet("harness study crossover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak.harness-crossover-study/v1alpha1 JSON")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study crossover --input STUDY.json")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study crossover: %v\n", err)
		return 1
	}
	study, err := harnesscrossover.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study crossover: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(harnesscrossover.Evaluate(study)); err != nil {
		fmt.Fprintf(stderr, "fak harness study crossover: %v\n", err)
		return 1
	}
	return 0
}

func runHarnessCreationStudy(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness study creation", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak.harness-creation-study/v1alpha1 JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study creation --input STUDY.json")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	study, err := harnesscreationstudy.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	if err := verifyHarnessStudyReceiptSources(*input, study); err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(harnesscreationstudy.Evaluate(study)); err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	return 0
}

func runHarnessCreationReceipt(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness study receipt", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak.harness-creation-receipt/v1alpha1 JSON")
	studyPath := fs.String("study", "", "optional study JSON for duplicate run/participant refusal")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study receipt --input RECEIPT.json")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study receipt: %v\n", err)
		return 1
	}
	receipt, err := harnesscreationreceipt.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study receipt: %v\n", err)
		return 1
	}
	result := harnesscreationreceipt.Evaluate(receipt)
	if *studyPath != "" {
		studyAbs, absErr := filepath.Abs(*studyPath)
		if absErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", absErr)
			return 1
		}
		receiptAbs, absErr := filepath.Abs(*input)
		if absErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", absErr)
			return 1
		}
		rel, relErr := filepath.Rel(filepath.Dir(studyAbs), receiptAbs)
		if relErr != nil || filepath.IsAbs(rel) {
			fmt.Fprintln(stderr, "fak harness study receipt: source receipt must be relative to study directory")
			return 1
		}
		sum := sha256.Sum256(raw)
		result.Row.SourceReceipt = filepath.ToSlash(rel)
		result.Row.SourceDigest = "sha256:" + hex.EncodeToString(sum[:])
		studyRaw, readErr := os.ReadFile(*studyPath)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", readErr)
			return 1
		}
		if uniqueErr := harnesscreationreceipt.CheckUnique(studyRaw, result.Row); uniqueErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", uniqueErr)
			return 1
		}
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak harness study receipt: %v\n", err)
		return 1
	}
	return 0
}

func verifyHarnessStudyReceiptSources(studyPath string, study harnesscreationstudy.Study) error {
	studyAbs, err := filepath.Abs(studyPath)
	if err != nil {
		return err
	}
	base := filepath.Dir(studyAbs)
	for _, run := range study.Runs {
		// Inline fixture rows keep parser/aggregate tests hermetic; production receipt paths never use this prefix.
		if run.PairID == "" || (run.SourceReceipt == "" && run.SourceDigest == "" && strings.HasPrefix(run.Receipt, "fixture:")) || (run.ParticipantClass == "maintainer-calibration" && run.SourceReceipt == "" && run.SourceDigest == "") {
			continue
		}
		if run.SourceReceipt == "" || !strings.HasPrefix(run.SourceDigest, "sha256:") {
			return fmt.Errorf("run %q requires source_receipt and source_digest", run.ID)
		}
		source := filepath.Clean(filepath.Join(base, filepath.FromSlash(run.SourceReceipt)))
		rel, err := filepath.Rel(base, source)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
			return fmt.Errorf("run %q source_receipt escapes study directory", run.ID)
		}
		raw, err := os.ReadFile(source)
		if err != nil {
			return fmt.Errorf("run %q source_receipt: %w", run.ID, err)
		}
		sum := sha256.Sum256(raw)
		if "sha256:"+hex.EncodeToString(sum[:]) != run.SourceDigest {
			return fmt.Errorf("run %q source_digest does not match archived receipt", run.ID)
		}
		receipt, err := harnesscreationreceipt.Parse(raw)
		if err != nil {
			return fmt.Errorf("run %q source_receipt: %w", run.ID, err)
		}
		row := harnesscreationreceipt.Evaluate(receipt).Row
		if row.ID != run.ID || row.ParticipantID != run.ParticipantID || row.Track != run.Track || row.Arm != run.Arm || row.PairID != run.PairID || row.TaskDigest != run.TaskDigest || row.MachineID != run.MachineID || row.PairOrder != run.PairOrder || row.ArmPosition != run.ArmPosition || row.ParticipantClass != run.ParticipantClass || row.Independent != run.Independent || row.OS != run.OS || row.CPU != run.CPU || row.NetworkState != run.NetworkState || row.CacheState != run.CacheState || row.Outcome != run.Outcome || row.ElapsedSeconds != run.ElapsedSeconds || row.Receipt != run.Receipt {
			return fmt.Errorf("run %q does not match archived receipt projection", run.ID)
		}
	}
	return nil
}
