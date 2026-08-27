package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/studyprio"
)

type studyPriorityOperations struct {
	build          func(studyprio.BuildOptions) (studyprio.Ledger, studyprio.Summary, error)
	marshalLedger  func(studyprio.Ledger) ([]byte, error)
	marshalSummary func(studyprio.Summary) ([]byte, error)
	write          func(string, []byte) error
	validate       func(studyprio.ValidateOptions) error
}

var defaultStudyPriorityOperations = studyPriorityOperations{
	build:          studyprio.Build,
	marshalLedger:  studyprio.MarshalLedger,
	marshalSummary: studyprio.MarshalSummary,
	write:          writeStudyPriorityFile,
	validate:       studyprio.ValidateFiles,
}

func runStudyPriority(stdout, stderr io.Writer, args []string) int {
	return runStudyPriorityWithOperations(stdout, stderr, args, defaultStudyPriorityOperations)
}

func runStudyPriorityWithOperations(stdout, stderr io.Writer, args []string, ops studyPriorityOperations) int {
	if len(args) == 0 {
		studyPriorityUsage(stderr)
		return 2
	}
	switch args[0] {
	case "build":
		return runStudyPriorityBuild(stdout, stderr, args[1:], ops)
	case "validate":
		return runStudyPriorityValidate(stdout, stderr, args[1:], ops)
	default:
		studyPriorityUsage(stderr)
		return 2
	}
}

func runStudyPriorityBuild(stdout, stderr io.Writer, args []string, ops studyPriorityOperations) int {
	fs := flag.NewFlagSet("study-priority build", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sourcePath := fs.String("source-ledger", "", "study-link source ledger path (required)")
	ledgerPath := fs.String("ledger", "", "output priority ledger path (required)")
	summaryPath := fs.String("summary", "", "output Markdown summary path (required)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || anyStudyPriorityPathMissing(*sourcePath, *ledgerPath, *summaryPath) {
		fmt.Fprintln(stderr, "usage: fak study-priority build --source-ledger PATH --ledger PATH --summary PATH")
		return 2
	}

	ledger, summary, err := ops.build(studyprio.BuildOptions{SourceLedgerPath: *sourcePath})
	if err != nil {
		fmt.Fprintf(stderr, "study-priority: build: %v\n", err)
		return 1
	}
	ledgerData, err := ops.marshalLedger(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "study-priority: marshal ledger: %v\n", err)
		return 1
	}
	summaryData, err := ops.marshalSummary(summary)
	if err != nil {
		fmt.Fprintf(stderr, "study-priority: marshal summary: %v\n", err)
		return 1
	}
	if err := ops.write(*ledgerPath, ledgerData); err != nil {
		fmt.Fprintf(stderr, "study-priority: write ledger: %v\n", err)
		return 1
	}
	if err := ops.write(*summaryPath, summaryData); err != nil {
		fmt.Fprintf(stderr, "study-priority: write summary: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "built study-priority ledger %s and summary %s\n", *ledgerPath, *summaryPath)
	return 0
}

func runStudyPriorityValidate(stdout, stderr io.Writer, args []string, ops studyPriorityOperations) int {
	fs := flag.NewFlagSet("study-priority validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sourcePath := fs.String("source-ledger", "", "study-link source ledger path (required)")
	ledgerPath := fs.String("ledger", "", "priority ledger path (required)")
	summaryPath := fs.String("summary", "", "priority Markdown summary path (required)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || anyStudyPriorityPathMissing(*sourcePath, *ledgerPath, *summaryPath) {
		fmt.Fprintln(stderr, "usage: fak study-priority validate --source-ledger PATH --ledger PATH --summary PATH")
		return 2
	}

	if err := ops.validate(studyprio.ValidateOptions{
		SourceLedgerPath: *sourcePath,
		LedgerPath:       *ledgerPath,
		SummaryPath:      *summaryPath,
	}); err != nil {
		fmt.Fprintf(stderr, "study-priority: validate: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "valid study-priority ledger %s and summary %s\n", *ledgerPath, *summaryPath)
	return 0
}

func anyStudyPriorityPathMissing(paths ...string) bool {
	for _, path := range paths {
		if path == "" {
			return true
		}
	}
	return false
}

func studyPriorityUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak study-priority <build|validate> [flags]")
	fmt.Fprintln(w, "  build --source-ledger PATH --ledger PATH --summary PATH")
	fmt.Fprintln(w, "  validate --source-ledger PATH --ledger PATH --summary PATH")
}

func writeStudyPriorityFile(path string, data []byte) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".study-priority-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write temporary file: %w", err)
	}
	if err := tmp.Chmod(0o644); err != nil {
		tmp.Close()
		return fmt.Errorf("chmod temporary file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temporary file: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("replace output: %w", err)
	}
	return nil
}
