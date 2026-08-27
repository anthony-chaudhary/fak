package main

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/studytickets"
)

type studyTicketsOperations struct {
	build         func(studytickets.BuildOptions) (studytickets.Ledger, studytickets.Report, error)
	marshalLedger func(studytickets.Ledger) ([]byte, error)
	marshalReport func(studytickets.Report) ([]byte, error)
	write         func(string, []byte) error
	validate      func(studytickets.ValidateOptions) error
}

var defaultStudyTicketsOperations = studyTicketsOperations{
	build:         studytickets.Build,
	marshalLedger: studytickets.MarshalLedger,
	marshalReport: studytickets.MarshalReport,
	write:         writeStudyPriorityFile,
	validate:      studytickets.ValidateFiles,
}

func runStudyTickets(stdout, stderr io.Writer, args []string) int {
	return runStudyTicketsWithOperations(stdout, stderr, args, defaultStudyTicketsOperations)
}

func runStudyTicketsWithOperations(stdout, stderr io.Writer, args []string, ops studyTicketsOperations) int {
	if len(args) == 0 {
		studyTicketsUsage(stderr)
		return 2
	}
	switch args[0] {
	case "build":
		return runStudyTicketsBuild(stdout, stderr, args[1:], ops)
	case "validate":
		return runStudyTicketsValidate(stdout, stderr, args[1:], ops)
	default:
		studyTicketsUsage(stderr)
		return 2
	}
}

func runStudyTicketsBuild(stdout, stderr io.Writer, args []string, ops studyTicketsOperations) int {
	fs, opts, ledgerPath, reportPath := studyTicketsFlags("study-tickets build", stderr)
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || studyTicketsPathsMissing(*opts, *ledgerPath, *reportPath) {
		fmt.Fprintln(stderr, studyTicketsBuildUsage)
		return 2
	}
	ledger, report, err := ops.build(*opts)
	if err != nil {
		fmt.Fprintf(stderr, "study-tickets: build: %v\n", err)
		return 1
	}
	ledgerData, err := ops.marshalLedger(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "study-tickets: marshal ledger: %v\n", err)
		return 1
	}
	reportData, err := ops.marshalReport(report)
	if err != nil {
		fmt.Fprintf(stderr, "study-tickets: marshal report: %v\n", err)
		return 1
	}
	if err := ops.write(*ledgerPath, ledgerData); err != nil {
		fmt.Fprintf(stderr, "study-tickets: write ledger: %v\n", err)
		return 1
	}
	if err := ops.write(*reportPath, reportData); err != nil {
		fmt.Fprintf(stderr, "study-tickets: write report: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "built study-tickets ledger %s and report %s\n", *ledgerPath, *reportPath)
	return 0
}

func runStudyTicketsValidate(stdout, stderr io.Writer, args []string, ops studyTicketsOperations) int {
	fs, opts, ledgerPath, reportPath := studyTicketsFlags("study-tickets validate", stderr)
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || studyTicketsPathsMissing(*opts, *ledgerPath, *reportPath) {
		fmt.Fprintln(stderr, studyTicketsValidateUsage)
		return 2
	}
	if err := ops.validate(studytickets.ValidateOptions{BuildOptions: *opts, LedgerPath: *ledgerPath, ReportPath: *reportPath}); err != nil {
		fmt.Fprintf(stderr, "study-tickets: validate: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "valid study-tickets ledger %s and report %s\n", *ledgerPath, *reportPath)
	return 0
}

const studyTicketsBuildUsage = "usage: fak study-tickets build --priority PATH --join PATH --forge PATH --adjacency PATH --classification PATH --ledger PATH --report PATH"
const studyTicketsValidateUsage = "usage: fak study-tickets validate --priority PATH --join PATH --forge PATH --adjacency PATH --classification PATH --ledger PATH --report PATH"

func studyTicketsFlags(name string, stderr io.Writer) (*flag.FlagSet, *studytickets.BuildOptions, *string, *string) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	opts := &studytickets.BuildOptions{}
	fs.StringVar(&opts.PriorityPath, "priority", "", "study-priority ledger path (required)")
	fs.StringVar(&opts.JoinPath, "join", "", "study-link ledger path (required)")
	fs.StringVar(&opts.ForgePath, "forge", "", "final FAK forge corpus path (required)")
	fs.StringVar(&opts.AdjacencyPath, "adjacency", "", "related-system adjacency manifest path (required)")
	fs.StringVar(&opts.ClassificationPath, "classification", "", "vLLM classification index path (required)")
	ledger := fs.String("ledger", "", "closure ledger path (required)")
	report := fs.String("report", "", "closure Markdown report path (required)")
	return fs, opts, ledger, report
}

func studyTicketsPathsMissing(opts studytickets.BuildOptions, paths ...string) bool {
	if opts.PriorityPath == "" || opts.JoinPath == "" || opts.ForgePath == "" || opts.AdjacencyPath == "" || opts.ClassificationPath == "" {
		return true
	}
	for _, path := range paths {
		if path == "" {
			return true
		}
	}
	return false
}

func studyTicketsUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak study-tickets <build|validate> [flags]")
	fmt.Fprintln(w, "  build --priority PATH --join PATH --forge PATH --adjacency PATH --classification PATH --ledger PATH --report PATH")
	fmt.Fprintln(w, "  validate --priority PATH --join PATH --forge PATH --adjacency PATH --classification PATH --ledger PATH --report PATH")
}
