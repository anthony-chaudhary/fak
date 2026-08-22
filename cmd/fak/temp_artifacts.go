package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/tempartifact"
)

func runTempArtifacts(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("temp-artifacts", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "temp-artifacts")
	minAge := fs.Duration("min-age", 0, "minimum artifact age required for selection (required and positive)")
	apply := fs.Bool("apply", false, "quarantine, recheck, and delete eligible exact files (default is preview)")
	jsonOutput := fs.Bool("json", false, "print the stable JSON receipt")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak temp-artifacts: unexpected positional arguments")
		return 2
	}
	if *minAge <= 0 {
		fmt.Fprintln(stderr, "fak temp-artifacts: --min-age is required and must be positive")
		return 2
	}

	report, err := tempartifact.Run(context.Background(), tempartifact.Config{
		MinAge: *minAge,
		Apply:  *apply,
	})
	if err != nil {
		fmt.Fprintln(stderr, "fak temp-artifacts:", err)
		return 1
	}
	if *jsonOutput {
		if code := encodeJSONOrFail(stdout, stderr, report, "fak temp-artifacts"); code != 0 {
			return code
		}
	} else {
		writeTempArtifactReport(stdout, report)
	}
	if hasTempArtifactFailure(report) {
		return 1
	}
	return 0
}

func writeTempArtifactReport(writer io.Writer, report tempartifact.Report) {
	fmt.Fprintf(writer, "temp-artifacts mode=%s root=%s min-age=%s inspection=%s\n",
		report.Mode, report.Root, time.Duration(report.MinAgeSeconds)*time.Second, report.Inspection)
	for _, item := range report.Items {
		fmt.Fprintf(writer, "  %s age=%s bytes=%d reason=%s", item.Path, time.Duration(item.AgeSeconds)*time.Second, item.Bytes, item.Reason)
		if item.QuarantinePath != "" {
			fmt.Fprintf(writer, " quarantine=%s", item.QuarantinePath)
		}
		fmt.Fprintln(writer)
	}
	fmt.Fprintf(writer,
		"summary matching=%d/%dB eligible=%d/%dB preserved=%d/%dB reaped=%d/%dB\n",
		report.Summary.MatchingCount, report.Summary.MatchingBytes,
		report.Summary.EligibleCount, report.Summary.EligibleBytes,
		report.Summary.PreservedCount, report.Summary.PreservedBytes,
		report.Summary.ReapedCount, report.Summary.ReapedBytes,
	)
}

func hasTempArtifactFailure(report tempartifact.Report) bool {
	for _, item := range report.Items {
		switch item.Reason {
		case tempartifact.ReasonMoveFailed,
			tempartifact.ReasonPostMoveRecheckFailed,
			tempartifact.ReasonDeleteFailed,
			tempartifact.ReasonQuarantineCreateFailed:
			return true
		}
	}
	return false
}
