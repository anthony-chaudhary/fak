package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func cmdTrajectory(args []string) {
	os.Exit(runTrajectory(os.Stdout, os.Stderr, args))
}

func runTrajectory(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		printTrajectoryUsage(stdout)
		return 0
	}
	switch args[0] {
	case "audit":
		return runTrajectoryAudit(stdout, stderr, args[1:])
	case "assurance":
		return runTrajectoryAssurance(os.Stdin, stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "fak trajectory: unknown subcommand %q\n", args[0])
		printTrajectoryUsage(stderr)
		return 2
	}
}

func runTrajectoryAudit(stdout, stderr io.Writer, args []string) int {
	flags := flag.NewFlagSet("fak trajectory audit", flag.ContinueOnError)
	flags.SetOutput(stderr)
	sinceText := flags.String("since", "7d", "include transcripts modified within this duration (7d, 12h, or 0 for all)")
	jsonlPath := flags.String("jsonl", "", "write versioned JSONL rows to this path")
	markdownPath := flags.String("md", "", "write the operator markdown report to this path")
	baselinePath := flags.String("baseline", "", "compare with a prior fak-trajectory-audit/1 JSONL artifact")
	userContains := flags.String("user-contains", "", "keep transcripts whose user-authored prompts contain this case-insensitive literal")
	claudeRoot := flags.String("claude-root", "", "override Claude projects root")
	codexRoot := flags.String("codex-root", "", "override Codex sessions root")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintf(stderr, "fak trajectory audit: unexpected arguments: %s\n", strings.Join(flags.Args(), " "))
		return 2
	}
	if *jsonlPath != "" && *markdownPath != "" && filepath.Clean(*jsonlPath) == filepath.Clean(*markdownPath) {
		fmt.Fprintln(stderr, "fak trajectory audit: --jsonl and --md must name different outputs")
		return 2
	}
	since, err := parseTrajectoryAuditSince(*sinceText)
	if err != nil {
		fmt.Fprintln(stderr, "fak trajectory audit:", err)
		return 2
	}

	sources := trajectory.DefaultAuditSources()
	for i := range sources {
		switch sources[i].Name {
		case trajectory.AuditSourceClaude:
			if strings.TrimSpace(*claudeRoot) != "" {
				sources[i].Root = *claudeRoot
			}
		case trajectory.AuditSourceCodex:
			if strings.TrimSpace(*codexRoot) != "" {
				sources[i].Root = *codexRoot
			}
		}
	}
	var baseline *trajectory.AuditSummaryRow
	if strings.TrimSpace(*baselinePath) != "" {
		baseline, err = trajectory.ReadAuditBaseline(*baselinePath)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
	}
	result, err := trajectory.RunAudit(trajectory.AuditOptions{Sources: sources, Since: since, Baseline: baseline, UserContains: strings.TrimSpace(*userContains)})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}

	if *jsonlPath == "" && *markdownPath == "" {
		if err := trajectory.WriteAuditMarkdown(stdout, result); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	} else {
		if *jsonlPath != "" {
			if err := writeTrajectoryAuditFile(*jsonlPath, func(w io.Writer) error { return trajectory.WriteAuditJSONL(w, result) }); err != nil {
				fmt.Fprintln(stderr, err)
				return 1
			}
		}
		if *markdownPath != "" {
			if err := writeTrajectoryAuditFile(*markdownPath, func(w io.Writer) error { return trajectory.WriteAuditMarkdown(w, result) }); err != nil {
				fmt.Fprintln(stderr, err)
				return 1
			}
		}
	}

	fmt.Fprintf(stderr, "trajectory audit: sessions=%d exact_usage=%d refused=%d\n", result.Summary.Transcripts, result.Summary.UsageRecordsExact, len(result.Refusals))
	if len(result.Refusals) > 0 {
		first := result.Refusals[0]
		fmt.Fprintf(stderr, "TRAJECTORY_SCHEMA_REFUSED %s %s:%d %s\n", first.Source, first.SourcePath, first.Line, first.Code)
		return 1
	}
	return 0
}

func parseTrajectoryAuditSince(value string) (time.Duration, error) {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" {
		return 0, fmt.Errorf("--since must not be empty")
	}
	if strings.HasSuffix(value, "d") {
		days, err := strconv.ParseFloat(strings.TrimSuffix(value, "d"), 64)
		if err != nil || days < 0 {
			return 0, fmt.Errorf("invalid --since %q", value)
		}
		return time.Duration(days * float64(24*time.Hour)), nil
	}
	duration, err := time.ParseDuration(value)
	if err != nil || duration < 0 {
		return 0, fmt.Errorf("invalid --since %q", value)
	}
	return duration, nil
}

func writeTrajectoryAuditFile(path string, write func(io.Writer) error) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("trajectory audit: create %s: %w", path, err)
	}
	writeErr := write(file)
	closeErr := file.Close()
	if writeErr != nil {
		return writeErr
	}
	if closeErr != nil {
		return fmt.Errorf("trajectory audit: close %s: %w", path, closeErr)
	}
	return nil
}

func printTrajectoryUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak trajectory <audit|assurance> [options]")
	fmt.Fprintln(w, "  audit      audit transcript usage and behavior")
	fmt.Fprintln(w, "  assurance  read typed evidence JSON on stdin and emit a shadow health receipt")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Audit exact Claude and Codex transcript usage, source denominators, behavior, and baseline regressions.")
}
