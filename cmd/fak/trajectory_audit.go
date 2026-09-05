package main

import (
	"encoding/json"
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
	case "incident":
		return runTrajectoryIncident(stdout, stderr, args[1:])
	case "nightly":
		return runTrajectoryNightly(stdout, stderr, args[1:])
	case "assurance":
		return runTrajectoryAssurance(os.Stdin, stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "fak trajectory: unknown subcommand %q\n", args[0])
		printTrajectoryUsage(stderr)
		return 2
	}
}

func runTrajectoryIncident(stdout, stderr io.Writer, args []string) int {
	flags := flag.NewFlagSet("fak trajectory incident", flag.ContinueOnError)
	flags.SetOutput(stderr)
	root := flags.String("root", "", "retained Codex session root")
	tag := flags.String("tag", "", "exact tag in the launch prompt")
	promptSHA256 := flags.String("prompt-sha256", "", "exact SHA-256 of the normalized launch prompt")
	sinceText := flags.String("since", "", "include sessions starting at or after RFC3339 time")
	untilText := flags.String("until", "", "include sessions starting at or before RFC3339 time")
	restartText := flags.String("restart", "", "bucket sessions around an RFC3339 restart boundary")
	maxFiles := flags.Int("max-files", trajectory.IncidentDefaultMaxFiles, "maximum JSONL files to inspect")
	maxFileBytes := flags.Int64("max-file-bytes", trajectory.IncidentDefaultMaxBytesPerFile, "maximum bytes read per JSONL file")
	maxTotalBytes := flags.Int64("max-total-bytes", trajectory.IncidentDefaultMaxBytesTotal, "maximum bytes read across all files")
	maxDuration := flags.Duration("max-duration", trajectory.IncidentDefaultMaxDuration, "maximum scan duration")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if len(flags.Args()) != 0 {
		fmt.Fprintf(stderr, "fak trajectory incident: unexpected arguments: %s\n", strings.Join(flags.Args(), " "))
		return 2
	}
	parseTime := func(name, value string) (time.Time, bool) {
		if value == "" {
			return time.Time{}, true
		}
		parsed, err := time.Parse(time.RFC3339, value)
		if err != nil {
			fmt.Fprintf(stderr, "fak trajectory incident: --%s must be RFC3339: %v\n", name, err)
			return time.Time{}, false
		}
		return parsed, true
	}
	since, ok := parseTime("since", *sinceText)
	if !ok {
		return 2
	}
	until, ok := parseTime("until", *untilText)
	if !ok {
		return 2
	}
	restart, ok := parseTime("restart", *restartText)
	if !ok {
		return 2
	}
	packet, err := trajectory.RunIncident(trajectory.IncidentOptions{Root: *root, Tag: *tag, PromptSHA256: *promptSHA256, Since: since, Until: until, Restart: restart, MaxFiles: *maxFiles, MaxBytesPerFile: *maxFileBytes, MaxBytesTotal: *maxTotalBytes, MaxDuration: *maxDuration})
	if err != nil {
		fmt.Fprintln(stderr, "fak trajectory incident:", err)
		return 1
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(packet); err != nil {
		fmt.Fprintln(stderr, "fak trajectory incident: encode:", err)
		return 1
	}
	return 0
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
	snapshotOut := flags.String("snapshot-out", "", "capture selected private inputs into a new replayable snapshot directory")
	snapshot := flags.String("snapshot", "", "verify and replay a private audit snapshot without reading live roots")
	snapshotUsageLedger := flags.String("snapshot-usage-ledger", "", "append privacy-safe capture/replay outcomes to this explicit JSONL file")
	snapshotUsageFold := flags.String("snapshot-usage-fold", "", "read this snapshot usage JSONL file and emit its deterministic ISO-week fold")
	progress := flags.Bool("progress", false, "emit bounded scan progress to stderr")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	explicit := map[string]bool{}
	flags.Visit(func(f *flag.Flag) { explicit[f.Name] = true })
	if strings.TrimSpace(*snapshotUsageFold) != "" {
		for name := range explicit {
			if name != "snapshot-usage-fold" {
				return trajectoryAuditSnapshotFlagRefusal(stderr, "--snapshot-usage-fold is a read-only mode and cannot be combined with --"+name)
			}
		}
		weeks, err := trajectory.ReadAuditSnapshotUsage(strings.TrimSpace(*snapshotUsageFold))
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if err := trajectory.WriteAuditSnapshotUsageFold(stdout, weeks); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
	operation := ""
	if strings.TrimSpace(*snapshotOut) != "" {
		operation = "capture"
	} else if strings.TrimSpace(*snapshot) != "" {
		operation = "replay"
	}
	usageLedgerPath := strings.TrimSpace(*snapshotUsageLedger)
	if usageLedgerPath != "" {
		if operation == "" {
			return trajectoryAuditSnapshotFlagRefusal(stderr, "--snapshot-usage-ledger requires --snapshot-out or --snapshot")
		}
		var err error
		usageLedgerPath, err = filepath.Abs(filepath.Clean(usageLedgerPath))
		if err != nil {
			return trajectoryAuditSnapshotFlagRefusal(stderr, "resolve --snapshot-usage-ledger target")
		}
		fmt.Fprintf(stderr, "OUT_OF_TREE_WRITE operation=trajectory-audit-snapshot-usage target=%q\n", usageLedgerPath)
	}
	finish := func(code int, outcome, reason string) int {
		if usageLedgerPath == "" || operation == "" {
			return code
		}
		err := trajectory.AppendAuditSnapshotUsage(usageLedgerPath, trajectory.AuditSnapshotUsageRow{
			ObservedAt: time.Now().UTC(), Operation: operation, Outcome: outcome, Reason: reason,
		})
		if err != nil {
			fmt.Fprintln(stderr, "TRAJECTORY_SNAPSHOT_USAGE_ERROR")
			fmt.Fprintln(stderr, err)
			return 1
		}
		return code
	}
	if flags.NArg() != 0 {
		fmt.Fprintf(stderr, "fak trajectory audit: unexpected arguments: %s\n", strings.Join(flags.Args(), " "))
		return finish(2, "refused", "UNEXPECTED_ARGUMENTS")
	}
	if *jsonlPath != "" && *markdownPath != "" && filepath.Clean(*jsonlPath) == filepath.Clean(*markdownPath) {
		fmt.Fprintln(stderr, "fak trajectory audit: --jsonl and --md must name different outputs")
		return finish(2, "refused", "OUTPUT_PATH_CONFLICT")
	}
	if strings.TrimSpace(*snapshotOut) != "" && strings.TrimSpace(*snapshot) != "" {
		return finish(trajectoryAuditSnapshotFlagRefusal(stderr, "--snapshot-out and --snapshot are mutually exclusive"), "refused", "SNAPSHOT_FLAGS_INCOMPATIBLE")
	}
	if strings.TrimSpace(*snapshot) != "" {
		for _, name := range []string{"since", "user-contains", "claude-root", "codex-root", "baseline"} {
			if explicit[name] {
				return finish(trajectoryAuditSnapshotFlagRefusal(stderr, "--snapshot rejects live selection flag --"+name), "refused", "SNAPSHOT_FLAGS_INCOMPATIBLE")
			}
		}
	}
	if strings.TrimSpace(*snapshotOut) != "" && explicit["baseline"] {
		return finish(trajectoryAuditSnapshotFlagRefusal(stderr, "--snapshot-out cannot capture an external baseline; capture the input corpus first"), "refused", "SNAPSHOT_FLAGS_INCOMPATIBLE")
	}
	since, err := parseTrajectoryAuditSince(*sinceText)
	if err != nil {
		fmt.Fprintln(stderr, "fak trajectory audit:", err)
		return finish(2, "refused", "SINCE_INVALID")
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
			return finish(2, "error", "BASELINE_READ_FAILED")
		}
	}
	var progressReporter trajectory.ProgressReporter
	if *progress && stderr != nil && stderr != io.Discard {
		progressReporter = trajectory.NewProgressReporter(stderr)
	}

	var result trajectory.AuditResult
	var snapshotManifest *trajectory.AuditSnapshotManifest
	switch {
	case strings.TrimSpace(*snapshotOut) != "":
		target, absErr := filepath.Abs(filepath.Clean(strings.TrimSpace(*snapshotOut)))
		if absErr != nil {
			return trajectoryAuditSnapshotFlagRefusal(stderr, "resolve --snapshot-out target")
		}
		for _, output := range []string{*jsonlPath, *markdownPath} {
			if output != "" && trajectoryAuditPathWithin(target, output) {
				return finish(trajectoryAuditSnapshotFlagRefusal(stderr, "audit output must be outside the private snapshot directory"), "refused", "SNAPSHOT_FLAGS_INCOMPATIBLE")
			}
		}
		if usageLedgerPath != "" && trajectoryAuditPathWithin(target, usageLedgerPath) {
			return trajectoryAuditSnapshotFlagRefusal(stderr, "snapshot usage ledger must be outside the private snapshot directory")
		}
		fmt.Fprintf(stderr, "OUT_OF_TREE_WRITE operation=trajectory-audit-snapshot target=%q\n", target)
		manifest, captured, captureErr := trajectory.CaptureAuditSnapshot(target, trajectory.AuditOptions{
			Sources:      sources,
			Since:        since,
			UserContains: strings.TrimSpace(*userContains),
			Progress:     progressReporter,
		})
		if captureErr != nil {
			reason := trajectory.AuditSnapshotRefusalCode(captureErr)
			return finish(trajectoryAuditSnapshotError(stderr, captureErr), trajectoryAuditSnapshotUsageOutcome(reason), reason)
		}
		result = captured
		snapshotManifest = &manifest
	case strings.TrimSpace(*snapshot) != "":
		manifest, replayed, replayErr := trajectory.ReplayAuditSnapshot(strings.TrimSpace(*snapshot))
		if replayErr != nil {
			reason := trajectory.AuditSnapshotRefusalCode(replayErr)
			return finish(trajectoryAuditSnapshotError(stderr, replayErr), trajectoryAuditSnapshotUsageOutcome(reason), reason)
		}
		result = replayed
		snapshotManifest = &manifest
	default:
		result, err = trajectory.RunAudit(trajectory.AuditOptions{
			Sources:      sources,
			Since:        since,
			Baseline:     baseline,
			UserContains: strings.TrimSpace(*userContains),
			Progress:     progressReporter,
		})
		if err != nil {
			fmt.Fprintln(stderr, err)
			return finish(1, "error", "AUDIT_FAILED")
		}
	}

	if *jsonlPath == "" && *markdownPath == "" {
		if err := trajectory.WriteAuditMarkdown(stdout, result); err != nil {
			fmt.Fprintln(stderr, err)
			return finish(1, "error", "OUTPUT_WRITE_FAILED")
		}
	} else {
		if *jsonlPath != "" {
			if err := writeTrajectoryAuditFile(*jsonlPath, func(w io.Writer) error { return trajectory.WriteAuditJSONL(w, result) }); err != nil {
				fmt.Fprintln(stderr, err)
				return finish(1, "error", "OUTPUT_WRITE_FAILED")
			}
		}
		if *markdownPath != "" {
			if err := writeTrajectoryAuditFile(*markdownPath, func(w io.Writer) error { return trajectory.WriteAuditMarkdown(w, result) }); err != nil {
				fmt.Fprintln(stderr, err)
				return finish(1, "error", "OUTPUT_WRITE_FAILED")
			}
		}
	}

	fmt.Fprintf(stderr, "trajectory audit: sessions=%d exact_usage=%d refused=%d schema_drift=%d schema_breaking=%d\n",
		result.Summary.Transcripts, result.Summary.UsageRecordsExact, len(result.Refusals), result.ConclusionStatus.SchemaDriftCount, result.ConclusionStatus.BreakingSchemaDrift)
	if len(result.Refusals) > 0 {
		first := result.Refusals[0]
		fmt.Fprintf(stderr, "TRAJECTORY_SCHEMA_REFUSED %s %s:%d %s\n", first.Source, first.SourcePath, first.Line, first.Code)
		return finish(1, "refused", "TRAJECTORY_SCHEMA_REFUSED")
	}
	if snapshotManifest != nil {
		fmt.Fprintf(stderr, "trajectory audit snapshot: corpus_sha256=%s files=%d verified=true\n", snapshotManifest.CorpusDigest, len(snapshotManifest.Files))
	}
	return finish(0, "success", "")
}

func trajectoryAuditSnapshotUsageOutcome(reason string) string {
	switch reason {
	case "SNAPSHOT_IO", "SNAPSHOT_PUBLISH_FAILED", "SNAPSHOT_AUDIT_REFUSED":
		return "error"
	default:
		return "refused"
	}
}

func trajectoryAuditSnapshotFlagRefusal(stderr io.Writer, detail string) int {
	fmt.Fprintln(stderr, "TRAJECTORY_SNAPSHOT_REFUSED SNAPSHOT_FLAGS_INCOMPATIBLE")
	fmt.Fprintln(stderr, "fak trajectory audit:", detail)
	return 2
}

func trajectoryAuditSnapshotError(stderr io.Writer, err error) int {
	fmt.Fprintln(stderr, "TRAJECTORY_SNAPSHOT_REFUSED", trajectory.AuditSnapshotRefusalCode(err))
	fmt.Fprintln(stderr, err)
	return 1
}

func trajectoryAuditPathWithin(root, candidate string) bool {
	root, err := filepath.Abs(filepath.Clean(root))
	if err != nil {
		return false
	}
	candidate, err = filepath.Abs(filepath.Clean(candidate))
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(root, candidate)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
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
	fmt.Fprintln(w, "usage: fak trajectory <audit|incident|nightly|assurance> [options]")
	fmt.Fprintln(w, "  audit      audit transcript usage and behavior")
	fmt.Fprintln(w, "  nightly    run the bounded attribution audit and append a scrubbed receipt")
	fmt.Fprintln(w, "  assurance  read typed evidence JSON on stdin and emit a shadow health receipt")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Audit exact Claude and Codex transcript usage, source denominators, behavior, and baseline regressions.")
}
