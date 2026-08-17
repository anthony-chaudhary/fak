package main

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func cmdDisambiguation(args []string) {
	os.Exit(runDisambiguation(os.Stdout, os.Stderr, args))
}

func runDisambiguation(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak disambiguation schema [--json] [--self-test]\n       fak disambiguation query <canonical-term> [--json]\n       fak disambiguation query --self-test [--json]\n       fak disambiguation stale-symbols-self-test [--json]\n       fak disambiguation coverage-self-test [--json]")
		return 2
	}
	switch args[0] {
	case "schema":
		return runDisambiguationSchema(stdout, stderr, args[1:])
	case "generate":
		return runDisambiguationGenerate(stdout, stderr, args[1:])
	case "version":
		return runDisambiguationVersion(stdout, stderr, args[1:])
	case "committed-freshness":
		return runDisambiguationCommittedFreshness(stdout, stderr, args[1:])
	case "query":
		return runDisambiguationQuery(stdout, stderr, args[1:])
	case "ownership":
		return runDisambiguationOwnership(stdout, stderr, args[1:])
	case "freshness":
		return runDisambiguationFreshness(stdout, stderr, args[1:])
	case "provenance":
		return runDisambiguationProvenance(stdout, stderr, args[1:])
	case "stale-symbols-self-test":
		return runDisambiguationStaleSymbolsSelfTest(stdout, stderr, args[1:])
	case "coverage-self-test":
		return runDisambiguationCoverageSelfTest(stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "fak disambiguation: unknown command %q (want schema, query, ownership, freshness, provenance, stale-symbols-self-test, or coverage-self-test)\n", args[0])
		return 2
	}
}

const defaultDisambiguationIndexPath = "docs/generated/disambiguation-index.json"

type disambiguationGenerateReport struct {
	Schema  string `json:"schema"`
	Path    string `json:"path"`
	SHA256  string `json:"sha256"`
	Bytes   int    `json:"bytes"`
	Changed bool   `json:"changed"`
	Check   bool   `json:"check"`
}

func runDisambiguationGenerate(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation generate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	outputPath := fs.String("output", defaultDisambiguationIndexPath, "generated index path")
	check := fs.Bool("check", false, "fail when the tracked artifact differs")
	jsonOutput := fs.Bool("json", false, "emit JSON report")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation generate: unexpected positional arguments")
		return 2
	}
	generated, err := disambiguation.GeneratePublicIndex()
	if err != nil {
		fmt.Fprintf(stderr, "fak disambiguation generate: %v\n", err)
		return 1
	}
	existing, readErr := os.ReadFile(*outputPath)
	changed := readErr != nil || !bytes.Equal(existing, generated)
	if *check {
		if changed {
			fmt.Fprintf(stderr, "fak disambiguation generate: stale artifact %s; rerun without --check\n", *outputPath)
			return 1
		}
	} else if changed {
		if err := os.MkdirAll(filepath.Dir(*outputPath), 0o755); err != nil {
			fmt.Fprintf(stderr, "fak disambiguation generate: create output directory: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*outputPath, generated, 0o644); err != nil {
			fmt.Fprintf(stderr, "fak disambiguation generate: write %s: %v\n", *outputPath, err)
			return 1
		}
	}
	digest := sha256.Sum256(generated)
	report := disambiguationGenerateReport{Schema: "fak-disambiguation-generate/1", Path: filepath.ToSlash(*outputPath), SHA256: hex.EncodeToString(digest[:]), Bytes: len(generated), Changed: changed, Check: *check}
	if *jsonOutput {
		if err := json.NewEncoder(stdout).Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak disambiguation generate: encode report: %v\n", err)
			return 1
		}
	} else {
		verb := "unchanged"
		if changed && !*check {
			verb = "wrote"
		}
		fmt.Fprintf(stdout, "%s %s sha256:%s\n", verb, report.Path, report.SHA256)
	}
	return 0
}

var committedDisambiguationProbe = probeCommittedDisambiguation

func runDisambiguationCommittedFreshness(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation committed-freshness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation committed-freshness: unexpected positional arguments")
		return 2
	}
	committed, probeErr := committedDisambiguationProbe()
	overlay, generationErr := disambiguation.GeneratePublicIndex()
	if generationErr != nil && probeErr == nil {
		probeErr = generationErr
	}
	report := disambiguation.EvaluateCommittedFreshness(committed, overlay, probeErr)
	if *jsonOutput {
		if err := json.NewEncoder(stdout).Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak disambiguation committed-freshness: encode: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, report.Verdict)
	}
	if report.Verdict == disambiguation.CommittedFreshnessUnavailable {
		return 1
	}
	return 0
}

func probeCommittedDisambiguation() ([]byte, error) {
	temp, err := os.MkdirTemp("", "fak-disambiguation-committed-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(temp)
	archivePath := filepath.Join(temp, "tip.zip")
	archive := exec.Command("git", "archive", "--format=zip", "-o", archivePath, "HEAD")
	configureDispatchHelperCommand(archive)
	if output, err := archive.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("archive committed tip: %w: %s", err, bytes.TrimSpace(output))
	}
	root := filepath.Join(temp, "tip")
	if err := extractDisambiguationArchive(archivePath, root); err != nil {
		return nil, err
	}
	check := exec.Command("go", "run", "./cmd/fak", "disambiguation", "generate", "--check", "--json")
	check.Dir = root
	configureDispatchHelperCommand(check)
	if output, err := check.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("committed-tip regenerate check: %w: %s", err, bytes.TrimSpace(output))
	}
	artifact, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(defaultDisambiguationIndexPath)))
	if err != nil {
		return nil, fmt.Errorf("read committed artifact: %w", err)
	}
	return artifact, nil
}

func extractDisambiguationArchive(archivePath, root string) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("open committed archive: %w", err)
	}
	defer reader.Close()
	for _, file := range reader.File {
		target := filepath.Join(root, filepath.FromSlash(file.Name))
		if !strings.HasPrefix(filepath.Clean(target), filepath.Clean(root)+string(os.PathSeparator)) {
			return fmt.Errorf("archive path escapes root: %s", file.Name)
		}
		if file.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		source, err := file.Open()
		if err != nil {
			return err
		}
		destination, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, file.Mode())
		if err != nil {
			source.Close()
			return err
		}
		_, copyErr := io.Copy(destination, source)
		closeErr := destination.Close()
		source.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

func runDisambiguationVersion(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation version", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation version: unexpected positional arguments")
		return 2
	}
	version, err := disambiguation.CurrentIndexVersion()
	if err != nil {
		fmt.Fprintf(stderr, "fak disambiguation version: %v\n", err)
		return 1
	}
	if *jsonOutput {
		if err := json.NewEncoder(stdout).Encode(version); err != nil {
			fmt.Fprintf(stderr, "fak disambiguation version: encode: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "%s entries=%d source=%s sha256:%s\n", version.IndexSchema, version.EntryCount, version.SourceRevision, version.ContentSHA256)
	return 0
}

func runDisambiguationSchema(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak disambiguation schema", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit the schema descriptor as JSON")
	selfTest := fs.Bool("self-test", false, "run the hermetic complete/required-omission contract witness")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation schema: positional arguments are not accepted")
		return 2
	}

	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if *selfTest {
		report, err := disambiguation.RunSelfTest()
		if err != nil {
			fmt.Fprintf(stderr, "disambiguation schema self-test: FAIL: %v\n", err)
			return 1
		}
		if *jsonOutput {
			if err := enc.Encode(report); err != nil {
				fmt.Fprintf(stderr, "encode self-test report: %v\n", err)
				return 1
			}
			return 0
		}
		fmt.Fprintf(stdout, "PASS %s: complete record accepted; %d required omissions rejected\n", report.Schema, len(report.OmissionsRejected))
		return 0
	}

	if *jsonOutput {
		if err := enc.Encode(disambiguation.Descriptor()); err != nil {
			fmt.Fprintf(stderr, "encode schema descriptor: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "%s (strict exact-version JSON; run with --json for required fields or --self-test for the contract witness)\n", disambiguation.EntrySchemaVersion)
	return 0
}

func runDisambiguationStaleSymbolsSelfTest(stdout, stderr io.Writer, args []string) int {
	flags := flag.NewFlagSet("disambiguation stale-symbols-self-test", flag.ContinueOnError)
	flags.SetOutput(stderr)
	jsonOutput := flags.Bool("json", false, "emit structured JSON")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak disambiguation stale-symbols-self-test [--json]")
		return 2
	}
	report := disambiguation.StaleSymbolsSelfCheck()
	if *jsonOutput {
		encoder := json.NewEncoder(stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(report); err != nil {
			fmt.Fprintf(stderr, "disambiguation stale-symbols-self-test: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "stale-symbols-self-test passed=%t fresh=%s stale=%s reason=%s\n", report.Passed, report.Fresh.Verdict, report.Stale.Verdict, report.Stale.ReasonCode)
	}
	if !report.Passed {
		return 1
	}
	return 0
}

func runDisambiguationCoverageSelfTest(stdout, stderr io.Writer, args []string) int {
	flags := flag.NewFlagSet("disambiguation coverage-self-test", flag.ContinueOnError)
	flags.SetOutput(stderr)
	jsonOutput := flags.Bool("json", false, "emit structured JSON")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak disambiguation coverage-self-test [--json]")
		return 2
	}
	report := disambiguation.CoverageSelfCheck()
	if *jsonOutput {
		encoder := json.NewEncoder(stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(report); err != nil {
			fmt.Fprintf(stderr, "disambiguation coverage-self-test: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "coverage-self-test passed=%t detected=%t covered=%t absent_from_query=%t classification=%s reason=%s\n", report.Passed, report.Detected, report.Covered, report.AbsentFromQuery, report.Classification, report.ClassificationReason)
	}
	if !report.Passed {
		return 1
	}
	return 0
}

func runDisambiguationQuery(stdout, stderr io.Writer, args []string) int {
	var jsonOutput, selfTest bool
	var scope disambiguation.Scope
	var terms []string
	for n := 0; n < len(args); n++ {
		arg := args[n]
		switch arg {
		case "--json":
			jsonOutput = true
		case "--self-test":
			selfTest = true
		case "--scope-kind", "--scope-value":
			if n+1 >= len(args) {
				fmt.Fprintf(stderr, "fak disambiguation query: %s requires a value\n", arg)
				return 2
			}
			n++
			if arg == "--scope-kind" {
				scope.Kind = args[n]
			} else {
				scope.Value = args[n]
			}
		default:
			if strings.HasPrefix(arg, "-") {
				fmt.Fprintf(stderr, "fak disambiguation query: unknown option %q\n", arg)
				return 2
			}
			terms = append(terms, arg)
		}
	}
	if (scope.Kind == "") != (scope.Value == "") {
		fmt.Fprintln(stderr, "fak disambiguation query: --scope-kind and --scope-value are required together")
		return 2
	}
	if selfTest {
		if len(terms) != 0 || scope.Kind != "" {
			fmt.Fprintln(stderr, "fak disambiguation query: --self-test does not accept a term or scope")
			return 2
		}
		report, err := disambiguation.RunQuerySelfTest()
		if err != nil {
			fmt.Fprintf(stderr, "disambiguation query self-test: FAIL: %v\n", err)
			return 1
		}
		if jsonOutput {
			return encodeDisambiguationJSON(stdout, stderr, report)
		}
		fmt.Fprintf(stdout, "PASS %s: alias %q resolved to canonical term %q with a complete %s record\n", report.Schema, report.MatchedAlias, report.CanonicalTerm, report.EntrySchema)
		return 0
	}
	if len(terms) != 1 {
		fmt.Fprintln(stderr, "usage: fak disambiguation query <term> [--json]")
		return 2
	}
	var response disambiguation.QueryResponse
	var err error
	if scope.Kind != "" {
		response, err = disambiguation.ResolveScoped(terms[0], scope)
	} else {
		response, err = disambiguation.Resolve(terms[0])
	}
	if err != nil {
		if errors.Is(err, disambiguation.ErrScopeRequired) {
			fmt.Fprintf(stderr, "fak disambiguation query: %v; use --scope-kind and --scope-value\n", err)
			return 3
		}
		if errors.Is(err, disambiguation.ErrCanonicalTermNotFound) {
			fmt.Fprintf(stderr, "fak disambiguation query: %v\n", err)
			return 3
		}
		fmt.Fprintf(stderr, "fak disambiguation query: %v\n", err)
		return 1
	}
	if jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, response)
	}
	fmt.Fprintf(stdout, "%s — %s\n", response.Entry.Identity.CanonicalTerm, response.Entry.Definition)
	return 0
}

func encodeDisambiguationJSON(stdout, stderr io.Writer, value any) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(value); err != nil {
		fmt.Fprintf(stderr, "encode disambiguation JSON: %v\n", err)
		return 1
	}
	return 0
}

func runDisambiguationProvenance(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation provenance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	selfTest := fs.Bool("self-test", false, "run strict public provenance acceptance and rejection fixtures")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !*selfTest || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak disambiguation provenance --self-test [--json]")
		return 2
	}
	report := disambiguation.ProvenanceSelfCheck()
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "disambiguation provenance: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "disambiguation provenance self-test: ok=%t round_trip=%t reject_absolute=%t reject_escape=%t reject_kind=%t\n", report.OK, report.RoundTrip, report.RejectedAbsolute, report.RejectedEscape, report.RejectedSourceKind)
	}
	if !report.OK {
		return 1
	}
	return 0
}

func runDisambiguationFreshness(stdout, stderr io.Writer, args []string) int {
	jsonOutput, selfTest := false, false
	for _, arg := range args {
		switch arg {
		case "--json":
			jsonOutput = true
		case "--self-test":
			selfTest = true
		default:
			fmt.Fprintf(stderr, "fak disambiguation freshness: unknown option %q\n", arg)
			return 2
		}
	}
	if !selfTest {
		fmt.Fprintln(stderr, "usage: fak disambiguation freshness --self-test [--json]")
		return 2
	}
	report := disambiguation.FreshnessSelfCheck()
	if jsonOutput {
		if code := encodeDisambiguationJSON(stdout, stderr, report); code != 0 {
			return code
		}
	} else {
		for _, row := range report.Cases {
			fmt.Fprintf(stdout, "%s\t%s\t%t\n", row.Verdict, row.ReasonCode, row.Passed)
		}
	}
	if !report.Passed {
		return 1
	}
	return 0
}

func runDisambiguationOwnership(stdout, stderr io.Writer, args []string) int {
	jsonOutput, selfTest := false, false
	for _, arg := range args {
		switch arg {
		case "--json":
			jsonOutput = true
		case "--self-test":
			selfTest = true
		default:
			fmt.Fprintf(stderr, "fak disambiguation ownership: unknown option %q\n", arg)
			return 2
		}
	}
	if !selfTest {
		fmt.Fprintln(stderr, "usage: fak disambiguation ownership --self-test [--json]")
		return 2
	}
	report := disambiguation.OwnershipSelfCheck()
	if jsonOutput {
		if code := encodeDisambiguationJSON(stdout, stderr, report); code != 0 {
			return code
		}
	} else {
		fmt.Fprintf(stdout, "ownership self-test: accepted=%t rejected_leaf=%t rejected_lane=%t\n", report.AcceptedFixture, report.RejectedLeaf, report.RejectedLane)
	}
	if !report.OK {
		return 1
	}
	return 0
}
