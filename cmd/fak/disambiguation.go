package main

import (
	"archive/zip"
	"bytes"
	"context"
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
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func cmdDisambiguation(args []string) {
	os.Exit(runDisambiguation(os.Stdout, os.Stderr, args))
}

func runDisambiguation(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak disambiguation issue-suggest-self-test [--json]\n       fak disambiguation migration-self-test [--json]\n       fak disambiguation diff --before FILE --after FILE [--json]\n       fak disambiguation metrics [--json]\n       fak disambiguation schema [--json] [--self-test]\n       fak disambiguation query <canonical-term> [--json]\n       fak disambiguation query --self-test [--json]\n       fak disambiguation search <term> [--json]\n       fak disambiguation reverse --kind source-path|symbol|cli-token|reason-code <locator> [--json]\n       fak disambiguation reverse --self-test [--json]\n       fak disambiguation cli-source [--json] [--self-test]\n       fak disambiguation docs [--output-dir DIR] [--check] [--json]\n       fak disambiguation ownership-source-self-test [--json]\n       fak disambiguation go-source-self-test [--json]\n       fak disambiguation lifecycle-source-self-test [--json]\n       fak disambiguation claims-source-self-test [--json]\n       fak disambiguation policy-source-self-test [--json]\n       fak disambiguation fleet-source-self-test [--json]\n       fak disambiguation runtime-source-self-test [--json]\n       fak disambiguation reason-source-self-test [--json]\n       fak disambiguation cache-source-self-test [--json]\n       fak disambiguation session-source-self-test [--json]\n       fak disambiguation stale-symbols-self-test [--json]\n       fak disambiguation coverage-self-test [--json]")
		return 2
	}
	switch args[0] {
	case "issue-suggest-self-test":
		return runDisambiguationIssueSuggestSelfTest(stdout, stderr, args[1:])
	case "migration-self-test":
		return runDisambiguationMigrationSelfTest(stdout, stderr, args[1:])
	case "diff":
		return runDisambiguationDiff(stdout, stderr, args[1:])
	case "metrics":
		return runDisambiguationMetrics(stdout, stderr, args[1:])
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
	case "search":
		return runDisambiguationSearch(stdout, stderr, args[1:])
	case "reverse":
		return runDisambiguationReverse(stdout, stderr, args[1:])
	case "cli-source":
		return runDisambiguationCLISource(stdout, stderr, args[1:])
	case "docs":
		return runDisambiguationDocs(stdout, stderr, args[1:])
	case "explain":
		return runDisambiguationExplain(stdout, stderr, args[1:])
	case "ownership":
		return runDisambiguationOwnership(stdout, stderr, args[1:])
	case "freshness":
		return runDisambiguationFreshness(stdout, stderr, args[1:])
	case "provenance":
		return runDisambiguationProvenance(stdout, stderr, args[1:])
	case "ownership-source-self-test":
		return runDisambiguationOwnershipSourceSelfTest(stdout, stderr, args[1:])
	case "go-source-self-test":
		return runDisambiguationGoSourceSelfTest(stdout, stderr, args[1:])
	case "lifecycle-source-self-test":
		return runDisambiguationLifecycleSourceSelfTest(stdout, stderr, args[1:])
	case "claims-source-self-test":
		return runDisambiguationClaimsSourceSelfTest(stdout, stderr, args[1:])
	case "policy-source-self-test":
		return runDisambiguationPolicySourceSelfTest(stdout, stderr, args[1:])
	case "fleet-source-self-test":
		return runDisambiguationFleetSourceSelfTest(stdout, stderr, args[1:])
	case "runtime-source-self-test":
		return runDisambiguationRuntimeSourceSelfTest(stdout, stderr, args[1:])
	case "reason-source-self-test":
		return runDisambiguationReasonSourceSelfTest(stdout, stderr, args[1:])
	case "cache-source-self-test":
		return runDisambiguationCacheSourceSelfTest(stdout, stderr, args[1:])
	case "session-source-self-test":
		return runDisambiguationSessionSourceSelfTest(stdout, stderr, args[1:])
	case "stale-symbols-self-test":
		return runDisambiguationStaleSymbolsSelfTest(stdout, stderr, args[1:])
	case "coverage-self-test":
		return runDisambiguationCoverageSelfTest(stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "fak disambiguation: unknown command %q (want issue-suggest-self-test, migration-self-test, diff, metrics, schema, query, search, reverse, cli-source, docs, explain, ownership, freshness, provenance, ownership-source-self-test, go-source-self-test, lifecycle-source-self-test, claims-source-self-test, policy-source-self-test, fleet-source-self-test, runtime-source-self-test, reason-source-self-test, cache-source-self-test, session-source-self-test, stale-symbols-self-test, or coverage-self-test)\n", args[0])
		return 2
	}
}

func runDisambiguationIssueSuggestSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation issue-suggest-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation issue-suggest-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunIssueSuggestionSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation issue-suggest self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: title=%q unsafe rejected=%t auto-file=%t\n", report.Schema, report.Suggestion.Title, report.UnsafeRejected, !report.NoAutoFile)
	return 0
}

func runDisambiguationMigrationSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation migration-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation migration-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunMigrationSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation migration self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: silent removal rejected=%t; versioned alias accepted=%t replacement=%s\n", report.Schema, report.SilentRemovalRejected, report.VersionedAliasAccepted, report.ReplacementTarget)
	return 0
}

func runDisambiguationDiff(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation diff", flag.ContinueOnError)
	fs.SetOutput(stderr)
	beforePath := fs.String("before", "", "generated index before change")
	afterPath := fs.String("after", "", "generated index after change")
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *beforePath == "" || *afterPath == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation diff: --before and --after are required")
		return 2
	}
	beforeData, err := os.ReadFile(*beforePath)
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation diff: %v\n", err)
		return 1
	}
	afterData, err := os.ReadFile(*afterPath)
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation diff: %v\n", err)
		return 1
	}
	before, err := disambiguation.DecodeGeneratedIndex(beforeData)
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation diff: %v\n", err)
		return 1
	}
	after, err := disambiguation.DecodeGeneratedIndex(afterData)
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation diff: %v\n", err)
		return 1
	}
	report := disambiguation.DiffIndexes(before, after)
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	for _, change := range report.Changes {
		fmt.Fprintf(stdout, "%s %s impact=%s - %s\n", change.Kind, change.CanonicalTerm, change.QueryImpact, change.Detail)
	}
	return 0
}

func runDisambiguationMetrics(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation metrics", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation metrics: unexpected positional arguments")
		return 2
	}
	report := disambiguation.PublicMetrics()
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "total=%d freshness=%d owners=%d source_families=%d uncovered_classes=%d\n", report.Total, disambiguation.SumMetrics(report.Freshness), disambiguation.SumMetrics(report.Owners), len(report.SourceFamilies), len(report.UncoveredCandidateClasses))
	return 0
}

func runDisambiguationOwnershipSourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation ownership-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation ownership-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunOwnershipSourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation ownership-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: %s leaf=%s lane=%s stamp=%s; mismatches typed=%t/%t/%t\n", report.Schema, report.Binding.ModuleAtRev, report.Binding.Leaf, report.Binding.Lane, report.Binding.Stamp, report.LeafMismatchTyped, report.LaneMismatchTyped, report.StampMismatchTyped)
	return 0
}

func runDisambiguationGoSourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation go-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation go-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunGoSourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation go-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: %d candidates; deterministic=%t; tests/generated/unexported excluded=%t/%t/%t\n", report.Schema, len(report.Candidates), report.Deterministic, report.TestsExcluded, report.GeneratedExcluded, report.UnexportedExcluded)
	return 0
}

func runDisambiguationLifecycleSourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation lifecycle-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation lifecycle-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunLifecycleSourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation lifecycle-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: %d ladders; incompatible spellings rejected=%t\n", report.Schema, len(report.Ladders), report.IncompatibleSpellingsRejected)
	return 0
}

func runDisambiguationClaimsSourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation claims-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation claims-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunClaimsSourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation claims-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: %d terms; missing baseline/provenance/scope rejected=%t/%t/%t\n", report.Schema, len(report.CanonicalTerms), report.MissingBaselineRejected, report.MissingProvenanceRejected, report.MissingScopeRejected)
	return 0
}

func runDisambiguationPolicySourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation policy-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation policy-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunPolicySourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation policy-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: %d concepts; structural/model separated=%t; capability/verdict separated=%t; reason collision rejected=%t\n", report.Schema, len(report.Resolutions), report.StructuralBeforeModel, report.CapabilityNotVerdict, report.IncompatibleReasonRejected)
	return 0
}

func runDisambiguationFleetSourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation fleet-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation fleet-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunFleetSourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation fleet-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: %d concepts; narration rejected=%t; structured identity accepted=%t\n", report.Schema, len(report.Resolutions), report.NarrationRejected, report.StructuredAccepted)
	return 0
}

func runDisambiguationRuntimeSourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation runtime-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation runtime-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunRuntimeSourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation runtime-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: unscoped ambiguous=%t; %d scoped runtimes resolved\n", report.Schema, report.UnscopedAmbiguous, len(report.Choices))
	return 0
}

func runDisambiguationReasonSourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation reason-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation reason-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunReasonSourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation reason-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: %d declarations; incompatible duplicate rejected=%t; alias allowed=%t\n", report.Schema, len(report.Terms), report.IncompatibleRejected, report.CrossPackageAliasAllowed)
	return 0
}

func runDisambiguationCacheSourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation cache-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation cache-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunCacheSourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation cache-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: %d cache concepts resolved; pairwise contrasts=%t\n", report.Schema, len(report.Resolutions), report.Pairwise)
	return 0
}

func runDisambiguationSessionSourceSelfTest(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation session-source-self-test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation session-source-self-test: unexpected positional arguments")
		return 2
	}
	report, err := disambiguation.RunSessionSourceSelfTest()
	if err != nil {
		fmt.Fprintf(stderr, "disambiguation session-source self-test: FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "PASS %s: %d terms resolved; resume/recovery distinct=%t; compaction/checkpoint distinct=%t\n", report.Schema, len(report.Resolutions), report.ResumeRecoveryConflation, report.CompactionCheckpointConflation)
	return 0
}

func runDisambiguationReverse(stdout, stderr io.Writer, args []string) int {
	var jsonOutput, selfTest bool
	var kind disambiguation.ReverseLocatorKind
	var values []string
	for n := 0; n < len(args); n++ {
		switch args[n] {
		case "--json":
			jsonOutput = true
		case "--self-test":
			selfTest = true
		case "--kind":
			if n+1 >= len(args) {
				fmt.Fprintln(stderr, "fak disambiguation reverse: --kind requires a value")
				return 2
			}
			n++
			kind = disambiguation.ReverseLocatorKind(args[n])
		default:
			if strings.HasPrefix(args[n], "-") {
				fmt.Fprintf(stderr, "fak disambiguation reverse: unknown option %q\n", args[n])
				return 2
			}
			values = append(values, args[n])
		}
	}
	if selfTest {
		if kind != "" || len(values) != 0 {
			fmt.Fprintln(stderr, "fak disambiguation reverse: --self-test does not accept --kind or a locator")
			return 2
		}
		report, err := disambiguation.RunReverseSelfTest()
		if err != nil {
			fmt.Fprintf(stderr, "disambiguation reverse self-test: FAIL: %v\n", err)
			return 1
		}
		if jsonOutput {
			return encodeDisambiguationJSON(stdout, stderr, report)
		}
		fmt.Fprintf(stdout, "PASS %s: %d locator kinds resolved; unknown input rejected=%t\n", report.Schema, len(report.Cases), report.UnknownRejected)
		return 0
	}
	if kind == "" || len(values) != 1 {
		fmt.Fprintln(stderr, "usage: fak disambiguation reverse --kind source-path|symbol|cli-token|reason-code <locator> [--json]")
		return 2
	}
	response, err := disambiguation.ReverseLookup(kind, values[0])
	if err != nil {
		if jsonOutput {
			_ = json.NewEncoder(stdout).Encode(response)
		}
		fmt.Fprintf(stderr, "fak disambiguation reverse: %v\n", err)
		return 1
	}
	if jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, response)
	}
	for _, match := range response.Matches {
		fmt.Fprintf(stdout, "%s\t%s:%s\t%s\n", match.Entry.Identity.CanonicalTerm, match.Entry.Scope.Kind, match.Entry.Scope.Value, match.MatchedValue)
	}
	return 0
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

func runDisambiguationExplain(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation explain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	scopeKind := fs.String("scope-kind", "", "scope qualifier kind")
	scopeValue := fs.String("scope-value", "", "scope qualifier value")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: fak disambiguation explain <term> [--scope-kind KIND --scope-value VALUE]")
		return 2
	}
	var result disambiguation.QueryResponse
	var err error
	if *scopeKind != "" || *scopeValue != "" {
		result, err = disambiguation.ResolveScoped(fs.Arg(0), disambiguation.Scope{Kind: *scopeKind, Value: *scopeValue})
	} else {
		result, err = disambiguation.Resolve(fs.Arg(0))
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak disambiguation explain: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, disambiguation.Explain(result))
	return 0
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

type disambiguationDocsFile struct {
	Path    string `json:"path"`
	SHA256  string `json:"sha256"`
	Bytes   int    `json:"bytes"`
	Changed bool   `json:"changed"`
}

type disambiguationDocsReport struct {
	Schema string                   `json:"schema"`
	Check  bool                     `json:"check"`
	Stale  []string                 `json:"stale,omitempty"`
	Files  []disambiguationDocsFile `json:"files"`
}

func runDisambiguationDocs(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation docs", flag.ContinueOnError)
	fs.SetOutput(stderr)
	outputDir := fs.String("output-dir", filepath.Join("docs", "generated", "disambiguation"), "generated documentation directory")
	check := fs.Bool("check", false, "fail when the generated tree differs")
	jsonOutput := fs.Bool("json", false, "emit JSON report")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation docs: unexpected positional arguments")
		return 2
	}
	pages, err := disambiguation.RenderPublicDocs()
	if err != nil {
		fmt.Fprintf(stderr, "fak disambiguation docs: %v\n", err)
		return 1
	}
	report := disambiguationDocsReport{Schema: "fak.disambiguation_docs.v1", Check: *check}
	expected := make(map[string]disambiguation.DocPage, len(pages))
	for _, page := range pages {
		expected[filepath.Clean(filepath.FromSlash(page.Path))] = page
	}
	actual, err := disambiguationDocFiles(*outputDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak disambiguation docs: inspect output: %v\n", err)
		return 1
	}
	for rel, page := range expected {
		target := filepath.Join(*outputDir, rel)
		current, readErr := os.ReadFile(target)
		changed := readErr != nil || !bytes.Equal(current, page.Content)
		report.Files = append(report.Files, disambiguationDocsFile{Path: filepath.ToSlash(target), Changed: changed})
		if *check && changed {
			report.Stale = append(report.Stale, filepath.ToSlash(target))
		}
	}
	for rel := range actual {
		if _, ok := expected[rel]; !ok {
			report.Stale = append(report.Stale, filepath.ToSlash(filepath.Join(*outputDir, rel)))
		}
	}
	if !*check {
		if err := os.RemoveAll(*outputDir); err != nil {
			fmt.Fprintf(stderr, "fak disambiguation docs: reset output: %v\n", err)
			return 1
		}
		for _, page := range pages {
			target := filepath.Join(*outputDir, filepath.FromSlash(page.Path))
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return 1
			}
			if err := os.WriteFile(target, page.Content, 0o644); err != nil {
				return 1
			}
		}
	}
	sort.Slice(report.Files, func(i, j int) bool { return report.Files[i].Path < report.Files[j].Path })
	sort.Strings(report.Stale)
	if *jsonOutput {
		if err := json.NewEncoder(stdout).Encode(report); err != nil {
			return 1
		}
	} else {
		for _, file := range report.Files {
			status := "unchanged"
			if file.Changed {
				status = "changed"
			}
			fmt.Fprintf(stdout, "%s %s\n", status, file.Path)
		}
		for _, extra := range report.Stale {
			fmt.Fprintf(stdout, "stale %s\n", extra)
		}
	}
	if *check && len(report.Stale) > 0 {
		fmt.Fprintf(stderr, "fak disambiguation docs: %d stale, missing, or extra file(s)\n", len(report.Stale))
		return 1
	}
	return 0
}

func disambiguationDocFiles(root string) (map[string]struct{}, error) {
	files := map[string]struct{}{}
	err := filepath.WalkDir(root, func(name string, entry os.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, os.ErrNotExist) && name == root {
				return nil
			}
			return err
		}
		if entry.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, name)
		if err != nil {
			return err
		}
		files[filepath.Clean(rel)] = struct{}{}
		return nil
	})
	return files, err
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
	ctxArchive, cancelArchive := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancelArchive()
	archive := exec.CommandContext(ctxArchive, "git", "archive", "--format=zip", "-o", archivePath, "HEAD")
	configureDispatchHelperCommand(archive)
	archive.WaitDelay = 5 * time.Second
	archive.Cancel = func() error {
		if archive.Process != nil && archive.Process.Pid > 0 {
			procguard.KillPID(archive.Process.Pid)
		}
		return nil
	}
	if output, err := archive.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("archive committed tip: %w: %s", err, bytes.TrimSpace(output))
	}
	root := filepath.Join(temp, "tip")
	if err := extractDisambiguationArchive(archivePath, root); err != nil {
		return nil, err
	}
	ctxCheck, cancelCheck := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancelCheck()
	check := exec.CommandContext(ctxCheck, "go", "run", "./cmd/fak", "disambiguation", "generate", "--check", "--json")
	check.Dir = root
	configureDispatchHelperCommand(check)
	check.WaitDelay = 5 * time.Second
	check.Cancel = func() error {
		if check.Process != nil && check.Process.Pid > 0 {
			procguard.KillPID(check.Process.Pid)
		}
		return nil
	}
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

func runDisambiguationCLISource(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("disambiguation cli-source", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit machine-readable JSON")
	selfTest := fs.Bool("self-test", false, "prove addition and stale-removal detection")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak disambiguation cli-source [--json] [--self-test]")
		return 2
	}
	help := usageWallText()
	var prior []disambiguation.CLITerm
	if *selfTest {
		prior = disambiguation.IndexCLISource(help+"  fak removed-fixture --json\n", nil).Terms
		help += "  fak added-fixture inspect --json\n"
	}
	report := disambiguation.IndexCLISource(help, prior)
	if *selfTest {
		if !cliSourceHasTerm(report.Terms, "added-fixture") || !cliSourceHasTerm(report.Stale, "removed-fixture") {
			fmt.Fprintln(stderr, "disambiguation cli-source self-test: FAIL")
			return 1
		}
	}
	if *jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, report)
	}
	fmt.Fprintf(stdout, "%s: %d terms, %d stale\n", report.Schema, len(report.Terms), len(report.Stale))
	return 0
}

func cliSourceHasTerm(terms []disambiguation.CLITerm, term string) bool {
	for _, candidate := range terms {
		if candidate.Term == term {
			return true
		}
	}
	return false
}

func runDisambiguationSearch(stdout, stderr io.Writer, args []string) int {
	var jsonOutput bool
	var terms []string
	for _, arg := range args {
		switch arg {
		case "--json":
			jsonOutput = true
		default:
			if strings.HasPrefix(arg, "-") {
				fmt.Fprintf(stderr, "fak disambiguation search: unknown option %q\n", arg)
				return 2
			}
			terms = append(terms, arg)
		}
	}
	if len(terms) != 1 {
		fmt.Fprintln(stderr, "usage: fak disambiguation search <term> [--json]")
		return 2
	}
	response := disambiguation.Search(terms[0])
	if jsonOutput {
		if code := encodeDisambiguationJSON(stdout, stderr, response); code != 0 {
			return code
		}
	} else {
		fmt.Fprintf(stdout, "%s: %s\n", response.Verdict, response.Query)
		for _, group := range []struct {
			name    string
			matches []disambiguation.SearchMatch
		}{
			{"exact", response.Groups.Exact},
			{"alias", response.Groups.Alias},
			{"prefix", response.Groups.Prefix},
		} {
			if len(group.matches) == 0 {
				continue
			}
			fmt.Fprintf(stdout, "%s:\n", group.name)
			for _, match := range group.matches {
				fmt.Fprintf(stdout, "- %s -> %s [%s=%s]\n", match.MatchedTerm, match.Entry.Identity.CanonicalTerm, match.Entry.Scope.Kind, match.Entry.Scope.Value)
			}
		}
	}
	if response.Verdict == disambiguation.SearchVerdictAmbiguous {
		return 3
	}
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
