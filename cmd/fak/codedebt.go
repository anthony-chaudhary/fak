package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/codedebt"
)

func cmdCodeDebt(argv []string) {
	os.Exit(runCodeDebt(os.Stdout, os.Stderr, argv))
}

func runCodeDebt(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak code-debt", flag.ContinueOnError)
	fs.SetOutput(stderr)

	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	kpi := fs.String("kpi", "", "filter code debt by KPI (e.g. architecture, tests, format, deps, honesty)")
	category := fs.String("category", "", "filter code debt by category (modularity, internal_consistency, internal_coherence)")
	pathFilter := fs.String("path", "", "filter code debt by file or package path")
	search := fs.String("search", "", "filter code debt by substring search in defect text")
	limit := fs.Int("limit", 0, "limit max defects returned (0 = all)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	countOnly := fs.Bool("count", false, "print matching defect count only")
	summaryOnly := fs.Bool("summary", false, "print aggregated summary of debt categories and KPIs")
	deterministic := fs.Bool("deterministic", true, "tree-deterministic mode (ignore sliding commit-history window)")
	fromFile := fs.String("from", "", "query an existing scorecard JSON file instead of scanning")
	_ = fs.Bool("native", true, "run native Go code-debt engine instead of Python checker")

	if !parseFlags(fs, argv) {
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	// Positional arguments can provide a path filter: `fak code-debt internal/gateway`
	if *pathFilter == "" && fs.NArg() > 0 {
		*pathFilter = fs.Arg(0)
	}

	var report *codedebt.Report
	var err error

	if *fromFile != "" {
		data, rErr := os.ReadFile(*fromFile)
		if rErr != nil {
			fmt.Fprintf(stderr, "fak code-debt: read %s: %v\n", *fromFile, rErr)
			return 1
		}
		report, err = codedebt.ParsePayload(data)
		if err != nil {
			fmt.Fprintf(stderr, "fak code-debt: parse %s: %v\n", *fromFile, err)
			return 1
		}
	} else {
		report, err = codedebt.ScanTree(codedebt.ScanOptions{
			Workspace:     root,
			Deterministic: *deterministic,
		})
		if err != nil {
			fmt.Fprintf(stderr, "fak code-debt: scan workspace %s: %v\n", root, err)
			return 1
		}
	}

	queryOpts := codedebt.QueryOptions{
		KPI:           *kpi,
		Category:      codedebt.Category(*category),
		Path:          *pathFilter,
		Package:       *pathFilter,
		Search:        *search,
		Limit:         *limit,
		Deterministic: *deterministic,
	}

	result := report.Query(queryOpts)

	if *countOnly {
		fmt.Fprintln(stdout, result.MatchedDebt)
		if result.MatchedDebt > 0 {
			return 1
		}
		return 0
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak code-debt: encode json: %v\n", err)
			return 1
		}
		if result.MatchedDebt > 0 {
			return 1
		}
		return 0
	}

	if *summaryOnly {
		fmt.Fprint(stdout, result.FormatSummary())
		if result.MatchedDebt > 0 {
			return 1
		}
		return 0
	}

	fmt.Fprint(stdout, result.FormatText())
	if result.MatchedDebt > 0 {
		return 1
	}
	return 0
}
