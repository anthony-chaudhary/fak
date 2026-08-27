package main

// qaprocessscore.go wires the `fak score qa-process` verb over internal/qaprocessscore:
// the E-testing-quality track's "is our test process honest?" card. It folds the QA-process
// KPIs -- today regression_catch (#3841): every reverted landing that came back off the trunk
// with no accompanying regression test -- into one control-pane payload and a worst-first work
// list.
//
// The git-history read (exec is a CLI concern) stays HERE, reusing the same bounded
// --commits window reader the brittleness card uses, degrading to "no history findings" when
// git is unavailable. The pure KPI folds live in internal/qaprocessscore.
//
// Per the #2247 consolidation convention this card lands ONLY under `fak score qa-process`,
// never as a top-level *-scorecard verb (score.go, scoreRoutes).

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/mutationefficacy"
	"github.com/anthony-chaudhary/fak/internal/qaprocessscore"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cmdQAProcessScorecard(argv []string) {
	os.Exit(runQAProcessScorecard(os.Stdout, os.Stderr, argv))
}

func runQAProcessScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score qa-process", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	commitsN := fs.Int("commits", 200, "size of the recent-commit window scanned for reverted landings (0 disables)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit the committed snapshot body")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	coverProfile := fs.String("coverprofile", "", "path to a `go test -coverprofile` file; when set, folds in the coverage_discipline KPI")
	coverBaseline := fs.String("coverage-baseline", "", "path to a JSON coverage ratchet {floor, epsilon, per_package}")
	coverFloor := fs.Float64("coverage-floor", 0, "absolute statement-coverage floor % (a --coverage-baseline that sets floor wins)")
	racedCSV := fs.String("raced", "", "comma-separated packages exercised under -race (empty => race unmeasured: one SOFT note, never a HARD fail)")
	var mutatePkgs stringList
	fs.Var(&mutatePkgs, "mutate-pkg", "package dir to mutation-probe for the mutation_efficacy KPI (repeatable; empty => KPI omitted); relative paths join --workspace")
	mutateCap := fs.Int("mutate-cap", 8, "per-package mutant cap for --mutate-pkg (iteration cap; 0 => uncapped)")
	mutateTimeout := fs.Int("mutate-timeout", 120, "per-mutant `go test` timeout in seconds for --mutate-pkg (time cap)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak score qa-process: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *commitsN < 0 {
		fmt.Fprintln(stderr, "fak score qa-process: --commits must be >= 0")
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	// The recent-commit window feeds the git-history KPIs. A git failure (not a repo, git
	// absent) is not fatal: the card just carries no history findings and scores clean.
	commits := readCommitWindowForBrittleness(root, *commitsN)
	kpis := []scorecard.KPI{
		qaprocessscore.RegressionCatch(commits),
	}

	// coverage_discipline (#3834) is opt-in on a supplied --coverprofile: an unmeasured card
	// omits the KPI rather than presenting a hollow 100 -- absence is absence, not a green
	// (the anti-fail-open discipline the epic turns on, #3833).
	if *coverProfile != "" {
		kpi, err := coverageDisciplineKPI(*coverProfile, *coverBaseline, *coverFloor, *racedCSV)
		if err != nil {
			fmt.Fprintf(stderr, "fak score qa-process: %v\n", err)
			return 1
		}
		kpis = append(kpis, kpi)
	}

	// mutation_efficacy (#3845) is opt-in on a supplied --mutate-pkg allow-list: it MUTATES
	// real source on disk (restore-always) and shells `go test`, so it never runs unless an
	// operator names the packages. With no allow-list the KPI is omitted (absence is absence,
	// not a green). Survivors are SOFT -- the probe can never gate the card.
	if len(mutatePkgs) > 0 {
		kpis = append(kpis, mutationEfficacyKPI(root, []string(mutatePkgs), *mutateCap, *mutateTimeout))
	}

	payload := qaprocessscore.Compose(kpis)

	return emitScorecard(stdout, stderr, "fak score qa-process", qaprocessscore.DebtKey, payload,
		*comparePath, *asJSON, *asMarkdown, scorecard.MarkdownDoc{
			Title: "fak qa-process scorecard - is our test process honest?",
			Description: "fak's deterministic QA-process card: not whether the tests pass, but whether the " +
				"process around them holds -- a reverted landing that came back with no regression test to catch " +
				"its return (regression_catch), a skipped/disabled test carrying no ticket (skip_debt), coverage " +
				"that slipped below its ratchet (coverage_discipline). Ranked worst-first as an in-tree-mendable " +
				"work list.",
			Heading: "QA-process scorecard",
			AutoGen: "Auto-generated by `fak score qa-process --markdown`. Do not hand-edit; re-run the tool.",
			Law: "The law: a green suite is not an honest suite. A bug that reached the trunk and was reverted, " +
				"with nothing added to catch its return, is a hole the next regression falls through -- so each " +
				"such gap is captured worst-first as HARD, fixable debt, not a passing grade.",
			DebtKey:     qaprocessscore.DebtKey,
			HeaderExtra: fmt.Sprintf(" - %d commit(s) scanned", len(commits)),
		})
}

// mutationEfficacyKPI runs the bounded mutation probe over the --mutate-pkg allow-list and folds
// the survivors into the mutation_efficacy SOFT KPI (#3845). It is the CLI's I/O boundary: the
// mutate/restore + `go test` runner live in internal/mutationefficacy; this seam only resolves
// each package path against the workspace root and picks the real `go test` runner. Relative
// paths join root so `--mutate-pkg internal/foo` probes the in-tree package.
func mutationEfficacyKPI(root string, pkgs []string, capN, timeoutSec int) scorecard.KPI {
	if timeoutSec <= 0 {
		timeoutSec = 120
	}
	run := mutationefficacy.GoTestRunner(time.Duration(timeoutSec) * time.Second)
	results := make([]mutationefficacy.PackageResult, 0, len(pkgs))
	for _, p := range pkgs {
		dir := p
		if !filepath.IsAbs(dir) {
			dir = filepath.Join(root, p)
		}
		results = append(results, mutationefficacy.ProbePackage(dir, run, capN))
	}
	return mutationefficacy.Fold(results)
}

// coverageDisciplineKPI reads the coverage profile (and optional ratchet baseline) off disk and
// folds them into the coverage_discipline KPI (#3834). All the arithmetic lives in
// internal/qaprocessscore; this seam is the CLI's I/O boundary. A --coverage-baseline JSON that
// omits "floor" falls back to the --coverage-floor flag, so the simple case needs only the flag.
func coverageDisciplineKPI(profilePath, baselinePath string, floor float64, racedCSV string) (scorecard.KPI, error) {
	cov, base, err := readCoverageInputs(profilePath, baselinePath, floor)
	if err != nil {
		return scorecard.KPI{}, err
	}

	// An empty --raced set means race was not measured (nil -> one honest SOFT note), NOT that
	// every package raced dirty. Only a non-empty list narrows to a measured raced set.
	var raced map[string]bool
	if fields := splitCSV(racedCSV); len(fields) > 0 {
		raced = make(map[string]bool, len(fields))
		for _, f := range fields {
			raced[f] = true
		}
	}

	return qaprocessscore.CoverageDiscipline(cov, base, raced), nil
}

// readCoverageInputs reads the coverage profile (and optional ratchet baseline) off disk into
// the pure fold inputs (per-package coverage + the baseline). It is the shared I/O boundary for
// both the coverage_discipline KPI and the qa-process-debt dispatcher, so the two read the
// ratchet identically. A --coverage-baseline JSON that omits "floor" falls back to the flag.
func readCoverageInputs(profilePath, baselinePath string, floor float64) (map[string]float64, qaprocessscore.CoverageBaseline, error) {
	prof, err := os.ReadFile(profilePath)
	if err != nil {
		return nil, qaprocessscore.CoverageBaseline{}, fmt.Errorf("reading --coverprofile: %w", err)
	}
	base := qaprocessscore.CoverageBaseline{Floor: floor}
	if baselinePath != "" {
		raw, err := os.ReadFile(baselinePath)
		if err != nil {
			return nil, qaprocessscore.CoverageBaseline{}, fmt.Errorf("reading --coverage-baseline: %w", err)
		}
		if err := json.Unmarshal(raw, &base); err != nil {
			return nil, qaprocessscore.CoverageBaseline{}, fmt.Errorf("parsing --coverage-baseline JSON: %w", err)
		}
		if base.Floor == 0 { // a baseline file that omits floor inherits the flag
			base.Floor = floor
		}
	}
	return qaprocessscore.ParseCoverProfile(string(prof)), base, nil
}

func cmdQAProcessDebtDispatch(argv []string) {
	os.Exit(runQAProcessDebtDispatch(os.Stdout, os.Stderr, argv))
}

// runQAProcessDebtDispatch is the fan-out half of the qa-process card (#3844): one deduped,
// scoped GitHub issue per HARD qa_process_debt gap (a reverted landing with no regression test,
// a package below its coverage ratchet), via the internal/dogfoodissues backlog bridge. Dry-run
// by default; --live files and dedups. It mirrors runUnwiredDebtDispatch exactly -- same
// dedup/cap/existing-json/fetch-existing/live seam -- adding only the --commits window and the
// coverage inputs the gap set is folded from.
func runQAProcessDebtDispatch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak qa-process-debt-dispatch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	project := addDogfoodProjectFlags(fs)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	commitsN := fs.Int("commits", 200, "size of the recent-commit window scanned for reverted landings (0 disables)")
	coverProfile := fs.String("coverprofile", "", "path to a `go test -coverprofile` file; when set, also fans coverage_discipline gaps")
	coverBaseline := fs.String("coverage-baseline", "", "path to a JSON coverage ratchet {floor, epsilon, per_package}")
	coverFloor := fs.Float64("coverage-floor", 0, "absolute statement-coverage floor % (a --coverage-baseline that sets floor wins)")
	capN := fs.Int("cap", 10, "maximum qa-process gaps to fan out into issues in one run")
	repo := fs.String("repo", "", "owner/repo for gh; default is the current repo")
	limit := fs.Int("limit", 300, "existing-issue scan limit for live/fetch modes")
	existingJSON := fs.String("existing-json", "", "fixture list of existing gh issues for dry-run dedup tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to dedup against existing issue bodies")
	live := fs.Bool("live", false, "create/update GitHub issues with gh")
	asJSON := fs.Bool("json", false, "emit machine-readable plan/result")
	var labels stringList
	fs.Var(&labels, "label", "label to add to newly-created issues; repeatable")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak qa-process-debt-dispatch: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *capN < 0 {
		fmt.Fprintln(stderr, "fak qa-process-debt-dispatch: --cap must be >= 0")
		return 2
	}
	if *commitsN < 0 {
		fmt.Fprintln(stderr, "fak qa-process-debt-dispatch: --commits must be >= 0")
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	commits := readCommitWindowForBrittleness(root, *commitsN)

	// Coverage gaps are opt-in on a supplied --coverprofile, exactly like the score verb; with
	// no profile the dispatcher fans only the regression gaps (absence is absence, never a
	// manufactured coverage gap).
	var cov map[string]float64
	base := qaprocessscore.CoverageBaseline{Floor: *coverFloor}
	if *coverProfile != "" {
		var err error
		cov, base, err = readCoverageInputs(*coverProfile, *coverBaseline, *coverFloor)
		if err != nil {
			fmt.Fprintf(stderr, "fak qa-process-debt-dispatch: %v\n", err)
			return 1
		}
	}

	gaps := qaprocessscore.Gaps(commits, cov, base)
	// Cap bounds how many gaps become issues in one run (creates + updates), so a first live run
	// over a large backlog can't open a flood; dedup makes re-runs converge.
	if *capN >= 0 && len(gaps) > *capN {
		gaps = gaps[:*capN]
	}
	items := qaprocessscore.ActionItems(gaps, "fak score qa-process --json")

	var existing []dogfoodissues.Issue
	switch {
	case *existingJSON != "":
		b, err := os.ReadFile(*existingJSON)
		if err != nil {
			fmt.Fprintf(stderr, "fak qa-process-debt-dispatch: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(b, &existing); err != nil {
			fmt.Fprintf(stderr, "fak qa-process-debt-dispatch: --existing-json must be a JSON list: %v\n", err)
			return 2
		}
	case *live || *fetchExisting:
		var err error
		existing, err = dogfoodissues.FetchExistingIssues(*repo, *limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak qa-process-debt-dispatch: %v\n", err)
			return 2
		}
	}

	buildOpt := dogfoodissues.BuildOptions{
		Live:           *live,
		DedupeChecked:  *existingJSON != "" || *fetchExisting || *live,
		DedupeCap:      *capN,
		ParentIssue:    *project.parent,
		ParentBaseline: project.baseline(), CompletionStandard: *project.standard, TargetEnvelope: *project.target, WitnessedEnvelope: *project.witnessed,
	}
	plan, skipped := dogfoodissues.BuildPlanWithOptions(items, existing, buildOpt)
	cohort := dogfoodissues.CohortPlan(items, buildOpt)

	mode := "dry-run"
	if *live {
		mode = "live"
	}
	result := dogfoodissues.Result{
		Schema:  dogfoodissues.Schema,
		Mode:    mode,
		Report:  "fak score qa-process --json",
		Planned: plan,
		Synced:  []dogfoodissues.SyncRow{},
		Skipped: skipped,
		Cohort:  &cohort,
	}

	exit := 0
	if *live && len(plan) > 0 {
		result.Synced = dogfoodissues.Sync(plan, *repo, []string(labels), nil)
		for _, row := range result.Synced {
			if !row.OK {
				exit = 1
			}
		}
	}

	if code := emitJSONOrPrintln(stdout, stderr, "fak qa-process-debt-dispatch", *asJSON, result, dogfoodissues.Render(result)); code != 0 {
		return code
	}
	return exit
}
