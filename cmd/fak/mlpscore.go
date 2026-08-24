package main

// fak mlp-score reports the witnessed first-lovable-cut grade for epic #3256,
// milestone #17. The pure fold lives in internal/mlpscore (which rides the shared
// pkg/scorecard kernel); this shell captures a committed HEAD snapshot and owns the
// family-standard output surface -- --json (the stable payload, now carrying the
// control-pane corpus), --markdown (the committed snapshot for the published page) and
// --compare (the prove-the-debt-drop regression gate against a prior --json run) -- plus
// the exit behavior, which stays owned by --check.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/internal/mlpscore"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cmdMLPScore(argv []string) {
	os.Exit(runMLPScore(os.Stdout, os.Stderr, argv))
}

func runMLPScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak mlp-score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit the stable scorecard JSON")
	// --markdown is defined in the family-standard form (the shape every sibling card uses)
	// with --md kept as a real alias, so the two flags stay independent bools OR'd together.
	markdownFlag := fs.Bool("markdown", false, "emit the markdown rollup")
	mdAlias := fs.Bool("md", false, "alias for --markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	check := fs.Bool("check", false, "exit 1 when the scorecard is not-yet")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak mlp-score: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	asMarkdown := *markdownFlag || *mdAlias
	if *asJSON && asMarkdown {
		fmt.Fprintln(stderr, "fak mlp-score: choose only one of --json or --markdown")
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	score, err := collectMLPScore(root, time.Now().UTC())
	if err != nil {
		fmt.Fprintf(stderr, "fak mlp-score: %v\n", err)
		return 1
	}

	compareExit := 0
	if *check && !score.Lovable {
		compareExit = 1
	}
	if code, done := emitScorecardComparison(stdout, stderr, "fak mlp-score", *comparePath, compareExit, func(base map[string]any) string {
		return scorecard.Compare(mlpscore.KernelPayload(score), base, mlpscore.DebtKey)
	}); done {
		return code
	}

	switch {
	case *asJSON:
		if err := writeIndentedJSONNoEscape(stdout, score); err != nil {
			fmt.Fprintf(stderr, "fak mlp-score: encode json: %v\n", err)
			return 1
		}
	case asMarkdown:
		fmt.Fprint(stdout, mlpscore.RenderMarkdown(score))
	default:
		fmt.Fprintln(stdout, mlpscore.Render(score))
	}

	if *check && !score.Lovable {
		return 1
	}
	return 0
}

func collectMLPScore(root string, now time.Time) (mlpscore.Score, error) {
	snapshot, err := mlpscore.LoadGitSnapshot(root)
	if err != nil {
		return mlpscore.Score{}, err
	}
	return mlpscore.Grade(snapshot, mlpscore.FoldOpts{
		Workspace:   root,
		Commit:      snapshot.Commit(),
		GeneratedAt: now.Format(time.RFC3339),
		Date:        now.Format("2006-01-02"),
	}), nil
}

func mlpMilestoneScorecard(score mlpscore.Score) milestonereport.ProgramScorecard {
	card := milestonereport.ProgramScorecard{
		Key:       "mlp",
		Milestone: 17,
		Title:     "MLP first lovable cut",
		Verdict:   score.MLPVerdict,
		Witnessed: score.Witnessed,
		Total:     score.Total,
	}
	for _, row := range score.Criteria {
		card.Criteria = append(card.Criteria, milestonereport.ProgramCriterion{
			Workstream: row.Workstream,
			Title:      row.Title,
			Grade:      row.Grade,
			WitnessRef: row.WitnessRef,
		})
	}
	return card
}
