// This file holds the READOUT hop for the eight flow KPIs (#6198, epic #6194):
// the argv-to-stdout half of `fak score flow`.
//
// WHY IT LIVES IN THE PACKAGE RATHER THAN IN cmd/fak. Everything above it —
// the gather shell, the fold, the grading, and RenderAging — is here, so a
// handler in cmd/fak would be a fourth place that knows the order of the four
// calls and the two failure modes between them. Keeping the whole hop here
// makes `cmd/fak/score_flow.go` a two-line delegate and, more importantly,
// makes the hop testable: score_test.go runs the real argv path against a
// fixture repository, which a test under cmd/fak could not do while that
// package's test binary is blocked on unrelated work.
//
// IT IS NOT A GATE. Flow debt is a measurement, and emitting an ACTION verdict
// with a non-zero debt is the CORRECT behaviour here; RunScore still returns 0
// so a control-pane collection or a superloop walk reads the payload rather
// than a failure. Refusal on a threshold is a separate child of #6194.
//
// The JSON payload is the Report value marshalled AS IS. It is not re-shaped
// into a bespoke struct or rebuilt as a map on the way out, because the control
// pane indexes each KPI's `defects` and `soft` arrays and Build already
// guarantees they marshal as `[]` and never as `null`. Any re-shaping here
// would be a second place for that invariant to rot.

package flowmetrics

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"
)

// RunScore is the `fak score flow` handler: it parses argv, gathers, folds, and
// prints. defaultRoot is the workspace to measure when --workspace is not given
// (cmd/fak passes the repo root); it is a parameter rather than a cwd lookup so
// a test can point the whole hop at a fixture repository.
//
// The return value is a process exit code: 0 on a completed readout (whatever
// its verdict), 2 on a usage error or a gather that could not run at all.
func RunScore(ctx context.Context, stdout, stderr io.Writer, argv []string, defaultRoot string) int {
	fs := flag.NewFlagSet("fak score flow", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the machine payload (the flowmetrics.Report envelope)")
	window := fs.Int("window", 0, "days of history the rate/efficiency axes fold over (0 = 30)")
	limit := fs.Int("limit", 0, "max issues to fetch (0 = every issue)")
	commits := fs.Int("commits", 0, "max commits to read from git log (0 = the whole history)")
	issuesFile := fs.String("issues-file", "", "read issues from a saved `gh issue list --json` dump instead of calling gh")
	agingLimit := fs.Int("aging-limit", 0, "cap on the emitted aging-WIP list (0 = 25)")
	probeBuild := fs.Bool("probe-build", false, "also compile the tree so the local-WIP axis can grade buildability")
	workspace := fs.String("workspace", "", "workspace root (default: the repo root)")
	var aboutToTouch []string
	fs.Func("touch", "repository-relative path this session is about to edit (repeatable)", func(path string) error {
		if path = strings.TrimSpace(path); path == "" {
			return errors.New("--touch requires a non-empty path")
		}
		aboutToTouch = append(aboutToTouch, path)
		return nil
	})
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	// A positional here is a typo'd flag. Folding a corpus nobody asked for is
	// worse than refusing, because the reading would still look authoritative.
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "score flow: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = defaultRoot
	}

	// Issues come from the plain `gh issue list --limit` path, which paginates
	// over GraphQL and is exact. The --issues-file escape replays a saved dump
	// so a run can be offline and deterministic; neither path adds a --search
	// filter, which would silently cap the corpus at 1000 results and shrink
	// every denominator without saying so.
	var (
		issues []Issue
		err    error
	)
	if *issuesFile != "" {
		issues, err = LoadIssuesFile(*issuesFile)
	} else {
		issues, err = GatherIssues(ctx, root, "all", *limit)
	}
	if err != nil {
		fmt.Fprintf(stderr, "score flow: gather issues: %v\n", err)
		return 2
	}
	commitRows, err := GatherCommits(ctx, root, *commits)
	if err != nil {
		fmt.Fprintf(stderr, "score flow: gather commits: %v\n", err)
		return 2
	}

	now := time.Now().UTC()
	// A tree census that cannot be taken leaves TreeWIP at its zero value, which
	// the fold reports as UNMEASURED rather than clean. That is the fail-safe: a
	// skipped gather must never read as an empty working tree.
	tree, terr := GatherTree(ctx, root, now)
	if terr != nil {
		fmt.Fprintf(stderr, "score flow: working-tree census unavailable (%v); local_wip stays unmeasured\n", terr)
		tree = TreeWIP{}
	} else if *probeBuild {
		ProbeBuild(ctx, root, &tree)
	}

	rep := Build(Input{
		Issues:       issues,
		Commits:      commitRows,
		Tree:         tree,
		Now:          now,
		WindowDays:   *window,
		AgingLimit:   *agingLimit,
		Workspace:    root,
		AboutToTouch: aboutToTouch,
	})

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "score flow: encode json: %v\n", err)
			return 2
		}
		return 0
	}
	RenderScore(stdout, rep, now)
	return 0
}

// RenderScore writes the human readout for a report Build already produced.
//
// flow_debt is a COUNT of tripped axes, not a percentage, so it is rendered bare
// and labelled "defect(s)" — printing it with a percent sign would make a debt
// of 7 read as 7%.
//
// now must be the instant the report was built at: the aging section measures
// every age against it, so taking a second clock reading here would print ages
// that disagree with the graded ones.
func RenderScore(w io.Writer, rep Report, now time.Time) {
	fmt.Fprintf(w, "flow metrics — %s: %s (grade %v, %v defect(s))\n",
		rep.Verdict, rep.Finding, rep.Corpus["grade"], rep.Corpus["flow_debt"])
	fmt.Fprintf(w, "corpus: %v issues, %v commits, %v spans over a %v-day window\n",
		rep.Corpus["issues"], rep.Corpus["commits"], rep.Corpus["spans"], rep.Corpus["window_days"])
	for _, k := range rep.KPIs {
		mark := "ok"
		if len(k.Defects) > 0 {
			mark = "DEFECT"
		}
		fmt.Fprintf(w, "  %-19s %3d/100 %-6s %s\n", k.KPI, k.Score, mark, k.Detail)
		for _, d := range k.Defects {
			fmt.Fprintf(w, "      ! %s\n", d)
		}
		for _, s := range k.Soft {
			fmt.Fprintf(w, "      ~ %s\n", s)
		}
	}
	// The aging list is printed unconditionally, right above `next`, because the
	// aging_wip axis is the only one whose remedy is a NAMED set of issues: every
	// other axis is answered by changing intake, but this one is answered by
	// finishing specific rotting work, and a count alone names nothing.
	RenderAging(w, rep, now)
	RenderArrivalServiceReadout(w, rep, now)
	fmt.Fprintf(w, "reason: %s\nnext: %s\n", rep.Reason, rep.NextAction)
}
