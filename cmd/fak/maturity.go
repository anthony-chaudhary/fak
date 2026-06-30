package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/maturity"
)

func cmdMaturity(argv []string) { os.Exit(runMaturity(os.Stdout, os.Stderr, argv)) }

// runMaturity scores where each declared fak capability sits on its lifecycle
// maturity ladder (proposed → prototyped → tested → dogfooded → benchmarked →
// default) and emits the next work item that would mature each one. Immaturity is
// never a defect; a ladder-skip (high-rung evidence over an unmet lower rung) is.
// Exit codes: 0 no ladder-skips, 1 carries skip-debt, 2 usage/IO error.
func runMaturity(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 {
		switch argv[0] {
		case "route", "issues":
			return runMaturityRoute(stdout, stderr, argv[1:])
		}
	}
	fs := flag.NewFlagSet("fak maturity", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	next := fs.Bool("next", false, "emit only the next-work backlog (ladder-skips first)")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	// `fak maturity next` is the natural spelling of the backlog view.
	if fs.NArg() == 1 && fs.Arg(0) == "next" {
		*next = true
	} else if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak maturity: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := maturity.Build(maturity.Options{Root: root})

	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak maturity", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, maturity.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak maturity: encode json: %v\n", err)
			return 1
		}
	case *asMarkdown:
		fmt.Fprint(stdout, maturity.Markdown(payload))
	case *next:
		fmt.Fprintln(stdout, maturity.RenderNext(payload))
	default:
		fmt.Fprintln(stdout, maturity.Render(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}

// runMaturityRoute is the backlog bridge: it turns the top `fak maturity next`
// items into stable, deduped GitHub issue plans. It is dry-run by default; only
// --live shells out to gh issue create/edit.
func runMaturityRoute(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak maturity route", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	repo := fs.String("repo", "", "owner/repo for gh; default is current repo")
	limit := fs.Int("limit", 3, "number of ranked maturity backlog items to route (0 = all)")
	existingJSON := fs.String("existing-json", "", "fixture/list of existing gh issues for dry-run tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to classify create vs update")
	live := fs.Bool("live", false, "create/update GitHub issues with gh")
	asJSON := fs.Bool("json", false, "emit machine-readable plan/result")
	var labels stringList
	fs.Var(&labels, "label", "label to add to newly-created issues; repeatable")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak maturity route: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := maturity.Build(maturity.Options{Root: root})
	projection := maturity.ProjectIssueItems(payload, *limit, []string(labels))
	items := projection.Items

	var existing []maturity.ExistingIssue
	var err error
	switch {
	case *existingJSON != "":
		b, err := os.ReadFile(*existingJSON)
		if err != nil {
			fmt.Fprintf(stderr, "fak maturity route: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(b, &existing); err != nil {
			fmt.Fprintf(stderr, "fak maturity route: --existing-json must contain a JSON list: %v\n", err)
			return 2
		}
	case *live || *fetchExisting:
		existing, err = maturity.FetchExistingIssues(*repo, 300)
		if err != nil {
			fmt.Fprintf(stderr, "fak maturity route: %v\n", err)
			return 2
		}
	}

	plan := maturity.BuildIssuePlan(items, existing)
	mode := "dry-run"
	if *live {
		mode = "live"
	}
	result := maturity.IssueResult{
		Schema:    maturity.IssueSchema,
		Mode:      mode,
		Workspace: payload.Workspace,
		Maturity: map[string]any{
			"score":         payload.Corpus["score"],
			"grade":         payload.Corpus["grade"],
			"maturity_debt": payload.Corpus["maturity_debt"],
			"backlog":       payload.Corpus["backlog"],
		},
		Planned: plan,
		Synced:  []maturity.IssueSyncRow{},
		Skipped: projection.Skipped,
	}
	if *live && len(plan) > 0 {
		result.Synced = maturity.SyncIssuePlan(plan, *repo, []string(labels), nil)
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak maturity route: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, maturity.RenderIssueResult(result))
	}

	if *live {
		for _, row := range result.Synced {
			if !row.OK {
				return 1
			}
		}
	}
	return 0
}
