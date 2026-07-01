package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/issuecohort"
	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

// runIssueCohort plans a whole batch of issue candidates at creation time:
// which are concurrency-safe to dispatch together (waves), which must be split
// first, which are triage-only, and which duplicate an existing marker key.
func runIssueCohort(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue cohort", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fromPlan := fs.String("from-plan", "", "issue-plan JSON file (an array, or {candidates|items|candidate})")
	file := fs.String("file", "", "alias for --from-plan")
	fromIssues := fs.String("from-issues", "", "GitHub issue JSON (gh issue list --json number,title,body,labels) — plan waves over existing open issues")
	live := fs.Bool("live", false, "review each candidate as an armed live/scheduled producer")
	dedupeChecked := fs.Bool("dedupe-checked", false, "producer proved marker dedupe against existing issues")
	dedupeCap := fs.Int("dedupe-cap", 0, "bounded issue scan cap proven before live sync")
	maxWave := fs.Int("max-wave", 0, "cap leaves per concurrency-safe wave (0 = disjoint-tree bound only)")
	asJSON := fs.Bool("json", false, "emit the machine-readable cohort plan")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	planPath := *fromPlan
	if planPath == "" {
		planPath = *file
	}
	sources := 0
	if planPath != "" {
		sources++
	}
	if *fromIssues != "" {
		sources++
	}
	if fs.NArg() != 0 || sources != 1 || (*fromPlan != "" && *file != "" && *fromPlan != *file) {
		fmt.Fprintln(stderr, "fak issue cohort: pass exactly one of --from-plan PLAN.json or --from-issues ISSUES.json")
		return 2
	}

	inputPath := planPath
	if *fromIssues != "" {
		inputPath = *fromIssues
	}
	abs, err := filepath.Abs(inputPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue cohort: %v\n", err)
		return 2
	}
	b, err := os.ReadFile(abs)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue cohort: %v\n", err)
		return 2
	}

	var candidates []issuecontract.Candidate
	if *fromIssues != "" {
		issues, derr := decodeIssueContractIssues(b)
		if derr != nil {
			fmt.Fprintf(stderr, "fak issue cohort: %v\n", derr)
			return 2
		}
		candidates = make([]issuecontract.Candidate, 0, len(issues))
		for _, d := range issues {
			candidates = append(candidates, issuecontract.CandidateFromIssueDraft(d))
		}
	} else {
		candidates, err = decodeIssueContractCandidates(b)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue cohort: %v\n", err)
			return 2
		}
	}

	plan := issuecohort.Build(candidates, issuecohort.Options{
		Options: issuecontract.Options{
			Live:          *live,
			DedupeChecked: *dedupeChecked,
			DedupeCap:     *dedupeCap,
		},
		MaxWave: *maxWave,
	})

	if *asJSON {
		if err := writeIndentedJSON(stdout, plan); err != nil {
			fmt.Fprintf(stderr, "fak issue cohort: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintln(stdout, issuecohort.Render(plan))
	return 0
}
