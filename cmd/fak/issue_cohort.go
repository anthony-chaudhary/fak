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
	live := fs.Bool("live", false, "review each candidate as an armed live/scheduled producer")
	dedupeChecked := fs.Bool("dedupe-checked", false, "producer proved marker dedupe against existing issues")
	dedupeCap := fs.Int("dedupe-cap", 0, "bounded issue scan cap proven before live sync")
	maxWave := fs.Int("max-wave", 0, "cap leaves per concurrency-safe wave (0 = disjoint-tree bound only)")
	asJSON := fs.Bool("json", false, "emit the machine-readable cohort plan")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	path := *fromPlan
	if path == "" {
		path = *file
	}
	if fs.NArg() != 0 || path == "" || (*fromPlan != "" && *file != "" && *fromPlan != *file) {
		fmt.Fprintln(stderr, "fak issue cohort: pass one --from-plan PLAN.json (an issue-candidate array)")
		return 2
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue cohort: %v\n", err)
		return 2
	}
	b, err := os.ReadFile(abs)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue cohort: %v\n", err)
		return 2
	}
	candidates, err := decodeIssueContractCandidates(b)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue cohort: %v\n", err)
		return 2
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
