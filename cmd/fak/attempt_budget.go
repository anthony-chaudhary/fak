package main

// attempt_budget.go is the thin I/O shell for `fak dispatch attempt-budget`: it
// reads per-issue attempt-history facts from --in JSON (or stdin), folds each
// through internal/attemptbudget's pure Decide, and renders which issues are
// still dispatchable versus HELD for triage once they cross the configured
// attempt budget -- so a repeatedly failing issue stops burning workers
// instead of being re-offered forever (#1777). It never spawns a worker or
// mutates GitHub; it only classifies facts the caller already gathered.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/attemptbudget"
)

func runDispatchAttemptBudget(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch attempt-budget", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "read per-issue attempt facts from this JSON file (default: stdin)")
	budget := fs.Int("budget", 0, "attempt budget applied to any issue whose own input omits one (0 = no default, per-issue budget only)")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}

	raw, code := readDispatchInput(stderr, *in)
	if code != 0 {
		return code
	}
	inputs, err := parseAttemptBudgetInputs(raw, *budget)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch attempt-budget: parse input: %v\n", err)
		return 1
	}

	rep := attemptbudget.DecideAll(inputs)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch attempt-budget")
	}
	renderAttemptBudget(stdout, rep)
	return 0
}

// parseAttemptBudgetInputs accepts either a bare JSON array of issue inputs or
// an object with an "issues" field, applying defaultBudget to any entry whose
// own "budget" is omitted/zero so a single flag can cover a whole fixture.
func parseAttemptBudgetInputs(raw []byte, defaultBudget int) ([]attemptbudget.Input, error) {
	var obj struct {
		Issues []attemptbudget.Input `json:"issues"`
	}
	var inputs []attemptbudget.Input
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Issues != nil {
		inputs = obj.Issues
	} else {
		var arr []attemptbudget.Input
		if err := json.Unmarshal(raw, &arr); err != nil {
			return nil, err
		}
		inputs = arr
	}
	for i := range inputs {
		if inputs[i].Budget == 0 {
			inputs[i].Budget = defaultBudget
		}
	}
	return inputs, nil
}

// renderAttemptBudget prints the batch verdict as an aligned, scannable table.
func renderAttemptBudget(w io.Writer, rep attemptbudget.Report) {
	fmt.Fprintf(w, "attempt budget -- %d dispatchable, %d held\n\n", rep.DispatchableCount, rep.HeldCount)
	fmt.Fprintf(w, "%-10s %-12s %8s %8s %s\n", "issue", "status", "attempts", "budget", "last_failure_class")
	for _, d := range rep.Decisions {
		lastClass := d.LastFailureClass
		if lastClass == "" {
			lastClass = "-"
		}
		fmt.Fprintf(w, "%-10s %-12s %8d %8d %s\n", d.IssueID, d.Status, d.AttemptCount, d.Budget, lastClass)
	}
}
