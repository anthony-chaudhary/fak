package main

// attempt_budget.go is the thin I/O shell for `fak dispatch attempt-budget`: it
// reads per-issue attempt-history facts from --in JSON (or stdin), folds each
// through internal/attemptbudget's pure Decide, and renders which issues are
// still dispatchable, COOLING_DOWN under their last failure's class-specific
// backoff window, or HELD for triage once they cross the configured attempt
// budget -- so a repeatedly failing issue stops burning workers instead of
// being re-offered forever (#1777), and so auth/merge/test/ambiguous-scope
// failures cool down differently instead of sharing one window (#1778). It
// never spawns a worker or mutates GitHub; it only classifies facts the
// caller already gathered -- this candidate report is the one place that
// policy is actually reflected, not just defined.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/attemptbudget"
)

func runDispatchAttemptBudget(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch attempt-budget", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "read per-issue attempt facts from this JSON file (default: stdin)")
	budget := fs.Int("budget", 0, "attempt budget applied to any issue whose own input omits one (0 = no default, per-issue budget only)")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for failure-class backoff math (0 = current time)")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}

	raw, code := readDispatchInput(stderr, *in)
	if code != 0 {
		return code
	}
	now := *nowUnix
	if now == 0 {
		now = time.Now().Unix()
	}
	inputs, err := parseAttemptBudgetInputs(raw, *budget, now)
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
// own "budget" is omitted/zero so a single flag can cover a whole fixture, and
// stamping nowUnix onto every entry whose own "now_unix" is omitted so the
// failure-class backoff window is evaluated against one consistent clock.
func parseAttemptBudgetInputs(raw []byte, defaultBudget int, nowUnix int64) ([]attemptbudget.Input, error) {
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
		if inputs[i].NowUnix == 0 {
			inputs[i].NowUnix = nowUnix
		}
	}
	return inputs, nil
}

// renderAttemptBudget prints the batch verdict as an aligned, scannable table,
// including the failure-class backoff policy actually applied to each issue's
// last recorded attempt so the report -- not just the code -- carries the
// #1778 policy.
func renderAttemptBudget(w io.Writer, rep attemptbudget.Report) {
	fmt.Fprintf(w, "attempt budget -- %d dispatchable, %d cooling down, %d held\n\n",
		rep.DispatchableCount, rep.CoolingDownCount, rep.HeldCount)
	fmt.Fprintf(w, "%-10s %-14s %8s %8s %-16s %-12s %s\n",
		"issue", "status", "attempts", "budget", "last_failure_class", "backoff_class", "backoff_window")
	for _, d := range rep.Decisions {
		lastClass := d.LastFailureClass
		if lastClass == "" {
			lastClass = "-"
		}
		backoffClass := string(d.BackoffClass)
		backoffWindow := "-"
		if backoffClass == "" {
			backoffClass = "-"
		} else {
			backoffWindow = (time.Duration(d.BackoffSeconds) * time.Second).String()
		}
		fmt.Fprintf(w, "%-10s %-14s %8d %8d %-16s %-12s %s\n",
			d.IssueID, d.Status, d.AttemptCount, d.Budget, lastClass, backoffClass, backoffWindow)
	}
}
