package main

// close_batch.go is the thin I/O shell for `fak dispatch close-batch`: it
// reads witnessed-closeable issue numbers (plus an optional rate-limit
// budget) as JSON, folds them through internal/closebatch's pure Plan, and
// renders the resulting dry-run batch report. This leaf never closes an
// issue -- it only plans the batches an operator would arm before a future
// --live close arm runs (#1826).

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/closebatch"
	"github.com/anthony-chaudhary/fak/internal/mutationbudget"
)

// defaultCloseBatchBudget is used when the --in payload carries no "budget"
// object: GitHub's standard authenticated REST hourly allowance, reset an hour
// out, so a caller who only cares about the batching shape (not rate-limit
// modeling) still gets an ALLOW-everywhere report.
var defaultCloseBatchBudget = mutationbudget.Budget{Remaining: 5000, Limit: 5000, ResetAtUnix: 3600}

func runDispatchCloseBatch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch close-batch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "read witnessed issue numbers (and optional budget) from this JSON file (default: stdin)")
	batchSize := fs.Int("batch-size", closebatch.DefaultBatchSize, "max issues per dry-run batch")
	reserve := fs.Int("reserve", 50, "minimum GitHub API budget that must survive each batch")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for rate-limit reset math (0 = current time)")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}

	raw, code := readDispatchInput(stderr, *in)
	if code != 0 {
		return code
	}
	issues, budget, err := parseCloseBatchInput(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch close-batch: %v\n", err)
		return 1
	}

	now := *nowUnix
	if now == 0 {
		now = closeBatchNow()
	}

	rep := closebatch.Plan(closebatch.Input{
		IssueNumbers: issues,
		BatchSize:    *batchSize,
		Budget:       budget,
		Reserve:      *reserve,
		NowUnix:      now,
	})

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch close-batch")
	}
	renderCloseBatch(stdout, rep)
	return 0
}

// closeBatchNow is a var so tests can pin the clock without a --now flag.
var closeBatchNow = func() int64 { return time.Now().Unix() }

// closeBatchInputJSON mirrors the accepted --in shapes: a bare JSON array of
// issue numbers, or an object naming "issues" and an optional "budget".
type closeBatchInputJSON struct {
	Issues []int `json:"issues"`
	Budget *struct {
		Remaining   int   `json:"remaining"`
		Limit       int   `json:"limit"`
		ResetAtUnix int64 `json:"reset_unix"`
	} `json:"budget"`
}

// parseCloseBatchInput is the pure half of the shell: turning the --in JSON
// into (issue numbers, budget) is deterministic and needs no process I/O, so
// it is tested directly against canned JSON.
func parseCloseBatchInput(raw []byte) ([]int, mutationbudget.Budget, error) {
	var arr []int
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, defaultCloseBatchBudget, nil
	}
	var obj closeBatchInputJSON
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, mutationbudget.Budget{}, fmt.Errorf("parse close-batch input json: %w", err)
	}
	budget := defaultCloseBatchBudget
	if obj.Budget != nil {
		budget = mutationbudget.Budget{
			Remaining:   obj.Budget.Remaining,
			Limit:       obj.Budget.Limit,
			ResetAtUnix: obj.Budget.ResetAtUnix,
		}
	}
	return obj.Issues, budget, nil
}

// renderCloseBatch prints the dry-run plan as an aligned, scannable table: one
// row per batch with its issue count, mutation cost, rate-limit verdict, and
// rollback note, then a totals line.
func renderCloseBatch(w io.Writer, rep closebatch.Report) {
	fmt.Fprintf(w, "close-batch dry-run -- %d witnessed issue(s), %d batch(es) of up to %d\n\n",
		rep.TotalIssues, rep.TotalBatches, rep.BatchSize)
	for _, b := range rep.Batches {
		// b.RateLimit.Reason already leads with ALLOW:/HOLD:, so it alone names
		// the verdict -- no separate prefix needed.
		fmt.Fprintf(w, "batch %d: %d issue(s) %v\n  cost=%d %s\n  %s\n\n",
			b.Index, len(b.Issues), b.Issues, b.MutationCost, b.RateLimit.Reason, b.Rollback)
	}
	if rep.TotalBatches == 0 {
		fmt.Fprintln(w, "(no witnessed issues -- nothing to batch)")
		return
	}
	if rep.AllAllow {
		fmt.Fprintln(w, "every batch fits the rate-limit budget -- safe to arm a live close arm batch by batch")
	} else {
		fmt.Fprintln(w, "one or more batches are HELD -- wait for the rate-limit window to reset (or shrink --batch-size) before arming live")
	}
}
