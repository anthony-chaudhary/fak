package main

// budget.go — `fak budget`: the inline per-task budget readout (#2091). An agent
// mid-task has no running "you've spent N, budget M, here is where it went"
// signal, so it can only discover it over-explored (or under-verified) in
// retrospect. This verb answers that from REAL usage records: it reads the
// gateway-usage ledger (internal/gatewayusageledger), picks the current task's
// latest counter snapshot, and folds it against an operator's soft target into a
// per-category breakdown (model tokens / served turns / adjudicated tool calls).
//
// The pure shape + fold live in internal/metrics (budget.go); this file only
// reads the ledger, maps the real counters into the pure BudgetSpend, and wires
// the exit code — the same pure-shape / CLI-fold split spend.go uses.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

func cmdBudget(argv []string) { os.Exit(runBudget(os.Stdout, os.Stderr, argv)) }

func runBudget(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("budget", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the fak-task-budget-readout/1 JSON envelope instead of a table")
	targetTokens := fs.Uint64("target-tokens", 0, "operator soft target for total (input+output) spend tokens; 0 = no target")
	targetTurns := fs.Uint64("target-turns", 0, "operator soft target for served turns; 0 = no target")
	sessionID := fs.String("session", "", "report this session id (default: the most recent task/session in the ledger)")
	ledgerPath := fs.String("ledger", "", "gateway-usage ledger path (default: .fak/nightrun/gateway-usage.jsonl under the repo root)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	path := *ledgerPath
	if path == "" {
		cwd, _ := os.Getwd()
		repoRoot := findRepoRoot(cwd)
		path = filepath.Join(repoRoot, filepath.FromSlash(gatewayusageledger.DefaultLedgerRel))
	}

	rows := gatewayusageledger.ReadLedgerFile(path)
	row, ok := latestTaskRow(rows, *sessionID)
	if !ok {
		if *sessionID != "" {
			fmt.Fprintf(stderr, "fak budget: no usage records for session %q in %s\n", *sessionID, path)
		} else {
			fmt.Fprintf(stderr, "fak budget: no usage records yet in %s (run a served/guard session first)\n", path)
		}
		return 1
	}

	spend := metrics.BudgetSpend{
		InputTokens:  row.Counters.InputTokens,
		OutputTokens: row.Counters.OutputTokens,
		CachedTokens: row.Counters.CachedPromptTokens,
		Turns:        row.Counters.ObservedTurns,
		ToolCalls:    budgetToolCalls(row.Counters),
	}
	target := metrics.BudgetTarget{Tokens: *targetTokens, Turns: *targetTurns}
	readout := metrics.FoldBudget(row.SessionID, spend, target)

	if defects := metrics.GateBudgetLabeled(readout); len(defects) > 0 {
		fmt.Fprintln(stderr, "fak budget: GATE FAILED — unlabeled budget category(ies):")
		for _, d := range defects {
			fmt.Fprintln(stderr, "  "+d)
		}
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(readout); err != nil {
			fmt.Fprintln(stderr, "fak budget: "+err.Error())
			return 1
		}
		return 0
	}

	printBudget(stdout, readout)
	return 0
}

// latestTaskRow selects the usage row for the task to report: the latest row (by
// wall-clock) of the requested session, or — when no session is named — the
// single most recent row in the ledger, whose SessionID is the current task. A
// ledger row is a CUMULATIVE counter snapshot, so the latest row of a session
// already holds that task's running total; there is nothing to sum across a
// session's rows.
func latestTaskRow(rows []gatewayusageledger.Row, session string) (gatewayusageledger.Row, bool) {
	var best gatewayusageledger.Row
	found := false
	for _, r := range rows {
		if session != "" && r.SessionID != session {
			continue
		}
		if !found || r.UnixMillis > best.UnixMillis {
			best = r
			found = true
		}
	}
	return best, found
}

// budgetToolCalls picks the best available tool-call count from a counter
// snapshot. The adjudication Total counts every fak_syscall the gateway decided
// on; on the guard proxy path the kernel Submits counter is 0, so Total is the
// truer tool-call figure there, with Submits then Admitted as fallbacks for a
// pure kernel session that recorded no adjudication total.
func budgetToolCalls(c gatewayusageledger.Counters) uint64 {
	if c.Total > 0 {
		return c.Total
	}
	if c.Submits > 0 {
		return uint64(c.Submits)
	}
	if c.Admitted > 0 {
		return uint64(c.Admitted)
	}
	return 0
}

func printBudget(w io.Writer, r metrics.BudgetReadout) {
	fmt.Fprintf(w, "fak budget — per-task readout (%s)\n", r.Schema)
	if r.Session != "" {
		fmt.Fprintf(w, "task/session: %s\n", r.Session)
	}
	printBudgetAxis(w, "tokens", r.Tokens, "tokens")
	printBudgetAxis(w, "turns", r.Turns, "turns")
	fmt.Fprintln(w, "breakdown (where it went):")
	for _, c := range r.Categories {
		fmt.Fprintf(w, "  %-22s %12d %-7s %-9s %s\n", c.Name, c.Spent, c.Unit, string(c.Provenance), c.Note)
	}
	if r.Note != "" {
		fmt.Fprintln(w, "note: "+r.Note)
	}
}

func printBudgetAxis(w io.Writer, label string, a metrics.BudgetAxis, unit string) {
	if a.HasTarget {
		state := "within target"
		if a.Over {
			state = "OVER target"
		}
		fmt.Fprintf(w, "%-7s spent %d / target %d %s (%.1f%% used, %d remaining) — %s\n",
			label, a.Spent, a.Target, unit, a.PercentUsed, a.Remaining, state)
	} else {
		fmt.Fprintf(w, "%-7s spent %d %s (no soft target set — pass --target-%s N)\n", label, a.Spent, unit, label)
	}
}
