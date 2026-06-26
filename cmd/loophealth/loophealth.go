// Command loophealth is the checking layer for fleet's RSI self-improve loop (#382).
//
// A self-improvement loop with no checking layer reward-hacks its own improvement
// metric: the backlog grows while closure sits at zero. The loop emits a typed verdict
// per cycle to `.dos/verdict-journal.jsonl` (the `dos improve --observe` trail), but
// nothing folds those verdicts into a loop-health number — so a run can report "busy"
// (many cycles, many reverts) while measurably closing nothing.
//
// loophealth folds that journal — and ONLY that journal, never a self-narrated status —
// into two loop-health rates over a window:
//
//	closure_rate    = verify SHIPPED / (verify SHIPPED + verify NOT_SHIPPED)
//	                  routed findings that MEASURABLY closed (a `dos verify` SHIPPED
//	                  verdict carries git evidence; NOT_SHIPPED / source=none does not).
//	regression_rate = improve REVERT / (improve KEEP + improve REVERT)
//	                  the fraction of resolved candidates the loop had to undo —
//	                  the "KEEP-then-reverted over total KEEP-decisions" health signal.
//
// It is READ-ONLY (it opens the journal for reading and writes nothing) and exit-coded:
// exit 0 once the metric is computed (even all-zero is honest — the point is the metric
// exists and is tracked); exit 2 on a usage / IO error; and, only when --gate is passed,
// exit 3 if the current window is worse than the recorded baseline on either axis — so
// the loop itself can be gated the way `dos improve` gates a single cycle.
//
// Usage:
//
//	go run ./cmd/loophealth                 # human one-liner over .dos/verdict-journal.jsonl
//	go run ./cmd/loophealth --json          # the {closure_rate, regression_rate, window, baseline} fold
//	go run ./cmd/loophealth --window 50     # over the most-recent 50 journal rows
//	go run ./cmd/loophealth --gate          # exit 3 if worse than baseline (for a control-pane loop)
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// Schema is the versioned payload tag so a consumer can pin the shape it folds.
const Schema = "self-improve-loop-health/1"

// defaultJournal is fleet's verdict-journal path, relative to the workspace root — the
// `dos improve --observe --run-id` trail of KEEP/REVERT/ESCALATE + verify SHIPPED/NOT.
const defaultJournal = ".dos/verdict-journal.jsonl"

// Baseline is the loop-health anchor recorded when this metric was introduced
// (2026-06-26), computed from the real .dos/verdict-journal.jsonl on that date:
// 9 verify SHIPPED of 11 verify decisions, and 2 REVERT of 4 KEEP-decisions. It is a
// FIXED reference the current window is compared against — not a recomputed value — so a
// later run can see whether the loop's closure improved or its regression worsened.
// Recording it (even were it 0.0) is the point: the metric now exists and is tracked.
var Baseline = baseline{
	ClosureRate:    0.818,
	RegressionRate: 0.5,
	Recorded:       "2026-06-26",
	Note:           "anchor folded from .dos/verdict-journal.jsonl at #382 introduction (9/11 verify SHIPPED; 2/4 improve KEEP-decisions reverted)",
}

// Fold is the pure count of the journal rows in the window, by the two syscall classes
// the loop-health rates are built from. Every rate is a function of these integers, so
// the same journal bytes always yield the same numbers (no wall-clock, no RNG).
type Fold struct {
	RowsConsidered  int `json:"rows_considered"`    // total journal rows in the window (all syscalls)
	Keep            int `json:"keep"`               // improve cycles that KEPT
	Revert          int `json:"revert"`             // improve cycles that REVERTED
	RegressedRevert int `json:"regressed_revert"`   // subset of Revert whose detail.revert_cause == "regressed"
	Escalate        int `json:"escalate"`           // improve cycles that ESCALATED (excluded from regression_rate)
	VerifyShipped   int `json:"verify_shipped"`     // verify decisions that resolved SHIPPED (measurably closed)
	VerifyNot       int `json:"verify_not_shipped"` // verify decisions that resolved NOT_SHIPPED
}

// ClosureRate is the fraction of routed findings that measurably closed: verify SHIPPED
// over all verify decisions in the window. 0.0 when no finding was routed for verify.
func (f Fold) ClosureRate() float64 {
	routed := f.VerifyShipped + f.VerifyNot
	if routed == 0 {
		return 0
	}
	return round3(float64(f.VerifyShipped) / float64(routed))
}

// RegressionRate is the fraction of resolved improve candidates the loop had to undo:
// REVERT over (KEEP + REVERT) in the window. ESCALATE is neither kept nor reverted, so
// it is excluded. 0.0 when the loop resolved no keep/revert decision.
func (f Fold) RegressionRate() float64 {
	decided := f.Keep + f.Revert
	if decided == 0 {
		return 0
	}
	return round3(float64(f.Revert) / float64(decided))
}

type baseline struct {
	ClosureRate    float64 `json:"closure_rate"`
	RegressionRate float64 `json:"regression_rate"`
	Recorded       string  `json:"recorded"`
	Note           string  `json:"note"`
}

// Window describes the slice of the journal the fold ran over.
type Window struct {
	Cap            int `json:"cap"`             // --window N (0 = all rows)
	RowsConsidered int `json:"rows_considered"` // journal rows actually folded
	ImproveCycles  int `json:"improve_cycles"`  // KEEP+REVERT+ESCALATE in the window
	VerifyChecks   int `json:"verify_checks"`   // verify decisions in the window
}

// Health is the read-only loop-health payload: the issue's {closure_rate,
// regression_rate, window, baseline} plus the auditable counts behind each rate and the
// delta against the recorded baseline.
type Health struct {
	Schema         string   `json:"schema"`
	Journal        string   `json:"journal"`
	Window         Window   `json:"window"`
	ClosureRate    float64  `json:"closure_rate"`
	RegressionRate float64  `json:"regression_rate"`
	Counts         Fold     `json:"counts"`
	Baseline       baseline `json:"baseline"`
	Delta          delta    `json:"delta"`
	Healthy        bool     `json:"healthy"` // current window not worse than baseline on either axis
	Note           string   `json:"note"`
}

type delta struct {
	ClosureRate    float64 `json:"closure_rate"`    // current - baseline (higher is better)
	RegressionRate float64 `json:"regression_rate"` // current - baseline (lower is better)
}

// foldJournal reads the JSONL verdict journal and folds the window into counts. windowCap
// caps the number of MOST-RECENT rows considered (0 = all); rows are read in file order,
// which is append order. Malformed lines and rows of other syscalls are skipped, not
// fatal — the journal is an evolving append log and a future row shape must not brick the
// checking layer. A missing journal folds to an all-zero (honest) result, not an error.
func foldJournal(path string, windowCap int) (Fold, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Fold{}, nil
		}
		return Fold{}, err
	}
	var rows []map[string]any
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue // tolerate a malformed line rather than brick the fold
		}
		rows = append(rows, row)
	}
	if windowCap > 0 && len(rows) > windowCap {
		rows = rows[len(rows)-windowCap:]
	}
	var f Fold
	for _, row := range rows {
		f.RowsConsidered++
		syscall := strings.ToLower(asString(row["syscall"]))
		verdict := strings.ToUpper(strings.TrimSpace(asString(row["verdict"])))
		switch syscall {
		case "improve":
			switch verdict {
			case "KEEP":
				f.Keep++
			case "REVERT":
				f.Revert++
				if detailString(row, "revert_cause") == "regressed" {
					f.RegressedRevert++
				}
			case "ESCALATE":
				f.Escalate++
			}
		case "verify":
			switch verdict {
			case "SHIPPED":
				f.VerifyShipped++
			case "NOT_SHIPPED":
				f.VerifyNot++
			}
		}
	}
	return f, nil
}

// computeHealth folds the journal and assembles the read-only Health payload.
func computeHealth(path string, windowCap int) (Health, error) {
	f, err := foldJournal(path, windowCap)
	if err != nil {
		return Health{}, err
	}
	closure := f.ClosureRate()
	regression := f.RegressionRate()
	dClosure := round3(closure - Baseline.ClosureRate)
	dRegression := round3(regression - Baseline.RegressionRate)
	// Healthy = not worse than the baseline on either axis: closure no lower, regression
	// no higher. (Within a 1e-9 epsilon so an exact-equal baseline reads as healthy.)
	healthy := dClosure >= -1e-9 && dRegression <= 1e-9
	return Health{
		Schema:  Schema,
		Journal: path,
		Window: Window{
			Cap:            windowCap,
			RowsConsidered: f.RowsConsidered,
			ImproveCycles:  f.Keep + f.Revert + f.Escalate,
			VerifyChecks:   f.VerifyShipped + f.VerifyNot,
		},
		ClosureRate:    closure,
		RegressionRate: regression,
		Counts:         f,
		Baseline:       Baseline,
		Delta:          delta{ClosureRate: dClosure, RegressionRate: dRegression},
		Healthy:        healthy,
		Note:           healthNote(f, closure, regression),
	}, nil
}

func healthNote(f Fold, closure, regression float64) string {
	if f.Keep+f.Revert+f.Escalate == 0 && f.VerifyShipped+f.VerifyNot == 0 {
		return "no improve/verify rows in the window — the loop has not yet emitted a checkable cycle; closure_rate/regression_rate are an honest 0.0 baseline"
	}
	return fmt.Sprintf("closure_rate %.3g (%d/%d verify SHIPPED), regression_rate %.3g (%d/%d KEEP-decisions reverted; %d cause=regressed)",
		closure, f.VerifyShipped, f.VerifyShipped+f.VerifyNot,
		regression, f.Revert, f.Keep+f.Revert, f.RegressedRevert)
}

// render is the human one-liner form.
func render(h Health) string {
	lines := []string{
		"self-improve loop-health (the checking layer) — " + h.Journal,
		fmt.Sprintf("  closure_rate    %.3g   (baseline %.3g, delta %+.3g)   %d/%d verify SHIPPED",
			h.ClosureRate, h.Baseline.ClosureRate, h.Delta.ClosureRate, h.Counts.VerifyShipped, h.Window.VerifyChecks),
		fmt.Sprintf("  regression_rate %.3g   (baseline %.3g, delta %+.3g)   %d/%d KEEP-decisions reverted",
			h.RegressionRate, h.Baseline.RegressionRate, h.Delta.RegressionRate, h.Counts.Revert, h.Counts.Keep+h.Counts.Revert),
		fmt.Sprintf("  window: %d row(s) considered (cap %d) — %d improve cycle(s), %d verify check(s)",
			h.Window.RowsConsidered, h.Window.Cap, h.Window.ImproveCycles, h.Window.VerifyChecks),
		"  -> " + h.Note,
	}
	if h.Healthy {
		lines = append(lines, "  healthy: yes (not worse than baseline on either axis)")
	} else {
		lines = append(lines, "  healthy: NO (worse than baseline — closure fell or regression rose)")
	}
	return strings.Join(lines, "\n")
}

func round3(v float64) float64 { return math.Round(v*1000) / 1000 }

func asString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

// detailString pulls a string field out of the row's nested "detail" object, "" if absent.
func detailString(row map[string]any, key string) string {
	detail, ok := row["detail"].(map[string]any)
	if !ok {
		return ""
	}
	return strings.TrimSpace(asString(detail[key]))
}

// resolveJournal returns the journal path: the explicit --journal, else defaultJournal
// joined to the workspace root (cwd).
func resolveJournal(explicit, root string) string {
	if explicit != "" {
		return explicit
	}
	return filepath.Join(root, defaultJournal)
}
