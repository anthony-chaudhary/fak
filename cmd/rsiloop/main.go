// Command rsiloop is fak's TRUE recursive-self-improvement loop — the closed-loop
// companion to cmd/rsicycle's one-shot. Where rsicycle takes the keep-bit's
// witnesses (before/after/suite-green/truth-clean) AS FLAGS, rsiloop DERIVES every
// one of them from a real measurement it runs itself, in an isolated worktree off
// `main`, and folds the result through the same non-forgeable keep-bit
// (internal/shipgate). The loop author cannot move a KEEP by narrating a number.
//
// Two modes:
//
//	-mode improve  run the closed loop: propose candidate cache sizes, measure each
//	               in a worktree, keep-or-revert on the keep-bit, advance the running
//	               baseline on every KEEP (the recursion), escalate after K non-keeps.
//	-mode track    record ONE measurement of the KPI on `main` to the journal — the
//	               ongoing benchmark-against-latest-main series, with regression
//	               detection vs the last recorded point.
//
// Exit codes mirror the dos improve verdict: 0 = normal, 3 = ESCALATE (improve) or
// a detected regression (track) — the "hand this to a human / alert" signal.
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

func main() {
	mode := flag.String("mode", "improve", "improve | track")
	repo := flag.String("repo", ".", "the fak module root (where go.mod lives)")
	journalPath := flag.String("journal", "-", "append-only JSONL journal path ('-' = stdout)")
	baselineRef := flag.String("baseline-ref", "main", "the ref the baseline + candidates fork from")
	candidates := flag.String("candidates", "6,8,8,10", "comma-separated DefaultCacheSize values to propose")
	k := flag.Int("k", 3, "escalation breaker: stop after K consecutive non-keeps")
	maxCycles := flag.Int("max-cycles", 0, "cap on candidates tried (0 = all)")
	probePkg := flag.String("probe", "./cmd/kpiprobe", "the KPI probe package path")
	suitePkgs := flag.String("suite-pkgs", "./...", "package pattern the suite-green gate builds+vets")
	flag.Parse()

	cfg := rsiloop.WorktreeConfig{
		Repo:        *repo,
		BaselineRef: *baselineRef,
		Candidates:  parseInts(*candidates),
		ProbePkg:    *probePkg,
		SuitePkgs:   *suitePkgs,
	}
	h := rsiloop.NewWorktreeHarness(cfg)

	j, err := rsiloop.NewJournal(*journalPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "journal:", err)
		os.Exit(1)
	}
	defer j.Close()

	switch *mode {
	case "track":
		os.Exit(runTrack(h, j, *journalPath))
	case "improve":
		os.Exit(runImprove(h, j, *k, *maxCycles))
	default:
		fmt.Fprintf(os.Stderr, "unknown -mode %q (want improve|track)\n", *mode)
		os.Exit(2)
	}
}

// runImprove drives the closed loop and prints a per-cycle trace + a summary.
func runImprove(h rsiloop.Harness, j *rsiloop.Journal, k, maxCycles int) int {
	res, err := rsiloop.Run(h, j, k, maxCycles)
	if err != nil {
		fmt.Fprintln(os.Stderr, "rsiloop:", err)
		return 1
	}
	fmt.Printf("baseline %s@%s = %.6f\n", h.MetricName, res.BaselineRef, baseOf(res))
	for _, r := range res.Rows {
		cand := fmt.Sprintf("%.6f", r.Candidate_)
		if !r.Measured {
			cand = "(not measured)"
		}
		fmt.Printf("  cycle %d  %-22s base=%.6f cand=%s improved=%v suite=%v truth=%v -> %s (kept=%v, breaker=%d)%s\n",
			r.Cycle, r.Candidate, r.Baseline, cand, r.Improved, r.SuiteGreen, r.TruthClean,
			r.Decision, r.Kept, r.BreakerCount, noteSuffix(r.Note))
	}
	fmt.Printf("SUMMARY cycles=%d kept=%d final=%s final_baseline=%.6f escalated=%v\n",
		res.Cycles, res.Kept, res.Final.String(), res.FinalBaseline, res.Escalated)
	if res.Escalated {
		return 3 // breaker tripped — hand to a human
	}
	return 0
}

// runTrack records one main measurement and compares it to the last recorded one.
func runTrack(h rsiloop.Harness, j *rsiloop.Journal, journalPath string) int {
	var prev rsiloop.Row
	havePrev := false
	if journalPath != "-" && journalPath != "" {
		p, ok, err := rsiloop.LastTrack(journalPath)
		if err != nil {
			// A genuine read failure must NOT silently disable the alert.
			fmt.Fprintln(os.Stderr, "warning: could not read prior track point:", err)
		}
		prev, havePrev = p, ok
	}
	row, err := rsiloop.Track(h, j)
	if err != nil {
		fmt.Fprintln(os.Stderr, "rsiloop track:", err)
		return 1
	}
	fmt.Printf("track %s@%s = %.6f\n", row.MetricName, row.BaselineRef, row.Baseline)
	if !havePrev {
		fmt.Println("  (no prior track point — this is the first)")
		return 0
	}
	// Only compare points measured against the SAME symbolic ref — a delta across
	// different refs (e.g. main vs an old tag) is meaningless, so don't emit a
	// spurious regression alert on it.
	if prev.RefName != row.RefName {
		fmt.Printf("  prior point was measured @%s, this run @%s — refs differ, skipping regression verdict\n",
			refLabel(prev.RefName), refLabel(row.RefName))
		return 0
	}
	delta := row.Baseline - prev.Baseline
	regressed := regression(prev.Baseline, row.Baseline, row.LowerBetter)
	fmt.Printf("  vs last (%s@%s=%.6f): delta=%+.6f regressed=%v\n",
		prev.MetricName, prev.BaselineRef, prev.Baseline, delta, regressed)
	if regressed {
		return 3 // a regression on main — alert
	}
	return 0
}

func refLabel(r string) string {
	if r == "" {
		return "(unknown)"
	}
	return r
}

func regression(prev, cur float64, lowerBetter bool) bool {
	if lowerBetter {
		return cur > prev
	}
	return cur < prev
}

func baseOf(res rsiloop.Result) float64 {
	if len(res.Rows) > 0 {
		return res.Rows[0].Baseline
	}
	return res.FinalBaseline
}

func noteSuffix(note string) string {
	if note == "" {
		return ""
	}
	return "  [" + note + "]"
}

func parseInts(s string) []int {
	var out []int
	for _, tok := range strings.Split(s, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		if n, err := strconv.Atoi(tok); err == nil {
			out = append(out, n)
		}
	}
	return out
}
