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
// Exit codes: 0 = normal (completed without escalation), 1 = error, 3 = ESCALATE (the
// breaker tripped after K consecutive non-keeps — hand to a human) or, in track mode, a
// detected regression on `main` (alert).
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
	harness := flag.String("harness", "worktree", "which REAL subsystem the loop drives: "+
		"worktree (rewrite DefaultCacheSize, measured in an isolated worktree) | "+
		"rulesynth (synthesize an adjudicator deny-rule from the frozen near-miss corpus) | "+
		"sessionobs (drive the session->outcome S0 loop-index score as the RSI objective)")
	dosObserve := flag.Bool("dos-observe", false, "also emit a `dos improve --observe` "+
		"receipt of each keep/revert verdict to the DOS audit journal (record-only "+
		"telemetry; never re-gates the loop; no-op when dos is absent) — #588")
	flag.Parse()

	h, herr := selectHarness(*harness, *repo, *baselineRef, *candidates, *probePkg, *suitePkgs)
	if herr != nil {
		fmt.Fprintln(os.Stderr, herr)
		os.Exit(2)
	}

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
		var obs rsiloop.Observer
		if *dosObserve {
			obs = dosObserveReceipt(*repo, *k) // nil (a no-op) when dos is absent
		}
		os.Exit(runImprove(h, j, *k, *maxCycles, obs))
	default:
		fmt.Fprintf(os.Stderr, "unknown -mode %q (want improve|track)\n", *mode)
		os.Exit(2)
	}
}

// selectHarness builds the real subsystem the loop drives from the -harness flag.
// Two are wired today, both folded through the SAME non-forgeable keep-bit (Run +
// shipgate.Evaluate): the worktree harness rewrites the DefaultCacheSize literal and
// measures the candidate in an isolated git worktree off main; the rulesynth harness
// synthesizes an adjudicator deny-rule from the frozen near-miss corpus and proves it
// against the real model-free adjudicator; the sessionobs harness makes the full S0
// loop-index score, with its Learn stage derived from sessionobs, the measured RSI
// objective. A
// second REAL subsystem — not a second knob on the same one — is what makes the loop
// a general improver rather than a cache-size demo (#586).
func selectHarness(kind, repo, baselineRef, candidates, probePkg, suitePkgs string) (rsiloop.Harness, error) {
	switch kind {
	case "worktree":
		return rsiloop.NewWorktreeHarness(rsiloop.WorktreeConfig{
			Repo:        repo,
			BaselineRef: baselineRef,
			Candidates:  parseInts(candidates),
			ProbePkg:    probePkg,
			SuitePkgs:   suitePkgs,
		}), nil
	case "rulesynth":
		// The corpus is the committed frozen fixture (rulesynth_corpus.go), mined
		// deterministically through the real Detect predicate so a KEEP reproduces
		// bit-for-bit. The worktree flags (repo/baseline-ref/candidates) do not apply:
		// this harness needs no git fork — its baseline is the zero-catch floor and its
		// replay is a pure adjudicator call.
		return rsiloop.NewRuleSynthHarness(rsiloop.FrozenRuleSynthCorpus()), nil
	case "sessionobs":
		// Deterministic S0 witness for #1161: a no-op sessionobs toolchain proposal
		// reverts, while the closed session->outcome->consuming-loop state keeps only
		// after the S0 loop-index strictly rises to 100.
		return rsiloop.NewSessionObsDemoHarness(), nil
	default:
		return rsiloop.Harness{}, fmt.Errorf("unknown -harness %q (want worktree|rulesynth|sessionobs)", kind)
	}
}

// runImprove drives the closed loop and prints a per-cycle trace + a summary. obs is an
// optional telemetry sink (nil = none) that mirrors each verdict to the DOS journal.
func runImprove(h rsiloop.Harness, j *rsiloop.Journal, k, maxCycles int, obs rsiloop.Observer) int {
	res, err := rsiloop.RunObserved(h, j, k, maxCycles, obs)
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
		fmt.Printf("  cycle %d  %-22s base=%.6f cand=%s improved=%v suite=%v truth=%v -> %s (kept=%v, breaker=%d)%s%s\n",
			r.Cycle, r.Candidate, r.Baseline, cand, r.Improved, r.SuiteGreen, r.TruthClean,
			r.Decision, r.Kept, r.BreakerCount, scoreSuffix(r.Score), noteSuffix(r.Note))
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

func scoreSuffix(score *rsiloop.Scorecard) string {
	if score == nil {
		return ""
	}
	return "  [" + scoreSummary(score) + "]"
}

func scoreSummary(score *rsiloop.Scorecard) string {
	name := score.Name
	if name == "" {
		name = "score"
	}
	parts := []string{fmt.Sprintf("score %s=%s", name, formatScoreValue(score.Value))}
	if score.Grade != "" {
		parts = append(parts, "grade="+score.Grade)
	}
	for _, c := range score.Components {
		if scoreSummaryComponent(c.Name) {
			parts = append(parts, fmt.Sprintf("%s=%s", c.Name, formatScoreValue(c.Value)))
		}
	}
	return strings.Join(parts, " ")
}

func scoreSummaryComponent(name string) bool {
	if strings.Contains(name, "ratio") ||
		strings.Contains(name, "debt") ||
		strings.HasSuffix(name, "_tokens") {
		return true
	}
	switch name {
	case "loop_consumes",
		"caught",
		"regressed",
		"support",
		"catches_cluster",
		"self_modify",
		"cache_size",
		"trace_len",
		"working_set":
		return true
	default:
		return false
	}
}

func formatScoreValue(v float64) string {
	i := int64(v)
	if v == float64(i) {
		return strconv.FormatInt(i, 10)
	}
	return fmt.Sprintf("%.3f", v)
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
