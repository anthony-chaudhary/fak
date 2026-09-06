// Command rsiloop is fak's TRUE recursive-self-improvement loop — the closed-loop
// companion to cmd/rsicycle's one-shot. Where rsicycle takes the keep-bit's
// witnesses (before/after/suite-green/truth-clean) AS FLAGS, rsiloop DERIVES every
// one of them from a real measurement it runs itself, in an isolated worktree off
// `main`, and folds the result through the same non-forgeable keep-bit
// (internal/shipgate). The loop author cannot move a KEEP by narrating a number.
//
// Three modes:
//
//	-mode improve  run the closed loop: propose candidate cache sizes, measure each
//	               in a worktree, keep-or-revert on the keep-bit, advance the running
//	               baseline on every KEEP (the recursion), escalate after K non-keeps.
//	-mode track    record ONE measurement of the KPI on `main` to the journal — the
//	               ongoing benchmark-against-latest-main series, with regression
//	               detection vs the last recorded point.
//	-mode meta     the APEX rung (#1195): read the rsiloop journal, fold a BOUNDED
//	               keep-policy proposal from clustered ESCALATEs, and print it as JSON.
//	               Propose-only by default (nothing mutates); --apply with a required
//	               --witness-journal witnesses the proposal through the SAME non-forgeable
//	               keep-bit and lands it only on a witnessed KEEP. This is RSI applied to
//	               the keep-GATE itself — the loop-author cannot move a KEEP by narrating.
//
// Exit codes: 0 = normal (completed without escalation; in meta mode: proposal emitted,
// no proposal, or a witnessed KEEP), 1 = error, 2 = usage (in meta mode: --apply refused
// for a missing --witness-journal), 3 = ESCALATE (the breaker tripped after K consecutive
// non-keeps — hand to a human) or, in track mode, a detected regression on `main`, or, in
// meta mode, a witnessed REVERT (the proposal was rejected by the keep-bit; policy untouched).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

func main() {
	mode := flag.String("mode", "improve", "improve | track | meta")
	repo := flag.String("repo", ".", "the fak module root (where go.mod lives)")
	journalPath := flag.String("journal", "-", "append-only JSONL journal path ('-' = stdout). "+
		"In -mode meta this is read as INPUT: the rsiloop journal the fold scans.")
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
	maxTransientRetries := flag.Int("max-transient-retries", rsiloop.DefaultTransientMeasurementRecoveryLimit,
		"max transient measurement recovery retries per candidate (negative disables)")
	transientBudget := flag.Int("transient-budget", 0,
		"total transient recovery retry budget across the run (0 = unlimited)")

	// -mode meta flags (#3975): the apex meta-RSI fold over the journal. The four
	// config knobs bound the proposal; --apply + --witness-journal witness and land it.
	dc := rsiloop.DefaultMetaConfig()
	metaWindow := flag.Int("meta-window", dc.Window, "meta: # of most-recent improve cycles the fold scans")
	metaMinEsc := flag.Int("meta-min-escalations", dc.MinEscalations, "meta: escalations within the window that trigger a proposal")
	metaGainStep := flag.Float64("meta-gain-step", dc.GainStep, "meta: bounded increment applied to the strict-gain bar")
	metaGainCeiling := flag.Float64("meta-gain-ceiling", dc.GainCeiling, "meta: the strict-gain bar is never proposed above this")
	metaGain := flag.Float64("meta-gain-threshold", 0, "meta: the CURRENT keep-gate strict-gain bar the fold proposes to raise")
	apply := flag.Bool("apply", false, "meta: witness the folded proposal through the keep-bit and LAND it on a witnessed KEEP (requires --witness-journal); default is propose-only")
	witnessJournal := flag.String("witness-journal", "", "meta: path to the journal OBSERVED under the proposed policy — the non-author witness --apply re-measures against")
	flag.Parse()

	if *mode == "meta" {
		os.Exit(runMeta(metaOptions{
			journalPath:    *journalPath,
			witnessJournal: *witnessJournal,
			apply:          *apply,
			cur:            rsiloop.KeepPolicy{GainThreshold: *metaGain},
			cfg: rsiloop.MetaConfig{
				Window:         *metaWindow,
				MinEscalations: *metaMinEsc,
				GainStep:       *metaGainStep,
				GainCeiling:    *metaGainCeiling,
			},
		}, os.Stdout, os.Stderr))
	}

	h, herr := selectHarness(*harness, *repo, *baselineRef, *candidates, *probePkg, *suitePkgs)
	if herr != nil {
		fmt.Fprintln(os.Stderr, herr)
		os.Exit(2)
	}
	applyTransientFlags(&h, *maxTransientRetries, *transientBudget)

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
		fmt.Fprintf(os.Stderr, "unknown -mode %q (want improve|track|meta)\n", *mode)
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

// applyTransientFlags applies the CLI flags -max-transient-retries and -transient-budget
// onto the target Harness.
func applyTransientFlags(h *rsiloop.Harness, maxTransientRetries, transientBudget int) {
	if maxTransientRetries <= 0 {
		h.MaxTransientRetries = -1
		h.TransientMeasurementRecoveryLimit = -1
	} else {
		h.MaxTransientRetries = maxTransientRetries
		h.TransientMeasurementRecoveryLimit = maxTransientRetries
	}
	h.TransientBudget = transientBudget
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

// metaOptions carries the parsed -mode meta inputs so runMeta is testable without the
// flag package: the journal to fold, the witness journal --apply re-measures against,
// the explicit allow bit, the current keep-policy, and the bounded fold config.
type metaOptions struct {
	journalPath    string
	witnessJournal string
	apply          bool
	cur            rsiloop.KeepPolicy
	cfg            rsiloop.MetaConfig
}

// metaProposalView is the readable JSON projection of a Proposal (the library's Knob is
// an opaque int; here it is its stable token). It is the hypothesis, never an authority.
type metaProposalView struct {
	Knob          string  `json:"knob"`
	Before        float64 `json:"before"`
	After         float64 `json:"after"`
	Escalations   int     `json:"escalations"`
	WindowScanned int     `json:"window_scanned"`
	Rationale     string  `json:"rationale"`
}

func proposalView(p rsiloop.Proposal) metaProposalView {
	return metaProposalView{
		Knob: p.Knob.String(), Before: p.Before, After: p.After,
		Escalations: p.Escalations, WindowScanned: p.Window, Rationale: p.Rationale,
	}
}

// runMeta reads the rsiloop journal, folds a BOUNDED keep-policy proposal, and prints a
// self-describing JSON object to out. It is propose-only unless opts.apply is set, in
// which case it witnesses the proposal through the library's non-forgeable keep-bit
// (ApplyProposalWithWitness) against the --witness-journal and lands it ONLY on a
// witnessed KEEP. Nothing mutates on the propose-only or no-proposal paths. Returns the
// process exit code: 0 (proposal/no-proposal/witnessed KEEP), 1 (read error),
// 2 (--apply refused for a missing witness — the library's own error), 3 (witnessed
// REVERT: the keep-bit rejected the proposal; the policy is left untouched).
func runMeta(opts metaOptions, out, errOut io.Writer) int {
	if p := strings.TrimSpace(opts.journalPath); p == "" || p == "-" {
		// meta reads -journal as INPUT; the append-journal default ('-') is almost
		// certainly a mis-invocation. Warn on stderr, keep stdout pure JSON.
		fmt.Fprintln(errOut, "rsiloop meta: -journal is the INPUT journal to fold; pass a real path (got "+strconv.Quote(opts.journalPath)+")")
	}
	rows, err := rsiloop.ReadJournal(opts.journalPath)
	if err != nil {
		fmt.Fprintln(errOut, "rsiloop meta: read journal:", err)
		return 1
	}

	p, ok := rsiloop.Fold(rows, opts.cur, opts.cfg)
	if !ok {
		writeMetaJSON(out, map[string]any{
			"has_proposal": false,
			"journal_rows": len(rows),
			"reason":       "no clustered escalation in window — nothing to propose",
		})
		return 0
	}

	if !opts.apply {
		writeMetaJSON(out, map[string]any{
			"has_proposal": true,
			"witnessed":    false,
			"applied":      false,
			"journal_rows": len(rows),
			"proposal":     proposalView(p),
		})
		return 0
	}

	// --apply: witness the proposal through the keep-bit. A missing witness ref surfaces
	// the library's own refusal (exit 2). The witness journal is the downstream journal
	// OBSERVED under the proposed policy; measure ignores the derived policy and returns
	// that pre-recorded witness (the file-journal witness the issue specifies).
	measure := func(rsiloop.KeepPolicy) ([]rsiloop.Row, error) {
		return rsiloop.ReadJournal(opts.witnessJournal)
	}
	rec, err := rsiloop.ApplyProposalWithWitness(opts.cur, rows, p, opts.apply, opts.witnessJournal, measure)
	if err != nil {
		fmt.Fprintln(errOut, "rsiloop meta apply:", err)
		return 2
	}

	writeMetaJSON(out, map[string]any{
		"has_proposal": true,
		"witnessed":    true,
		"applied":      rec.Applied,
		"decision":     rec.Decision.String(),
		"witness_ref":  rec.WitnessRef,
		"before_rate":  rec.BeforeRate,
		"after_rate":   rec.AfterRate,
		"before_rows":  rec.BeforeRows,
		"after_rows":   rec.AfterRows,
		"policy": map[string]any{
			"gain_threshold": rec.Policy.GainThreshold,
			"breaker_k":      rec.Policy.BreakerK,
			"throttle":       rec.Policy.Throttle,
		},
		"log":      rec.Log,
		"proposal": proposalView(p),
	})
	if rec.Decision != shipgate.KEEP {
		return 3 // witnessed REVERT — proposal rejected, policy left untouched
	}
	return 0
}

// writeMetaJSON emits v as indented JSON (a single object per run) followed by a newline.
func writeMetaJSON(out io.Writer, v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Fprintln(out, "{}")
		return
	}
	fmt.Fprintln(out, string(b))
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
