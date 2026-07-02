package main

// hooklat.go — `fak hooklat`: the guard-hook latency rollup command (issue #1993).
//
// Guard adjudication fires a pretool + posttool hook around every tool call the
// wrapped agent makes, and the DOS hook-observation stream records each firing's
// latency_ms — but until this command nothing aggregated it: the dominant
// guard-imposed per-tool-call wall-clock cost was written to disk and never read.
// This is the read side: discover the observation streams, fold them through
// internal/turntaxmeter's percentile rollup, and judge the tail against the
// declared p99 budget. A breach names the closed GATE_LATENCY_REGRESSION token and
// exits 1 so the command is CI-gateable; a thin sample (< MinHookAlarmSamples)
// prints its stats but abstains from alarming, per #1993's own small-n caveat.
//
// The same fold renders the one-line hook-latency row in the `fak guard` exit
// summary (guard_child.go), so the tax is visible at the surface every guarded
// session already prints — not only to operators who know to ask.

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

func cmdHookLat(argv []string) {
	fs := flag.NewFlagSet("hooklat", flag.ExitOnError)
	budget := fs.Float64("budget-p99-ms", turntaxmeter.DefaultHookP99BudgetMS,
		"p99 latency budget in milliseconds the folded tail is judged against; a breach prints GATE_LATENCY_REGRESSION and exits 1. 0 disables the alarm (report-only)")
	since := fs.Duration("since", 0,
		"only fold observations newer than this age (e.g. 2h); 0 folds the whole stream. Rows without a timestamp cannot witness the window and are excluded when set")
	jsonOut := fs.Bool("json", false, "emit the rollup + verdict as one JSON object instead of the table")
	_ = fs.Parse(argv)

	paths := fs.Args()
	if len(paths) == 0 {
		paths = discoverHookObservationStreams(".")
	}
	if len(paths) == 0 {
		fmt.Println("fak hooklat: no hook-observation streams found (looked for .dos/metrics/observations.jsonl and .dispatch-runs/*/.dos/metrics/observations.jsonl); pass stream paths explicitly")
		return
	}

	var obs []turntaxmeter.HookObservation
	type sourceCount struct {
		path          string
		rows, skipped int
	}
	sources := make([]sourceCount, 0, len(paths))
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak hooklat: skip %s: %v\n", p, err)
			continue
		}
		rows, skipped, err := turntaxmeter.ParseHookObservations(f)
		f.Close()
		must(err)
		obs = append(obs, rows...)
		sources = append(sources, sourceCount{p, len(rows), skipped})
	}
	if *since > 0 {
		obs = turntaxmeter.FilterHookObservationsSince(obs, time.Now().Add(-*since))
	}

	rollup := turntaxmeter.FoldHookLatency(obs)
	verdict := turntaxmeter.JudgeHookLatency(rollup.Total, *budget)

	if *jsonOut {
		out, err := json.MarshalIndent(struct {
			Rollup  turntaxmeter.HookLatencyRollup  `json:"rollup"`
			Verdict turntaxmeter.HookLatencyVerdict `json:"verdict"`
		}{rollup, verdict}, "", "  ")
		must(err)
		fmt.Println(string(out))
	} else {
		fmt.Println("== fak hooklat: guard-hook latency rollup (hook-observation v1) ==")
		for _, s := range sources {
			fmt.Printf("source: %s (%d rows, %d skipped)\n", s.path, s.rows, s.skipped)
		}
		fmt.Println()
		fmt.Print(formatHookLatencyTable(rollup))
		fmt.Println()
		fmt.Print(formatHookLatencyVerdict(verdict))
	}
	if !verdict.OK {
		os.Exit(1)
	}
}

// discoverHookObservationStreams returns the hook-observation streams a workspace
// conventionally carries: its own .dos/metrics/observations.jsonl plus one per
// dispatch-run workspace under .dispatch-runs/. Missing paths are simply absent —
// discovery never errors, it just finds fewer streams.
func discoverHookObservationStreams(root string) []string {
	var paths []string
	own := filepath.Join(root, ".dos", "metrics", "observations.jsonl")
	if _, err := os.Stat(own); err == nil {
		paths = append(paths, own)
	}
	runs, _ := filepath.Glob(filepath.Join(root, ".dispatch-runs", "*", ".dos", "metrics", "observations.jsonl"))
	sort.Strings(runs)
	return append(paths, runs...)
}

// formatHookLatencyTable renders the per-verb + total percentile rows. Pure
// (rollup in, string out) so the table shape is unit-testable.
func formatHookLatencyTable(r turntaxmeter.HookLatencyRollup) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-9s %6s %9s %9s %9s %9s %9s\n", "verb", "n", "mean", "p50", "p90", "p99", "max")
	row := func(s turntaxmeter.HookLatencyStats, label string) {
		fmt.Fprintf(&b, "%-9s %6d %8.1fms %8.1fms %8.1fms %8.1fms %8.1fms\n",
			label, s.Count, s.MeanMS, s.P50MS, s.P90MS, s.P99MS, s.MaxMS)
	}
	for _, s := range r.ByVerb {
		row(s, s.Verb)
	}
	if r.Total.Count > 0 {
		row(r.Total, "all")
	} else {
		b.WriteString("(no measured hook observations)\n")
	}
	return b.String()
}

// formatHookLatencyVerdict renders the budget judgment: the closed token on a
// breach, an explicit abstention on a thin sample, and the within-budget line
// otherwise — never a silent pass.
func formatHookLatencyVerdict(v turntaxmeter.HookLatencyVerdict) string {
	switch {
	case !v.OK:
		return fmt.Sprintf("verdict: %s — p99 %.1fms exceeds the %.0fms budget (n=%d); the guard hook path taxes every tool call — reduce it (#2073), don't normalize it\n",
			v.Reason, v.ObservedP99MS, v.BudgetP99MS, v.Count)
	case v.Thin:
		return fmt.Sprintf("verdict: THIN — n=%d < %d samples; percentiles above are real but the alarm abstains until the rollup accumulates a trustworthy tail\n",
			v.Count, turntaxmeter.MinHookAlarmSamples)
	case v.BudgetP99MS <= 0:
		return fmt.Sprintf("verdict: REPORT-ONLY — no p99 budget declared (n=%d, p99 %.1fms)\n", v.Count, v.ObservedP99MS)
	default:
		return fmt.Sprintf("verdict: OK — p99 %.1fms within the %.0fms budget (n=%d)\n",
			v.ObservedP99MS, v.BudgetP99MS, v.Count)
	}
}

// formatGuardHookLatencyLine is the exit-summary render: ONE line with the
// session's hook tax, or "" when nothing was measured (a hook-less session stays
// quiet rather than printing a vacuous zero row). windowLabel names the fold's
// honest scope — "session window" when the rows were since-filtered to this run,
// "all recorded runs" when no session anchor existed.
func formatGuardHookLatencyLine(r turntaxmeter.HookLatencyRollup, v turntaxmeter.HookLatencyVerdict, windowLabel string) string {
	if r.Total.Count == 0 {
		return ""
	}
	state := fmt.Sprintf("p99 budget %.0fms: OK", v.BudgetP99MS)
	switch {
	case !v.OK:
		state = fmt.Sprintf("p99 budget %.0fms: %s — see `fak hooklat` and #2073", v.BudgetP99MS, v.Reason)
	case v.Thin:
		state = fmt.Sprintf("alarm abstains: n=%d < %d", v.Count, turntaxmeter.MinHookAlarmSamples)
	}
	return fmt.Sprintf("fak guard: hook-latency — n=%d p50=%.0fms p90=%.0fms p99=%.0fms max=%.0fms per hook firing, pre+post per tool call (%s; %s)\n",
		r.Total.Count, r.Total.P50MS, r.Total.P90MS, r.Total.P99MS, r.Total.MaxMS, windowLabel, state)
}

// guardHookLatencySummaryLine reads the workspace's own hook-observation stream
// and folds it for the exit summary. window > 0 scopes the fold to the trailing
// session window (rows must carry a timestamp inside it); window == 0 folds the
// whole stream and says so. Best-effort by contract: no stream, an unreadable
// stream, or zero measured rows all return "" — the exit summary never grows an
// error line for an observability nicety.
func guardHookLatencySummaryLine(window time.Duration, now time.Time) string {
	f, err := os.Open(filepath.Join(".dos", "metrics", "observations.jsonl"))
	if err != nil {
		return ""
	}
	defer f.Close()
	obs, _, err := turntaxmeter.ParseHookObservations(f)
	if err != nil {
		return ""
	}
	windowLabel := "all recorded runs"
	if window > 0 {
		obs = turntaxmeter.FilterHookObservationsSince(obs, now.Add(-window))
		windowLabel = "session window"
	}
	rollup := turntaxmeter.FoldHookLatency(obs)
	verdict := turntaxmeter.JudgeHookLatency(rollup.Total, turntaxmeter.DefaultHookP99BudgetMS)
	return formatGuardHookLatencyLine(rollup, verdict, windowLabel)
}
