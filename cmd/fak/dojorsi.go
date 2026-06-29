package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/dojocal"
	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

const dojoRSIMetricName = "dojo_fold_calibrable"

func cmdDojoRSI(argv []string) { os.Exit(runDojoRSI(os.Stdout, os.Stderr, argv)) }

func runDojoRSI(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		dojoRSIUsage(stderr)
		return 2
	}
	cmd, rest := argv[0], argv[1:]
	switch cmd {
	case "fold":
		return runDojoRSIFold(stdout, stderr, rest)
	case "propose":
		return runDojoRSIPropose(stdout, stderr, rest)
	case "rewrite":
		return runDojoRSIRewrite(stdout, stderr, rest)
	case "run":
		return runDojoRSIRun(stdout, stderr, rest)
	case "loop":
		return runDojoRSILoop(stdout, stderr, rest)
	case "trend":
		return runDojoRSITrend(stdout, stderr, rest)
	case "-h", "--help", "help":
		dojoRSIUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak dojo-rsi: unknown command %q\n", cmd)
		dojoRSIUsage(stderr)
		return 2
	}
}

func dojoRSIUsage(w io.Writer) {
	fmt.Fprint(w, `fak dojo-rsi - self-pacing RSI loop over a dojo report

usage:
  fak dojo-rsi fold    --report report.json [--json]
  fak dojo-rsi propose --report report.json [--journal FILE] [--json]
  fak dojo-rsi rewrite --report report.json [--cell lever/metric] [--claims PATH] [--json]
  fak dojo-rsi run     --report report.json [--witness JSON] [--journal FILE] [--json] [--check]
  fak dojo-rsi loop    --report report.json [--ticks N] [--witness JSON] [--journal FILE] [--json]
  fak dojo-rsi trend   [--journal FILE] [--json]

The loop reads a committed dojo report, picks the next cell by novelty x value x
staleness, appends docs/dojo/rsi-journal.jsonl, and stops saturated instead of
thrashing a fresh cell. RECALIBRATE can KEEP; REPROJECT/HARVEST/floor cells route
to the agent arm and can ESCALATE through the breaker.

rewrite previews the exact one-literal change a RECALIBRATE candidate would make to
the claim registry (internal/dojo/claims.go) — a DRY RUN that writes nothing, opens
no worktree, and commits nothing. It is the pure core the Phase-2 worktree arm
(#1024) applies inside a throwaway worktree before re-measuring FoldCalibrable.
`)
}

func runDojoRSIFold(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dojo-rsi fold", flag.ContinueOnError)
	fs.SetOutput(stderr)
	reportPath := fs.String("report", "", "dojo report JSON from `fak dojo run --json`")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dojo-rsi fold: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	r, ok := readDojoRSIReport(stderr, *reportPath)
	if !ok {
		return 2
	}
	fold := dojo.FoldCalibrable(r.Episodes)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, fold, "fak dojo-rsi fold")
	}
	fmt.Fprintf(stdout, "dojo-rsi fold: value %.4g  estimates %.4g/%d  floor %.4g/%d  measured %d\n",
		fold.Value, fold.EstimateMeanCalibErr, fold.EstimateCount, fold.FloorBreachErr, fold.FloorCount, fold.Measured)
	return 0
}

func runDojoRSIPropose(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dojo-rsi propose", flag.ContinueOnError)
	fs.SetOutput(stderr)
	reportPath := fs.String("report", "", "dojo report JSON from `fak dojo run --json`")
	journalPath := fs.String("journal", "", "dojo-RSI journal (default: <root>/"+dojocal.DefaultJournalRel+")")
	nowStr := fs.String("now", "", "pin selector time (RFC3339 or YYYY-MM-DD)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dojo-rsi propose: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	r, ok := readDojoRSIReport(stderr, *reportPath)
	if !ok {
		return 2
	}
	now, ok := parseDojoRSITime(stderr, *nowStr)
	if !ok {
		return 2
	}
	rows := readDojoRSIJournal(journalPathFor(*journalPath))
	payload := dojocal.ProposeRecals(r)
	ranked := dojocal.RankCandidates(payload.Candidates, rows, dojocal.SelectOptions{Now: now})
	wake := dojocal.ScheduleWakeup(ranked, now)
	env := dojoRSIPlan{Schema: "fak-dojo-rsi.plan/1", Propose: payload, Ranked: ranked, Wakeup: wake}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, env, "fak dojo-rsi propose")
	}
	renderDojoRSIPlan(stdout, env)
	return 0
}

// dojoRSIRewritePreview is the --json envelope for `fak dojo-rsi rewrite`: the
// targeted cell, the old/new claim, and the exact one-line diff an apply would
// write — DryRun is always true (this verb never writes, never opens a worktree,
// never commits).
type dojoRSIRewritePreview struct {
	Schema       string  `json:"schema"`
	Lever        string  `json:"lever"`
	Metric       string  `json:"metric"`
	Kind         string  `json:"kind"`
	OldClaimed   float64 `json:"old_claimed"`
	NewClaimed   float64 `json:"new_claimed"`
	MeasuredMean float64 `json:"measured_mean"`
	Sample       int     `json:"sample"`
	Verdict      string  `json:"verdict"`
	CalibErr     float64 `json:"calib_err"`
	ClaimsPath   string  `json:"claims_path"`
	LineNo       int     `json:"line_no"`
	Before       string  `json:"before"`
	After        string  `json:"after"`
	DryRun       bool    `json:"dry_run"`
	Note         string  `json:"note"`
}

// runDojoRSIRewrite previews the single-literal change a RECALIBRATE candidate would
// make to the claim registry. It is the pure-core preview of the Phase-2 worktree arm
// (#1024): it reads the report, proposes recalibrations, picks the worst mechanical
// RECALIBRATE cell (or the --cell the operator names), and renders the exact anchored
// diff `dojocal.RewriteClaim` would apply — WITHOUT writing the file, opening a
// worktree, or committing. A floor/REPROJECT/HARVEST cell is refused with the reason
// it routes to the agent arm instead of being recalibrated, so the preview can never
// suggest erasing a guard the dojo defends.
func runDojoRSIRewrite(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dojo-rsi rewrite", flag.ContinueOnError)
	fs.SetOutput(stderr)
	reportPath := fs.String("report", "", "dojo report JSON from `fak dojo run --json`")
	cellSel := fs.String("cell", "", "target a specific cell as lever/metric (default: the worst RECALIBRATE candidate)")
	claimsPath := fs.String("claims", "", "claim registry path (default: <root>/"+dojocal.ClaimsRelPath+")")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dojo-rsi rewrite: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	report, ok := readDojoRSIReport(stderr, *reportPath)
	if !ok {
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	payload := dojocal.ProposeRecals(report)
	cand, ok := selectRewriteCandidate(stderr, payload.Candidates, *cellSel)
	if !ok {
		return 1
	}

	path := *claimsPath
	if path == "" {
		path = filepath.Join(root, filepath.FromSlash(dojocal.ClaimsRelPath))
	}
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak dojo-rsi rewrite: read claims registry: %v\n", err)
		return 1
	}
	lineNo, before, after, err := dojocal.ClaimChangeLine(src, cand.Lever, cand.Metric, cand.NewClaimed)
	if err != nil {
		fmt.Fprintf(stderr, "fak dojo-rsi rewrite: %v\n", err)
		return 1
	}

	prev := dojoRSIRewritePreview{
		Schema:       "fak-dojo-rsi.rewrite/1",
		Lever:        cand.Lever,
		Metric:       cand.Metric,
		Kind:         string(cand.Kind),
		OldClaimed:   cand.OldClaimed,
		NewClaimed:   cand.NewClaimed,
		MeasuredMean: cand.MeasuredMean,
		Sample:       cand.Sample,
		Verdict:      cand.Verdict,
		CalibErr:     cand.CalibErr,
		ClaimsPath:   filepath.ToSlash(dojocal.ClaimsRelPath),
		LineNo:       lineNo,
		Before:       strings.TrimSpace(before),
		After:        strings.TrimSpace(after),
		DryRun:       true,
		Note:         "dry run: nothing written. The Phase-2 worktree arm (#1024) applies this exact swap inside a throwaway worktree, re-measures FoldCalibrable on two disjoint shards, and only then lands it by-path.",
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, prev, "fak dojo-rsi rewrite")
	}
	renderDojoRSIRewrite(stdout, prev)
	return 0
}

// selectRewriteCandidate picks the cell to preview: an explicit lever/metric from
// --cell, else the worst RECALIBRATE candidate (the candidates are sorted worst-first
// with RECALIBRATE highest-priority, so the first one is the worst mechanical cell).
// A named cell that is not RECALIBRATE — a floor, an UNDER_CLAIM harvest, or a pinned
// REPROJECT — is refused with the reason it routes to the agent arm, never previewed
// as a claim swap.
func selectRewriteCandidate(stderr io.Writer, candidates []dojocal.Recal, cellSel string) (dojocal.Recal, bool) {
	if sel := strings.TrimSpace(cellSel); sel != "" {
		lever, metric, found := strings.Cut(sel, "/")
		if !found {
			fmt.Fprintf(stderr, "fak dojo-rsi rewrite: --cell %q must be lever/metric\n", sel)
			return dojocal.Recal{}, false
		}
		for _, c := range candidates {
			if c.Lever == lever && c.Metric == metric {
				if c.Kind != dojocal.RecalibrateKind {
					fmt.Fprintf(stderr, "fak dojo-rsi rewrite: %s/%s is %s, not a mechanical RECALIBRATE — %s\n", lever, metric, c.Kind, c.Reason)
					return dojocal.Recal{}, false
				}
				return c, true
			}
		}
		fmt.Fprintf(stderr, "fak dojo-rsi rewrite: no candidate cell %s in the report (see `fak dojo-rsi propose`)\n", sel)
		return dojocal.Recal{}, false
	}
	for _, c := range candidates {
		if c.Kind == dojocal.RecalibrateKind {
			return c, true
		}
	}
	// No mechanical recalibration available — name the worst candidate and why it routes.
	if len(candidates) > 0 {
		w := candidates[0]
		fmt.Fprintf(stderr, "fak dojo-rsi rewrite: no mechanical RECALIBRATE candidate; worst cell is %s/%s (%s) — %s\n", w.Lever, w.Metric, w.Kind, w.Reason)
	} else {
		fmt.Fprintln(stderr, "fak dojo-rsi rewrite: the report produced no candidates (nothing measured to recalibrate)")
	}
	return dojocal.Recal{}, false
}

// renderDojoRSIRewrite prints the human preview: the cell, the claim delta, and the
// anchored one-line diff, with the dry-run note last so it is the parting word.
func renderDojoRSIRewrite(w io.Writer, p dojoRSIRewritePreview) {
	fmt.Fprintf(w, "dojo-rsi rewrite (DRY RUN) — %s/%s  %s\n", p.Lever, p.Metric, p.Kind)
	fmt.Fprintf(w, "  re-point claim %.3g -> %.3g (corpus mean %.3g over %d sample(s); worst %s, calib_err %.3g)\n",
		p.OldClaimed, p.NewClaimed, p.MeasuredMean, p.Sample, p.Verdict, p.CalibErr)
	fmt.Fprintf(w, "  %s:%d\n", p.ClaimsPath, p.LineNo)
	fmt.Fprintf(w, "    - %s\n", p.Before)
	fmt.Fprintf(w, "    + %s\n", p.After)
	fmt.Fprintf(w, "  %s\n", p.Note)
}

func runDojoRSIRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dojo-rsi run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	opts := parseDojoRSILoopFlags(fs, true)
	check := fs.Bool("check", false, "honesty-check a kept iteration before accepting it")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dojo-rsi run: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	opts.Ticks = 1
	res, code := runDojoRSITicks(stdout, stderr, opts)
	if code != 0 && code != 3 {
		return code
	}
	if *check {
		for _, it := range res.Iterations {
			if v := dojocal.CheckIteration(it); len(v) > 0 {
				fmt.Fprintln(stdout, "dojo-rsi --check: FAIL")
				for _, one := range v {
					fmt.Fprintf(stdout, "  - %s\n", one)
				}
				return 1
			}
		}
	}
	return code
}

func runDojoRSILoop(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dojo-rsi loop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	opts := parseDojoRSILoopFlags(fs, false)
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dojo-rsi loop: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	returnCode := 0
	_, returnCode = runDojoRSITicks(stdout, stderr, opts)
	return returnCode
}

func runDojoRSITrend(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dojo-rsi trend", flag.ContinueOnError)
	fs.SetOutput(stderr)
	journalPath := fs.String("journal", "", "dojo-RSI journal (default: <root>/"+dojocal.DefaultJournalRel+")")
	nowStr := fs.String("now", "", "pin trend time (RFC3339 or YYYY-MM-DD)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dojo-rsi trend: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	now, ok := parseDojoRSITime(stderr, *nowStr)
	if !ok {
		return 2
	}
	tr := dojocal.FoldTrend(readDojoRSIJournal(journalPathFor(*journalPath)), now)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, tr, "fak dojo-rsi trend")
	}
	fmt.Fprintln(stdout, dojocal.MarshalTrendText(tr))
	return 0
}

type dojoRSILoopOpts struct {
	ReportPath    string
	JournalPath   string
	WitnessJSON   string
	Now           string
	Ticks         int
	K             int
	MinSample     int
	AppendJournal bool
	AsJSON        bool
	DosObserve    bool
	DosArbitrate  bool
	Workspace     string
}

func parseDojoRSILoopFlags(fs *flag.FlagSet, oneShot bool) *dojoRSILoopOpts {
	opts := &dojoRSILoopOpts{}
	fs.StringVar(&opts.ReportPath, "report", "", "dojo report JSON from `fak dojo run --json`")
	fs.StringVar(&opts.JournalPath, "journal", "", "dojo-RSI journal (default: <root>/"+dojocal.DefaultJournalRel+")")
	fs.StringVar(&opts.WitnessJSON, "witness", "", "JSON witness object not authored by the loop, e.g. {\"ok\":true}")
	fs.StringVar(&opts.Now, "now", "", "pin run time (RFC3339 or YYYY-MM-DD)")
	fs.IntVar(&opts.K, "k", 3, "breaker: ESCALATE after K consecutive non-keeps")
	fs.IntVar(&opts.MinSample, "min-sample", dojocal.DefaultMinSample, "minimum measured samples for a RECALIBRATE KEEP")
	fs.BoolVar(&opts.AppendJournal, "append-journal", true, "append the tick(s) to the dojo-RSI journal")
	fs.BoolVar(&opts.AsJSON, "json", false, "emit JSON")
	fs.BoolVar(&opts.DosObserve, "dos-observe", false, "emit dos improve --observe receipts (record-only telemetry)")
	fs.StringVar(&opts.Workspace, "workspace", "", "workspace root (default: repo root)")
	if oneShot {
		opts.Ticks = 1
	} else {
		fs.IntVar(&opts.Ticks, "ticks", 1, "maximum unattended ticks to run")
		fs.BoolVar(&opts.DosArbitrate, "dos-arbitrate", true, "wrap each tick in `dos arbitrate` when dos is installed")
	}
	return opts
}

type dojoRSIPlan struct {
	Schema  string                 `json:"schema"`
	Propose dojocal.ProposePayload `json:"propose"`
	Ranked  []dojocal.ScoredCell   `json:"ranked"`
	Wakeup  dojocal.Wakeup         `json:"wakeup"`
}

type dojoRSILoopResult struct {
	Schema      string               `json:"schema"`
	GeneratedAt string               `json:"generated_at"`
	Ticks       int                  `json:"ticks"`
	Rows        []dojocal.JournalRow `json:"rows"`
	Iterations  []dojocal.Iteration  `json:"iterations,omitempty"`
	Wakeup      dojocal.Wakeup       `json:"wakeup"`
	Trend       dojocal.JournalTrend `json:"trend"`
}

func runDojoRSITicks(stdout, stderr io.Writer, opts *dojoRSILoopOpts) (dojoRSILoopResult, int) {
	root := opts.Workspace
	if root == "" {
		root = repoRoot()
	}
	report, ok := readDojoRSIReport(stderr, opts.ReportPath)
	if !ok {
		return dojoRSILoopResult{}, 2
	}
	now, ok := parseDojoRSITime(stderr, opts.Now)
	if !ok {
		return dojoRSILoopResult{}, 2
	}
	if opts.Ticks <= 0 {
		opts.Ticks = 1
	}
	if opts.K <= 0 {
		opts.K = 3
	}
	var witness map[string]any
	if opts.WitnessJSON != "" {
		if err := json.Unmarshal([]byte(opts.WitnessJSON), &witness); err != nil {
			fmt.Fprintf(stderr, "fak dojo-rsi: parse --witness: %v\n", err)
			return dojoRSILoopResult{}, 2
		}
	}

	journalPath := journalPathForRoot(opts.JournalPath, root)
	rows := readDojoRSIJournal(journalPath)
	breaker := carriedBreaker(rows)
	var obs rsiloop.Observer
	if opts.DosObserve {
		obs = dojoRSIDOSObserveReceipt(root, opts.K)
	}

	res := dojoRSILoopResult{Schema: "fak-dojo-rsi.loop/1", GeneratedAt: now.UTC().Format(time.RFC3339)}
	code := 0
	for i := 0; i < opts.Ticks; i++ {
		tickNow := now.Add(time.Duration(i) * time.Second)
		payload := dojocal.ProposeRecals(report)
		ranked := dojocal.RankCandidates(payload.Candidates, rows, dojocal.SelectOptions{Now: tickNow})
		scored, has := dojocal.NextCandidate(ranked)
		if !has {
			res.Wakeup = dojocal.ScheduleWakeup(ranked, tickNow)
			break
		}
		if opts.DosArbitrate {
			if err := admitDojoRSILane(root); err != nil {
				fmt.Fprintf(stderr, "fak dojo-rsi loop: dos arbitrate refused dojocal tick: %v\n", err)
				return res, 1
			}
		}

		it := dojocal.RunIteration(report, scored.Candidate, opts.MinSample, witness)
		decision := "REVERT"
		if it.Kept {
			decision = "KEEP"
			breaker = 0
		} else {
			breaker++
			if breaker >= opts.K {
				decision = "ESCALATE"
			}
		}
		row := dojocal.NewJournalRow(len(rows)+1, it, decision, breaker, tickNow, dojoHeadCommit(root), dojocal.Wakeup{})
		nextRows := append(append([]dojocal.JournalRow(nil), rows...), row)
		nextRanked := dojocal.RankCandidates(payload.Candidates, nextRows, dojocal.SelectOptions{Now: tickNow})
		wake := dojocal.ScheduleWakeup(nextRanked, tickNow)
		row.WakeupAt = wake.At
		rows = append(rows, row)
		res.Rows = append(res.Rows, row)
		res.Iterations = append(res.Iterations, it)
		res.Ticks++
		res.Wakeup = wake

		if opts.AppendJournal {
			if err := appendDojoRSIJournalRow(journalPath, row); err != nil {
				fmt.Fprintf(stderr, "fak dojo-rsi: append journal: %v\n", err)
				return res, 1
			}
		}
		if obs != nil {
			obs(dojoRSIRowToObserverRow(row, it, witnessOKForDojoRSI(witness)))
		}
		if decision == "ESCALATE" {
			code = 3
			break
		}
	}
	res.Trend = dojocal.FoldTrend(rows, now)
	if opts.AsJSON {
		_ = encodeJSONOrFail(stdout, stderr, res, "fak dojo-rsi loop")
	} else {
		renderDojoRSILoop(stdout, res)
	}
	return res, code
}

func readDojoRSIReport(stderr io.Writer, path string) (dojo.Report, bool) {
	if path == "" {
		fmt.Fprintln(stderr, "fak dojo-rsi: --report is required")
		return dojo.Report{}, false
	}
	var b []byte
	var err error
	if path == "-" {
		b, err = io.ReadAll(os.Stdin)
	} else {
		b, err = os.ReadFile(path)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak dojo-rsi: read --report: %v\n", err)
		return dojo.Report{}, false
	}
	var r dojo.Report
	if err := json.Unmarshal(b, &r); err == nil && r.Schema == dojo.Schema {
		return r, true
	}
	var env struct {
		Report dojo.Report `json:"report"`
	}
	if err := json.Unmarshal(b, &env); err == nil && env.Report.Schema == dojo.Schema {
		return env.Report, true
	}
	fmt.Fprintln(stderr, "fak dojo-rsi: --report is not a dojo report JSON")
	return dojo.Report{}, false
}

func parseDojoRSITime(stderr io.Writer, s string) (time.Time, bool) {
	if strings.TrimSpace(s) == "" {
		return time.Now().UTC(), true
	}
	for _, layout := range []string{time.RFC3339, "2006-01-02"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), true
		}
	}
	fmt.Fprintf(stderr, "fak dojo-rsi: --now %q is not RFC3339 or YYYY-MM-DD\n", s)
	return time.Time{}, false
}

func journalPathFor(path string) string {
	return journalPathForRoot(path, repoRoot())
}

func journalPathForRoot(path, root string) string {
	if path != "" {
		return path
	}
	return filepath.Join(root, filepath.FromSlash(dojocal.DefaultJournalRel))
}

func readDojoRSIJournal(path string) []dojocal.JournalRow {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return dojocal.ParseJournal(string(b))
}

func appendDojoRSIJournalRow(path string, row dojocal.JournalRow) error {
	line, err := dojocal.AppendJournalLine(row)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line + "\n")
	return err
}

func carriedBreaker(rows []dojocal.JournalRow) int {
	n := 0
	for i := len(rows) - 1; i >= 0; i-- {
		if rows[i].Decision == "KEEP" {
			break
		}
		n++
	}
	return n
}

func witnessOKForDojoRSI(w map[string]any) bool {
	if w == nil {
		return false
	}
	v, ok := w["ok"].(bool)
	return ok && v
}

func renderDojoRSIPlan(w io.Writer, p dojoRSIPlan) {
	fmt.Fprintf(w, "dojo-rsi propose: %d candidate(s); wake %s\n", len(p.Propose.Candidates), p.Wakeup.At)
	for i, s := range p.Ranked {
		fmt.Fprintf(w, "%d. %.3f  %s/%s  %s  value=%.3f stale=%.3f%s\n",
			i+1, s.Score, s.Candidate.Lever, s.Candidate.Metric, s.Candidate.Kind,
			s.ValueWeight, s.Staleness, saturatedSuffix(s.Saturated))
		fmt.Fprintf(w, "   %s\n", s.Reason)
	}
}

func renderDojoRSILoop(w io.Writer, res dojoRSILoopResult) {
	if len(res.Rows) == 0 {
		fmt.Fprintf(w, "dojo-rsi loop: no tick run; %s\n", res.Wakeup.Reason)
		fmt.Fprintln(w, dojocal.MarshalTrendText(res.Trend))
		return
	}
	for _, r := range res.Rows {
		fmt.Fprintf(w, "tick %d  %s/%s  %s -> %s  base=%.4g replay=%.4g delta=%.4g breaker=%d\n",
			r.Tick, r.Lever, r.Metric, r.Kind, r.Decision, r.Baseline, r.Replayed, r.MeasuredDelta, r.BreakerCount)
		if r.Reason != "" {
			fmt.Fprintf(w, "  %s\n", r.Reason)
		}
	}
	if res.Wakeup.At != "" {
		fmt.Fprintf(w, "wake: %s (%s)\n", res.Wakeup.At, res.Wakeup.Reason)
	}
	fmt.Fprintln(w, dojocal.MarshalTrendText(res.Trend))
}

func saturatedSuffix(v bool) string {
	if v {
		return " saturated"
	}
	return ""
}

func dojoRSIRowToObserverRow(row dojocal.JournalRow, it dojocal.Iteration, suiteGreen bool) rsiloop.Row {
	return rsiloop.Row{
		Cycle:        row.Tick,
		Mode:         "improve",
		Candidate:    fmt.Sprintf("%s/%s/%s", row.Lever, row.Metric, row.Kind),
		MetricName:   dojoRSIMetricName,
		Baseline:     it.BaselineValue,
		Candidate_:   it.ReplayedValue,
		Measured:     true,
		LowerBetter:  true,
		Improved:     it.MeasuredDelta > 0,
		SuiteGreen:   suiteGreen,
		TruthClean:   true,
		Decision:     row.Decision,
		Kept:         row.Kept,
		BreakerCount: row.BreakerCount,
		BaselineRef:  row.Commit,
		RefName:      "main",
		Note:         row.Reason,
	}
}

func dojoRSIDOSObserveReceipt(workspace string, maxReverts int) rsiloop.Observer {
	dosPath, err := exec.LookPath("dos")
	if err != nil {
		fmt.Fprintln(os.Stderr, "dojo-rsi --dos-observe: 'dos' not on PATH; emitting no receipts")
		return nil
	}
	return func(r rsiloop.Row) {
		cmd := exec.Command(dosPath, dojoRSIDOSImproveArgs(workspace, maxReverts, r)...)
		cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
		if runErr := cmd.Run(); runErr != nil {
			var exitErr *exec.ExitError
			if !errors.As(runErr, &exitErr) {
				fmt.Fprintf(os.Stderr, "dojo-rsi --dos-observe: dos improve did not run for tick %d: %v\n", r.Cycle, runErr)
			}
		}
	}
}

func dojoRSIDOSImproveArgs(workspace string, maxReverts int, r rsiloop.Row) []string {
	work, baselineWork := dojoRSIScaleWorkPair(r.Candidate_, r.Baseline, r.LowerBetter)
	args := []string{
		"improve", "--observe",
		"--work", strconv.FormatInt(work, 10),
		"--baseline-work", strconv.FormatInt(baselineWork, 10),
		"--consecutive-reverts", strconv.Itoa(r.BreakerCount),
		"--max-reverts", strconv.Itoa(maxReverts),
		"--lane", "dojocal",
		"--subject", fmt.Sprintf("dojo-rsi::%s::tick%d", r.Candidate, r.Cycle),
		"--narrated", fmt.Sprintf("dojo-rsi verdict=%s candidate=%q improved=%v measured=%v", r.Decision, r.Candidate, r.Improved, r.Measured),
	}
	if r.SuiteGreen {
		args = append(args, "--suite-passed")
	}
	if r.TruthClean {
		args = append(args, "--truth-clean")
	}
	if workspace != "" {
		args = append(args, "--workspace", workspace)
	}
	return args
}

func dojoRSIRoundUnit(x float64) int64 { return int64(math.Round(x * 1e9)) }

func dojoRSIScaleWorkPair(candidate, baseline float64, lowerBetter bool) (work, baselineWork int64) {
	if lowerBetter {
		c := math.Max(candidate, baseline)
		return dojoRSIRoundUnit(c - candidate), dojoRSIRoundUnit(c - baseline)
	}
	return dojoRSIRoundUnit(candidate), dojoRSIRoundUnit(baseline)
}

func admitDojoRSILane(root string) error {
	dosPath, err := exec.LookPath("dos")
	if err != nil {
		return nil
	}
	args := []string{
		"arbitrate",
		"--workspace", root,
		"--lane", "dojocal",
		"--mode", "exclusive",
		"--tree", "internal/dojo/**", "internal/dojocal/**", "cmd/fak/dojo*.go", "docs/dojo/**", ".github/workflows/dojo-rsi-feed.yml",
		"--output", "json",
	}
	cmd := exec.Command(dosPath, args...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}
