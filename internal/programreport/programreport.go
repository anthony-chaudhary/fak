package programreport

// programreport.go is the pure, unit-tested surface of the ONGOING-PROGRAM report:
// the dimension interpreters (one per worktype program), the fold, the durable-ledger
// parse/trend, render, and the advisory gate. The live runners (the cache-value +
// cache-frontier ledgers and the perf-lane git window) live in collect.go. The
// package doc is in doc.go.
//
// This report is the ongoing-program sibling of internal/milestonereport. Where the
// milestone roadmap measures DISCRETE epics by completion %, this report measures the
// ONGOING PROGRAMS (worktype.KernelOptimization, worktype.CacheOptimization, and
// worktype.HumanOperatorEffectiveness) the way they actually move: by a FRONTIER
// (the best number / state witnessed so far) and a TREND (is the frontier still
// advancing?). There is no completion % here on purpose — an ongoing program has no
// 100%.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/worktype"
)

// Schema is the stable control-pane schema identifier for the report envelope.
const Schema = "fak-program-report/1"

// LedgerSchema tags each durable history row so a reader can validate the line.
const LedgerSchema = "fak-program-ledger/1"

// DefaultLedgerRel is the committed, append-only history ledger (one JSONL row per
// program tick). It lives under docs/ so it is durable trunk evidence, not a
// regenerable build artifact.
const DefaultLedgerRel = "docs/programs/history.jsonl"

// --- the per-program dimension ----------------------------------------------

// Signal is one ongoing program's measured frontier-activity for a tick. It is
// deliberately NOT a completion %: Frontier is the best-witnessed number/state (e.g.
// the realized reuse ratio, or the trailing perf-ship count), Direction is whether
// the frontier is advancing, and Activity is the recent shipped work that moved it.
// A program whose signal could not be measured sets Err — never a silent zero, the
// same honesty seam the milestone roadmap carries for an unreadable epic.
type Signal struct {
	Class     worktype.Class `json:"class"`
	Label     string         `json:"label"`
	Doc       string         `json:"doc,omitempty"`    // the program's operating-plan doc
	Frontier  string         `json:"frontier"`         // the best-witnessed frontier reading (human string)
	Metric    float64        `json:"metric"`           // the numeric frontier reading the trend is computed on
	Direction string         `json:"direction"`        // advancing | holding | regressed | unknown
	Activity  int            `json:"activity"`         // recent shipped frontier-moves over the window
	Window    string         `json:"window,omitempty"` // the window the activity is counted over (e.g. "7d")
	Note      string         `json:"note,omitempty"`   // an honesty fence / caveat (e.g. the #1066 reuse-family label)
	Err       string         `json:"err,omitempty"`
	OK        bool           `json:"ok"`
}

// Programs is the PROGRAMS dimension: the per-program signals folded for a tick.
// Err is set only when EVERY program failed to measure (a true unmeasured dimension
// that gates); a partial failure records a non-gating PartialNote and leaves Err == ""
// so one flaky ledger read never reds the whole report.
type Programs struct {
	Signals     []Signal `json:"signals"`
	Tracked     int      `json:"tracked"`
	Measured    int      `json:"measured"`
	Advancing   int      `json:"advancing"`
	Regressed   int      `json:"regressed"`
	PartialNote string   `json:"partial_note,omitempty"`
	Err         string   `json:"err,omitempty"`
	OK          bool     `json:"ok"`
}

// InterpretPrograms folds the per-program signals into the PROGRAMS dimension. The
// signals arrive already measured (collect.go does the impure reads); this is the
// pure tally so the verdict logic and the JSON shape are unit-testable with no disk.
func InterpretPrograms(signals []Signal) Programs {
	p := Programs{Tracked: len(signals)}
	var failed int
	for _, s := range signals {
		if s.Err != "" {
			failed++
		} else {
			p.Measured++
			switch s.Direction {
			case "advancing":
				p.Advancing++
			case "regressed":
				p.Regressed++
			}
		}
		p.Signals = append(p.Signals, s)
	}
	switch {
	case p.Tracked == 0:
		p.Err = "no programs tracked"
	case p.Measured == 0:
		p.Err = "no program could be measured (every program's frontier signal failed to read)"
	case failed > 0:
		p.PartialNote = fmt.Sprintf("%d of %d program(s) had no readable frontier signal", failed, p.Tracked)
	}
	p.OK = p.Err == ""
	return p
}

// --- the fold ---------------------------------------------------------------

// Trend is the per-tick delta vs the previous ledger row.
type Trend struct {
	PrevDate          string  `json:"prev_date"`
	PrevCommit        string  `json:"prev_commit"`
	Direction         string  `json:"direction"` // improved | regressed | flat | new
	KernelMetricFrom  float64 `json:"kernel_metric_from"`
	KernelMetricTo    float64 `json:"kernel_metric_to"`
	KernelMetricDelta float64 `json:"kernel_metric_delta"`
	CacheMetricFrom   float64 `json:"cache_metric_from"`
	CacheMetricTo     float64 `json:"cache_metric_to"`
	CacheMetricDelta  float64 `json:"cache_metric_delta"`
	HumanMetricFrom   float64 `json:"human_metric_from"`
	HumanMetricTo     float64 `json:"human_metric_to"`
	HumanMetricDelta  float64 `json:"human_metric_delta"`
	AdvancingFrom     int     `json:"advancing_from"`
	AdvancingTo       int     `json:"advancing_to"`
	Summary           string  `json:"summary"`
}

// Report is one folded ongoing-program control-pane envelope.
type Report struct {
	Schema      string   `json:"schema"`
	OK          bool     `json:"ok"`
	Verdict     string   `json:"verdict"`
	Finding     string   `json:"finding"`
	Reason      string   `json:"reason"`
	NextAction  string   `json:"next_action"`
	Workspace   string   `json:"workspace"`
	Commit      string   `json:"commit"`
	GeneratedAt string   `json:"generated_at"`
	Date        string   `json:"date"`
	Programs    Programs `json:"programs"`
	Trend       *Trend   `json:"trend,omitempty"`
	GateExit    *int     `json:"gate_exit,omitempty"`
	GateMessage string   `json:"gate_message,omitempty"`
}

// FoldOpts carries the ambient context the fold stamps onto the envelope.
type FoldOpts struct {
	Workspace   string
	Commit      string
	GeneratedAt string
	Date        string
}

// Fold folds the programs dimension into one report envelope. The verdict ladder
// mirrors milestonereport's REPORT contract, not a second quality gate: it is ACTION
// only when the dimension could not be MEASURED (every program's frontier signal
// failed), and OK otherwise — a regressed program frontier is surfaced as an advisory
// line, never a gate (a frontier can dip for honest reasons; the per-program ratchet,
// e.g. the cache-value trend gate, owns the real regression gate).
func Fold(p Programs, opts FoldOpts) Report {
	r := Report{
		Schema:      Schema,
		Workspace:   opts.Workspace,
		Commit:      opts.Commit,
		GeneratedAt: opts.GeneratedAt,
		Date:        opts.Date,
		Programs:    p,
	}

	summary := summarize(p)
	switch {
	case p.Err != "":
		r.OK, r.Verdict, r.Finding = false, "ACTION", "programs_unmeasured"
		r.Reason = "program report incomplete — " + p.Err
		r.NextAction = "repair the failing program signal (the cache-value / cache-frontier ledgers or the perf-lane git window), then re-run `fak program report`"
	case p.Regressed > 0:
		r.OK, r.Verdict, r.Finding = true, "OK", "programs_advisory"
		r.Reason = "programs recorded; " + summary + fmt.Sprintf(" (advisory: %d program frontier(s) regressed — the per-program ratchet owns that gate)", p.Regressed)
		r.NextAction = "investigate the regressed program's frontier (e.g. `fak cachevalue report` for cache-opt, the perf-parity RSI loop for kernel-opt); the tick keeps recording the trend"
	default:
		r.OK, r.Verdict, r.Finding = true, "OK", "programs_recorded"
		r.Reason = "programs recorded; " + summary
		r.NextAction = "hold the line; the scheduled program tick keeps each ongoing program's frontier + trend recorded"
	}
	return r
}

// summarize renders the one-line programs summary (per-program frontier + direction).
func summarize(p Programs) string {
	parts := make([]string, 0, len(p.Signals))
	for _, s := range p.Signals {
		if s.Err != "" {
			parts = append(parts, fmt.Sprintf("%s: unmeasured", s.Label))
			continue
		}
		parts = append(parts, fmt.Sprintf("%s: %s (%s)", s.Label, s.Frontier, s.Direction))
	}
	return strings.Join(parts, "; ")
}

// withAdvisory rewrites a recorded report when the per-tick trend regressed — an
// advisory line, never a gate flip. Applied after the trend is attached so Fold stays
// trend-free + pure (the milestonereport split).
func (r Report) withAdvisory() Report {
	if r.Finding != "programs_recorded" || r.Trend == nil || r.Trend.Direction != "regressed" {
		return r
	}
	r.Finding = "programs_advisory"
	r.Reason += " (advisory: " + r.Trend.Summary + ")"
	r.NextAction = "investigate the regression — a program frontier metric fell across ticks; the tick keeps recording the trend"
	return r
}

// WithTrend attaches a per-tick trend and applies the advisory rewrite.
func (r Report) WithTrend(t Trend) Report {
	r.Trend = &t
	return r.withAdvisory()
}

// metricFor returns the numeric frontier metric for a program class in the signals
// (0 when the program is absent or unmeasured), so the ledger row + trend have a
// stable per-class column regardless of signal ordering.
func metricFor(p Programs, c worktype.Class) float64 {
	for _, s := range p.Signals {
		if s.Class == c && s.Err == "" {
			return s.Metric
		}
	}
	return 0
}

// --- the durable history ledger ---------------------------------------------

// LedgerRow is one durable, append-only history line (a flattened per-class
// projection so the ledger is a self-describing time series).
type LedgerRow struct {
	Schema       string  `json:"schema"`
	Date         string  `json:"date"`
	Commit       string  `json:"commit"`
	GeneratedAt  string  `json:"generated_at"`
	Verdict      string  `json:"verdict"`
	Tracked      int     `json:"tracked"`
	Measured     int     `json:"measured"`
	Advancing    int     `json:"advancing"`
	Regressed    int     `json:"regressed"`
	KernelMetric float64 `json:"kernel_metric"`
	KernelDir    string  `json:"kernel_dir,omitempty"`
	CacheMetric  float64 `json:"cache_metric"`
	CacheDir     string  `json:"cache_dir,omitempty"`
	HumanMetric  float64 `json:"human_metric"`
	HumanDir     string  `json:"human_dir,omitempty"`
}

// RowFromReport projects a folded report into one durable ledger row.
func RowFromReport(r Report) LedgerRow {
	return LedgerRow{
		Schema:       LedgerSchema,
		Date:         r.Date,
		Commit:       r.Commit,
		GeneratedAt:  r.GeneratedAt,
		Verdict:      r.Verdict,
		Tracked:      r.Programs.Tracked,
		Measured:     r.Programs.Measured,
		Advancing:    r.Programs.Advancing,
		Regressed:    r.Programs.Regressed,
		KernelMetric: metricFor(r.Programs, worktype.KernelOptimization),
		KernelDir:    dirFor(r.Programs, worktype.KernelOptimization),
		CacheMetric:  metricFor(r.Programs, worktype.CacheOptimization),
		CacheDir:     dirFor(r.Programs, worktype.CacheOptimization),
		HumanMetric:  metricFor(r.Programs, worktype.HumanOperatorEffectiveness),
		HumanDir:     dirFor(r.Programs, worktype.HumanOperatorEffectiveness),
	}
}

func dirFor(p Programs, c worktype.Class) string {
	for _, s := range p.Signals {
		if s.Class == c && s.Err == "" {
			return s.Direction
		}
	}
	return ""
}

// ParseLedger parses an append-only JSONL ledger, tolerating blank + garbled lines.
func ParseLedger(content string) []LedgerRow {
	var rows []LedgerRow
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row LedgerRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.Date == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// AppendLedgerLine renders the JSONL line for a row (no trailing newline).
func AppendLedgerLine(row LedgerRow) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// TrendVsLast computes the per-tick trend vs the most recent prior row. The direction
// is driven by the two program metrics: a rise in either frontier metric is
// "improved", a fall in either (with no rise) is "regressed". With no prior row it is
// "new".
func TrendVsLast(row LedgerRow, prior []LedgerRow) Trend {
	last, ok := latestBefore(row, prior)
	if !ok {
		return Trend{
			Direction:      "new",
			KernelMetricTo: row.KernelMetric,
			CacheMetricTo:  row.CacheMetric,
			HumanMetricTo:  row.HumanMetric,
			AdvancingTo:    row.Advancing,
			Summary: fmt.Sprintf("first program tick (kernel metric %.3f %s, cache metric %.3f %s, human metric %.3f %s; %d advancing)",
				row.KernelMetric, dashIfEmpty(row.KernelDir), row.CacheMetric, dashIfEmpty(row.CacheDir), row.HumanMetric, dashIfEmpty(row.HumanDir), row.Advancing),
		}
	}
	kDelta := round3(row.KernelMetric - last.KernelMetric)
	cDelta := round3(row.CacheMetric - last.CacheMetric)
	hDelta := round3(row.HumanMetric - last.HumanMetric)
	dir := "flat"
	switch {
	case kDelta > 0 || cDelta > 0 || hDelta > 0:
		dir = "improved"
	case kDelta < 0 || cDelta < 0 || hDelta < 0:
		dir = "regressed"
	}
	return Trend{
		PrevDate:          last.Date,
		PrevCommit:        last.Commit,
		Direction:         dir,
		KernelMetricFrom:  last.KernelMetric,
		KernelMetricTo:    row.KernelMetric,
		KernelMetricDelta: kDelta,
		CacheMetricFrom:   last.CacheMetric,
		CacheMetricTo:     row.CacheMetric,
		CacheMetricDelta:  cDelta,
		HumanMetricFrom:   last.HumanMetric,
		HumanMetricTo:     row.HumanMetric,
		HumanMetricDelta:  hDelta,
		AdvancingFrom:     last.Advancing,
		AdvancingTo:       row.Advancing,
		Summary: fmt.Sprintf("programs %s; kernel metric %+.3f (%.3f->%.3f), cache metric %+.3f (%.3f->%.3f), human metric %+.3f (%.3f->%.3f) vs %s",
			dir, kDelta, last.KernelMetric, row.KernelMetric, cDelta, last.CacheMetric, row.CacheMetric, hDelta, last.HumanMetric, row.HumanMetric, last.Date),
	}
}

// latestBefore returns the most recent prior row (by date, then generated_at),
// excluding a row with the exact same generated_at (idempotent re-append).
func latestBefore(row LedgerRow, prior []LedgerRow) (LedgerRow, bool) {
	cands := make([]LedgerRow, 0, len(prior))
	for _, p := range prior {
		if p.GeneratedAt != "" && p.GeneratedAt == row.GeneratedAt {
			continue
		}
		cands = append(cands, p)
	}
	if len(cands) == 0 {
		return LedgerRow{}, false
	}
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].Date != cands[j].Date {
			return cands[i].Date < cands[j].Date
		}
		return cands[i].GeneratedAt < cands[j].GeneratedAt
	})
	return cands[len(cands)-1], true
}

// --- render + gate ----------------------------------------------------------

// Render produces the human snapshot.
func Render(r Report) string {
	mark := func(ok bool, err string) string {
		if err != "" {
			return "x"
		}
		if ok {
			return "+"
		}
		return "."
	}
	lines := []string{
		fmt.Sprintf("program report — %s (%s)  @%s  %s", r.Verdict, r.Finding, r.Commit, r.Date),
		"",
		fmt.Sprintf("  %s programs    %d/%d measured; %d advancing, %d regressed",
			mark(r.Programs.OK, r.Programs.Err), r.Programs.Measured, r.Programs.Tracked, r.Programs.Advancing, r.Programs.Regressed),
		"      (ongoing programs — frontier + trend, never 'done')",
	}
	for _, s := range r.Programs.Signals {
		lines = append(lines, "      "+signalLine(s))
		if s.Note != "" {
			lines = append(lines, "          fence: "+s.Note)
		}
	}
	if r.Programs.PartialNote != "" {
		lines = append(lines, "      ("+r.Programs.PartialNote+")")
	}
	if r.Trend != nil {
		lines = append(lines, "", "  trend: "+r.Trend.Summary)
	}
	lines = append(lines, "", "  -> "+r.NextAction)
	return strings.Join(lines, "\n")
}

// signalLine renders one program's frontier line, surfacing an unreadable signal
// honestly rather than as a fabricated frontier.
func signalLine(s Signal) string {
	if s.Err != "" {
		return fmt.Sprintf("%s — signal unreadable (%s)", s.Label, s.Err)
	}
	win := ""
	if s.Window != "" {
		win = " over " + s.Window
	}
	return fmt.Sprintf("%s — frontier: %s; %s; %d shipped move(s)%s", s.Label, s.Frontier, s.Direction, s.Activity, win)
}

// CheckGate is the advisory CI gate over a folded report. It fails ONLY when the
// programs dimension could not be measured — the report is a mirror, not a second
// quality gate (a regressed frontier is a measured fact, not an incomplete report).
//
//	0  programs recorded (clear or advisory)
//	1  the programs dimension failed to measure
func CheckGate(r Report) (int, string) {
	if r.Finding == "programs_unmeasured" {
		return 1, "PROGRAM INCOMPLETE: " + r.Reason
	}
	return 0, "PROGRAM OK: " + r.Reason
}

// WithGate returns a copy reconciled to a CheckGate decision, for --check --json.
func (r Report) WithGate(code int, message string) Report {
	q := r
	q.OK = code == 0
	if code == 0 {
		q.Verdict = "OK"
	} else {
		q.Verdict = "ACTION"
	}
	c := code
	q.GateExit = &c
	q.GateMessage = message
	return q
}

// --- small shared helpers ---------------------------------------------------

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func round3(f float64) float64 {
	if f < 0 {
		return float64(int64(f*1000-0.5)) / 1000
	}
	return float64(int64(f*1000+0.5)) / 1000
}
