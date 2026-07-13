package milestoneburndown

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/trendreport"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// Schema is the report envelope schema id; LedgerSchema is the durable trend row.
const (
	Schema       = "fak-milestone-burndown/1"
	LedgerSchema = "fak-milestone-burndown-ledger/1"
)

// DefaultWindowDays is the trailing window over which recent closure velocity is
// measured. Weekly product milestones move slowly, so a 4-week window gives a
// stable issues/day pace without over-reacting to a single quiet week.
const DefaultWindowDays = 28

// DefaultLedgerRel is the committed, append-only history ledger (one JSONL row
// per cadence tick) the burndown trend accrues into — a sibling of the milestone
// climb/roadmap ledger, rooted at the same docs/milestones/ tree.
const DefaultLedgerRel = "docs/milestones/burndown.jsonl"

// soonDays is the "due imminently" threshold used only on the degraded no-velocity
// path: without a pace signal we cannot project, so a milestone with open work due
// within this many days is flagged AT_RISK out of caution (the note says why).
const soonDays = 7

// Status is the closed at-risk vocabulary. It is a schedule verdict, not a quality
// judgement: OVERDUE/AT_RISK/NO_DUE_DATE are MEASURED facts about the milestone's
// due date and pace, never a claim that the work is wrong.
const (
	StatusOnTrack   = "ON_TRACK"
	StatusAtRisk    = "AT_RISK"
	StatusOverdue   = "OVERDUE"
	StatusNoDueDate = "NO_DUE_DATE"
	StatusDone      = "DONE"
)

// Findings the fold stamps onto the envelope.
const (
	findingRecorded   = "burndown_recorded"
	findingUnmeasured = "burndown_unmeasured"
	findingAdvisory   = "burndown_advisory"
)

// Milestone is one GitHub milestone's raw schedule + progress signal handed to the
// pure fold. DueOn is the milestone's own due_on (RFC3339, e.g.
// "2026-07-13T00:00:00Z") or "" when the milestone has no due date. ClosedInWindow
// is the count of issues closed within the trailing WindowDays; a negative value
// means velocity could not be measured and the fold degrades honestly.
type Milestone struct {
	Number         int
	Title          string
	URL            string
	DueOn          string
	Open           int
	Closed         int
	ClosedInWindow int
	WindowDays     int
}

// Row is one milestone's folded schedule verdict.
type Row struct {
	Number       int     `json:"number"`
	Title        string  `json:"title"`
	URL          string  `json:"url,omitempty"`
	Open         int     `json:"open"`
	Closed       int     `json:"closed"`
	Total        int     `json:"total"`
	Pct          float64 `json:"pct"`
	DueOn        string  `json:"due_on,omitempty"` // "YYYY-MM-DD" or ""
	HasDue       bool    `json:"has_due"`
	DaysToDue    int     `json:"days_to_due"` // signed; negative = past due; 0 when !HasDue
	RequiredRate float64 `json:"required_rate"`
	ActualRate   float64 `json:"actual_rate"`
	HasVelocity  bool    `json:"has_velocity"`
	ProjDays     int     `json:"proj_days"`           // -1 when a projection is impossible
	ProjDate     string  `json:"proj_date,omitempty"` // "YYYY-MM-DD" or ""
	Status       string  `json:"status"`
	Note         string  `json:"note"`
}

// Portfolio is the folded view over every milestone: the per-milestone rows plus
// the counts and the single at-risk-debt integer the trend ledger tracks.
type Portfolio struct {
	Rows       []Row  `json:"rows"`
	Total      int    `json:"total"`
	OnTrack    int    `json:"on_track"`
	AtRisk     int    `json:"at_risk"`
	Overdue    int    `json:"overdue"`
	NoDueDate  int    `json:"no_due_date"`
	Done       int    `json:"done"`
	OpenTotal  int    `json:"open_total"`
	WindowDays int    `json:"window_days"`
	AtRiskDebt int    `json:"at_risk_debt"`
	Err        string `json:"err,omitempty"`
	OK         bool   `json:"ok"`
}

// Report is the embeddable control-pane envelope plus the folded portfolio and an
// optional week-over-week trend.
type Report struct {
	trendreport.Envelope
	Portfolio Portfolio `json:"portfolio"`
	Trend     *Trend    `json:"trend,omitempty"`
}

// FoldOpts carries the ambient stamp the fold writes onto the envelope.
type FoldOpts struct {
	Workspace   string
	Commit      string
	GeneratedAt string
	Date        string
}

// Interpret folds the raw milestones into a Portfolio as of `now`. It is pure: no
// process, no network, no clock read — `now` and the milestone data are the only
// inputs, so every verdict is reproducible from a fixture. `readErr` is non-empty
// only when the milestone list itself could not be read; that is the one condition
// that makes the portfolio unmeasured (Err set, OK false).
func Interpret(ms []Milestone, windowDays int, now time.Time, readErr string) Portfolio {
	if windowDays <= 0 {
		windowDays = DefaultWindowDays
	}
	p := Portfolio{WindowDays: windowDays, OK: readErr == ""}
	if readErr != "" {
		p.Err = readErr
		return p
	}
	rows := make([]Row, 0, len(ms))
	for _, m := range ms {
		rows = append(rows, interpretOne(m, windowDays, now))
	}
	sort.SliceStable(rows, func(i, j int) bool { return lessSeverity(rows[i], rows[j]) })
	p.Rows = rows
	p.Total = len(rows)
	for _, r := range rows {
		p.OpenTotal += r.Open
		switch r.Status {
		case StatusOverdue:
			p.Overdue++
		case StatusAtRisk:
			p.AtRisk++
		case StatusNoDueDate:
			p.NoDueDate++
		case StatusOnTrack:
			p.OnTrack++
		case StatusDone:
			p.Done++
		}
	}
	// One integer for the trend ledger: overdue is worst, then at-risk, then the
	// milder no-due-date hygiene gap. A lower number is a healthier portfolio.
	p.AtRiskDebt = 3*p.Overdue + 2*p.AtRisk + 1*p.NoDueDate
	return p
}

// interpretOne classifies a single milestone against `now`.
func interpretOne(m Milestone, windowDays int, now time.Time) Row {
	total := m.Open + m.Closed
	r := Row{
		Number:   m.Number,
		Title:    m.Title,
		URL:      m.URL,
		Open:     m.Open,
		Closed:   m.Closed,
		Total:    total,
		Pct:      pct(m.Closed, total),
		ProjDays: -1,
	}

	// Velocity: a non-negative ClosedInWindow over a positive window is a pace.
	if m.ClosedInWindow >= 0 && windowDays > 0 {
		r.HasVelocity = true
		r.ActualRate = round2(float64(m.ClosedInWindow) / float64(windowDays))
	}

	// DONE short-circuits every date question: no open work left to schedule.
	if m.Open == 0 {
		r.Status = StatusDone
		r.Note = "all issues closed"
		return r
	}

	due, hasDue := parseDue(m.DueOn)
	r.HasDue = hasDue
	if !hasDue {
		r.Status = StatusNoDueDate
		r.Note = fmt.Sprintf("%d open, no due date set", m.Open)
		return r
	}
	r.DueOn = due.Format("2006-01-02")
	remaining := due.Sub(now)
	r.DaysToDue = daysCeil(remaining)

	// Past its own due date with open work: OVERDUE, full stop — a measured fact.
	if remaining <= 0 {
		r.Status = StatusOverdue
		r.Note = fmt.Sprintf("%d open, due %s (%d days ago)", m.Open, r.DueOn, -r.DaysToDue)
		return r
	}

	// Future due date: required pace to finish in time.
	r.RequiredRate = round2(float64(m.Open) / float64(mathx.MaxInt(r.DaysToDue, 1)))

	if !r.HasVelocity {
		// Degraded path (no pace signal): assert only urgency. Live collection
		// always supplies velocity; this keeps the fold honest when it does not.
		if r.DaysToDue <= soonDays {
			r.Status = StatusAtRisk
			r.Note = fmt.Sprintf("%d open, due in %d days; velocity unmeasured", m.Open, r.DaysToDue)
		} else {
			r.Status = StatusOnTrack
			r.Note = fmt.Sprintf("%d open, due in %d days; velocity unmeasured", m.Open, r.DaysToDue)
		}
		return r
	}

	if r.ActualRate <= 0 {
		r.Status = StatusAtRisk
		r.Note = fmt.Sprintf("%d open, due in %d days; no closures in %dd", m.Open, r.DaysToDue, windowDays)
		return r
	}

	// Project the drain date at the recent pace and compare to the due date.
	r.ProjDays = int(math.Ceil(float64(m.Open) / r.ActualRate))
	projDate := now.Add(time.Duration(r.ProjDays) * 24 * time.Hour)
	r.ProjDate = projDate.Format("2006-01-02")
	if projDate.After(due) {
		r.Status = StatusAtRisk
		r.Note = fmt.Sprintf("%d open at %.2f/day finishes %s, past due %s", m.Open, r.ActualRate, r.ProjDate, r.DueOn)
	} else {
		r.Status = StatusOnTrack
		r.Note = fmt.Sprintf("%d open at %.2f/day finishes %s, by due %s", m.Open, r.ActualRate, r.ProjDate, r.DueOn)
	}
	return r
}

// Fold stamps the envelope over the folded portfolio. Verdict is ACTION/unmeasured
// only when the milestones could not be read; otherwise OK/recorded. The reason
// summarises the portfolio and names the single most urgent milestone so the
// operator's next action is one line, not a scan.
func Fold(p Portfolio, opts FoldOpts) Report {
	env := trendreport.Stamp(Schema, trendreport.Opts{
		Workspace:   opts.Workspace,
		Commit:      opts.Commit,
		GeneratedAt: opts.GeneratedAt,
		Date:        opts.Date,
	})
	rep := Report{Portfolio: p}
	if !p.OK {
		env.OK = false
		env.Verdict = trendreport.VerdictAction
		env.Finding = findingUnmeasured
		env.Reason = "milestones could not be read: " + p.Err
		env.NextAction = "check `gh` auth / repo context, then re-run"
		rep.Envelope = env
		return rep
	}
	env.OK = true
	env.Verdict = trendreport.VerdictOK
	env.Finding = findingRecorded
	env.Reason = fmt.Sprintf("%d milestones: %d overdue, %d at-risk, %d no-due-date, %d on-track, %d done",
		p.Total, p.Overdue, p.AtRisk, p.NoDueDate, p.OnTrack, p.Done)
	env.NextAction = nextAction(p)
	rep.Envelope = env
	return rep
}

// nextAction names the most urgent milestone (rows are already severity-sorted).
func nextAction(p Portfolio) string {
	for _, r := range p.Rows {
		switch r.Status {
		case StatusOverdue:
			return fmt.Sprintf("#%d %q is OVERDUE — %s", r.Number, r.Title, r.Note)
		case StatusAtRisk:
			return fmt.Sprintf("#%d %q is AT_RISK — %s", r.Number, r.Title, r.Note)
		}
	}
	if p.NoDueDate > 0 {
		for _, r := range p.Rows {
			if r.Status == StatusNoDueDate {
				return fmt.Sprintf("set a due date on #%d %q (%d open, untracked)", r.Number, r.Title, r.Open)
			}
		}
	}
	return "hold the line — every dated milestone is on track"
}

// CheckGate is the advisory CI gate: it fails ONLY when the portfolio is unmeasured
// (the milestone list could not be read). An OVERDUE milestone is a MEASURED fact,
// not an incomplete report, so it passes — the report mirrors schedule truth.
func CheckGate(rep Report) trendreport.GateVerdict {
	return trendreport.AdvisoryGate("BURNDOWN", rep.Finding, rep.Reason, findingUnmeasured)
}

// WithGate reconciles the report envelope to a gate decision for the --check --json
// envelope.
func (rep Report) WithGate(v trendreport.GateVerdict) Report {
	rep.Envelope = rep.Envelope.WithGate(v)
	return rep
}

// severityRank orders the status vocabulary worst-first for rendering and for the
// most-urgent pick.
func severityRank(status string) int {
	switch status {
	case StatusOverdue:
		return 0
	case StatusAtRisk:
		return 1
	case StatusNoDueDate:
		return 2
	case StatusOnTrack:
		return 3
	case StatusDone:
		return 4
	default:
		return 5
	}
}

// lessSeverity sorts worst-first, then soonest-due, then by number for stability.
func lessSeverity(a, b Row) bool {
	ra, rb := severityRank(a.Status), severityRank(b.Status)
	if ra != rb {
		return ra < rb
	}
	// Within a band, a sooner due date is more urgent. Rows without a due date
	// sort after dated ones.
	switch {
	case a.HasDue && b.HasDue && a.DueOn != b.DueOn:
		return a.DueOn < b.DueOn
	case a.HasDue != b.HasDue:
		return a.HasDue
	}
	return a.Number < b.Number
}

// Render is the human-readable snapshot: a summary line, the most-urgent action,
// then one line per milestone worst-first.
func Render(rep Report) string {
	var b strings.Builder
	p := rep.Portfolio
	fmt.Fprintf(&b, "MILESTONE BURNDOWN — %s\n", rep.Reason)
	if !p.OK {
		fmt.Fprintf(&b, "  UNMEASURED: %s\n", p.Err)
		return b.String()
	}
	fmt.Fprintf(&b, "  next: %s\n", rep.NextAction)
	fmt.Fprintf(&b, "  velocity window: %dd | open across all: %d | at-risk debt: %d\n", p.WindowDays, p.OpenTotal, p.AtRiskDebt)
	if rep.Trend != nil {
		fmt.Fprintf(&b, "  trend: %s — %s\n", rep.Trend.Direction, rep.Trend.Summary)
	}
	b.WriteString("\n")
	for _, r := range p.Rows {
		due := r.DueOn
		if due == "" {
			due = "—"
		}
		fmt.Fprintf(&b, "  [%-11s] #%-4d %-44s due %-10s %d/%d closed  %s\n",
			r.Status, r.Number, truncate(r.Title, 44), due, r.Closed, r.Total, r.Note)
	}
	return b.String()
}

// ---- ledger + trend -------------------------------------------------------

// LedgerRow is the durable JSONL trend row: the flattened portfolio counts keyed by
// date, with generated_at the same-day idempotency key.
type LedgerRow struct {
	Schema      string `json:"schema"`
	Date        string `json:"date"`
	Commit      string `json:"commit"`
	GeneratedAt string `json:"generated_at"`
	Verdict     string `json:"verdict"`
	Total       int    `json:"total"`
	Overdue     int    `json:"overdue"`
	AtRisk      int    `json:"at_risk"`
	NoDueDate   int    `json:"no_due_date"`
	OnTrack     int    `json:"on_track"`
	Done        int    `json:"done"`
	OpenTotal   int    `json:"open_total"`
	AtRiskDebt  int    `json:"at_risk_debt"`
}

// RowFromReport projects a report into its durable ledger row.
func RowFromReport(rep Report) LedgerRow {
	p := rep.Portfolio
	return LedgerRow{
		Schema:      LedgerSchema,
		Date:        rep.Date,
		Commit:      rep.Commit,
		GeneratedAt: rep.GeneratedAt,
		Verdict:     rep.Verdict,
		Total:       p.Total,
		Overdue:     p.Overdue,
		AtRisk:      p.AtRisk,
		NoDueDate:   p.NoDueDate,
		OnTrack:     p.OnTrack,
		Done:        p.Done,
		OpenTotal:   p.OpenTotal,
		AtRiskDebt:  p.AtRiskDebt,
	}
}

// ParseLedger parses an append-only JSONL ledger, tolerating blank lines and
// skipping any line that is not a valid row (so a hand-edit can't crash the
// reader). Rows are returned in file order. Mirrors milestonereport.ParseLedger.
func ParseLedger(content string) []LedgerRow {
	return jsonlledger.Parse(content, func(r LedgerRow) bool { return r.Date != "" })
}

// AppendLedgerLine renders the JSONL line for a row (no trailing newline); the
// caller appends the newline. Keeping the rendering pure makes the writer testable
// without touching disk.
func AppendLedgerLine(row LedgerRow) (string, error) {
	return trendreport.AppendLedgerLine(row)
}

// Trend is the week-over-week direction, driven by the at-risk debt integer (a
// falling debt is an improvement; a rising one a regression).
type Trend struct {
	Direction    string `json:"direction"` // improved | regressed | flat | new
	PrevDate     string `json:"prev_date,omitempty"`
	PrevCommit   string `json:"prev_commit,omitempty"`
	DebtFrom     int    `json:"debt_from"`
	DebtTo       int    `json:"debt_to"`
	DebtDelta    int    `json:"debt_delta"`
	OverdueFrom  int    `json:"overdue_from"`
	OverdueTo    int    `json:"overdue_to"`
	OverdueDelta int    `json:"overdue_delta"`
	AtRiskFrom   int    `json:"at_risk_from"`
	AtRiskTo     int    `json:"at_risk_to"`
	Summary      string `json:"summary"`
}

// TrendVsLast computes the per-tick trend of `row` against the most recent prior
// row. With no prior row the trend is "new". Mirrors milestonereport.TrendVsLast.
func TrendVsLast(row LedgerRow, prior []LedgerRow) Trend {
	last, ok := latestBefore(row, prior)
	if !ok {
		return Trend{
			Direction: "new",
			DebtTo:    row.AtRiskDebt,
			OverdueTo: row.Overdue,
			AtRiskTo:  row.AtRisk,
			Summary: fmt.Sprintf("first burndown tick (at-risk debt %d: %d overdue, %d at-risk, %d no-due-date)",
				row.AtRiskDebt, row.Overdue, row.AtRisk, row.NoDueDate),
		}
	}
	debtDelta := row.AtRiskDebt - last.AtRiskDebt
	dir := "flat"
	switch {
	case debtDelta < 0:
		dir = "improved"
	case debtDelta > 0:
		dir = "regressed"
	}
	return Trend{
		Direction:    dir,
		PrevDate:     last.Date,
		PrevCommit:   last.Commit,
		DebtFrom:     last.AtRiskDebt,
		DebtTo:       row.AtRiskDebt,
		DebtDelta:    debtDelta,
		OverdueFrom:  last.Overdue,
		OverdueTo:    row.Overdue,
		OverdueDelta: row.Overdue - last.Overdue,
		AtRiskFrom:   last.AtRisk,
		AtRiskTo:     row.AtRisk,
		Summary: fmt.Sprintf("at-risk debt %s %+d (%d->%d); overdue %+d, at-risk %+d vs %s",
			dir, debtDelta, last.AtRiskDebt, row.AtRiskDebt, row.Overdue-last.Overdue, row.AtRisk-last.AtRisk, last.Date),
	}
}

// latestBefore returns the most recent prior row, comparing by (date, then
// generated_at). A row with the exact same generated_at as `row` is excluded
// (idempotent re-append), mirroring milestonereport.
func latestBefore(row LedgerRow, prior []LedgerRow) (LedgerRow, bool) {
	return jsonlledger.LatestBefore(row, prior,
		func(r LedgerRow) string { return r.Date },
		func(r LedgerRow) string { return r.GeneratedAt })
}

// WithTrend attaches a trend and, on regression, downgrades the finding to advisory
// (never a gate flip — the advisory gate still only reds on unmeasured).
func (rep Report) WithTrend(t Trend) Report {
	rep.Trend = &t
	if t.Direction == "regressed" && rep.Finding == findingRecorded {
		rep.Finding = findingAdvisory
		rep.Reason = fmt.Sprintf("%s (at-risk debt up %d vs %s)", rep.Reason, t.DebtDelta, t.PrevDate)
	}
	return rep
}

// ---- small pure helpers ---------------------------------------------------

func parseDue(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, false
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.UTC(), true
	}
	// GitHub sometimes renders a date-only due_on; accept it as end-of-day UTC.
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return t.UTC(), true
	}
	return time.Time{}, false
}

// daysCeil renders a duration as whole days, rounding a positive remainder up so
// "23h left" reads as 1 day, and a past-due gap down (negative) symmetrically.
func daysCeil(d time.Duration) int {
	h := d.Hours() / 24
	if d >= 0 {
		return int(math.Ceil(h))
	}
	return int(math.Floor(h))
}

func pct(closed, total int) float64 {
	if total <= 0 {
		return 0
	}
	return round2(100 * float64(closed) / float64(total))
}

func round2(f float64) float64 { return math.Round(f*100) / 100 }

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}
