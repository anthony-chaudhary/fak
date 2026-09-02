package milestonereport

// milestonereport.go is the pure, unit-tested surface: the two dimension
// interpreters (Maturity = the climb, Epics = the roadmap), the fold, the
// durable-ledger parse/trend, render, and the advisory gate. The live runners
// (covmatrix.Grid + the `gh` shell + git HEAD) live in collect.go. The package
// doc is in doc.go.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/generation"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/reportledger"
	"github.com/anthony-chaudhary/fak/internal/supportmaturity"
	"github.com/anthony-chaudhary/fak/internal/trendreport"
	"github.com/anthony-chaudhary/fak/internal/worktype"
)

// Schema is the stable control-pane schema identifier for the report envelope.
const Schema = "fak-milestone-report/1"

// LedgerSchema tags each durable history row so a reader can validate the line.
const LedgerSchema = "fak-milestone-ledger/1"

// DefaultLedgerRel is the committed, append-only history ledger (one JSONL row
// per milestone tick). It lives under docs/ so it is durable trunk evidence, not
// a regenerable build artifact.
const DefaultLedgerRel = "docs/milestones/history.jsonl"

// maturedRung is the floor a cell must reach to count as MATURE support: M4Correct
// (correctness witnessed by a CI-runnable oracle). Cells below it are honest
// positions on the ladder but still support-maturity debt, matching
// supportmaturityscore's "a SUPPORTED cell is mature" bar.
const maturedRung = supportmaturity.M4Correct

// --- the CLIMB dimension ----------------------------------------------------

// Maturity is the CLIMB dimension: the model x backend grid's distribution across
// the closed M0–M7 support-maturity ladder. Every count is WITNESSED — lowered
// from the live covmatrix.Grid(), never self-reported. Err exists only for shape
// symmetry with the roadmap dimension; the grid is in-process and never fails to
// measure, so a populated grid always yields Err == "".
type Maturity struct {
	Cells       int            `json:"cells"`
	Dist        map[string]int `json:"dist"`         // "M0".."M7" -> count (every rung seeded)
	Matured     int            `json:"matured"`      // cells at M4Correct or above (the mature floor)
	Highest     string         `json:"highest"`      // highest rung any cell reached, e.g. "M5"
	HighestRank int            `json:"highest_rank"` // the rung's ladder index (for trend math)
	ProgressPct float64        `json:"progress_pct"` // 100 * mean(rank) / rank(M7), 0..100
	Worst       []string       `json:"worst,omitempty"`
	Err         string         `json:"err,omitempty"`
	OK          bool           `json:"ok"`
}

// InterpretMaturity lowers the live grid onto the ladder and folds the
// distribution. Each cell's rung is supportmaturity.FromSupport(cell.Support); the
// distribution seeds every M0–M7 key so an absent rung renders as 0, not a gap. An
// empty grid is an ERRORED dimension (we can witness nothing), never a silent zero.
func InterpretMaturity(cells []covmatrix.Cell) Maturity {
	m := Maturity{Dist: map[string]int{}}
	for _, r := range supportmaturity.Rungs {
		m.Dist[r.String()] = 0
	}
	if len(cells) == 0 {
		m.Err = "empty grid — no model x backend cells to measure"
		return m
	}
	m.Cells = len(cells)
	var rankSum int
	highest := supportmaturity.M0None
	type lowCell struct {
		name string
		rank supportmaturity.Rung
	}
	var lows []lowCell
	for _, c := range cells {
		r := supportmaturity.FromSupport(c.Support)
		m.Dist[r.String()]++
		rankSum += int(r)
		if highest.Less(r) {
			highest = r
		}
		if r.Less(maturedRung) {
			lows = append(lows, lowCell{name: c.Family + " x " + c.Backend, rank: r})
		} else {
			m.Matured++
		}
	}
	m.Highest = highest.String()
	m.HighestRank = int(highest)
	m.ProgressPct = round1(100 * float64(rankSum) / float64(len(cells)*int(supportmaturity.M7BeyondSOTA)))

	// The render-only worst list: the lowest-rung cells, lowest first then name, so an
	// operator sees what is dragging the climb. Capped so the card stays scannable.
	sort.SliceStable(lows, func(i, j int) bool {
		if lows[i].rank != lows[j].rank {
			return lows[i].rank < lows[j].rank
		}
		return lows[i].name < lows[j].name
	})
	const worstCap = 5
	for i, lc := range lows {
		if i >= worstCap {
			break
		}
		m.Worst = append(m.Worst, fmt.Sprintf("%s (%s %s)", lc.name, lc.rank.String(), lc.rank.Label()))
	}
	m.OK = true
	return m
}

// --- the ROADMAP dimension --------------------------------------------------

// EpicCounts is the raw child tally the `gh` shell hands the interpreter for one
// epic: how many children, how many closed, by which source — or an Err when no
// child signal could be witnessed. A failed read MUST set Err, never Total 0; the
// interpreter relies on that to tell "0 of 4 done" from "could not read". Source is
// the provenance label ("label" | "checklist" | "checklist+issue-state") so the fold
// can report HOW the number was witnessed (the conflation-honesty contract) — in
// particular whether a checklist count was cross-checked against the referenced
// children's real state or read off the hand-ticked boxes alone.
type EpicCounts struct {
	Number int
	Closed int
	Total  int
	Source string
	Err    string
}

// EpicRow is one tracked epic's completion in the folded report. Class is its
// work-class (worktype.ClassifyEpic): a DISCRETE epic's Pct is a meaningful "how
// far to done", while an ONGOING program's child tally is a frontier-activity
// signal — the report renders it WITHOUT a completion-% framing and excludes it
// from the roadmap's OverallPct, because an ongoing program has no 100%.
type EpicRow struct {
	Number     int            `json:"number"`
	Title      string         `json:"title"`
	Generation string         `json:"generation"`
	Class      worktype.Class `json:"class"`
	Closed     int            `json:"closed"`
	Total      int            `json:"total"`
	Pct        float64        `json:"pct"`
	Source     string         `json:"source,omitempty"`
	Err        string         `json:"err,omitempty"`
}

// Ongoing reports whether this row is an ongoing optimization program rather than a
// discrete deliverable — the one predicate the render branches on.
func (r EpicRow) Ongoing() bool { return r.Class.Ongoing() }

// Epics is the ROADMAP dimension: completion across the tracked epics, split by
// work class. The completion math (Closed/Total/OverallPct) folds DISCRETE epics
// ONLY — an ongoing optimization program has no 100%, so blending its child tally
// into a "roadmap % complete" would manufacture a false stall. The program rows are
// still measured and surfaced (Programs); they just trend as frontier activity, not
// as progress toward done. Err is set ONLY when EVERY tracked epic failed to read (a
// true unmeasured dimension that gates); a partial failure records a non-gating
// PartialNote and leaves Err == "", so one flaky `gh` query never reds the report.
type Epics struct {
	Rows        []EpicRow       `json:"rows"`
	Tracked     int             `json:"tracked"`
	Measured    int             `json:"measured"`
	Programs    int             `json:"programs"`    // count of ongoing-program rows (any read state)
	Discrete    int             `json:"discrete"`    // count of discrete-epic rows (any read state)
	Closed      int             `json:"closed"`      // summed over measured DISCRETE rows only
	Total       int             `json:"total"`       // summed over measured DISCRETE rows only
	OverallPct  float64         `json:"overall_pct"` // 100 * Closed/Total over measured DISCRETE rows
	Generations []GenerationRow `json:"generations"`
	PartialNote string          `json:"partial_note,omitempty"`
	Err         string          `json:"err,omitempty"`
	OK          bool            `json:"ok"`
}

// GenerationRow is the roadmap summary by product horizon. It mirrors the
// discrete-vs-ongoing honesty rule: OverallPct folds measured DISCRETE rows only,
// while ongoing programs are counted as frontier activity, not completion.
type GenerationRow struct {
	Generation          string  `json:"generation"`
	Tracked             int     `json:"tracked"`
	Measured            int     `json:"measured"`
	Programs            int     `json:"programs"`
	Discrete            int     `json:"discrete"`
	Closed              int     `json:"closed"`
	Total               int     `json:"total"`
	OverallPct          float64 `json:"overall_pct"`
	Errored             int     `json:"errored,omitempty"`
	DebtScore           int     `json:"debt_score"`
	StaleIssues         int     `json:"stale_issues,omitempty"`
	MissingWitnesses    int     `json:"missing_witnesses,omitempty"`
	UnpromotedBets      int     `json:"unpromoted_bets,omitempty"`
	LabelShipMismatches int     `json:"label_ship_mismatches,omitempty"`
	DebtReason          string  `json:"debt_reason,omitempty"`
}

// InterpretEpics folds the per-epic child tallies into the roadmap dimension. A
// whole-command runErr (the `gh` binary itself failed) errors every row. Each row's
// own Err excludes it from the overall pct. The dimension errors only when no epic
// could be measured.
func InterpretEpics(specs []EpicSpec, counts []EpicCounts, runErr string) Epics {
	e := Epics{Tracked: len(specs)}
	byNum := map[int]EpicCounts{}
	for _, c := range counts {
		byNum[c.Number] = c
	}
	var failed int
	for _, spec := range specs {
		// Classify by number via the canonical worktype map. The resolver's EpicSpec
		// is intentionally taxonomy-free (it only knows how to READ children); the
		// work-class lives in one table (internal/worktype) so a reclassification is a
		// one-line edit there, not a field threaded through the resolver.
		class := worktype.ClassifyEpic(spec.Number)
		row := EpicRow{Number: spec.Number, Title: spec.Title, Generation: normalizeGeneration(spec.Generation), Class: class}
		if class.Ongoing() {
			e.Programs++
		} else {
			e.Discrete++
		}
		c, ok := byNum[spec.Number]
		switch {
		case runErr != "":
			row.Err = runErr
		case !ok:
			row.Err = "no count returned"
		case c.Err != "":
			row.Err = c.Err
		default:
			row.Closed, row.Total, row.Source = c.Closed, c.Total, c.Source
			row.Pct = pct(c.Closed, c.Total)
			e.Measured++
			// Only DISCRETE epics fold into the roadmap completion %. An ongoing
			// program's child tally is read + surfaced but never blended into a
			// "% complete" — it has no 100%.
			if !row.Ongoing() {
				e.Closed += c.Closed
				e.Total += c.Total
			}
		}
		if row.Err != "" {
			failed++
		}
		e.Rows = append(e.Rows, row)
	}
	e.OverallPct = pct(e.Closed, e.Total)
	e.Generations = summarizeGenerations(e.Rows)
	switch {
	case e.Tracked == 0:
		e.Err = "no epics tracked"
	case e.Measured == 0:
		if runErr != "" {
			e.Err = runErr
		} else {
			e.Err = "no tracked epic could be read (no child signal via gh)"
		}
	case failed > 0:
		e.PartialNote = fmt.Sprintf("%d of %d tracked epic(s) had no readable child signal", failed, e.Tracked)
	}
	e.OK = e.Err == ""
	return e
}

var generationOrder = append(generation.Order(), "unclassified")

func normalizeGeneration(g string) string { return generation.Normalize(g) }

func summarizeGenerations(rows []EpicRow) []GenerationRow {
	by := make(map[string]*GenerationRow, len(generationOrder))
	for _, gen := range generationOrder {
		by[gen] = &GenerationRow{Generation: gen}
	}
	for _, row := range rows {
		gen := normalizeGeneration(row.Generation)
		lane, ok := by[gen]
		if !ok {
			lane = &GenerationRow{Generation: gen}
			by[gen] = lane
		}
		lane.Tracked++
		if row.Ongoing() {
			lane.Programs++
		} else {
			lane.Discrete++
		}
		if row.Err != "" {
			lane.Errored++
			lane.MissingWitnesses++
			continue
		}
		lane.Measured++
		if !row.Ongoing() {
			lane.Closed += row.Closed
			lane.Total += row.Total
			open := row.Total - row.Closed
			if open > 0 {
				lane.StaleIssues += open
				if gen != "now" && gen != "unclassified" {
					lane.UnpromotedBets += open
				}
			}
		} else if gen != "now" && gen != "unclassified" {
			lane.UnpromotedBets++
		}
		if gen == "unclassified" {
			lane.LabelShipMismatches++
		}
	}
	out := make([]GenerationRow, 0, len(by))
	seen := map[string]bool{}
	for _, gen := range generationOrder {
		lane := by[gen]
		lane.OverallPct = pct(lane.Closed, lane.Total)
		finalizeGenerationDebt(lane)
		out = append(out, *lane)
		seen[gen] = true
	}
	var extras []string
	for gen := range by {
		if !seen[gen] {
			extras = append(extras, gen)
		}
	}
	sort.Strings(extras)
	for _, gen := range extras {
		lane := by[gen]
		lane.OverallPct = pct(lane.Closed, lane.Total)
		finalizeGenerationDebt(lane)
		out = append(out, *lane)
	}
	return out
}

func finalizeGenerationDebt(row *GenerationRow) {
	if row == nil {
		return
	}
	row.DebtScore = row.StaleIssues + 3*row.MissingWitnesses + 2*row.UnpromotedBets + 2*row.LabelShipMismatches
	var reasons []string
	if row.StaleIssues > 0 {
		reasons = append(reasons, fmt.Sprintf("%d stale-risk issue(s)", row.StaleIssues))
	}
	if row.MissingWitnesses > 0 {
		reasons = append(reasons, fmt.Sprintf("%d missing witness(es)", row.MissingWitnesses))
	}
	if row.UnpromotedBets > 0 {
		reasons = append(reasons, fmt.Sprintf("%d unpromoted bet(s)", row.UnpromotedBets))
	}
	if row.LabelShipMismatches > 0 {
		reasons = append(reasons, fmt.Sprintf("%d label/ship mismatch(es)", row.LabelShipMismatches))
	}
	row.DebtReason = strings.Join(reasons, ", ")
}

// --- the fold ---------------------------------------------------------------

// Trend is the per-tick delta vs the previous ledger row.
type Trend struct {
	PrevDate      string  `json:"prev_date"`
	PrevCommit    string  `json:"prev_commit"`
	Direction     string  `json:"direction"` // improved | regressed | flat | new
	MaturityFrom  float64 `json:"maturity_from"`
	MaturityTo    float64 `json:"maturity_to"`
	MaturityDelta float64 `json:"maturity_delta"`
	HighestFrom   string  `json:"highest_from"`
	HighestTo     string  `json:"highest_to"`
	MaturedFrom   int     `json:"matured_from"`
	MaturedTo     int     `json:"matured_to"`
	MaturedDelta  int     `json:"matured_delta"`
	EpicPctFrom   float64 `json:"epic_pct_from"`
	EpicPctTo     float64 `json:"epic_pct_to"`
	EpicPctDelta  float64 `json:"epic_pct_delta"`
	Summary       string  `json:"summary"`
}

// Report is one folded milestone-report control-pane envelope. The common head
// (schema/ok/verdict/finding/reason/next_action, the ambient stamp, and the two
// --check --json gate fields) is the embedded trendreport.Envelope; only the
// milestone dimensions live here.
type Report struct {
	trendreport.Envelope
	Maturity          Maturity           `json:"maturity"`
	Epics             Epics              `json:"epics"`
	ProgramScorecards []ProgramScorecard `json:"program_scorecards,omitempty"`
	Trend             *Trend             `json:"trend,omitempty"`
	Velocity          *Velocity          `json:"velocity,omitempty"`
}

// ProgramScorecard is a neutral milestone projection of a program-specific
// scorecard. The producer owns the grading logic; milestone report only carries
// and renders the witnessed rows under the matching milestone.
type ProgramScorecard struct {
	Key       string             `json:"key"`
	Milestone int                `json:"milestone"`
	Title     string             `json:"title"`
	Verdict   string             `json:"verdict"` // lovable | not-yet
	Witnessed int                `json:"witnessed"`
	Total     int                `json:"total"`
	Criteria  []ProgramCriterion `json:"criteria"`
}

// ProgramCriterion is one witness-linked row in a program scorecard projection.
type ProgramCriterion struct {
	Workstream string `json:"workstream"`
	Title      string `json:"title"`
	Grade      string `json:"grade"`
	WitnessRef string `json:"witness_ref"`
}

// FoldOpts carries the ambient context the fold stamps onto the envelope. It is
// the shared trendreport.Opts; the alias keeps this package's fold signature
// stable for existing callers.
type FoldOpts = trendreport.Opts

// Fold folds the two dimensions into one milestone-report envelope. The verdict
// ladder mirrors cadencereport's REPORT contract, not a second quality gate: it is
// ACTION only when a dimension could not be MEASURED (the roadmap's `gh` read failed
// for every epic), and OK otherwise — surfacing a regressed climb or a stalled epic
// as an advisory line in the reason. The maturity dimension is pure and never the
// unmeasured one.
func Fold(m Maturity, e Epics, opts FoldOpts) Report {
	r := Report{
		Envelope: trendreport.Stamp(Schema, opts),
		Maturity: m,
		Epics:    e,
	}

	var unmeasured []string
	if m.Err != "" {
		unmeasured = append(unmeasured, "maturity ("+m.Err+")")
	}
	if e.Err != "" {
		unmeasured = append(unmeasured, "roadmap ("+e.Err+")")
	}

	climbLine := fmt.Sprintf("climb: %s, %d/%d matured (progress %.1f%%)", m.Highest, m.Matured, m.Cells, m.ProgressPct)
	// The roadmap % is over DISCRETE epics only; programs are reported alongside as
	// ongoing-frontier activity, never folded into a completion bar.
	roadLine := fmt.Sprintf("roadmap: %.1f%% over %d discrete epic(s); %d ongoing program(s)", e.OverallPct, e.Discrete, e.Programs)
	summary := climbLine + "; " + roadLine
	if e.PartialNote != "" {
		summary += " — " + e.PartialNote
	}

	switch {
	case len(unmeasured) > 0:
		r.OK, r.Verdict, r.Finding = false, "ACTION", "milestone_unmeasured"
		r.Reason = "milestone report incomplete — could not measure " + strings.Join(unmeasured, ", ")
		r.NextAction = "repair the failing dimension (check `gh auth status` / the tracked-epic child signal), then re-run `fak milestone report`"
	default:
		r.OK, r.Verdict, r.Finding = true, "OK", "milestone_recorded"
		r.Reason = "milestone recorded; " + summary
		r.NextAction = "hold the line; the scheduled milestone tick keeps the climb + roadmap trended"
	}
	return r
}

// withAdvisory rewrites a recorded report's finding/reason when the per-tick trend
// shows a regression — an advisory line, never a gate flip (the climb/roadmap can
// dip for honest reasons; the scorecard ratchet owns the real debt gate). It is
// applied after the trend is attached so the fold itself stays trend-free + pure.
func (r Report) withAdvisory() Report {
	if r.Finding != "milestone_recorded" || r.Trend == nil || r.Trend.Direction != "regressed" {
		return r
	}
	r.Finding = "milestone_advisory"
	r.Reason += " (advisory: " + r.Trend.Summary + ")"
	r.NextAction = "investigate the regression — a cell dropped a rung or an epic lost closed children; the tick keeps recording the trend"
	return r
}

// WithTrend attaches a per-tick trend and applies the advisory rewrite, returning
// the reconciled report. It is the one place the trend touches the verdict, keeping
// Fold pure of trend state (the cadencereport split).
func (r Report) WithTrend(t Trend) Report {
	r.Trend = &t
	return r.withAdvisory()
}

// WithProgramScorecard attaches or replaces a program card by milestone+key.
// Replacement keeps repeated collection idempotent while preserving stable order.
func (r Report) WithProgramScorecard(card ProgramScorecard) Report {
	for i := range r.ProgramScorecards {
		if r.ProgramScorecards[i].Milestone == card.Milestone && r.ProgramScorecards[i].Key == card.Key {
			r.ProgramScorecards[i] = card
			return r
		}
	}
	r.ProgramScorecards = append(r.ProgramScorecards, card)
	return r
}

// --- the durable history ledger ---------------------------------------------

// LedgerRow is one durable, append-only history line (a flattened projection of the
// two dimensions, so the ledger is a self-describing time series).
type LedgerRow struct {
	Schema           string  `json:"schema"`
	Date             string  `json:"date"`
	Commit           string  `json:"commit"`
	GeneratedAt      string  `json:"generated_at"`
	Verdict          string  `json:"verdict"`
	Cells            int     `json:"cells"`
	MaturityProgress float64 `json:"maturity_progress"`
	MaturityHighest  string  `json:"maturity_highest"`
	Matured          int     `json:"matured"`
	M0               int     `json:"m0"`
	M1               int     `json:"m1"`
	M2               int     `json:"m2"`
	M3               int     `json:"m3"`
	M4               int     `json:"m4"`
	M5               int     `json:"m5"`
	M6               int     `json:"m6"`
	M7               int     `json:"m7"`
	EpicsTracked     int     `json:"epics_tracked"`
	EpicsMeasured    int     `json:"epics_measured"`
	EpicOverallPct   float64 `json:"epic_overall_pct"`
}

// LedgerDate implements reportledger.Dated: the tick date the trend orders by.
func (r LedgerRow) LedgerDate() string { return r.Date }

// LedgerGeneratedAt implements reportledger.Dated: the same-run stamp that
// breaks same-day ties and excludes a row's own prior generation.
func (r LedgerRow) LedgerGeneratedAt() string { return r.GeneratedAt }

// RowFromReport projects a folded report into one durable ledger row.
func RowFromReport(r Report) LedgerRow {
	d := r.Maturity.Dist
	return LedgerRow{
		Schema:           LedgerSchema,
		Date:             r.Date,
		Commit:           r.Commit,
		GeneratedAt:      r.GeneratedAt,
		Verdict:          r.Verdict,
		Cells:            r.Maturity.Cells,
		MaturityProgress: r.Maturity.ProgressPct,
		MaturityHighest:  r.Maturity.Highest,
		Matured:          r.Maturity.Matured,
		M0:               d["M0"], M1: d["M1"], M2: d["M2"], M3: d["M3"],
		M4: d["M4"], M5: d["M5"], M6: d["M6"], M7: d["M7"],
		EpicsTracked:   r.Epics.Tracked,
		EpicsMeasured:  r.Epics.Measured,
		EpicOverallPct: r.Epics.OverallPct,
	}
}

// ParseLedger parses an append-only JSONL ledger, tolerating blank lines and
// skipping any line that is not a valid row (so a hand-edit can't crash the reader).
// Rows are returned in file order.
func ParseLedger(content string) []LedgerRow {
	return jsonlledger.Parse(content, func(r LedgerRow) bool { return r.Date != "" })
}

// AppendLedgerLine renders the JSONL line for a row (no trailing newline). The
// caller appends it with a newline; keeping the rendering pure makes the writer
// testable without touching disk.
func AppendLedgerLine(row LedgerRow) (string, error) {
	return trendreport.AppendLedgerLine(row)
}

// TrendVsLast computes the per-tick trend of `row` against the most recent prior
// row. The direction is driven by the maturity centroid (progress %), then by the
// roadmap pct — a climb up OR a roadmap gain is "improved"; a rung drop or lost
// children is "regressed". With no prior row the trend is "new".
func TrendVsLast(row LedgerRow, prior []LedgerRow) Trend {
	last, ok := reportledger.LatestBefore(row, prior)
	if !ok {
		return Trend{
			Direction:  "new",
			MaturityTo: row.MaturityProgress,
			HighestTo:  row.MaturityHighest,
			MaturedTo:  row.Matured,
			EpicPctTo:  row.EpicOverallPct,
			Summary:    fmt.Sprintf("first milestone tick (climb %s, progress %.1f%%, roadmap %.1f%%)", row.MaturityHighest, row.MaturityProgress, row.EpicOverallPct),
		}
	}
	matDelta := round1(row.MaturityProgress - last.MaturityProgress)
	epicDelta := round1(row.EpicOverallPct - last.EpicOverallPct)
	maturedDelta := row.Matured - last.Matured
	dir := "flat"
	switch {
	case matDelta > 0 || (matDelta == 0 && epicDelta > 0):
		dir = "improved"
	case matDelta < 0 || epicDelta < 0:
		dir = "regressed"
	}
	return Trend{
		PrevDate:      last.Date,
		PrevCommit:    last.Commit,
		Direction:     dir,
		MaturityFrom:  last.MaturityProgress,
		MaturityTo:    row.MaturityProgress,
		MaturityDelta: matDelta,
		HighestFrom:   last.MaturityHighest,
		HighestTo:     row.MaturityHighest,
		MaturedFrom:   last.Matured,
		MaturedTo:     row.Matured,
		MaturedDelta:  maturedDelta,
		EpicPctFrom:   last.EpicOverallPct,
		EpicPctTo:     row.EpicOverallPct,
		EpicPctDelta:  epicDelta,
		Summary: fmt.Sprintf("climb %s progress %+.1f%% (%.1f->%.1f), %+d matured; roadmap %+.1f%% (%.1f->%.1f) vs %s",
			dir, matDelta, last.MaturityProgress, row.MaturityProgress, maturedDelta,
			epicDelta, last.EpicOverallPct, row.EpicOverallPct, last.Date),
	}
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
		fmt.Sprintf("milestone report — %s (%s)  @%s  %s", r.Verdict, r.Finding, r.Commit, r.Date),
		"",
		fmt.Sprintf("  %s climb       %s: %d/%d cell(s) matured (M4+); progress %.1f%%",
			mark(r.Maturity.OK, r.Maturity.Err), dashIfEmpty(r.Maturity.Highest), r.Maturity.Matured, r.Maturity.Cells, r.Maturity.ProgressPct),
		"      ladder: " + RenderDist(r.Maturity.Dist),
	}
	if len(r.Maturity.Worst) > 0 {
		lines = append(lines, "      lowest: "+strings.Join(r.Maturity.Worst, ", "))
	}
	// The roadmap renders in two sections so an ongoing optimization PROGRAM is never
	// read as a stalled deliverable: discrete epics carry a completion %, programs
	// carry a frontier-activity line (closed/open children, no "% complete").
	var programs, discrete []EpicRow
	for _, row := range r.Epics.Rows {
		if row.Ongoing() {
			programs = append(programs, row)
		} else {
			discrete = append(discrete, row)
		}
	}
	lines = append(lines, fmt.Sprintf("  %s roadmap     %.1f%% across %d discrete epic(s); %d ongoing program(s)",
		mark(r.Epics.OK, r.Epics.Err), r.Epics.OverallPct, r.Epics.Discrete, r.Epics.Programs))
	if len(r.Epics.Generations) > 0 {
		lines = append(lines, "      generation lanes:")
		for _, row := range r.Epics.Generations {
			lines = append(lines, "        "+generationRowLine(row))
		}
	}
	if len(discrete) > 0 {
		lines = append(lines, "      discrete epics (-> done):")
		for _, row := range discrete {
			lines = append(lines, "        "+epicRowLine(row))
		}
	}
	if len(programs) > 0 {
		lines = append(lines, "      ongoing programs (frontier + trend, never 'done'):")
		for _, row := range programs {
			lines = append(lines, "        "+programRowLine(row))
		}
	}
	// The code-movement lens renders beside the roadmap: epic bars say what CLOSED,
	// module velocity says where the code actually MOVED (#2494). append(_, nil...) is
	// a no-op, so a report with no lens attached renders exactly as before.
	lines = append(lines, renderVelocity(r.Velocity)...)
	for _, card := range r.ProgramScorecards {
		mark := "."
		if card.Verdict == "lovable" {
			mark = "+"
		}
		lines = append(lines, fmt.Sprintf("  %s milestone #%d %s: %s (%d/%d witnessed)",
			mark, card.Milestone, card.Title, strings.ToUpper(card.Verdict), card.Witnessed, card.Total))
		for _, criterion := range card.Criteria {
			criterionMark := "."
			if criterion.Grade == "witnessed" {
				criterionMark = "+"
			}
			lines = append(lines, fmt.Sprintf("      %s [%s] %s - %s - %s",
				criterionMark, criterion.Workstream, criterion.Title, criterion.Grade, criterion.WitnessRef))
		}
	}
	if r.Epics.PartialNote != "" {
		lines = append(lines, "      ("+r.Epics.PartialNote+")")
	}
	if r.Trend != nil {
		lines = append(lines, "", "  trend: "+r.Trend.Summary)
	}
	lines = append(lines, "", "  -> "+r.NextAction)
	return strings.Join(lines, "\n")
}

// RenderDist renders the M0..M7 distribution as a compact one-liner ("M0:2 M1:0 …"),
// every rung in ladder order so an absent rung shows as 0. Exported so the Slack card
// (internal/milestonepost) renders the same ladder line without re-iterating the rungs.
func RenderDist(dist map[string]int) string {
	parts := make([]string, 0, len(supportmaturity.Rungs))
	for _, r := range supportmaturity.Rungs {
		id := r.String()
		parts = append(parts, fmt.Sprintf("%s:%d", id, dist[id]))
	}
	return strings.Join(parts, " ")
}

// epicRowLine renders one epic completion line for the human snapshot, honestly
// surfacing an unreadable epic as "gh read failed" rather than a fabricated 0%.
func epicRowLine(row EpicRow) string {
	if row.Err != "" {
		return fmt.Sprintf("#%d %s — gh read failed (%s)", row.Number, row.Title, row.Err)
	}
	src := ""
	if row.Source != "" {
		src = " [" + row.Source + "]"
	}
	return fmt.Sprintf("#%d %s — %.0f%% (%d/%d)%s", row.Number, row.Title, row.Pct, row.Closed, row.Total, src)
}

// programRowLine renders one ONGOING-program row. It deliberately does NOT show a
// completion % — an optimization program has no 100%. Instead it shows frontier
// ACTIVITY: closed children (shipped frontier moves) vs open (in-flight), plus the
// program label, so an operator reads "is this frontier still advancing?" not "how
// far to done?". An unreadable program is surfaced honestly, like a discrete epic.
func programRowLine(row EpicRow) string {
	label := " [" + row.Class.Label() + "]"
	if row.Err != "" {
		return fmt.Sprintf("#%d %s%s — gh read failed (%s)", row.Number, row.Title, label, row.Err)
	}
	open := row.Total - row.Closed
	src := ""
	if row.Source != "" {
		src = " {" + row.Source + "}"
	}
	return fmt.Sprintf("#%d %s%s — %d shipped / %d in-flight%s", row.Number, row.Title, label, row.Closed, open, src)
}

func generationRowLine(row GenerationRow) string {
	if row.Tracked == 0 {
		return fmt.Sprintf("%s: 0 tracked", row.Generation)
	}
	parts := []string{fmt.Sprintf("%s: %d tracked", row.Generation, row.Tracked)}
	if row.Discrete > 0 {
		parts = append(parts, fmt.Sprintf("%.1f%% over %d discrete", row.OverallPct, row.Discrete))
	} else {
		parts = append(parts, "no discrete completion")
	}
	if row.Programs > 0 {
		parts = append(parts, fmt.Sprintf("%d ongoing", row.Programs))
	}
	if row.Errored > 0 {
		parts = append(parts, fmt.Sprintf("%d unreadable", row.Errored))
	}
	if row.DebtScore > 0 {
		if row.DebtReason != "" {
			parts = append(parts, fmt.Sprintf("debt %d (%s)", row.DebtScore, row.DebtReason))
		} else {
			parts = append(parts, fmt.Sprintf("debt %d", row.DebtScore))
		}
	}
	return strings.Join(parts, "; ")
}

// CheckGate is the advisory CI gate over a folded report (pure: exit code + message).
// It fails ONLY when a dimension could not be measured — the milestone report is a
// mirror, not a second quality gate (a 0%-complete epic is a measured fact, not an
// incomplete report).
//
//	0  milestone recorded (clear or advisory)
//	1  a dimension failed to measure (the report is incomplete)
func CheckGate(r Report) (int, string) {
	v := trendreport.AdvisoryGate("MILESTONE", r.Finding, r.Reason, "milestone_unmeasured")
	return v.Exit, v.Message
}

// CheckGateTriaged is CheckGate with the decenter-the-human fold applied at the
// source: an INCOMPLETE report whose NextAction is a runnable regenerate routes to
// the fleet instead of paging. Soaks behind enforce (the CLI reads
// FAK_MILESTONE_TRIAGE_GATE); enforce=false is byte-for-byte CheckGate.
func CheckGateTriaged(r Report, enforce bool) (int, string) {
	v := trendreport.AdvisoryGateTriaged("MILESTONE", r.Finding, r.Reason, r.NextAction, "milestone_unmeasured", enforce)
	return v.Exit, v.Message
}

// TriageSelfcheck proves the milestone report's source-level page-vs-act fold with
// no I/O, delegating to the shared trendreport proof.
func TriageSelfcheck() error { return trendreport.TriageSelfcheck() }

// WithGate returns a copy reconciled to a CheckGate decision, for --check --json.
func (r Report) WithGate(code int, message string) Report {
	r.Envelope = r.Envelope.WithGate(trendreport.GateVerdict{Exit: code, Message: message})
	return r
}

// --- small shared helpers ---------------------------------------------------

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func pct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return round1(100 * float64(n) / float64(total))
}

func round1(f float64) float64 {
	return float64(int64(f*10+0.5)) / 10
}
