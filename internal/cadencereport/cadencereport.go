package cadencereport

// cadencereport.go holds the pure, unit-tested surface: the dimension
// interpreters, the fold, the durable-ledger parse/trend, render, and the gate.
// The live runners (Python sub-tools + git) live in collect.go. The package doc
// is in doc.go.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Schema is the stable control-pane schema identifier for the report envelope.
const Schema = "fak-cadence-report/1"

// LedgerSchema tags each durable history row so a reader can validate the line.
const LedgerSchema = "fak-cadence-ledger/1"

// DefaultLedgerRel is the committed, append-only history ledger (one JSONL row
// per cadence tick). It lives under docs/ so it is durable trunk evidence, not a
// regenerable build artifact.
const DefaultLedgerRel = "docs/cadence/history.jsonl"

// DefaultWindowDays is the trailing window the WORK-DONE dimension counts over.
const DefaultWindowDays = 7

// Scores is the SCORES dimension, distilled from the scorecard control pane.
type Scores struct {
	Debt           int    `json:"debt"`
	Measured       int    `json:"measured"`
	TrendDirection string `json:"trend_direction"`
	TrendSummary   string `json:"trend_summary"`
	OK             bool   `json:"ok"`
	Err            string `json:"err,omitempty"`
}

// Work is the WORK-DONE dimension, derived from git over a trailing window: the
// commit count and the subset whose SUBJECT carries a real per-leaf ship-stamp
// (the same `(fak <leaf>)` trailer / `fak/<leaf>:` direct grammar the pre-commit
// lint binds to — see hooks.StampOf). ByLane buckets those ships by leaf; it is
// report/render-only and intentionally NOT persisted to the ledger, so the
// fak-cadence-ledger/1 row schema stays byte-stable.
type Work struct {
	WindowDays int            `json:"window_days"`
	Commits    int            `json:"commits"`
	Ships      int            `json:"ships"`
	ByLane     map[string]int `json:"by_lane,omitempty"`
	Err        string         `json:"err,omitempty"`
}

// Releases is the RELEASES dimension, distilled from the release-status fold.
type Releases struct {
	Version      string `json:"version"`
	ActionKind   string `json:"action_kind"`
	ActionDetail string `json:"action_detail"`
	Verdict      string `json:"verdict"`
	OK           bool   `json:"ok"`
	Err          string `json:"err,omitempty"`
}

// Trend is the per-tick delta vs the previous ledger row (the durable history's
// reason for existing: a trend across ticks, not against one pinned baseline).
type Trend struct {
	PrevDate         string `json:"prev_date"`
	PrevCommit       string `json:"prev_commit"`
	Direction        string `json:"direction"` // improved | regressed | flat | new
	DebtFrom         int    `json:"debt_from"`
	DebtTo           int    `json:"debt_to"`
	DebtDelta        int    `json:"debt_delta"`
	WorkCommitsFrom  int    `json:"work_commits_from"`
	WorkCommitsTo    int    `json:"work_commits_to"`
	WorkCommitsDelta int    `json:"work_commits_delta"`
	WorkShipsFrom    int    `json:"work_ships_from"`
	WorkShipsTo      int    `json:"work_ships_to"`
	WorkShipsDelta   int    `json:"work_ships_delta"`
	ShipsSince       int    `json:"ships_since"`
	Summary          string `json:"summary"`
}

// Report is one folded cadence-report control-pane envelope.
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
	Scores      Scores   `json:"scores"`
	Work        Work     `json:"work"`
	Releases    Releases `json:"releases"`
	Trend       *Trend   `json:"trend,omitempty"`
	// gate fields, set only for the --check --json envelope.
	GateExit    *int   `json:"gate_exit,omitempty"`
	GateMessage string `json:"gate_message,omitempty"`
}

// LedgerRow is one durable, append-only history line (a flattened projection of
// the three dimensions, so the ledger is a self-describing time series).
type LedgerRow struct {
	Schema         string `json:"schema"`
	Date           string `json:"date"`
	Commit         string `json:"commit"`
	GeneratedAt    string `json:"generated_at"`
	Verdict        string `json:"verdict"`
	ScoresDebt     int    `json:"scores_debt"`
	ScoresTrend    string `json:"scores_trend"`
	WorkWindowDays int    `json:"work_window_days"`
	WorkCommits    int    `json:"work_commits"`
	WorkShips      int    `json:"work_ships"`
	ReleaseVersion string `json:"release_version"`
	ReleaseAction  string `json:"release_action"`
}

// --- pure interpretation of the sub-tool payloads --------------------------

// InterpretScores distills a scorecard_control_pane.py --json payload into the
// SCORES dimension. The debt integer and trend live at the top level of that
// fold; a missing/garbled payload degrades to an errored dimension, never a
// silent zero.
func InterpretScores(payload map[string]any, runErr string) Scores {
	if runErr != "" || payload == nil {
		return Scores{Err: orNoPayload(runErr), TrendDirection: "unknown"}
	}
	s := Scores{
		Debt:           asInt(payload["total_debt"]),
		Measured:       asInt(payload["measured"]),
		TrendDirection: "unknown",
	}
	if tr, ok := payload["trend"].(map[string]any); ok {
		s.TrendDirection = asString(tr["direction"])
		s.TrendSummary = asString(tr["summary"])
	}
	if s.TrendDirection == "" {
		s.TrendDirection = "unknown"
	}
	// A measurement is only trustworthy when every scorecard reported. The fold
	// carries `errored`; a nonzero count means the portfolio number is partial.
	if errored := asInt(payload["errored"]); errored > 0 {
		s.Err = fmt.Sprintf("%d scorecard(s) unmeasured", errored)
	}
	// SCORES is healthy when the ratchet did not regress (debt may hold or fall).
	s.OK = s.Err == "" && s.TrendDirection != "regressed"
	return s
}

// InterpretReleases distills a release_status.py --json payload into the
// RELEASES dimension.
func InterpretReleases(payload map[string]any, runErr string) Releases {
	if runErr != "" || payload == nil {
		return Releases{Err: orNoPayload(runErr), Verdict: "ERROR"}
	}
	r := Releases{
		Verdict: asString(payload["verdict"]),
		OK:      asBool(payload["ok"]),
	}
	if rolling, ok := payload["rolling"].(map[string]any); ok {
		r.Version = asString(rolling["last_tag"])
	}
	if action, ok := payload["next_action"].(map[string]any); ok {
		r.ActionKind = asString(action["kind"])
		r.ActionDetail = asString(action["detail"])
	}
	if r.Version == "" {
		r.Version = "(none)"
	}
	if r.Verdict == "" {
		if r.OK {
			r.Verdict = "OK"
		} else {
			r.Verdict = "ACTION"
		}
	}
	return r
}

// --- the fold --------------------------------------------------------------

// Fold folds the three dimensions into one cadence-report control-pane envelope.
//
// The verdict ladder is deliberately a REPORT contract, not a second quality
// gate: the scorecard ratchet (ci.yml) already gates debt regressions, so the
// cadence report must not double-gate them. It is ACTION only when a dimension
// could not be MEASURED — i.e. when the report itself is incomplete — and OK
// otherwise, surfacing a regressed score or a pending release as an advisory
// line in the reason. This mirrors fresh_status's advisory contract.
func Fold(scores Scores, work Work, releases Releases, opts FoldOpts) Report {
	r := Report{
		Schema:      Schema,
		Workspace:   opts.Workspace,
		Commit:      opts.Commit,
		GeneratedAt: opts.GeneratedAt,
		Date:        opts.Date,
		Scores:      scores,
		Work:        work,
		Releases:    releases,
	}

	var unmeasured []string
	if scores.Err != "" {
		unmeasured = append(unmeasured, "scores ("+scores.Err+")")
	}
	if work.Err != "" {
		unmeasured = append(unmeasured, "work ("+work.Err+")")
	}
	if releases.Err != "" {
		unmeasured = append(unmeasured, "releases ("+releases.Err+")")
	}

	scoreLine := fmt.Sprintf("scores: debt %d (%s)", scores.Debt, scores.TrendDirection)
	workLine := fmt.Sprintf("work: %d commit(s)/%d ship(s) in %dd", work.Commits, work.Ships, work.WindowDays)
	relLine := fmt.Sprintf("releases: %s -> %s", releases.Version, releases.ActionKind)
	summary := strings.Join([]string{scoreLine, workLine, relLine}, "; ")

	switch {
	case len(unmeasured) > 0:
		r.OK, r.Verdict, r.Finding = false, "ACTION", "cadence_unmeasured"
		r.Reason = "cadence report incomplete — could not measure " + strings.Join(unmeasured, ", ")
		r.NextAction = "repair the failing dimension(s) so the cadence report is whole, then re-run `fak cadence`"
	case scores.TrendDirection == "regressed":
		r.OK, r.Verdict, r.Finding = true, "OK", "cadence_advisory"
		r.Reason = "cadence recorded; " + summary + " (advisory: score debt regressed — the scorecard ratchet owns that gate)"
		r.NextAction = "retire the regressed scorecard worst-first; the cadence tick keeps recording the trend"
	default:
		r.OK, r.Verdict, r.Finding = true, "OK", "cadence_recorded"
		r.Reason = "cadence recorded; " + summary
		r.NextAction = "hold the line; the scheduled cadence tick keeps scores/work/releases trended"
	}
	return r
}

// FoldOpts carries the ambient context the fold stamps onto the envelope.
type FoldOpts struct {
	Workspace   string
	Commit      string
	GeneratedAt string
	Date        string
}

// RowFromReport projects a folded report into one durable ledger row.
func RowFromReport(r Report) LedgerRow {
	return LedgerRow{
		Schema:         LedgerSchema,
		Date:           r.Date,
		Commit:         r.Commit,
		GeneratedAt:    r.GeneratedAt,
		Verdict:        r.Verdict,
		ScoresDebt:     r.Scores.Debt,
		ScoresTrend:    r.Scores.TrendDirection,
		WorkWindowDays: r.Work.WindowDays,
		WorkCommits:    r.Work.Commits,
		WorkShips:      r.Work.Ships,
		ReleaseVersion: r.Releases.Version,
		ReleaseAction:  r.Releases.ActionKind,
	}
}

// --- the durable history ledger --------------------------------------------

// ParseLedger parses an append-only JSONL ledger, tolerating blank lines and
// skipping any line that is not a valid row (so a hand-edit can't crash the
// reader). Rows are returned in file order.
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

// TrendVsLast computes the per-tick trend of `row` against the most recent prior
// row in `prior` (the rows already on the ledger). With no prior row the trend
// is "new" (the first tick establishes the series).
func TrendVsLast(row LedgerRow, prior []LedgerRow) Trend {
	last, ok := latestBefore(row, prior)
	if !ok {
		return Trend{
			Direction:     "new",
			DebtTo:        row.ScoresDebt,
			WorkCommitsTo: row.WorkCommits,
			WorkShipsTo:   row.WorkShips,
			ShipsSince:    row.WorkShips,
			Summary:       fmt.Sprintf("first cadence tick (debt %d, %d ship(s) in %dd)", row.ScoresDebt, row.WorkShips, row.WorkWindowDays),
		}
	}
	debtDelta := row.ScoresDebt - last.ScoresDebt
	workCommitsDelta := row.WorkCommits - last.WorkCommits
	workShipsDelta := row.WorkShips - last.WorkShips
	dir := "flat"
	if debtDelta < 0 {
		dir = "improved"
	} else if debtDelta > 0 {
		dir = "regressed"
	}
	return Trend{
		PrevDate:         last.Date,
		PrevCommit:       last.Commit,
		Direction:        dir,
		DebtFrom:         last.ScoresDebt,
		DebtTo:           row.ScoresDebt,
		DebtDelta:        debtDelta,
		WorkCommitsFrom:  last.WorkCommits,
		WorkCommitsTo:    row.WorkCommits,
		WorkCommitsDelta: workCommitsDelta,
		WorkShipsFrom:    last.WorkShips,
		WorkShipsTo:      row.WorkShips,
		WorkShipsDelta:   workShipsDelta,
		ShipsSince:       row.WorkShips,
		Summary: fmt.Sprintf("debt %s %+d (%d->%d); work %s %+d commit(s) (%d->%d), %s %+d ship(s) (%d->%d) vs %s; %d ship(s) in the last %dd",
			dir, debtDelta, last.ScoresDebt, row.ScoresDebt,
			directionWord(workCommitsDelta), workCommitsDelta, last.WorkCommits, row.WorkCommits,
			directionWord(workShipsDelta), workShipsDelta, last.WorkShips, row.WorkShips,
			last.Date, row.WorkShips, row.WorkWindowDays),
	}
}

// directionWord renders the sign of a per-tick delta as a trend word
// (up | down | flat). Shared by the commit and ship deltas in TrendVsLast.
func directionWord(delta int) string {
	if delta > 0 {
		return "up"
	}
	if delta < 0 {
		return "down"
	}
	return "flat"
}

// latestBefore returns the most recent prior row, comparing by (date, then
// generated_at) so a same-day re-run trends against the earlier same-day tick.
// A row with the exact same generated_at as `row` is excluded (idempotent
// re-append).
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

// AppendLedgerLine renders the JSONL line for a row (no trailing newline). The
// caller appends it to the ledger file with a newline; keeping the rendering
// pure makes the writer testable without touching disk.
func AppendLedgerLine(row LedgerRow) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// --- render + gate ---------------------------------------------------------

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
		fmt.Sprintf("cadence report — %s (%s)  @%s  %s", r.Verdict, r.Finding, r.Commit, r.Date),
		"",
		fmt.Sprintf("  %s scores      debt %d across %d scorecard(s); trend %s",
			mark(r.Scores.OK, r.Scores.Err), r.Scores.Debt, r.Scores.Measured, dashIfEmpty(r.Scores.TrendSummary)),
		fmt.Sprintf("  %s work        %d commit(s) / %d ship(s) in the last %dd",
			mark(r.Work.Err == "", r.Work.Err), r.Work.Commits, r.Work.Ships, r.Work.WindowDays),
		fmt.Sprintf("  %s releases    %s; next: %s — %s",
			mark(r.Releases.OK, r.Releases.Err), r.Releases.Version, dashIfEmpty(r.Releases.ActionKind), dashIfEmpty(r.Releases.ActionDetail)),
	}
	if len(r.Work.ByLane) > 0 {
		leaves := make([]string, 0, len(r.Work.ByLane))
		for leaf := range r.Work.ByLane {
			leaves = append(leaves, leaf)
		}
		sort.Strings(leaves)
		parts := make([]string, len(leaves))
		for i, leaf := range leaves {
			parts[i] = fmt.Sprintf("%s %d", leaf, r.Work.ByLane[leaf])
		}
		lines = append(lines, "      by lane: "+strings.Join(parts, ", "))
	}
	if r.Trend != nil {
		lines = append(lines, "", "  trend: "+r.Trend.Summary)
	}
	lines = append(lines, "", "  -> "+r.NextAction)
	return strings.Join(lines, "\n")
}

// CheckGate is the advisory CI gate over a folded report (pure: exit code +
// message). It fails ONLY when a dimension could not be measured — the cadence
// report is a mirror, not a second quality gate (the scorecard ratchet owns debt
// regressions). Mirrors fresh_status's advisory contract.
//
//	0  cadence recorded (clear or score-regression advisory)
//	1  a dimension failed to measure (the report is incomplete)
func CheckGate(r Report) (int, string) {
	if r.Finding == "cadence_unmeasured" {
		return 1, "CADENCE INCOMPLETE: " + r.Reason
	}
	return 0, "CADENCE OK: " + r.Reason
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

// --- small tolerant decoders (shared shape with internal/gardenbundle) ------

func orNoPayload(runErr string) string {
	if runErr != "" {
		return runErr
	}
	return "no payload"
}

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func asBool(v any) bool {
	b, ok := v.(bool)
	return ok && b
}

func asString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

func asInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	case string:
		var i int
		_, _ = fmt.Sscanf(strings.TrimSpace(n), "%d", &i)
		return i
	default:
		return 0
	}
}
