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

	maturityscore "github.com/anthony-chaudhary/fak/internal/maturity"
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

const (
	standingBase                  = 1000
	standingHealthBPScale         = 10000
	standingGradeDebtPerScorecard = 8
)

// Scores is the SCORES dimension, distilled from the scorecard control pane.
type Scores struct {
	Debt           int    `json:"debt"`
	GradeDebt      int    `json:"grade_debt"`
	Measured       int    `json:"measured"`
	TrendDirection string `json:"trend_direction"`
	TrendSummary   string `json:"trend_summary"`
	OK             bool   `json:"ok"`
	Err            string `json:"err,omitempty"`
}

// Maturity is the feature-lifecycle dimension distilled from `fak maturity`.
// It makes the capability ladder visible in the regular cadence report, not just
// as an on-demand scorecard: debt is the ratcheted ladder-skip count, while
// score/backlog/distribution are the human/agent work-shaping signals.
type Maturity struct {
	Debt                int            `json:"debt"`
	Score               int            `json:"score"`
	Grade               string         `json:"grade"`
	Capabilities        int            `json:"capabilities"`
	LadderSkips         int            `json:"ladder_skips"`
	Backlog             int            `json:"backlog"`
	Distribution        map[string]int `json:"distribution,omitempty"`
	NextLane            string         `json:"next_lane,omitempty"`
	NextItem            string         `json:"next_item,omitempty"`
	RouteKey            string         `json:"route_key,omitempty"`
	RouteLane           string         `json:"route_lane,omitempty"`
	RouteItem           string         `json:"route_item,omitempty"`
	RouteWitness        string         `json:"route_witness,omitempty"`
	RouteSkippedPrivate int            `json:"route_skipped_private,omitempty"`
	OK                  bool           `json:"ok"`
	Err                 string         `json:"err,omitempty"`
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

// Releases is the RELEASES dimension, distilled from the release-status fold and
// (for the publish-staleness fields) the Go-native releasestale signal.
type Releases struct {
	Version      string `json:"version"`
	ActionKind   string `json:"action_kind"`
	ActionDetail string `json:"action_detail"`
	Verdict      string `json:"verdict"`
	OK           bool   `json:"ok"`
	// Publish-staleness: how far the latest published vX.Y.Z tag — what
	// `go install ...@latest` resolves to — lags HEAD. Layered on by
	// WithPublishStaleness from the releasestale signal; informational, never gating.
	CommitsBehind  int     `json:"commits_behind,omitempty"`
	DaysBehind     float64 `json:"days_behind,omitempty"`
	PublishVerdict string  `json:"publish_verdict,omitempty"`
	Err            string  `json:"err,omitempty"`
}

// WithPublishStaleness layers the Go-native publish-staleness verdict onto a Releases
// dimension: how far the latest published tag lags HEAD, in commits AND days, plus the
// fresh/stale/very_stale/unknown verdict. The git gathering lives in collect.go (the
// impure shell); this projection is pure so the render/ledger wiring stays testable.
//
// It is INFORMATIONAL: it never flips Releases.OK or the cadence verdict. The cadence
// report gates only on UN-MEASURED dimensions (it must not double-gate), and `fak
// release-staleness --check` is the dedicated gate for the lag itself. This just makes
// the lag visible in the fold the fleet already runs and trendable in the ledger.
func WithPublishStaleness(r Releases, commitsBehind int, daysBehind float64, verdict string) Releases {
	r.CommitsBehind = commitsBehind
	r.DaysBehind = daysBehind
	r.PublishVerdict = verdict
	return r
}

// Trend is the per-tick delta vs the previous ledger row (the durable history's
// reason for existing: a trend across ticks, not against one pinned baseline).
type Trend struct {
	PrevDate                string `json:"prev_date"`
	PrevCommit              string `json:"prev_commit"`
	Direction               string `json:"direction"` // improved | regressed | flat | new
	DebtFrom                int    `json:"debt_from"`
	DebtTo                  int    `json:"debt_to"`
	DebtDelta               int    `json:"debt_delta"`
	GradeDebtFrom           int    `json:"grade_debt_from"`
	GradeDebtTo             int    `json:"grade_debt_to"`
	GradeDebtDelta          int    `json:"grade_debt_delta"`
	StandingFrom            int    `json:"standing_from"`
	StandingTo              int    `json:"standing_to"`
	StandingDelta           int    `json:"standing_delta"`
	StandingHealthFromBP    int    `json:"standing_health_from_bp"`
	StandingHealthToBP      int    `json:"standing_health_to_bp"`
	StandingHealthDeltaBP   int    `json:"standing_health_delta_bp"`
	StandingDifficultyFrom  int    `json:"standing_difficulty_from"`
	StandingDifficultyTo    int    `json:"standing_difficulty_to"`
	StandingDifficultyDelta int    `json:"standing_difficulty_delta"`
	WorkCommitsFrom         int    `json:"work_commits_from"`
	WorkCommitsTo           int    `json:"work_commits_to"`
	WorkCommitsDelta        int    `json:"work_commits_delta"`
	WorkShipsFrom           int    `json:"work_ships_from"`
	WorkShipsTo             int    `json:"work_ships_to"`
	WorkShipsDelta          int    `json:"work_ships_delta"`
	ShipsSince              int    `json:"ships_since"`
	MaturityScoreFrom       int    `json:"maturity_score_from"`
	MaturityScoreTo         int    `json:"maturity_score_to"`
	MaturityScoreDelta      int    `json:"maturity_score_delta"`
	MaturityDebtFrom        int    `json:"maturity_debt_from"`
	MaturityDebtTo          int    `json:"maturity_debt_to"`
	MaturityDebtDelta       int    `json:"maturity_debt_delta"`
	MaturityBacklogFrom     int    `json:"maturity_backlog_from"`
	MaturityBacklogTo       int    `json:"maturity_backlog_to"`
	MaturityBacklogDelta    int    `json:"maturity_backlog_delta"`
	Summary                 string `json:"summary"`
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
	Maturity    Maturity `json:"maturity"`
	Work        Work     `json:"work"`
	Releases    Releases `json:"releases"`
	Trend       *Trend   `json:"trend,omitempty"`
	// gate fields, set only for the --check --json envelope.
	GateExit    *int   `json:"gate_exit,omitempty"`
	GateMessage string `json:"gate_message,omitempty"`
}

// LedgerRow is one durable, append-only history line (a flattened projection of
// the cadence dimensions, so the ledger is a self-describing time series).
type LedgerRow struct {
	Schema                  string `json:"schema"`
	Date                    string `json:"date"`
	Commit                  string `json:"commit"`
	GeneratedAt             string `json:"generated_at"`
	Verdict                 string `json:"verdict"`
	ScoresDebt              int    `json:"scores_debt"`
	ScoresGradeDebt         int    `json:"scores_grade_debt,omitempty"`
	ScoresMeasured          int    `json:"scores_measured,omitempty"`
	ScoresTrend             string `json:"scores_trend"`
	StandingScore           int    `json:"standing_score,omitempty"`
	StandingDelta           int    `json:"standing_delta,omitempty"`
	StandingHealthBP        int    `json:"standing_health_bp,omitempty"`
	StandingDifficulty      int    `json:"standing_difficulty,omitempty"`
	StandingDifficultyDelta int    `json:"standing_difficulty_delta,omitempty"`
	MaturityDebt            int    `json:"maturity_debt"`
	MaturityScore           int    `json:"maturity_score"`
	MaturityGrade           string `json:"maturity_grade,omitempty"`
	MaturityBacklog         int    `json:"maturity_backlog"`
	MaturityCapabilities    int    `json:"maturity_capabilities"`
	MaturityProposed        int    `json:"maturity_proposed"`
	MaturityPrototyped      int    `json:"maturity_prototyped"`
	MaturityTested          int    `json:"maturity_tested"`
	MaturityDogfooded       int    `json:"maturity_dogfooded"`
	MaturityDefault         int    `json:"maturity_default"`
	MaturityRouteKey        string `json:"maturity_route_key,omitempty"`
	MaturityRouteLane       string `json:"maturity_route_lane,omitempty"`
	MaturityRouteSkipped    int    `json:"maturity_route_skipped_private,omitempty"`
	WorkWindowDays          int    `json:"work_window_days"`
	WorkCommits             int    `json:"work_commits"`
	WorkShips               int    `json:"work_ships"`
	ReleaseVersion          string `json:"release_version"`
	ReleaseAction           string `json:"release_action"`
	ReleaseCommitsBehind    int    `json:"release_commits_behind,omitempty"`
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
		GradeDebt:      asInt(payload["grade_debt"]),
		Measured:       asInt(payload["measured"]),
		TrendDirection: "unknown",
	}
	if _, ok := payload["grade_debt"]; !ok && s.Debt > 0 && s.Measured > 0 {
		s.GradeDebt = minInt(s.Debt, s.Measured*standingGradeDebtPerScorecard)
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

// MaturityFromScorecard distills the Go-native maturity scorecard payload into
// the cadence dimension. It accepts the typed payload so the live path does not
// JSON round-trip just to read corpus fields.
func MaturityFromScorecard(payload maturityscore.ScorecardPayload) Maturity {
	c := payload.Corpus
	m := Maturity{
		Debt:         asInt(c["maturity_debt"]),
		Score:        asInt(c["score"]),
		Grade:        asString(c["grade"]),
		Capabilities: asInt(c["capabilities"]),
		LadderSkips:  asInt(c["ladder_skips"]),
		Backlog:      asInt(c["backlog"]),
		Distribution: asIntMap(c["distribution"]),
		OK:           payload.OK,
	}
	if len(payload.Backlog) > 0 {
		m.NextLane = payload.Backlog[0].Lane
		m.NextItem = payload.Backlog[0].Title
	}
	projection := maturityscore.ProjectIssueItems(payload, 1, nil)
	if len(projection.Items) > 0 {
		item := projection.Items[0]
		m.RouteKey = item.Key
		m.RouteLane = item.Lane
		m.RouteItem = item.Title
		m.RouteWitness = item.Witness
	}
	m.RouteSkippedPrivate = len(projection.Skipped)
	return m
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

// Fold folds the original three dimensions into one cadence-report control-pane
// envelope. Production uses FoldWithMaturity so the maturity dimension is also
// visible; this wrapper keeps older pure tests and callers stable.
//
// The verdict ladder is deliberately a REPORT contract, not a second quality
// gate: the scorecard ratchet (ci.yml) already gates debt regressions, so the
// cadence report must not double-gate them. It is ACTION only when a dimension
// could not be MEASURED — i.e. when the report itself is incomplete — and OK
// otherwise, surfacing a regressed score or a pending release as an advisory
// line in the reason. This mirrors fresh_status's advisory contract.
func Fold(scores Scores, work Work, releases Releases, opts FoldOpts) Report {
	return FoldWithMaturity(scores, Maturity{}, work, releases, opts)
}

// FoldWithMaturity folds the cadence dimensions into one report, including the
// feature-maturity ladder. The no-maturity Fold wrapper is kept for older tests
// and callers that only exercise the original three-dimension fold.
func FoldWithMaturity(scores Scores, maturity Maturity, work Work, releases Releases, opts FoldOpts) Report {
	r := Report{
		Schema:      Schema,
		Workspace:   opts.Workspace,
		Commit:      opts.Commit,
		GeneratedAt: opts.GeneratedAt,
		Date:        opts.Date,
		Scores:      scores,
		Maturity:    maturity,
		Work:        work,
		Releases:    releases,
	}

	var unmeasured []string
	if scores.Err != "" {
		unmeasured = append(unmeasured, "scores ("+scores.Err+")")
	}
	if maturity.Err != "" {
		unmeasured = append(unmeasured, "maturity ("+maturity.Err+")")
	}
	if work.Err != "" {
		unmeasured = append(unmeasured, "work ("+work.Err+")")
	}
	if releases.Err != "" {
		unmeasured = append(unmeasured, "releases ("+releases.Err+")")
	}

	scoreLine := fmt.Sprintf("scores: debt %d (%s)", scores.Debt, scores.TrendDirection)
	maturityLine := fmt.Sprintf("maturity: index %d/100, debt %d, backlog %d", maturity.Score, maturity.Debt, maturity.Backlog)
	if maturity.RouteLane != "" {
		maturityLine += fmt.Sprintf(", route %s", maturity.RouteLane)
	}
	if maturity.RouteSkippedPrivate > 0 {
		maturityLine += fmt.Sprintf(", %d private skip(s)", maturity.RouteSkippedPrivate)
	}
	workLine := fmt.Sprintf("work: %d commit(s)/%d ship(s) in %dd", work.Commits, work.Ships, work.WindowDays)
	relLine := fmt.Sprintf("releases: %s -> %s", releases.Version, releases.ActionKind)
	if releases.CommitsBehind > 0 {
		relLine += fmt.Sprintf(" (@latest %d behind", releases.CommitsBehind)
		if releases.PublishVerdict != "" {
			relLine += ", " + releases.PublishVerdict
		}
		relLine += ")"
	}
	summary := strings.Join([]string{scoreLine, maturityLine, workLine, relLine}, "; ")

	switch {
	case len(unmeasured) > 0:
		r.OK, r.Verdict, r.Finding = false, "ACTION", "cadence_unmeasured"
		r.Reason = "cadence report incomplete — could not measure " + strings.Join(unmeasured, ", ")
		r.NextAction = "repair the failing dimension(s) so the cadence report is whole, then re-run `fak cadence`"
	case scores.TrendDirection == "regressed":
		r.OK, r.Verdict, r.Finding = true, "OK", "cadence_advisory"
		r.Reason = "cadence recorded; " + summary + " (advisory: score debt regressed — the scorecard ratchet owns that gate)"
		r.NextAction = "retire the regressed scorecard worst-first; the cadence tick keeps recording the trend"
	case maturity.Debt > 0:
		r.OK, r.Verdict, r.Finding = true, "OK", "cadence_advisory"
		r.Reason = "cadence recorded; " + summary + " (advisory: maturity ladder-skip debt exists — the scorecard ratchet owns that gate)"
		if maturity.RouteLane != "" {
			r.NextAction = fmt.Sprintf("run `fak maturity route --fetch-existing --limit 3` to seed dispatch with %s; retire ladder-skips first", maturity.RouteLane)
		} else {
			r.NextAction = "run `fak maturity next` and retire ladder-skips first; the cadence tick keeps recording the trend"
		}
	default:
		r.OK, r.Verdict, r.Finding = true, "OK", "cadence_recorded"
		r.Reason = "cadence recorded; " + summary
		if maturity.RouteLane != "" {
			r.NextAction = fmt.Sprintf("run `fak maturity route --fetch-existing --limit 3` to seed dispatch with %s; the scheduled cadence tick keeps the trend", maturity.RouteLane)
		} else {
			r.NextAction = "hold the line; the scheduled cadence tick keeps scores/maturity/work/releases trended"
		}
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
	dist := r.Maturity.Distribution
	return LedgerRow{
		Schema:               LedgerSchema,
		Date:                 r.Date,
		Commit:               r.Commit,
		GeneratedAt:          r.GeneratedAt,
		Verdict:              r.Verdict,
		ScoresDebt:           r.Scores.Debt,
		ScoresGradeDebt:      r.Scores.GradeDebt,
		ScoresMeasured:       r.Scores.Measured,
		ScoresTrend:          r.Scores.TrendDirection,
		MaturityDebt:         r.Maturity.Debt,
		MaturityScore:        r.Maturity.Score,
		MaturityGrade:        r.Maturity.Grade,
		MaturityBacklog:      r.Maturity.Backlog,
		MaturityCapabilities: r.Maturity.Capabilities,
		MaturityProposed:     dist["proposed"],
		MaturityPrototyped:   dist["prototyped"],
		MaturityTested:       dist["tested"],
		MaturityDogfooded:    dist["dogfooded"],
		MaturityDefault:      dist["default"],
		MaturityRouteKey:     r.Maturity.RouteKey,
		MaturityRouteLane:    r.Maturity.RouteLane,
		MaturityRouteSkipped: r.Maturity.RouteSkippedPrivate,
		WorkWindowDays:       r.Work.WindowDays,
		WorkCommits:          r.Work.Commits,
		WorkShips:            r.Work.Ships,
		ReleaseVersion:       r.Releases.Version,
		ReleaseAction:        r.Releases.ActionKind,
		ReleaseCommitsBehind: r.Releases.CommitsBehind,
	}
}

// ProjectStanding adds the durable, unbounded standing fields to a ledger row.
// The raw scorecards still use bounded local scores (usually 0..100), but the
// cadence ledger needs a value that can keep climbing or fall across ticks. The
// formula first normalizes the current row into a 0..10000 health reading so a
// harder tick (more scorecards or more maturity capabilities) is visible as
// difficulty rather than as an eyeballed "100". Only the delta in normalized
// health moves standing; the starting point is an arbitrary 1000-base index.
func ProjectStanding(row LedgerRow, prior []LedgerRow) LedgerRow {
	row.StandingHealthBP, row.StandingDifficulty = standingHealthBP(row)
	last, ok := latestBefore(row, prior)
	if !ok {
		row.StandingScore = standingBase
		return row
	}
	lastHealth, lastDifficulty := last.StandingHealthBP, last.StandingDifficulty
	if lastHealth == 0 && lastDifficulty == 0 {
		lastHealth, lastDifficulty = standingHealthBP(last)
	}
	row.StandingDifficultyDelta = row.StandingDifficulty - lastDifficulty
	if last.StandingScore == 0 {
		row.StandingScore = standingBase
		return row
	}
	row.StandingDelta = standingPointDelta(row.StandingHealthBP - lastHealth)
	row.StandingScore = last.StandingScore + row.StandingDelta
	return row
}

func standingHealthBP(row LedgerRow) (healthBP int, difficulty int) {
	var components []int
	if row.ScoresMeasured > 0 {
		capacity := row.ScoresMeasured * standingGradeDebtPerScorecard
		gradeDebt := row.ScoresGradeDebt
		if gradeDebt == 0 && row.ScoresDebt > 0 {
			gradeDebt = minInt(row.ScoresDebt, capacity)
		}
		gradeDebt = clampInt(gradeDebt, 0, capacity)
		components = append(components, standingHealthBPScale-(gradeDebt*standingHealthBPScale)/capacity)
		difficulty += capacity
	}
	if row.MaturityCapabilities > 0 || row.MaturityScore > 0 {
		score := clampInt(row.MaturityScore, 0, 100)
		components = append(components, score*100)
		if row.MaturityCapabilities > 0 {
			difficulty += row.MaturityCapabilities
		} else {
			difficulty++
		}
	}
	if len(components) == 0 {
		return 0, 0
	}
	total := 0
	for _, c := range components {
		total += c
	}
	return (total + len(components)/2) / len(components), difficulty
}

func standingPointDelta(healthDeltaBP int) int {
	if healthDeltaBP >= 0 {
		return (healthDeltaBP + 50) / 100
	}
	return -((-healthDeltaBP + 50) / 100)
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
	if row.StandingScore == 0 {
		row = ProjectStanding(row, prior)
	}
	last, ok := latestBefore(row, prior)
	if !ok {
		return Trend{
			Direction:            "new",
			DebtTo:               row.ScoresDebt,
			GradeDebtTo:          row.ScoresGradeDebt,
			StandingTo:           row.StandingScore,
			StandingHealthToBP:   row.StandingHealthBP,
			StandingDifficultyTo: row.StandingDifficulty,
			WorkCommitsTo:        row.WorkCommits,
			WorkShipsTo:          row.WorkShips,
			ShipsSince:           row.WorkShips,
			MaturityScoreTo:      row.MaturityScore,
			MaturityDebtTo:       row.MaturityDebt,
			MaturityBacklogTo:    row.MaturityBacklog,
			Summary: fmt.Sprintf("first cadence tick (standing %d, health %s, difficulty %d; debt %d, maturity %d/100 with %d backlog item(s), %d ship(s) in %dd)",
				row.StandingScore, formatBP(row.StandingHealthBP), row.StandingDifficulty,
				row.ScoresDebt, row.MaturityScore, row.MaturityBacklog, row.WorkShips, row.WorkWindowDays),
		}
	}
	debtDelta := row.ScoresDebt - last.ScoresDebt
	gradeDebtDelta := row.ScoresGradeDebt - last.ScoresGradeDebt
	workCommitsDelta := row.WorkCommits - last.WorkCommits
	workShipsDelta := row.WorkShips - last.WorkShips
	maturityScoreDelta := row.MaturityScore - last.MaturityScore
	maturityDebtDelta := row.MaturityDebt - last.MaturityDebt
	maturityBacklogDelta := row.MaturityBacklog - last.MaturityBacklog
	standingDelta := row.StandingDelta
	if row.StandingScore != 0 && last.StandingScore != 0 {
		standingDelta = row.StandingScore - last.StandingScore
	}
	standingHealthDelta := row.StandingHealthBP - last.StandingHealthBP
	standingDifficultyDelta := row.StandingDifficulty - last.StandingDifficulty
	standingComparable := last.StandingScore != 0 && row.StandingScore != 0
	dir := trendDirection(debtDelta, standingDelta, standingComparable)
	summary := fmt.Sprintf("debt %s %+d (%d->%d); maturity score %+d (%d->%d), debt %+d (%d->%d), backlog %+d (%d->%d); work %s %+d commit(s) (%d->%d), %s %+d ship(s) (%d->%d) vs %s; %d ship(s) in the last %dd",
		dir, debtDelta, last.ScoresDebt, row.ScoresDebt,
		maturityScoreDelta, last.MaturityScore, row.MaturityScore,
		maturityDebtDelta, last.MaturityDebt, row.MaturityDebt,
		maturityBacklogDelta, last.MaturityBacklog, row.MaturityBacklog,
		directionWord(workCommitsDelta), workCommitsDelta, last.WorkCommits, row.WorkCommits,
		directionWord(workShipsDelta), workShipsDelta, last.WorkShips, row.WorkShips,
		last.Date, row.WorkShips, row.WorkWindowDays)
	if standingComparable {
		summary = fmt.Sprintf("standing %+d (%d->%d; health %s->%s; difficulty %+d, %d->%d); ",
			standingDelta, last.StandingScore, row.StandingScore,
			formatBP(last.StandingHealthBP), formatBP(row.StandingHealthBP),
			standingDifficultyDelta, last.StandingDifficulty, row.StandingDifficulty) + summary
	}
	return Trend{
		PrevDate:                last.Date,
		PrevCommit:              last.Commit,
		Direction:               dir,
		DebtFrom:                last.ScoresDebt,
		DebtTo:                  row.ScoresDebt,
		DebtDelta:               debtDelta,
		GradeDebtFrom:           last.ScoresGradeDebt,
		GradeDebtTo:             row.ScoresGradeDebt,
		GradeDebtDelta:          gradeDebtDelta,
		StandingFrom:            last.StandingScore,
		StandingTo:              row.StandingScore,
		StandingDelta:           standingDelta,
		StandingHealthFromBP:    last.StandingHealthBP,
		StandingHealthToBP:      row.StandingHealthBP,
		StandingHealthDeltaBP:   standingHealthDelta,
		StandingDifficultyFrom:  last.StandingDifficulty,
		StandingDifficultyTo:    row.StandingDifficulty,
		StandingDifficultyDelta: standingDifficultyDelta,
		WorkCommitsFrom:         last.WorkCommits,
		WorkCommitsTo:           row.WorkCommits,
		WorkCommitsDelta:        workCommitsDelta,
		WorkShipsFrom:           last.WorkShips,
		WorkShipsTo:             row.WorkShips,
		WorkShipsDelta:          workShipsDelta,
		ShipsSince:              row.WorkShips,
		MaturityScoreFrom:       last.MaturityScore,
		MaturityScoreTo:         row.MaturityScore,
		MaturityScoreDelta:      maturityScoreDelta,
		MaturityDebtFrom:        last.MaturityDebt,
		MaturityDebtTo:          row.MaturityDebt,
		MaturityDebtDelta:       maturityDebtDelta,
		MaturityBacklogFrom:     last.MaturityBacklog,
		MaturityBacklogTo:       row.MaturityBacklog,
		MaturityBacklogDelta:    maturityBacklogDelta,
		Summary:                 summary,
	}
}

func trendDirection(debtDelta, standingDelta int, standingComparable bool) string {
	if standingComparable {
		if standingDelta > 0 {
			return "improved"
		}
		if standingDelta < 0 {
			return "regressed"
		}
		return "flat"
	}
	if debtDelta < 0 {
		return "improved"
	}
	if debtDelta > 0 {
		return "regressed"
	}
	return "flat"
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
		fmt.Sprintf("  %s scores      debt %d; grade-debt %d across %d scorecard(s); trend %s",
			mark(r.Scores.OK, r.Scores.Err), r.Scores.Debt, r.Scores.GradeDebt, r.Scores.Measured, dashIfEmpty(r.Scores.TrendSummary)),
		fmt.Sprintf("  %s maturity    index %d/100 [%s]; debt %d; backlog %d%s",
			mark(r.Maturity.OK, r.Maturity.Err), r.Maturity.Score, dashIfEmpty(r.Maturity.Grade),
			r.Maturity.Debt, r.Maturity.Backlog, maturityNextSuffix(r.Maturity)+maturityRouteSuffix(r.Maturity)),
		fmt.Sprintf("  %s work        %d commit(s) / %d ship(s) in the last %dd",
			mark(r.Work.Err == "", r.Work.Err), r.Work.Commits, r.Work.Ships, r.Work.WindowDays),
		fmt.Sprintf("  %s releases    %s%s; next: %s — %s",
			mark(r.Releases.OK, r.Releases.Err), r.Releases.Version, publishLagSuffix(r.Releases), dashIfEmpty(r.Releases.ActionKind), dashIfEmpty(r.Releases.ActionDetail)),
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
		if r.Trend.StandingTo != 0 {
			lines = append(lines, fmt.Sprintf("  standing   score %d (%+d); health %s; difficulty %d (%+d)",
				r.Trend.StandingTo, r.Trend.StandingDelta, formatBP(r.Trend.StandingHealthToBP),
				r.Trend.StandingDifficultyTo, r.Trend.StandingDifficultyDelta))
		}
		lines = append(lines, "", "  trend: "+r.Trend.Summary)
	}
	lines = append(lines, "", "  -> "+r.NextAction)
	return strings.Join(lines, "\n")
}

// publishLagSuffix renders the @latest-vs-HEAD lag for the releases render line, or ""
// when @latest is current (or the lag was not measured). Kept tiny + pure.
func publishLagSuffix(r Releases) string {
	if r.CommitsBehind <= 0 {
		return ""
	}
	suffix := fmt.Sprintf("  [@latest %d behind", r.CommitsBehind)
	if r.PublishVerdict != "" {
		suffix += ", " + r.PublishVerdict
	}
	return suffix + "]"
}

func maturityNextSuffix(m Maturity) string {
	if m.NextLane == "" || m.NextItem == "" {
		return ""
	}
	return "; next " + m.NextLane + ": " + m.NextItem
}

func maturityRouteSuffix(m Maturity) string {
	skipped := ""
	if m.RouteSkippedPrivate > 0 {
		skipped = fmt.Sprintf(" (%d private skipped)", m.RouteSkippedPrivate)
	}
	if m.RouteLane == "" || m.RouteItem == "" {
		if skipped != "" {
			return "; route none" + skipped
		}
		return ""
	}
	return "; route " + m.RouteLane + ": " + m.RouteItem + skipped
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

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func clampInt(n, lo, hi int) int {
	if n < lo {
		return lo
	}
	if n > hi {
		return hi
	}
	return n
}

func formatBP(bp int) string {
	return fmt.Sprintf("%.1f%%", float64(bp)/100)
}

func asIntMap(v any) map[string]int {
	out := map[string]int{}
	switch m := v.(type) {
	case map[string]int:
		for k, n := range m {
			out[k] = n
		}
	case map[string]any:
		for k, n := range m {
			out[k] = asInt(n)
		}
	}
	return out
}
