package dojocal

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

const (
	// JournalSchema tags committed dojo-RSI journal rows.
	JournalSchema = "fak-dojo-rsi-journal/1"
	// TrendSchema tags a folded KEEP/REVERT trend over the journal.
	TrendSchema = "fak-dojo-rsi-trend/1"
	// DefaultJournalRel is the committed dojo-RSI ledger the loop and CI feed read.
	DefaultJournalRel = "docs/dojo/rsi-journal.jsonl"
	// DefaultCellRecheckDays is the staleness horizon used by the Phase-3 selector.
	DefaultCellRecheckDays = 7
)

const (
	selectWNovelty   = 0.45
	selectWValue     = 0.35
	selectWStaleness = 0.20
)

// JournalRow is one durable dojo-RSI tick. It stores the measured keep/revert
// facts and enough routing context for the CI feed to trend mechanical KEEPs,
// agent-route REVERTs, and floor ESCALATEs without re-running a corpus.
type JournalRow struct {
	Schema        string    `json:"schema"`
	Tick          int       `json:"tick"`
	Date          string    `json:"date"`
	GeneratedAt   string    `json:"generated_at"`
	Commit        string    `json:"commit,omitempty"`
	Lever         string    `json:"lever"`
	Metric        string    `json:"metric,omitempty"`
	Kind          RecalKind `json:"kind"`
	Decision      string    `json:"decision"`
	Kept          bool      `json:"kept"`
	AgentArm      bool      `json:"agent_arm,omitempty"`
	Baseline      float64   `json:"baseline_value"`
	Replayed      float64   `json:"replayed_value"`
	MeasuredDelta float64   `json:"measured_delta"`
	BreakerCount  int       `json:"breaker_nonkeeps"`
	WakeupAt      string    `json:"wakeup_at,omitempty"`
	Reason        string    `json:"reason,omitempty"`
}

// ScoredCell is one candidate scored by novelty x value x staleness.
type ScoredCell struct {
	Candidate    Recal   `json:"candidate"`
	Score        float64 `json:"score"`
	LastTouched  string  `json:"last_touched,omitempty"`
	NextEligible string  `json:"next_eligible,omitempty"`
	AgeDays      float64 `json:"age_days"`
	Novelty      float64 `json:"novelty"`
	ValueWeight  float64 `json:"value_weight"`
	Staleness    float64 `json:"staleness"`
	Saturated    bool    `json:"saturated,omitempty"`
	Reason       string  `json:"reason"`
}

// SelectOptions parameterizes candidate ranking.
type SelectOptions struct {
	Now         time.Time
	RecheckDays int
}

// Wakeup is the loop's next self-pacing decision. Command layers can translate it
// to an MCP ScheduleWakeup call; pure code only computes the time.
type Wakeup struct {
	At      string `json:"at"`
	DelayS  int64  `json:"delay_seconds"`
	Reason  string `json:"reason"`
	Pending bool   `json:"pending"`
}

// JournalTrend folds the committed journal into the CI feed payload.
type JournalTrend struct {
	Schema          string      `json:"schema"`
	GeneratedAt     string      `json:"generated_at"`
	Rows            int         `json:"rows"`
	Keep            int         `json:"keep"`
	Revert          int         `json:"revert"`
	Escalate        int         `json:"escalate"`
	MechanicalKeep  int         `json:"mechanical_keep"`
	AgentRoutes     int         `json:"agent_routes"`
	ReprojectRoutes int         `json:"reproject_routes"`
	HarvestRoutes   int         `json:"harvest_routes"`
	FloorEscalates  int         `json:"floor_escalates"`
	Latest          *JournalRow `json:"latest,omitempty"`
	Summary         string      `json:"summary"`
}

// RankCandidates reuses nightrun's selector shape for dojo cells: novelty
// (never touched), value (calibration gap), and staleness (last touch age) are
// blended into one deterministic priority score. A fresh touched cell is marked
// Saturated so a loop can stop instead of thrashing the same row.
func RankCandidates(candidates []Recal, rows []JournalRow, opts SelectOptions) []ScoredCell {
	now := opts.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	recheck := opts.RecheckDays
	if recheck <= 0 {
		recheck = DefaultCellRecheckDays
	}

	out := make([]ScoredCell, 0, len(candidates))
	for _, c := range candidates {
		s := ScoredCell{Candidate: c, AgeDays: -1, ValueWeight: valueWeight(c)}
		last, ok := lastCellTouch(rows, c.Lever, c.Metric)
		if ok {
			s.LastTouched = last.GeneratedAt
			if t, valid := parseStamp(last.GeneratedAt, last.Date); valid {
				age := now.Sub(t).Hours() / 24
				if age < 0 {
					age = 0
				}
				s.AgeDays = age
				next := t.Add(time.Duration(recheck) * 24 * time.Hour)
				s.NextEligible = next.UTC().Format(time.RFC3339)
				s.Staleness = clamp01(age / float64(recheck))
			}
		} else {
			s.Novelty = 1
		}
		s.Score = selectWNovelty*s.Novelty + selectWValue*s.ValueWeight + selectWStaleness*s.Staleness
		if ok && s.Staleness < 1 {
			s.Saturated = true
		}
		s.Reason = selectReason(s, recheck)
		out = append(out, s)
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Saturated != b.Saturated {
			return !a.Saturated
		}
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		if candidatePriority(a.Candidate.Kind) != candidatePriority(b.Candidate.Kind) {
			return candidatePriority(a.Candidate.Kind) > candidatePriority(b.Candidate.Kind)
		}
		if a.Candidate.CalibErr != b.Candidate.CalibErr {
			return a.Candidate.CalibErr > b.Candidate.CalibErr
		}
		if a.Candidate.Lever != b.Candidate.Lever {
			return a.Candidate.Lever < b.Candidate.Lever
		}
		return a.Candidate.Metric < b.Candidate.Metric
	})
	return out
}

// NextCandidate returns the highest-ranked non-saturated cell.
func NextCandidate(ranked []ScoredCell) (ScoredCell, bool) {
	for _, s := range ranked {
		if !s.Saturated {
			return s, true
		}
	}
	return ScoredCell{}, false
}

// ScheduleWakeup returns now when there is runnable work, or the earliest
// next-eligible time when every candidate is saturated.
func ScheduleWakeup(ranked []ScoredCell, now time.Time) Wakeup {
	if now.IsZero() {
		now = time.Now().UTC()
	}
	if next, ok := NextCandidate(ranked); ok {
		return Wakeup{
			At:      now.UTC().Format(time.RFC3339),
			DelayS:  0,
			Pending: true,
			Reason:  "next dojo-RSI cell is runnable now: " + next.Candidate.Lever + "/" + next.Candidate.Metric,
		}
	}
	var earliest time.Time
	for _, s := range ranked {
		if s.NextEligible == "" {
			continue
		}
		t, err := time.Parse(time.RFC3339, s.NextEligible)
		if err != nil {
			continue
		}
		if earliest.IsZero() || t.Before(earliest) {
			earliest = t
		}
	}
	if earliest.IsZero() {
		return Wakeup{At: now.UTC().Format(time.RFC3339), Reason: "no dojo-RSI candidates", Pending: false}
	}
	delay := int64(earliest.Sub(now).Seconds())
	if delay < 0 {
		delay = 0
	}
	return Wakeup{
		At:     earliest.UTC().Format(time.RFC3339),
		DelayS: delay,
		Reason: "all dojo-RSI cells are fresh; wake when the first cell becomes stale",
	}
}

// NewJournalRow projects one iteration into the durable row schema.
func NewJournalRow(tick int, it Iteration, decision string, breaker int, now time.Time, commit string, wake Wakeup) JournalRow {
	if now.IsZero() {
		now = time.Now().UTC()
	}
	if decision == "" {
		decision = "REVERT"
		if it.Kept {
			decision = "KEEP"
		}
	}
	c := it.Candidate
	return JournalRow{
		Schema:        JournalSchema,
		Tick:          tick,
		Date:          now.UTC().Format("2006-01-02"),
		GeneratedAt:   now.UTC().Format(time.RFC3339),
		Commit:        commit,
		Lever:         c.Lever,
		Metric:        c.Metric,
		Kind:          c.Kind,
		Decision:      decision,
		Kept:          it.Kept,
		AgentArm:      c.Kind == ReprojectKind || c.Kind == HarvestKind || c.Kind == RouteFloor,
		Baseline:      it.BaselineValue,
		Replayed:      it.ReplayedValue,
		MeasuredDelta: it.MeasuredDelta,
		BreakerCount:  breaker,
		WakeupAt:      wake.At,
		Reason:        it.Reason,
	}
}

// AppendJournalLine renders a journal row as one JSONL line.
func AppendJournalLine(row JournalRow) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ParseJournal parses committed dojo-RSI JSONL, skipping torn or foreign rows.
func ParseJournal(content string) []JournalRow {
	var rows []JournalRow
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row JournalRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.Schema != JournalSchema {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// FoldTrend summarizes the KEEP/REVERT journal for the CI feed.
func FoldTrend(rows []JournalRow, now time.Time) JournalTrend {
	if now.IsZero() {
		now = time.Now().UTC()
	}
	tr := JournalTrend{Schema: TrendSchema, GeneratedAt: now.UTC().Format(time.RFC3339), Rows: len(rows)}
	for i := range rows {
		r := rows[i]
		switch r.Decision {
		case "KEEP":
			tr.Keep++
			if r.Kind == RecalibrateKind {
				tr.MechanicalKeep++
			}
		case "ESCALATE":
			tr.Escalate++
			if r.Kind == RouteFloor {
				tr.FloorEscalates++
			}
		default:
			tr.Revert++
		}
		if r.AgentArm {
			tr.AgentRoutes++
		}
		switch r.Kind {
		case ReprojectKind:
			tr.ReprojectRoutes++
		case HarvestKind:
			tr.HarvestRoutes++
		}
	}
	if len(rows) > 0 {
		latest := rows[len(rows)-1]
		tr.Latest = &latest
	}
	tr.Summary = fmt.Sprintf("dojo-RSI trend: %d row(s), KEEP %d (%d mechanical), REVERT %d, ESCALATE %d; agent routes reproject=%d harvest=%d floor-escalate=%d",
		tr.Rows, tr.Keep, tr.MechanicalKeep, tr.Revert, tr.Escalate, tr.ReprojectRoutes, tr.HarvestRoutes, tr.FloorEscalates)
	return tr
}

// TreeChangedWithin is the REPROJECT path gate: every changed path must be within
// the candidate's declared path set. Entries ending in "/" or "/**" are treated
// as prefixes; other entries are exact files.
func TreeChangedWithin(changed, declared []string) bool {
	if len(changed) == 0 || len(declared) == 0 {
		return false
	}
	for _, ch := range changed {
		if !pathAllowed(ch, declared) {
			return false
		}
	}
	return true
}

func pathAllowed(p string, declared []string) bool {
	p = cleanSlashPath(p)
	if p == "." || strings.HasPrefix(p, "../") {
		return false
	}
	for _, d := range declared {
		raw := strings.TrimSpace(strings.ReplaceAll(d, "\\", "/"))
		globPrefix := strings.HasSuffix(raw, "/**")
		dirPrefix := strings.HasSuffix(raw, "/")
		d = cleanSlashPath(d)
		switch {
		case globPrefix:
			prefix := strings.TrimSuffix(d, "/**") + "/"
			if strings.HasPrefix(p, prefix) {
				return true
			}
		case dirPrefix:
			prefix := strings.TrimSuffix(d, "/") + "/"
			if strings.HasPrefix(p, prefix) {
				return true
			}
		case p == d:
			return true
		}
	}
	return false
}

func cleanSlashPath(p string) string {
	p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
	p = strings.TrimPrefix(p, "./")
	if p == "" {
		return "."
	}
	return path.Clean(p)
}

func lastCellTouch(rows []JournalRow, lever, metric string) (JournalRow, bool) {
	var out JournalRow
	found := false
	for _, r := range rows {
		if r.Lever != lever || r.Metric != metric {
			continue
		}
		if !found || newer(r, out) {
			out, found = r, true
		}
	}
	return out, found
}

func newer(a, b JournalRow) bool {
	ta, oka := parseStamp(a.GeneratedAt, a.Date)
	tb, okb := parseStamp(b.GeneratedAt, b.Date)
	if oka && okb {
		return ta.After(tb)
	}
	if oka != okb {
		return oka
	}
	return a.Tick > b.Tick
}

func parseStamp(generatedAt, date string) (time.Time, bool) {
	if t, err := time.Parse(time.RFC3339, generatedAt); err == nil {
		return t.UTC(), true
	}
	if t, err := time.Parse("2006-01-02", date); err == nil {
		return t.UTC(), true
	}
	return time.Time{}, false
}

func valueWeight(c Recal) float64 {
	v := c.CalibErr / dojo.MaxCalibErr
	if c.Kind == RouteUnmeasured {
		v = 0
	}
	return clamp01(v)
}

func selectReason(s ScoredCell, recheckDays int) string {
	cell := s.Candidate.Lever + "/" + s.Candidate.Metric
	if s.LastTouched == "" {
		return fmt.Sprintf("%s never touched; value %.3f", cell, s.ValueWeight)
	}
	if s.Saturated {
		return fmt.Sprintf("%s touched %s (%.1fd ago, recheck %dd); fresh, skip to avoid thrash", cell, s.LastTouched, s.AgeDays, recheckDays)
	}
	return fmt.Sprintf("%s touched %s (%.1fd ago, recheck %dd); stale enough to revisit", cell, s.LastTouched, s.AgeDays, recheckDays)
}

func clamp01(f float64) float64 {
	if f < 0 {
		return 0
	}
	if f > 1 {
		return 1
	}
	return f
}

// MarshalTrendText renders the trend as a compact text card for Slack/CI.
func MarshalTrendText(tr JournalTrend) string {
	var b bytes.Buffer
	b.WriteString(tr.Summary)
	if tr.Latest != nil {
		fmt.Fprintf(&b, "\nlatest: tick %d %s/%s %s -> %s",
			tr.Latest.Tick, tr.Latest.Lever, tr.Latest.Metric, tr.Latest.Kind, tr.Latest.Decision)
	}
	return b.String()
}
