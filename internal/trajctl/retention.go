package trajctl

// retention.go — issue #2570, the retention/compaction + schema-pinning rung of
// the trajectory-control epic (#2533). The append-only ledger (trajctl.go) grows
// one row per scorer per objective per turn end forever; a steering substrate that
// bloats gets disabled in practice, and then nothing steers. This file bounds the
// ledger two ways without changing its schema:
//
//   - Retention/compaction: a CLOSED objective (met|abandoned) whose most recent
//     activity is older than the retention window collapses its many per-turn score
//     rows into ONE summary row per scoring method. The summary carries the
//     compaction-invariant reduction of that method's curve — count, value sum,
//     first/last value and their timestamps — so a fold over the compacted history
//     matches the fold over the full history within float tolerance, and CurveFor
//     still folds a curve for the closed objective (its Latest is preserved).
//   - A documented growth bound: after compaction the ledger's steady-state row
//     count is O(open objectives × turns retained) + O(closed objectives × methods)
//     — the closed tail is bounded by ONE row per (objective, method) regardless of
//     how long the objective ran. The active file is additionally byte-bounded at
//     the write site by jsonlledger.AppendBounded (DefaultActiveBytes rotation).
//
// Schema pinning + migration stance: summary rows ride the SAME pinned Schema
// ("fak-trajctl/1", SchemaVersion 1) as every other ledger row — compaction is an
// in-schema REDUCTION, not a format change. The migration stance is forward-only
// and additive: a reader pins to Schema and ParseLedger drops any foreign- or
// newer-schema row fail-open (a torn or newer append can never poison the fold), so
// a future /2 bump is introduced beside /1 and old readers simply skip it. No
// in-place row rewrite is ever required.

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

const (
	// KindSummary is the ledger row kind for a compacted objective's per-method
	// summary. It carries a [SummaryRow] and rides the pinned Schema.
	KindSummary = "summary"

	// SchemaVersion is the integer major version encoded in Schema
	// ("fak-trajctl/1"). It is exposed so a consumer can assert the pin without
	// string-parsing the schema id.
	SchemaVersion = 1

	// DefaultRetentionDays is the compaction window used when CompactOptions leaves
	// RetentionDays unset: a closed objective idle longer than this compacts.
	DefaultRetentionDays = 14

	millisPerDay = int64(24 * 60 * 60 * 1000)
)

// IsCurrentSchema reports whether schema is the pinned trajctl ledger schema. A row
// carrying any other schema is foreign or newer and is skipped fail-open by
// ParseLedger — the forward-only migration stance in one predicate.
func IsCurrentSchema(schema string) bool { return schema == Schema }

// SummaryRow is the compacted stand-in for a closed objective's many score rows for
// one scoring method. It preserves the fold: Count and Sum reproduce the mean, and
// First/Last (with their timestamps) reproduce the curve's endpoints so CurveFor's
// Latest survives compaction. It is not a witnessed score — it is the reduction of
// the score rows that were collapsed, so it never gates steering.
type SummaryRow struct {
	ObjectiveID     string      `json:"objective_id"`
	Method          string      `json:"method"`
	Version         string      `json:"version"`
	Witness         WitnessRung `json:"witness"`
	Count           int         `json:"count"`
	Sum             float64     `json:"sum"`
	First           float64     `json:"first"`
	Last            float64     `json:"last"`
	FirstUnixMillis int64       `json:"first_unix_millis,omitempty"`
	LastUnixMillis  int64       `json:"last_unix_millis,omitempty"`
}

// Mean is the average value across the collapsed score rows.
func (r SummaryRow) Mean() float64 {
	if r.Count == 0 {
		return 0
	}
	return r.Sum / float64(r.Count)
}

// SummaryRecord builds a ledger row for a SummaryRow.
func SummaryRecord(s SummaryRow) Row {
	return Row{Schema: Schema, Kind: KindSummary, Summary: &s}
}

// validateSummary checks that a summary row is well-formed enough to fold.
func validateSummary(s SummaryRow) error {
	if s.ObjectiveID == "" {
		return errors.New("trajctl: summary objective id is required")
	}
	if s.Method == "" {
		return errors.New("trajctl: summary method is required")
	}
	if s.Version == "" {
		return errors.New("trajctl: summary version is required")
	}
	if !validWitness(s.Witness) {
		return fmt.Errorf("trajctl: invalid witness rung %q", s.Witness)
	}
	if s.Count <= 0 {
		return fmt.Errorf("trajctl: summary count %d must be positive", s.Count)
	}
	return nil
}

// MethodDigest is the compaction-invariant reduction of one method's history for one
// objective — the fold that must round-trip across compaction. Count and Sum give
// the mean; First/Last (with their timestamps) give the curve endpoints.
type MethodDigest struct {
	ObjectiveID     string
	Method          string
	Count           int
	Sum             float64
	First           float64
	Last            float64
	FirstUnixMillis int64
	LastUnixMillis  int64
}

// Mean is the average value across the digested rows.
func (d MethodDigest) Mean() float64 {
	if d.Count == 0 {
		return 0
	}
	return d.Sum / float64(d.Count)
}

// methodAgg accumulates score points and summary rows into one reduction.
type methodAgg struct {
	version                 string
	witness                 WitnessRung
	count                   int
	sum                     float64
	first, last             float64
	firstMillis, lastMillis int64
	hasFirst, hasLast       bool
}

func (a *methodAgg) addPoint(value float64, version string, witness WitnessRung, millis int64) {
	a.count++
	a.sum += value
	if !a.hasFirst || millis < a.firstMillis {
		a.first, a.firstMillis, a.hasFirst = value, millis, true
	}
	if !a.hasLast || millis >= a.lastMillis {
		a.last, a.lastMillis, a.hasLast = value, millis, true
		a.version, a.witness = version, witness
	}
}

func (a *methodAgg) addSummary(s SummaryRow) {
	a.count += s.Count
	a.sum += s.Sum
	if !a.hasFirst || s.FirstUnixMillis < a.firstMillis {
		a.first, a.firstMillis, a.hasFirst = s.First, s.FirstUnixMillis, true
	}
	if !a.hasLast || s.LastUnixMillis >= a.lastMillis {
		a.last, a.lastMillis, a.hasLast = s.Last, s.LastUnixMillis, true
		a.version, a.witness = s.Version, s.Witness
	}
}

const aggKeySep = "\x00"

func aggKey(objectiveID, method string) string { return objectiveID + aggKeySep + method }

func splitAggKey(k string) (objectiveID, method string) {
	if i := strings.IndexByte(k, 0); i >= 0 {
		return k[:i], k[i+1:]
	}
	return k, ""
}

// DigestLedger folds every score and summary row into one MethodDigest per
// (objective, method), in (objective id, method) order. It is the deterministic
// fold the compaction round-trip is measured against: DigestLedger(full) and
// DigestLedger(Compact(full)) agree within float tolerance because a summary row
// carries exactly the count, sum and endpoints the collapsed score rows contributed.
func DigestLedger(rows []Row) []MethodDigest {
	aggs := map[string]*methodAgg{}
	order := make([]string, 0)
	get := func(objectiveID, method string) *methodAgg {
		k := aggKey(objectiveID, method)
		a, ok := aggs[k]
		if !ok {
			a = &methodAgg{}
			aggs[k] = a
			order = append(order, k)
		}
		return a
	}
	for _, row := range rows {
		switch row.Kind {
		case KindScore:
			if row.Score != nil {
				get(row.Score.ObjectiveID, row.Score.Method).addPoint(row.Score.Value, row.Score.Version, row.Score.Witness, row.Score.UnixMillis)
			}
		case KindSummary:
			if row.Summary != nil {
				get(row.Summary.ObjectiveID, row.Summary.Method).addSummary(*row.Summary)
			}
		}
	}
	out := make([]MethodDigest, 0, len(order))
	for _, k := range order {
		a := aggs[k]
		obj, method := splitAggKey(k)
		out = append(out, MethodDigest{
			ObjectiveID:     obj,
			Method:          method,
			Count:           a.count,
			Sum:             a.sum,
			First:           a.first,
			Last:            a.last,
			FirstUnixMillis: a.firstMillis,
			LastUnixMillis:  a.lastMillis,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ObjectiveID != out[j].ObjectiveID {
			return out[i].ObjectiveID < out[j].ObjectiveID
		}
		return out[i].Method < out[j].Method
	})
	return out
}

// CompactOptions parameterizes a compaction pass. NowUnixMillis is the clock the
// retention window is measured against (the impurity stays at the call site);
// RetentionDays is the closed-objective idle window, defaulting to
// DefaultRetentionDays when non-positive.
type CompactOptions struct {
	NowUnixMillis int64
	RetentionDays int
}

// CompactionStats reports what a compaction pass did, for observability.
type CompactionStats struct {
	ScoreRowsCompacted  int
	SummaryRowsWritten  int
	ObjectivesCompacted int
}

// Compact collapses the score history of every closed (met|abandoned) objective
// whose most recent activity is older than the retention window into one summary
// row per scoring method, leaving every open or recently-closed objective and every
// non-score row untouched. It is pure and deterministic: it reads the folded status
// from rows, never touches a clock (NowUnixMillis is injected), and emits the merged
// summaries in (objective id, method) order. Running it again is a fold-preserving
// no-op — an existing summary row is re-folded into an identical summary — so a
// nightly compactor is safe to run repeatedly.
func Compact(rows []Row, opts CompactOptions) ([]Row, CompactionStats) {
	days := opts.RetentionDays
	if days <= 0 {
		days = DefaultRetentionDays
	}
	cutoff := opts.NowUnixMillis - int64(days)*millisPerDay

	statuses := foldStatuses(rows)

	lastActivity := map[string]int64{}
	seen := map[string]bool{}
	note := func(objectiveID string, millis int64) {
		if !seen[objectiveID] || millis > lastActivity[objectiveID] {
			lastActivity[objectiveID], seen[objectiveID] = millis, true
		}
	}
	for _, row := range rows {
		switch row.Kind {
		case KindScore:
			if row.Score != nil {
				note(row.Score.ObjectiveID, row.Score.UnixMillis)
			}
		case KindSummary:
			if row.Summary != nil {
				note(row.Summary.ObjectiveID, row.Summary.LastUnixMillis)
			}
		}
	}
	eligible := func(objectiveID string) bool {
		st, ok := statuses[objectiveID]
		if !ok || (st != StatusMet && st != StatusAbandoned) {
			return false
		}
		return seen[objectiveID] && lastActivity[objectiveID] < cutoff
	}

	aggs := map[string]*methodAgg{}
	order := make([]string, 0)
	get := func(objectiveID, method string) *methodAgg {
		k := aggKey(objectiveID, method)
		a, ok := aggs[k]
		if !ok {
			a = &methodAgg{}
			aggs[k] = a
			order = append(order, k)
		}
		return a
	}

	out := make([]Row, 0, len(rows))
	stats := CompactionStats{}
	compacted := map[string]bool{}
	for _, row := range rows {
		switch row.Kind {
		case KindScore:
			if row.Score != nil && eligible(row.Score.ObjectiveID) {
				get(row.Score.ObjectiveID, row.Score.Method).addPoint(row.Score.Value, row.Score.Version, row.Score.Witness, row.Score.UnixMillis)
				stats.ScoreRowsCompacted++
				compacted[row.Score.ObjectiveID] = true
				continue
			}
		case KindSummary:
			if row.Summary != nil && eligible(row.Summary.ObjectiveID) {
				get(row.Summary.ObjectiveID, row.Summary.Method).addSummary(*row.Summary)
				compacted[row.Summary.ObjectiveID] = true
				continue
			}
		}
		out = append(out, row)
	}

	sort.Strings(order)
	for _, k := range order {
		a := aggs[k]
		obj, method := splitAggKey(k)
		out = append(out, SummaryRecord(SummaryRow{
			ObjectiveID:     obj,
			Method:          method,
			Version:         a.version,
			Witness:         a.witness,
			Count:           a.count,
			Sum:             a.sum,
			First:           a.first,
			Last:            a.last,
			FirstUnixMillis: a.firstMillis,
			LastUnixMillis:  a.lastMillis,
		}))
		stats.SummaryRowsWritten++
	}
	stats.ObjectivesCompacted = len(compacted)
	return out, stats
}

// foldStatuses returns the latest status per objective id, cheaply, without the
// full curve fold — the only fact Compact needs to decide eligibility.
func foldStatuses(rows []Row) map[string]ObjectiveStatus {
	out := map[string]ObjectiveStatus{}
	for _, row := range rows {
		if row.Kind == KindObjective && row.Objective != nil {
			out[row.Objective.ID] = row.Objective.Status
		}
	}
	return out
}

// summaryPoints reconstructs the representative curve points for a compacted
// method so CurveFor still folds a curve for a closed objective: the first and last
// witnessed values at their timestamps (or the single endpoint when only one row was
// collapsed). Fold appends these into State.Scores so the compacted objective's
// Latest matches its pre-compaction Latest.
func summaryPoints(s SummaryRow) []ScoreRow {
	base := ScoreRow{ObjectiveID: s.ObjectiveID, Method: s.Method, Version: s.Version, Witness: s.Witness}
	if s.Count <= 1 {
		p := base
		p.Value, p.UnixMillis = s.Last, s.LastUnixMillis
		return []ScoreRow{p}
	}
	first := base
	first.Value, first.UnixMillis = s.First, s.FirstUnixMillis
	last := base
	last.Value, last.UnixMillis = s.Last, s.LastUnixMillis
	return []ScoreRow{first, last}
}
