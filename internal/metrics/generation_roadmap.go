package metrics

import (
	"strconv"
	"strings"
)

// Generational roadmap dashboard (issue #1660, gen/second-next).
//
// A long-horizon portfolio cannot be read as one milestone percent-complete bar:
// the horizons past "now" are architectural OPTIONS, not deliverables converging on
// 100%. Their honest measure is the same shape internal/worktype draws for an
// ongoing program — a FRONTIER (the leading edge witnessed at that horizon) and a
// TREND (is that edge still advancing?) — plus the lifecycle signals that move an
// option between horizons: promotion candidates (options with promotion evidence),
// stale assumptions (the demotion/retirement signal), and current velocity.
//
// This file is the SHAPE of that dashboard: a pure, stdlib-only lifecycle model
// that names the closed set of rows and renders a deterministic table from a
// caller-supplied snapshot. It is architectural exploration behind no default
// exposure — nothing here is wired into a live command or a runtime feature gate;
// a future agent promotes it by folding a real generation snapshot (from
// internal/devindex generations + the milestone/velocity ledgers) into GenerationRoadmap
// and mounting Render on an observability surface.
//
// Generation stays orthogonal to three things, and the dashboard is built to keep
// them separate (see OrthogonalityNote, rendered in the header):
//
//   - PRIORITY — a horizon is not a priority. gen/future is a horizon label, not a
//     value judgment; a gen/future row can carry higher-priority work than a gen/now
//     row. The dashboard sorts columns by horizon (now -> future), never by priority.
//   - SHARED TRUNK — every generation lives on main. There is no branch or worktree
//     per generation; the roadmap reads one trunk, so a column is a label filter, not
//     a separate line of history.
//   - RUNTIME FEATURE GATES — a gen/* label describes WHEN an option is expected to
//     mature, not WHETHER it is exposed at runtime. Default exposure is decided by an
//     explicit feature gate, independent of the generation label.
//
// Promotion / demotion / invalidating-assumption for this artifact itself:
//   - Promotion evidence: a real generation snapshot renders through Render() on an
//     operator surface and an agent reads it instead of the whole generation epic.
//   - Demotion/retirement evidence: retire this shape if the milestone roadmap
//     (internal/milestonereport) grows a horizon-aware view that subsumes it, or if
//     the gen/* label vocabulary is dropped.
//   - Invalidating assumption: that the four-horizon vocabulary (now/next/second-next/
//     future) and its now->future ordering stay stable and cheap to read. If the
//     horizon set drifts, RoadmapGenerations and the column ordering must move first.

// RoadmapRow is the closed set of rows the generational roadmap dashboard shows.
// A value outside this set is a bug, not a new row — the same closed-vocabulary
// discipline internal/worktype applies to a work class.
type RoadmapRow string

const (
	// RowFrontier is the leading edge witnessed at a horizon — the best/most
	// advanced option currently active there, not a completion percent.
	RowFrontier RoadmapRow = "frontier"
	// RowTrend is whether the horizon's frontier is advancing, holding, or slipping.
	RowTrend RoadmapRow = "trend"
	// RowPromotionCandidates counts options at a horizon that carry promotion
	// evidence — ready to move toward a nearer horizon.
	RowPromotionCandidates RoadmapRow = "promotion-candidates"
	// RowStaleAssumptions counts options whose witness has gone stale — the
	// demotion/retirement signal.
	RowStaleAssumptions RoadmapRow = "stale-assumptions"
	// RowVelocity is the current witnessed throughput at a horizon (work items
	// landed per period), so a stalled horizon is visible against an advancing one.
	RowVelocity RoadmapRow = "velocity"
)

// RoadmapRows is the ordered, closed row vocabulary. The renderer iterates this so
// a new row is added here (and in the column struct), never in the render loop.
var RoadmapRows = []RoadmapRow{
	RowFrontier,
	RowTrend,
	RowPromotionCandidates,
	RowStaleAssumptions,
	RowVelocity,
}

// Label is the short human label for a row, for a render line or a Slack card.
func (r RoadmapRow) Label() string {
	switch r {
	case RowFrontier:
		return "Frontier"
	case RowTrend:
		return "Trend"
	case RowPromotionCandidates:
		return "Promotion candidates"
	case RowStaleAssumptions:
		return "Stale assumptions"
	case RowVelocity:
		return "Velocity"
	default:
		return string(r)
	}
}

// RoadmapGenerations is the ordered, closed horizon vocabulary the dashboard shows
// as columns — now -> future, matching internal/devindex generationRank. Ordering
// is by horizon, never by priority (see OrthogonalityNote).
var RoadmapGenerations = []string{"now", "next", "second-next", "future"}

// Trend is the closed set of frontier-direction values for RowTrend.
type Trend string

const (
	// TrendAdvancing means the horizon's frontier moved forward in the period.
	TrendAdvancing Trend = "advancing"
	// TrendHolding means the frontier held — neither advanced nor slipped.
	TrendHolding Trend = "holding"
	// TrendSlipping means the frontier regressed or a promotion candidate demoted.
	TrendSlipping Trend = "slipping"
	// TrendUnknown means no witness was available to decide the trend.
	TrendUnknown Trend = "unknown"
)

// GenerationColumn is one horizon column of the dashboard: the generation identity
// plus one cell for each RoadmapRow. It is a plain data snapshot — a caller folds
// real telemetry into it; this package does not read disk.
type GenerationColumn struct {
	// Stream is the horizon key (one of RoadmapGenerations): now|next|second-next|future.
	Stream string `json:"stream"`
	// Label is the gen/* label for the horizon (e.g. "gen/second-next").
	Label string `json:"label"`
	// Frontier is the leading edge witnessed at this horizon (RowFrontier).
	Frontier string `json:"frontier"`
	// Trend is the frontier direction (RowTrend).
	Trend Trend `json:"trend"`
	// PromotionCandidates counts options with promotion evidence (RowPromotionCandidates).
	PromotionCandidates int `json:"promotion_candidates"`
	// StaleAssumptions counts options whose witness went stale (RowStaleAssumptions).
	StaleAssumptions int `json:"stale_assumptions"`
	// Velocity is the witnessed throughput at this horizon, items/period (RowVelocity).
	Velocity float64 `json:"velocity"`
}

// GenerationRoadmap is the full dashboard snapshot: one column per horizon, plus
// the period the velocity/trend cells were measured over.
type GenerationRoadmap struct {
	// Period names the measurement window for the trend/velocity cells (e.g. "7d").
	Period string `json:"period"`
	// Columns are the per-horizon snapshots, expected in RoadmapGenerations order.
	Columns []GenerationColumn `json:"columns"`
}

// OrthogonalityNote is the one-paragraph statement, rendered in the dashboard
// header, of how a generation label stays orthogonal to priority, shared trunk,
// and runtime feature gates. It is exported so the render and a test both bind to
// the same text.
const OrthogonalityNote = "Generation is a horizon, not a priority: gen/future is not lower-priority by default. " +
	"Every generation lives on the shared trunk (main) — no branch or worktree per generation. " +
	"A gen/* label says WHEN an option is expected to mature, not WHETHER it is exposed at runtime — " +
	"default exposure is an explicit feature gate, independent of the generation label."

// cell returns the string form of one row's value for a column.
func (c GenerationColumn) cell(row RoadmapRow) string {
	switch row {
	case RowFrontier:
		if c.Frontier == "" {
			return "-"
		}
		return c.Frontier
	case RowTrend:
		if c.Trend == "" {
			return string(TrendUnknown)
		}
		return string(c.Trend)
	case RowPromotionCandidates:
		return strconv.Itoa(c.PromotionCandidates)
	case RowStaleAssumptions:
		return strconv.Itoa(c.StaleAssumptions)
	case RowVelocity:
		return strconv.FormatFloat(c.Velocity, 'f', -1, 64)
	default:
		return "-"
	}
}

// Render produces a deterministic text dashboard: an orthogonality header, then a
// row-per-metric x column-per-generation table. It is pure (no clock, no disk) so a
// test can assert its exact bytes and an observability surface can mount it.
func (rm GenerationRoadmap) Render() string {
	var b strings.Builder
	b.WriteString("Generational roadmap (period: ")
	if rm.Period == "" {
		b.WriteString("n/a")
	} else {
		b.WriteString(rm.Period)
	}
	b.WriteString(")\n")
	b.WriteString(OrthogonalityNote)
	b.WriteString("\n\n")

	// Header row: metric label + one column per horizon, in RoadmapGenerations order.
	cols := rm.orderedColumns()
	b.WriteString(pad("Row", roadmapLabelWidth))
	for _, c := range cols {
		b.WriteString(" | ")
		b.WriteString(pad(horizonHeader(c), roadmapCellWidth))
	}
	b.WriteString("\n")

	for _, row := range RoadmapRows {
		b.WriteString(pad(row.Label(), roadmapLabelWidth))
		for _, c := range cols {
			b.WriteString(" | ")
			b.WriteString(pad(c.cell(row), roadmapCellWidth))
		}
		b.WriteString("\n")
	}
	return b.String()
}

// orderedColumns returns the roadmap's columns in RoadmapGenerations order,
// synthesizing an empty column for any missing horizon so the dashboard always
// shows the full closed horizon set.
func (rm GenerationRoadmap) orderedColumns() []GenerationColumn {
	byStream := make(map[string]GenerationColumn, len(rm.Columns))
	for _, c := range rm.Columns {
		byStream[c.Stream] = c
	}
	out := make([]GenerationColumn, 0, len(RoadmapGenerations))
	for _, stream := range RoadmapGenerations {
		if c, ok := byStream[stream]; ok {
			out = append(out, c)
			continue
		}
		out = append(out, GenerationColumn{Stream: stream, Label: "gen/" + stream, Trend: TrendUnknown})
	}
	return out
}

func horizonHeader(c GenerationColumn) string {
	if c.Label != "" {
		return c.Label
	}
	if c.Stream != "" {
		return "gen/" + c.Stream
	}
	return "gen/?"
}

const (
	roadmapLabelWidth = 20
	roadmapCellWidth  = 16
)

func pad(s string, width int) string {
	if len(s) >= width {
		return s
	}
	return s + strings.Repeat(" ", width-len(s))
}
