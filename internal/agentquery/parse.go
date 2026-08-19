package agentquery

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const QueryPlanSchema = "fak-agent-query-plan/1"

// QueryPlan is the closed relational plan shared by CLI flags and constrained query text.
type QueryPlan struct {
	Schema               string   `json:"schema"`
	Source               string   `json:"source"`
	HistoryWindowSeconds int64    `json:"history_window_seconds"`
	GroupBy              []string `json:"group_by"`
	Aggregates           []string `json:"aggregates"`
	OrderBy              string   `json:"order_by"`
	TimeColumn           string   `json:"time_column"`
}

var baseAggregates = []string{"count", "max_elapsed_ms"}
var fullAggregates = []string{"count", "min_elapsed_ms", "max_elapsed_ms", "sum_elapsed_ms", "avg_elapsed_ms"}

func GroupedPlan(window time.Duration) (QueryPlan, error) {
	if window <= 0 || window > 3650*24*time.Hour {
		return QueryPlan{}, fmt.Errorf("history window must be within 1ns..3650d")
	}
	return QueryPlan{Schema: QueryPlanSchema, Source: "history", HistoryWindowSeconds: int64(window / time.Second), GroupBy: []string{"lane", "state"}, Aggregates: append([]string(nil), baseAggregates...), OrderBy: "max_elapsed_ms_desc", TimeColumn: "started_at"}, nil
}

var groupedQueryRE = regexp.MustCompile(`(?i)^\s*select\s+lane\s*,\s*state\s*,\s*count\s*\(\s*\*\s*\)\s+as\s+agents\s*,\s*max\s*\(\s*elapsed_ms\s*\)\s+as\s+max_elapsed_ms\s+from\s+agents\s+where\s+(started_at|observed_at)\s*>=\s*now\s*\(\s*\)\s*-\s*interval\s+'([0-9]+)\s+day(?:s)?'\s+group\s+by\s+lane\s*,\s*state\s+order\s+by\s+max_elapsed_ms\s+desc\s*$`)

var fullGroupedQueryRE = regexp.MustCompile(`(?i)^\s*select\s+lane\s*,\s*state\s*,\s*count\s*\(\s*\*\s*\)\s+as\s+agents\s*,\s*min\s*\(\s*elapsed_ms\s*\)\s+as\s+min_elapsed_ms\s*,\s*max\s*\(\s*elapsed_ms\s*\)\s+as\s+max_elapsed_ms\s*,\s*sum\s*\(\s*elapsed_ms\s*\)\s+as\s+sum_elapsed_ms\s*,\s*avg\s*\(\s*elapsed_ms\s*\)\s+as\s+avg_elapsed_ms\s+from\s+agents\s+where\s+(started_at|observed_at)\s*>=\s*now\s*\(\s*\)\s*-\s*interval\s+'([0-9]+)\s+day(?:s)?'\s+group\s+by\s+lane\s*,\s*state\s+order\s+by\s+max_elapsed_ms\s+desc\s*$`)

// ParseQuery accepts one read-only bounded aggregate shape and rejects everything else.
// Full anchoring deliberately excludes comments, semicolons, extra statements, functions,
// columns, and clauses not represented by QueryPlan.
func ParseQuery(raw string) (QueryPlan, error) {
	if len(raw) > 4096 {
		return QueryPlan{}, fmt.Errorf("query exceeds 4096-byte cap")
	}
	trimmed := strings.TrimSpace(raw)
	m := fullGroupedQueryRE.FindStringSubmatch(trimmed)
	full := m != nil
	if !full {
		m = groupedQueryRE.FindStringSubmatch(trimmed)
	}
	if m == nil {
		return QueryPlan{}, fmt.Errorf("unsupported query: want a bounded lane,state typed aggregate shape")
	}
	days, err := strconv.Atoi(m[2])
	if err != nil || days < 1 || days > 3650 {
		return QueryPlan{}, fmt.Errorf("interval must be 1..3650 days")
	}
	plan, err := GroupedPlan(time.Duration(days) * 24 * time.Hour)
	if err != nil {
		return QueryPlan{}, err
	}
	plan.TimeColumn = strings.ToLower(m[1])
	if full {
		plan.Aggregates = append([]string(nil), fullAggregates...)
	}
	return plan, nil
}

func (p QueryPlan) Window() time.Duration { return time.Duration(p.HistoryWindowSeconds) * time.Second }
