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

func GroupedPlan(window time.Duration) (QueryPlan, error) {
	if window <= 0 || window > 3650*24*time.Hour {
		return QueryPlan{}, fmt.Errorf("history window must be within 1ns..3650d")
	}
	return QueryPlan{Schema: QueryPlanSchema, Source: "history", HistoryWindowSeconds: int64(window / time.Second), GroupBy: []string{"lane", "state"}, Aggregates: []string{"count", "max_elapsed_ms"}, OrderBy: "max_elapsed_ms_desc", TimeColumn: "started_at"}, nil
}

var groupedQueryRE = regexp.MustCompile(`(?i)^\s*select\s+lane\s*,\s*state\s*,\s*count\s*\(\s*\*\s*\)\s+as\s+agents\s*,\s*max\s*\(\s*elapsed_ms\s*\)\s+as\s+max_elapsed_ms\s+from\s+agents\s+where\s+(started_at|observed_at)\s*>=\s*now\s*\(\s*\)\s*-\s*interval\s+'([0-9]+)\s+day(?:s)?'\s+group\s+by\s+lane\s*,\s*state\s+order\s+by\s+max_elapsed_ms\s+desc\s*$`)

// ParseQuery accepts one read-only bounded aggregate shape and rejects everything else.
// Full anchoring deliberately excludes comments, semicolons, extra statements, functions,
// columns, and clauses not represented by QueryPlan.
func ParseQuery(raw string) (QueryPlan, error) {
	if len(raw) > 4096 {
		return QueryPlan{}, fmt.Errorf("query exceeds 4096-byte cap")
	}
	m := groupedQueryRE.FindStringSubmatch(strings.TrimSpace(raw))
	if m == nil {
		return QueryPlan{}, fmt.Errorf("unsupported query: want the bounded lane,state count/max shape")
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
	return plan, nil
}

func (p QueryPlan) Window() time.Duration { return time.Duration(p.HistoryWindowSeconds) * time.Second }
