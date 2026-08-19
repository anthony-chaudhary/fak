package agentquery

import (
	"math"
	"sort"
	"time"
)

const GroupSchema = "fak-agent-groups/2"

type GroupRow struct {
	Lane         *string  `json:"lane"`
	State        string   `json:"state"`
	Count        int      `json:"count"`
	MinElapsedMS *int64   `json:"min_elapsed_ms"`
	MaxElapsedMS *int64   `json:"max_elapsed_ms"`
	SumElapsedMS *int64   `json:"sum_elapsed_ms"`
	AvgElapsedMS *float64 `json:"avg_elapsed_ms"`
}

type GroupMetadata struct {
	Schema      string        `json:"schema"`
	GroupBy     []string      `json:"group_by"`
	Source      string        `json:"source"`
	Since       string        `json:"since"`
	ObservedAt  string        `json:"observed_at"`
	InputRows   int           `json:"input_rows"`
	MatchedRows int           `json:"matched_rows"`
	History     *SourceHealth `json:"history"`
	Plan        QueryPlan     `json:"plan"`
}

type GroupResult struct {
	Metadata GroupMetadata `json:"metadata"`
	Rows     []GroupRow    `json:"rows"`
}

// GroupLaneState folds rows started within [since, observed] into deterministic lane/state groups.
// Missing lane remains a typed null and sorts after named lanes.
func GroupLaneState(rows []Row, since, observed time.Time, source string, health *SourceHealth) GroupResult {
	plan, _ := GroupedPlan(observed.Sub(since))
	return GroupLaneStatePlan(rows, plan, observed, source, health)
}

// GroupLaneStatePlan executes the validated closed plan used by both flags and query text.
func GroupLaneStatePlan(rows []Row, plan QueryPlan, observed time.Time, source string, health *SourceHealth) GroupResult {
	since := observed.Add(-plan.Window())
	type key struct {
		lane, state string
		hasLane     bool
	}
	groups := map[key]*GroupRow{}
	numericCounts := map[key]int64{}
	matched := 0
	for _, r := range rows {
		timeValue := r.StartedAt
		if plan.TimeColumn == "observed_at" {
			timeValue = &r.ObservedAt
		}
		if timeValue == nil {
			continue
		}
		rowTime, err := time.Parse(time.RFC3339, *timeValue)
		if err != nil || rowTime.Before(since) || rowTime.After(observed) {
			continue
		}
		k := key{state: r.State}
		if r.Lane != nil {
			k.lane = *r.Lane
			k.hasLane = true
		}
		g := groups[k]
		if g == nil {
			g = &GroupRow{State: r.State}
			if k.hasLane {
				lane := k.lane
				g.Lane = &lane
			}
			groups[k] = g
		}
		g.Count++
		matched++
		if r.ElapsedMS != nil {
			v := *r.ElapsedMS
			if g.MinElapsedMS == nil || v < *g.MinElapsedMS {
				x := v
				g.MinElapsedMS = &x
			}
			if g.MaxElapsedMS == nil || v > *g.MaxElapsedMS {
				x := v
				g.MaxElapsedMS = &x
			}
			if numericCounts[k] < 0 {
				// A prior overflow keeps sum and average typed null.
			} else if g.SumElapsedMS == nil {
				x := v
				g.SumElapsedMS = &x
			} else if (v > 0 && *g.SumElapsedMS > math.MaxInt64-v) || (v < 0 && *g.SumElapsedMS < math.MinInt64-v) {
				g.SumElapsedMS = nil
				numericCounts[k] = -1
			} else if numericCounts[k] >= 0 {
				x := *g.SumElapsedMS + v
				g.SumElapsedMS = &x
			}
			if numericCounts[k] >= 0 {
				numericCounts[k]++
			}
		}
	}
	out := make([]GroupRow, 0, len(groups))
	for k, g := range groups {
		if n := numericCounts[k]; n > 0 && g.SumElapsedMS != nil {
			avg := float64(*g.SumElapsedMS) / float64(n)
			g.AvgElapsedMS = &avg
		}
		if !hasAggregate(plan, "min_elapsed_ms") {
			g.MinElapsedMS = nil
		}
		if !hasAggregate(plan, "sum_elapsed_ms") {
			g.SumElapsedMS = nil
		}
		if !hasAggregate(plan, "avg_elapsed_ms") {
			g.AvgElapsedMS = nil
		}
		out = append(out, *g)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Lane == nil {
			return false
		}
		if out[j].Lane == nil {
			return true
		}
		if *out[i].Lane != *out[j].Lane {
			return *out[i].Lane < *out[j].Lane
		}
		return out[i].State < out[j].State
	})
	return GroupResult{Metadata: GroupMetadata{Schema: GroupSchema, GroupBy: plan.GroupBy, Source: source, Since: since.UTC().Format(time.RFC3339), ObservedAt: observed.UTC().Format(time.RFC3339), InputRows: len(rows), MatchedRows: matched, History: health, Plan: plan}, Rows: out}
}

func hasAggregate(plan QueryPlan, want string) bool {
	for _, aggregate := range plan.Aggregates {
		if aggregate == want {
			return true
		}
	}
	return false
}
