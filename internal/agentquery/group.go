package agentquery

import (
	"sort"
	"time"
)

const GroupSchema = "fak-agent-groups/1"

type GroupRow struct {
	Lane         *string `json:"lane"`
	State        string  `json:"state"`
	Count        int     `json:"count"`
	MaxElapsedMS *int64  `json:"max_elapsed_ms"`
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
		if r.ElapsedMS != nil && (g.MaxElapsedMS == nil || *r.ElapsedMS > *g.MaxElapsedMS) {
			v := *r.ElapsedMS
			g.MaxElapsedMS = &v
		}
	}
	out := make([]GroupRow, 0, len(groups))
	for _, g := range groups {
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
