package worktype

import (
	"sort"
	"strings"
)

const SpendSchema = "fak-worktype-attribution/1"

type SpendRow struct {
	SessionID    string   `json:"session_id"`
	TraceID      string   `json:"trace_id,omitempty"`
	PatternID    string   `json:"pattern_id,omitempty"`
	Subpatterns  []string `json:"subpatterns,omitempty"`
	Tokens       uint64   `json:"tokens"`
	Cost         float64  `json:"cost"`
	Outcome      string   `json:"outcome"`
	OutcomeProof string   `json:"outcome_provenance,omitempty"`
}
type SpendGroup struct {
	PatternID    string         `json:"pattern_id"`
	Sessions     int            `json:"sessions"`
	Tokens       uint64         `json:"tokens"`
	Cost         float64        `json:"cost"`
	Outcomes     map[string]int `json:"outcomes"`
	AcceptedRate float64        `json:"accepted_witness_rate"`
	Drilldown    []SpendRow     `json:"drilldown"`
}
type SpendCoverage struct {
	RowCount       int     `json:"total_sessions"`
	ClassifiedRows int     `json:"covered_sessions"`
	TotalTokens    uint64  `json:"total_tokens"`
	CoveredTokens  uint64  `json:"covered_tokens"`
	SessionRate    float64 `json:"session_rate"`
	TokenRate      float64 `json:"token_rate"`
}
type SpendReport struct {
	Schema            string        `json:"schema"`
	CostUnit          string        `json:"cost_unit"`
	Coverage          SpendCoverage `json:"coverage"`
	HighestSpendClass string        `json:"highest_spend_class"`
	Groups            []SpendGroup  `json:"groups"`
}

func FoldSpend(rows []SpendRow, valid map[string]bool) SpendReport {
	r := SpendReport{Schema: SpendSchema, CostUnit: "input_token_equivalent"}
	gs := map[string]*SpendGroup{}
	for _, x := range rows {
		x.SessionID = strings.TrimSpace(x.SessionID)
		x.TraceID = strings.TrimSpace(x.TraceID)
		x.PatternID = strings.TrimSpace(x.PatternID)
		if x.Outcome == "" {
			x.Outcome = "unknown"
		}
		if !valid[x.PatternID] {
			x.PatternID = "unknown"
		}
		r.Coverage.RowCount++
		r.Coverage.TotalTokens += x.Tokens
		if x.PatternID != "unknown" {
			r.Coverage.ClassifiedRows++
			r.Coverage.CoveredTokens += x.Tokens
		}
		g := gs[x.PatternID]
		if g == nil {
			g = &SpendGroup{PatternID: x.PatternID, Outcomes: map[string]int{}}
			gs[x.PatternID] = g
		}
		g.Sessions++
		g.Tokens += x.Tokens
		g.Cost += x.Cost
		g.Outcomes[x.Outcome]++
		g.Drilldown = append(g.Drilldown, x)
	}
	if r.Coverage.RowCount > 0 {
		r.Coverage.SessionRate = float64(r.Coverage.ClassifiedRows) / float64(r.Coverage.RowCount)
	}
	if r.Coverage.TotalTokens > 0 {
		r.Coverage.TokenRate = float64(r.Coverage.CoveredTokens) / float64(r.Coverage.TotalTokens)
	}
	for _, g := range gs {
		g.AcceptedRate = float64(g.Outcomes["accepted_witness"]) / float64(g.Sessions)
		sort.Slice(g.Drilldown, func(i, j int) bool { return g.Drilldown[i].SessionID < g.Drilldown[j].SessionID })
		r.Groups = append(r.Groups, *g)
	}
	sort.Slice(r.Groups, func(i, j int) bool {
		if r.Groups[i].Cost != r.Groups[j].Cost {
			return r.Groups[i].Cost > r.Groups[j].Cost
		}
		return r.Groups[i].PatternID < r.Groups[j].PatternID
	})
	if len(r.Groups) > 0 {
		r.HighestSpendClass = r.Groups[0].PatternID
	}
	return r
}
func SeedPatternIDs() map[string]bool {
	m := map[string]bool{}
	for _, p := range SeedPatternCatalog().Patterns {
		m[p.ID] = true
	}
	return m
}
