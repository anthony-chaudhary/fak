package agentquery

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

const ListPlanSchema = "fak-agent-list-plan/1"

type ListPlan struct {
	Schema        string  `json:"schema"`
	State         string  `json:"state,omitempty"`
	Liveness      string  `json:"liveness,omitempty"`
	Owner         string  `json:"owner,omitempty"`
	Host          string  `json:"host,omitempty"`
	Lane          string  `json:"lane,omitempty"`
	Group         string  `json:"group,omitempty"`
	Model         string  `json:"model,omitempty"`
	Provider      string  `json:"provider,omitempty"`
	RootID        string  `json:"root_id,omitempty"`
	ParentID      string  `json:"parent_id,omitempty"`
	StartedAfter  *string `json:"started_after"`
	StartedBefore *string `json:"started_before"`
	OrderBy       string  `json:"order_by"`
	Limit         int     `json:"limit"`
}

var listOrders = map[string]bool{"elapsed_desc": true, "elapsed_asc": true, "progress_age_desc": true, "progress_age_asc": true, "started_desc": true, "started_asc": true, "ended_desc": true, "ended_asc": true, "cost_desc": true, "cost_asc": true, "identity_asc": true, "identity_desc": true}

func ValidateListPlan(p ListPlan) error {
	if p.Schema != "" && p.Schema != ListPlanSchema {
		return fmt.Errorf("unsupported list plan schema")
	}
	if !listOrders[p.OrderBy] {
		return fmt.Errorf("unsupported order-by %q", p.OrderBy)
	}
	if p.Limit < 1 || p.Limit > 10000 {
		return fmt.Errorf("limit must be 1..10000")
	}
	for _, v := range []*string{p.StartedAfter, p.StartedBefore} {
		if v != nil {
			if _, err := time.Parse(time.RFC3339, *v); err != nil {
				return fmt.Errorf("time bounds must be RFC3339")
			}
		}
	}
	return nil
}

func ApplyListPlan(rows []Row, p ListPlan, observed time.Time) ([]Row, bool, error) {
	if p.Schema == "" {
		p.Schema = ListPlanSchema
	}
	if err := ValidateListPlan(p); err != nil {
		return nil, false, err
	}
	out := make([]Row, 0, len(rows))
	for _, r := range rows {
		if !rowMatches(r, p) {
			continue
		}
		out = append(out, r)
	}
	sort.SliceStable(out, func(i, j int) bool { return rowLess(out[i], out[j], p.OrderBy, observed) })
	truncated := len(out) > p.Limit
	if truncated {
		out = out[:p.Limit]
	}
	return out, truncated, nil
}
func rowMatches(r Row, p ListPlan) bool {
	return equalFilter(r.State, p.State) && equalFilter(r.Liveness, p.Liveness) && ptrFilter(r.Owner, p.Owner) && ptrFilter(r.Host, p.Host) && ptrFilter(r.Lane, p.Lane) && ptrFilter(r.Group, p.Group) && ptrFilter(r.Model, p.Model) && ptrFilter(r.Provider, p.Provider) && ptrFilter(r.RootID, p.RootID) && ptrFilter(r.ParentID, p.ParentID) && withinStarted(r.StartedAt, p.StartedAfter, p.StartedBefore)
}
func equalFilter(got, want string) bool { return want == "" || strings.EqualFold(got, want) }
func ptrFilter(got *string, want string) bool {
	return want == "" || (got != nil && strings.EqualFold(*got, want))
}
func withinStarted(got, after, before *string) bool {
	if after == nil && before == nil {
		return true
	}
	if got == nil {
		return false
	}
	t, e := time.Parse(time.RFC3339, *got)
	if e != nil {
		return false
	}
	if after != nil {
		a, _ := time.Parse(time.RFC3339, *after)
		if t.Before(a) {
			return false
		}
	}
	if before != nil {
		b, _ := time.Parse(time.RFC3339, *before)
		if t.After(b) {
			return false
		}
	}
	return true
}
func rowLess(a, b Row, order string, observed time.Time) bool {
	desc := strings.HasSuffix(order, "_desc")
	base := strings.TrimSuffix(strings.TrimSuffix(order, "_desc"), "_asc")
	an, bn := rowSortNull(a, base), rowSortNull(b, base)
	if an != bn {
		return !an
	}
	cmp := 0
	switch base {
	case "elapsed":
		cmp = cmpInt(a.ElapsedMS, b.ElapsedMS)
	case "progress_age":
		cmp = cmpAge(a.LastProgressAt, b.LastProgressAt, observed)
	case "started":
		cmp = cmpTime(a.StartedAt, b.StartedAt)
	case "ended":
		cmp = cmpTime(a.EndedAt, b.EndedAt)
	case "cost":
		cmp = cmpFloat(a.Cost, b.Cost)
	case "identity":
		cmp = strings.Compare(a.AgentID, b.AgentID)
	}
	if cmp == 0 {
		cmp = strings.Compare(a.AgentID, b.AgentID)
	}
	if desc {
		return cmp > 0
	}
	return cmp < 0
}
func rowSortNull(r Row, base string) bool {
	switch base {
	case "elapsed":
		return r.ElapsedMS == nil
	case "progress_age":
		return r.LastProgressAt == nil
	case "started":
		return r.StartedAt == nil
	case "ended":
		return r.EndedAt == nil
	case "cost":
		return r.Cost == nil
	}
	return false
}

func cmpOrdered[T int64 | float64](a, b *T) int {
	if a == nil && b == nil {
		return 0
	}
	if a == nil {
		return 1
	}
	if b == nil {
		return -1
	}
	if *a < *b {
		return -1
	}
	if *a > *b {
		return 1
	}
	return 0
}

func cmpInt(a, b *int64) int     { return cmpOrdered(a, b) }
func cmpFloat(a, b *float64) int { return cmpOrdered(a, b) }
func cmpTime(a, b *string) int {
	if a == nil && b == nil {
		return 0
	}
	if a == nil {
		return 1
	}
	if b == nil {
		return -1
	}
	ta, ea := time.Parse(time.RFC3339, *a)
	tb, eb := time.Parse(time.RFC3339, *b)
	if ea != nil && eb != nil {
		return 0
	}
	if ea != nil {
		return 1
	}
	if eb != nil {
		return -1
	}
	if ta.Before(tb) {
		return -1
	}
	if ta.After(tb) {
		return 1
	}
	return 0
}
func cmpAge(a, b *string, now time.Time) int {
	var aa, bb *int64
	if a != nil {
		if t, e := time.Parse(time.RFC3339, *a); e == nil {
			v := now.Sub(t).Milliseconds()
			aa = &v
		}
	}
	if b != nil {
		if t, e := time.Parse(time.RFC3339, *b); e == nil {
			v := now.Sub(t).Milliseconds()
			bb = &v
		}
	}
	return cmpInt(aa, bb)
}
