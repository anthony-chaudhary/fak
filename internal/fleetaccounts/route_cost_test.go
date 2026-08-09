package fleetaccounts

import (
	"encoding/json"
	"testing"
)

func floatp(v float64) *float64 { return &v }

func TestRouteAccountRanksPlanBilledSeatsByNominalCost(t *testing.T) {
	rows := capRoster()
	rows[0].Tag, rows[1].Tag = "expensive", "cheap"
	rows[0].RouteWeight, rows[1].RouteWeight = intp(10), intp(10)
	rows[0].LiveSessions, rows[1].LiveSessions = intp(0), intp(0)
	rows[0].ActiveSessions, rows[1].ActiveSessions = intp(0), intp(0)
	rows[0].RoutingCostPerMTok, rows[1].RoutingCostPerMTok = floatp(15), floatp(3)
	rows[0].BilledCostPerMTok, rows[1].BilledCostPerMTok = floatp(0), floatp(0)

	route := RouteAccount(rows, "implement the feature", "engineering", false, false, "claude", DefaultPolicy())
	if !route.OK || route.Account == nil {
		t.Fatalf("route = %+v, want selected account", route)
	}
	if route.Account.Tag != "cheap" {
		t.Fatalf("selected %q, want cheapest nominal seat", route.Account.Tag)
	}
	if got := derefFloat64(route.Account.BilledCostPerMTok); got != 0 {
		t.Fatalf("billed cost = %v, want 0 for plan-billed seat", got)
	}
	if got := derefFloat64(route.Account.RoutingCostPerMTok); got != 3 {
		t.Fatalf("routing cost = %v, want 3", got)
	}

	encoded, err := json.Marshal(route.Account)
	if err != nil {
		t.Fatal(err)
	}
	var reported struct {
		Routing float64 `json:"routing_cost_per_mtok"`
		Billed  float64 `json:"billed_cost_per_mtok"`
	}
	if err := json.Unmarshal(encoded, &reported); err != nil {
		t.Fatal(err)
	}
	if reported.Routing != 3 || reported.Billed != 0 {
		t.Fatalf("reported costs = %+v, want routing 3 and billed 0; json=%s", reported, encoded)
	}
}

func TestRouteRankKeepsHardTermsAboveNominalCost(t *testing.T) {
	rows := capRoster()
	rows[0].Tag, rows[1].Tag = "preferred", "cheap"
	rows[0].RouteWeight, rows[1].RouteWeight = intp(20), intp(10)
	rows[0].LiveSessions, rows[1].LiveSessions = intp(0), intp(0)
	rows[0].ActiveSessions, rows[1].ActiveSessions = intp(0), intp(0)
	rows[0].RoutingCostPerMTok, rows[1].RoutingCostPerMTok = floatp(30), floatp(1)

	route := RouteAccount(rows, "implement the feature", "engineering", false, false, "claude", DefaultPolicy())
	if !route.OK || route.Account == nil || route.Account.Tag != "preferred" {
		t.Fatalf("route = %+v, want higher route weight to remain authoritative", route)
	}
}
