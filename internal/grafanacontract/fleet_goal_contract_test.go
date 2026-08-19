package grafanacontract

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

type dashboard struct {
	Panels []panel `json:"panels"`
}
type panel struct {
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Targets     []target `json:"targets"`
}
type target struct {
	Expr string `json:"expr"`
}

func TestFleetOverviewCarriesRootGoalDrilldownContract(t *testing.T) {
	b, err := os.ReadFile("../../tools/grafana/dashboards/fak-fleet-overview.json")
	if err != nil {
		t.Fatal(err)
	}
	var d dashboard
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatal(err)
	}
	all := ""
	for _, p := range d.Panels {
		all += p.Title + "\n" + p.Description + "\n"
		for _, q := range p.Targets {
			all += q.Expr + "\n"
		}
	}
	for _, want := range []string{"Canonical goals", "Fleet -> canonical goal -> execution root -> session", "execution_root_only", "fak_fleet_canonical_goal_execution_roots", "fak_fleet_canonical_goal_attempts_total", "fak_fleet_canonical_goal_provider_billed_micro_usd_total", "fak_fleet_canonical_goal_cache_value_reuse_ratio", "fak_fleet_canonical_goal_binding_ratio", "fak_fleet_canonical_goal_efficiency_ready", "Starting goals", "fak_fleet_goal_info", "fak_fleet_goal_usage_attribution_ratio", "fak_fleet_goal_attempts_total", "fak_fleet_goal_provider_billed_micro_usd_total", "fak_fleet_goal_provider_cost_attribution_ratio", "fak_fleet_goal_cache_value_reused_tokens_total", "fak_fleet_goal_cache_value_reuse_ratio", "fak_fleet_goal_cache_value_attribution_ratio", `fak_fleet_goal_usage_rows{attribution="unattributed"}`, "bounded to root_registration, root_issue, task, state, and outcome"} {
		if !strings.Contains(all, want) {
			t.Errorf("dashboard missing %q", want)
		}
	}
}
