package grafanacontract

import (
	"encoding/json"
	"os"
	"testing"
)

// Invariant: Grafana overview dashboard contracts must preserve root goal drilldown panel definitions.
// Guard: TestDashboardSchema verifies that the JSON dashboard contains non-empty panels and targets.

func TestGrafanaContractLifecycle(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("../../tools/grafana/dashboards/fak-fleet-overview.json")
	if err != nil {
		t.Fatalf("failed reading dashboard file: %v", err)
	}

	var d Dashboard
	if err := json.Unmarshal(data, &d); err != nil {
		t.Fatalf("failed unmarshaling dashboard: %v", err)
	}

	if len(d.Panels) == 0 {
		t.Fatal("expected non-zero panels in fleet overview dashboard")
	}
}
