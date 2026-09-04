package grafanacontract

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Dashboard models the top-level Grafana dashboard JSON schema needed for contract verification.
type Dashboard struct {
	Title  string  `json:"title"`
	UID    string  `json:"uid"`
	Panels []Panel `json:"panels"`
}

// Panel represents a Grafana dashboard panel.
type Panel struct {
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Targets     []Target `json:"targets"`
}

// Target represents a metric target / expression within a panel.
type Target struct {
	Expr string `json:"expr"`
}

// CheckResult represents the outcome of verifying a dashboard contract.
type CheckResult struct {
	DashboardPath string   `json:"dashboard_path"`
	Passed        bool     `json:"passed"`
	Missing       []string `json:"missing,omitempty"`
}

// VerifyDashboardBytes checks raw JSON bytes against required string tokens.
func VerifyDashboardBytes(b []byte, requiredTokens []string) (CheckResult, error) {
	var d Dashboard
	if err := json.Unmarshal(b, &d); err != nil {
		return CheckResult{Passed: false}, fmt.Errorf("unmarshal dashboard: %w", err)
	}

	var sb strings.Builder
	sb.WriteString(d.Title)
	sb.WriteByte('\n')
	for _, p := range d.Panels {
		sb.WriteString(p.Title)
		sb.WriteByte('\n')
		sb.WriteString(p.Description)
		sb.WriteByte('\n')
		for _, q := range p.Targets {
			sb.WriteString(q.Expr)
			sb.WriteByte('\n')
		}
	}
	all := sb.String()

	var missing []string
	for _, want := range requiredTokens {
		if !strings.Contains(all, want) {
			missing = append(missing, want)
		}
	}

	return CheckResult{
		Passed:  len(missing) == 0,
		Missing: missing,
	}, nil
}

// VerifyDashboardFile reads a dashboard JSON file and checks required string tokens.
func VerifyDashboardFile(path string, requiredTokens []string) (CheckResult, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return CheckResult{DashboardPath: path, Passed: false}, err
	}
	res, err := VerifyDashboardBytes(b, requiredTokens)
	res.DashboardPath = path
	return res, err
}

// DefaultFleetOverviewTokens returns the canonical contract tokens required for the Run Operations / fleet overview dashboard.
func DefaultFleetOverviewTokens() []string {
	return []string{
		"FAK Run Operations",
		"Open / live runs (LIVE inventory)",
		"Completed runs (DURABLE registration history)",
		"fak_fleet_session_info",
		"fak_fleet_run_info",
		"fak_fleet_run_duration_seconds",
		"fak_fleet_registration_registry_readable",
	}
}

// VerifyFleetOverview checks the default fleet overview dashboard JSON against its contract.
func VerifyFleetOverview(path string) (CheckResult, error) {
	return VerifyDashboardFile(path, DefaultFleetOverviewTokens())
}
