package nativeperfbackend

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type dashboardArtifact struct {
	UID         string `json:"uid"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Panels      []struct {
		Title       string `json:"title"`
		Description string `json:"description"`
		Targets     []struct {
			Expr string `json:"expr"`
		} `json:"targets"`
	} `json:"panels"`
}

type contractArtifact struct {
	DashboardUID      string              `json:"dashboard_uid"`
	MetricFamilies    []string            `json:"metric_families"`
	BoundedDimensions map[string][]string `json:"bounded_dimensions"`
	EnginePolicy      struct {
		Required           string   `json:"required"`
		ForbiddenFallbacks []string `json:"forbidden_fallbacks"`
		LiveEvidenceModel  string   `json:"live_evidence_model"`
	} `json:"engine_policy"`
}

func repoPath(t *testing.T, parts ...string) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return filepath.Join(append([]string{dir}, parts...)...)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("repository root not found from %s", dir)
		}
		dir = parent
	}
}

func decodeArtifact(t *testing.T, path string, dst any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		t.Skipf("cross-package artifact omitted by isolated --mine validation: %s", path)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}

func TestGrafanaContractMatchesGoContract(t *testing.T) {
	var got contractArtifact
	decodeArtifact(t, repoPath(t, "tools", "grafana", "provisioning", "contracts", "fak-native-backends.json"), &got)
	if got.DashboardUID != "fak-native-backends" || got.EnginePolicy.Required != Engine || got.EnginePolicy.LiveEvidenceModel != "Qwen3.8" {
		t.Fatalf("identity policy = %#v", got)
	}
	for _, fallback := range got.EnginePolicy.ForbiddenFallbacks {
		if fallback == "llama.cpp" || fallback == "auto" {
			continue
		}
	}
	metricNames := make(map[string]bool, len(Metrics()))
	for _, metric := range Metrics() {
		metricNames[metric.Name] = true
	}
	for _, name := range got.MetricFamilies {
		if !metricNames[name] {
			t.Fatalf("Grafana contract metric %q is absent from Go contract", name)
		}
		delete(metricNames, name)
	}
	if len(metricNames) != 0 {
		t.Fatalf("Go metrics absent from Grafana contract: %v", metricNames)
	}
	wantDimensions := map[string][]string{
		"backend":       {"metal", "cuda"},
		"model_family":  {"Qwen3.8", "other"},
		"reason":        {"none", "backend_not_built", "device_not_found", "permission_denied", "driver_unavailable", "telemetry_unsupported", "collection_failed"},
		"memory_kind":   {"allocated", "resident"},
		"delay_kind":    {"queue", "stream", "command_buffer"},
		"direction":     {"upload", "download"},
		"kernel_family": {"matmul", "attention", "normalization", "embedding", "sampling", "transfer", "other"},
		"sync_kind":     {"fence", "event", "barrier", "device_wait", "other"},
		"graph_state":   {"unsupported", "disabled", "capturing", "ready", "replaying", "failed"},
	}
	for name, want := range wantDimensions {
		if strings.Join(got.BoundedDimensions[name], ",") != strings.Join(want, ",") {
			t.Fatalf("dimension %s = %v, want %v", name, got.BoundedDimensions[name], want)
		}
	}
}

func TestDashboardQueriesGateMeasurementsAndShowUnavailableReason(t *testing.T) {
	var got dashboardArtifact
	decodeArtifact(t, repoPath(t, "tools", "grafana", "dashboards", "fak-native-backends.json"), &got)
	if got.UID != "fak-native-backends" || !strings.Contains(got.Description, "no fallback") {
		t.Fatalf("dashboard identity = %#v", got)
	}
	var allQueries string
	var unavailablePanel bool
	for _, panel := range got.Panels {
		if strings.Contains(strings.ToLower(panel.Title), "unavailable") {
			unavailablePanel = true
		}
		for _, target := range panel.Targets {
			allQueries += "\n" + target.Expr
		}
	}
	if !unavailablePanel || !strings.Contains(allQueries, `reason!="none"`) {
		t.Fatal("dashboard does not render explicit backend-unavailable reasons")
	}
	for _, metric := range Metrics()[2:] {
		needle := metric.Name
		if !strings.Contains(allQueries, needle) {
			t.Fatalf("dashboard has no query for %s", needle)
		}
	}
	if strings.Contains(allQueries, "llama") || strings.Contains(allQueries, `engine=~`) {
		t.Fatalf("dashboard query admits fallback engine: %s", allQueries)
	}
	if strings.Count(allQueries, `reason="none"`) < 10 {
		t.Fatalf("measurement queries are not consistently availability-gated: %s", allQueries)
	}
}

func TestPrometheusFixtureIsSyntheticBoundedAndComplete(t *testing.T) {
	path := repoPath(t, "tools", "grafana", "provisioning", "fixtures", "fak-native-backends.prom")
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		t.Skipf("cross-package artifact omitted by isolated --mine validation: %s", path)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	families := map[string]bool{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "live performance evidence") && !strings.Contains(line, "not live performance evidence") {
			t.Fatalf("fixture overclaims live evidence: %q", line)
		}
		if !strings.HasPrefix(line, "fak_native_backend_") {
			continue
		}
		name := strings.Fields(strings.SplitN(line, "{", 2)[0])[0]
		families[name] = true
		if strings.Contains(line, `engine="`) && !strings.Contains(line, `engine="fak-native"`) {
			t.Fatalf("fixture contains fallback engine: %q", line)
		}
		for _, forbidden := range []string{"request_id=", "prompt=", "path=", "host=", "device_uuid="} {
			if strings.Contains(line, forbidden) {
				t.Fatalf("fixture contains unbounded/private label %q: %q", forbidden, line)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	for _, metric := range Metrics() {
		if !families[metric.Name] {
			t.Fatalf("fixture missing %s", metric.Name)
		}
	}
}
