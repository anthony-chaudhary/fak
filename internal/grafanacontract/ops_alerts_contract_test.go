package grafanacontract

import (
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// prometheusAlertsPath is the rule file the ops contract pins. The public repo
// carries no YAML dependency (go.mod requires only golang.org/x/term and
// golang.org/x/sys), so the parse below is a deliberate line-oriented scan over
// the small, well-shaped Prometheus rule grammar rather than a yaml.Unmarshal.
const prometheusAlertsPath = "../../tools/grafana/prometheus-alerts.yml"

const prometheusConfigPath = "../../tools/grafana/prometheus.yml"

// OpsAlert is one rule as the scanner recovers it from the flat YAML.
type OpsAlert struct {
	Name        string
	Expr        string
	Severity    string
	Summary     string
	Description string
}

// OpsGroup is a named rule group and the alerts recovered from it.
type OpsGroup struct {
	Name   string
	Alerts []OpsAlert
}

var (
	opsGroupLine = regexp.MustCompile(`^\s*-\s+name:\s*(\S+)\s*$`)
	opsAlertLine = regexp.MustCompile(`^\s*-\s+alert:\s*(\S+)\s*$`)
	opsFieldLine = regexp.MustCompile(`^(\s+)(expr|severity|summary|description):\s*(.*)$`)
)

// ParseAlertGroups scans a Prometheus rule file and returns its groups in file
// order. It understands exactly the shape prometheus-alerts.yml uses: a `groups:`
// list whose entries are `- name:`, each holding a `rules:` list of `- alert:`
// entries with `expr`, `for`, `labels.severity` and `annotations.*` beneath them.
// It is intentionally tolerant of comments and unrelated top-level keys; it is
// NOT a general YAML parser, and both the positive and negative controls below
// exercise it directly.
func ParseAlertGroups(text string) []OpsGroup {
	var groups []OpsGroup
	var cur *OpsGroup
	var alert *OpsAlert
	inAnnotations := false

	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimRight(raw, "\r")
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}

		if m := opsGroupLine.FindStringSubmatch(line); m != nil {
			if cur != nil {
				groups = append(groups, *cur)
			}
			alert = nil
			cur = &OpsGroup{Name: m[1]}
			continue
		}
		if m := opsAlertLine.FindStringSubmatch(line); m != nil {
			if cur == nil {
				continue
			}
			cur.Alerts = append(cur.Alerts, OpsAlert{Name: m[1]})
			alert = &cur.Alerts[len(cur.Alerts)-1]
			inAnnotations = false
			continue
		}
		if alert == nil {
			continue
		}

		// `annotations:` opens the block whose summary/description lines are
		// indented deeper than `labels:`. Track it so an unrelated `summary`
		// elsewhere cannot be mis-attributed.
		if strings.HasSuffix(trimmed, "annotations:") {
			inAnnotations = true
			continue
		}
		if strings.HasSuffix(trimmed, "labels:") || strings.HasSuffix(trimmed, "rules:") {
			inAnnotations = false
			continue
		}

		m := opsFieldLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		key, val := m[2], strings.TrimSpace(m[3])
		val = strings.Trim(val, `"'`)
		switch key {
		case "expr":
			alert.Expr = val
			inAnnotations = false
		case "summary":
			if inAnnotations {
				alert.Summary = val
			}
		case "description":
			if inAnnotations {
				alert.Description = val
			}
		case "severity":
			alert.Severity = val
		}
	}
	if cur != nil {
		groups = append(groups, *cur)
	}
	return groups
}

// opsAlertSpec is the pinned contract: the group name and the alert names it
// must contain. A future edit that deletes a rule or renames a group fails here
// instead of silently disarming the ops plane.
type opsAlertSpec struct {
	Group  string
	Alerts []string
}

func opsAlertContract() []opsAlertSpec {
	return []opsAlertSpec{
		{
			Group: "fak_ops_availability",
			Alerts: []string{
				"FakOpsExporterDown",
				"FakOpsWorkersExporterDown",
				"FakOpsSnapshotStale",
			},
		},
		{
			Group: "fak_ops_degradation",
			Alerts: []string{
				"FakOpsPlaneDegraded",
				"FakOpsStuckPlaneFailed",
				"FakOpsStateUnreadable",
				"FakOpsReportsLedgerMissing",
				"FakOpsGoalRegistryMissing",
				"FakOpsReportsWindowEmpty",
			},
		},
		{
			Group: "fak_ops_health",
			Alerts: []string{
				"FakOpsRoutinesStale",
				"FakOpsWorkersRoutinesStale",
				"FakOpsHighFailureRate",
				"FakOpsGoalClosureLow",
				"FakOpsIssueBacklogGrowing",
				"FakOpsThroughputCensusTruncated",
				"FakOpsWorktreePlaneDown",
			},
		},
	}
}

func groupByName(groups []OpsGroup, name string) (OpsGroup, bool) {
	for _, g := range groups {
		if g.Name == name {
			return g, true
		}
	}
	return OpsGroup{}, false
}

// TestOpsAlertGroupsExist pins the three ops alert groups and every alert in
// them. The ops plane is fail-closed: a deleted rule is an unwatched factory.
func TestOpsAlertGroupsExist(t *testing.T) {
	raw, err := os.ReadFile(prometheusAlertsPath)
	if err != nil {
		t.Fatalf("read %s: %v", prometheusAlertsPath, err)
	}
	groups := ParseAlertGroups(string(raw))

	for _, spec := range opsAlertContract() {
		g, ok := groupByName(groups, spec.Group)
		if !ok {
			t.Errorf("alert group %q not found in %s (was it deleted or renamed?)", spec.Group, prometheusAlertsPath)
			continue
		}
		got := map[string]bool{}
		for _, a := range g.Alerts {
			got[a.Name] = true
		}
		for _, want := range spec.Alerts {
			if !got[want] {
				t.Errorf("alert %q missing from group %q", want, spec.Group)
			}
		}
	}
}

// TestOpsAlertRulesAreWellFormed asserts every alert in the three ops groups
// carries the fields a rule needs to actually fire and to be readable when it
// does: a non-empty expr, a severity in the closed {info,warning,critical} set,
// and non-empty summary/description annotations.
func TestOpsAlertRulesAreWellFormed(t *testing.T) {
	raw, err := os.ReadFile(prometheusAlertsPath)
	if err != nil {
		t.Fatalf("read %s: %v", prometheusAlertsPath, err)
	}
	groups := ParseAlertGroups(string(raw))

	validSeverity := map[string]bool{"info": true, "warning": true, "critical": true}
	for _, spec := range opsAlertContract() {
		g, ok := groupByName(groups, spec.Group)
		if !ok {
			t.Errorf("alert group %q not found", spec.Group)
			continue
		}
		for _, a := range g.Alerts {
			if strings.TrimSpace(a.Expr) == "" {
				t.Errorf("%s/%s: empty expr", spec.Group, a.Name)
			}
			if !validSeverity[a.Severity] {
				t.Errorf("%s/%s: severity %q not in {info,warning,critical}", spec.Group, a.Name, a.Severity)
			}
			if strings.TrimSpace(a.Summary) == "" {
				t.Errorf("%s/%s: empty annotations.summary", spec.Group, a.Name)
			}
			if strings.TrimSpace(a.Description) == "" {
				t.Errorf("%s/%s: empty annotations.description", spec.Group, a.Name)
			}
		}
	}
}

// TestOpsAlertsParserNegativeControl is the negative control for the scanner:
// given the real rule file with one ops group's `- name:` line renamed away, the
// parser must report the group ABSENT and the contract must fail. It also proves
// an entirely-emptied document reads as no groups rather than a false pass.
func TestOpsAlertsParserNegativeControl(t *testing.T) {
	fixture := `groups:
  - name: fak_ops_availability
    rules:
      - alert: FakOpsExporterDown
        expr: up{job="fak_ops"} == 0
        labels:
          severity: warning
        annotations:
          summary: "s"
          description: "d"

  - name: fak_ops_degradation
    rules:
      - alert: FakOpsPlaneDegraded
        expr: fak_ops_partial == 1
        labels:
          severity: warning
        annotations:
          summary: "s"
          description: "d"
`
	cases := []struct {
		name      string
		text      string
		group     string
		wantFound bool
	}{
		{"present group is found", fixture, "fak_ops_availability", true},
		{"removed group is absent", fixture, "fak_ops_health", false},
		{"absent group is absent", fixture, "fak_ops_nonexistent", false},
		{"renamed group is absent", strings.Replace(fixture, "name: fak_ops_degradation", "name: not_ops", 1), "fak_ops_degradation", false},
		{"empty document has no groups", "", "fak_ops_availability", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, got := groupByName(ParseAlertGroups(tc.text), tc.group)
			if got != tc.wantFound {
				t.Errorf("group %q found=%v, want %v", tc.group, got, tc.wantFound)
			}
		})
	}
}

// TestPrometheusLoadsNativePerformanceAlerts pins that prometheus.yml actually
// loads native-performance-alerts.yml in its rule_files list. The file was
// previously named nowhere, so its rules existed but were never evaluated.
func TestPrometheusLoadsNativePerformanceAlerts(t *testing.T) {
	raw, err := os.ReadFile(prometheusConfigPath)
	if err != nil {
		t.Fatalf("read %s: %v", prometheusConfigPath, err)
	}
	text := string(raw)

	idx := strings.Index(text, "rule_files:")
	if idx < 0 {
		t.Fatalf("%s has no rule_files: block", prometheusConfigPath)
	}
	// The rule_files list is a short block of `- "file.yml"` entries; scan until
	// the next top-level key so a later mention elsewhere cannot satisfy this.
	rest := text[idx+len("rule_files:"):]
	var entries []string
	for _, line := range strings.Split(rest, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		if !strings.HasPrefix(trimmed, "-") {
			break
		}
		entries = append(entries, strings.Trim(strings.TrimSpace(strings.TrimPrefix(trimmed, "-")), `"'`))
	}
	sort.Strings(entries)

	for _, want := range []string{"native-performance-alerts.yml", "prometheus-alerts.yml"} {
		found := false
		for _, e := range entries {
			if e == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("rule_files does not load %q; entries=%v", want, entries)
		}
	}
}

// privateOpsExporterPaths are the companion-repo exporters whose instrumentation
// defines which fak_ops_* series exist. `go test` runs with the package directory
// as its working directory, so the candidate prefixes cover both that cwd
// (`internal/grafanacontract` -> ../../../fak-private) and a repo-root cwd. The
// public repo must not depend on the private checkout, so an unresolvable path
// SKIPs rather than fails.
var privateOpsExporterPaths = []string{
	"platform/ops/dashboard/dashboard.go",
	"cmd/fak-sync/ops_throughput_metrics.go",
}

var privateOpsRootPrefixes = []string{
	"../../../fak-private/",
	"../fak-private/",
	"../../fak-private/",
}

var fakOpsMetric = regexp.MustCompile(`\bfak_ops_[a-z0-9_]+`)

// readFirstExisting reads prefixes[0]+rel, falling back through the rest, and
// returns the bytes plus the path that actually resolved.
func readFirstExisting(prefixes []string, rel string) ([]byte, string, error) {
	var lastErr error
	for _, prefix := range prefixes {
		p := prefix + rel
		b, err := os.ReadFile(p)
		if err == nil {
			return b, p, nil
		}
		lastErr = err
	}
	return nil, "", lastErr
}

// TestOpsAlertsNameExportedMetrics asserts every fak_ops_* metric an ops alert
// expr names is actually exported by one of the two ops exporters. This catches
// an alert left pointing at a renamed series -- which would fire never, or fire
// permanently, without anyone noticing.
func TestOpsAlertsNameExportedMetrics(t *testing.T) {
	exporterText := ""
	for _, rel := range privateOpsExporterPaths {
		b, resolved, err := readFirstExisting(privateOpsRootPrefixes, rel)
		if err != nil {
			t.Skipf("SKIP: companion exporter %s not readable from this checkout (public repo must not depend on fak-private); tried %v: %v", rel, privateOpsRootPrefixes, err)
		}
		t.Logf("companion exporter resolved: %s", resolved)
		exporterText += string(b) + "\n"
	}
	if strings.TrimSpace(exporterText) == "" {
		t.Skip("SKIP: no companion ops exporter sources readable")
	}

	exported := map[string]bool{}
	for _, m := range fakOpsMetric.FindAllString(exporterText, -1) {
		exported[m] = true
	}

	raw, err := os.ReadFile(prometheusAlertsPath)
	if err != nil {
		t.Fatalf("read %s: %v", prometheusAlertsPath, err)
	}
	groups := ParseAlertGroups(string(raw))

	var missing []string
	seen := map[string]bool{}
	for _, spec := range opsAlertContract() {
		g, ok := groupByName(groups, spec.Group)
		if !ok {
			continue
		}
		for _, a := range g.Alerts {
			for _, m := range fakOpsMetric.FindAllString(a.Expr, -1) {
				if seen[m] {
					continue
				}
				seen[m] = true
				if !exported[m] {
					missing = append(missing, fmt.Sprintf("%s (expr of %s)", m, a.Name))
				}
			}
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Errorf("ops alert exprs name metrics no ops exporter exports: %v", missing)
	}
}
