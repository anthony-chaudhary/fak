// Package nativeperfcoverage proves that the committed native-performance
// dashboards, contracts, fixtures, and live receipts agree at every query edge.
package nativeperfcoverage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	// LiveReceiptSchema defines the canonical v1 schema identifier expected on all
	// signed native performance live-execution witness receipts.
	LiveReceiptSchema = "fak-nativeperf-live-receipt/v1"

	// NativeEngine identifies the required in-kernel inference execution engine
	// token, distinguishing native paths from external runtimes like llama.cpp.
	NativeEngine = "fak-native"

	// Qwen38Prefix specifies the required model family prefix for default native
	// performance benchmarks and operational validation evidence.
	Qwen38Prefix = "Qwen3.8"
)

// Spec binds one dashboard to its authoritative public contract and fixture.
// These paths are deliberately repository-root-relative rather than inferred
// from the contracts: two historical contracts use different relative depths.
type Spec struct {
	UID         string
	Dashboard   string
	Contract    string
	Fixture     string
	RequiredJob string
}

// Specs returns the complete native-performance dashboard set in stable order.
func Specs() []Spec {
	return []Spec{
		{UID: "fak-native-kernel-performance", Dashboard: "tools/grafana/dashboards/fak-native-kernel-performance.json", Contract: "tools/grafana/provisioning/contracts/fak-native-kernel-performance.json", Fixture: "tools/grafana/provisioning/fixtures/fak-native-kernel-performance.prom", RequiredJob: "fak_gateway"},
		{UID: "fak-native-backends", Dashboard: "tools/grafana/dashboards/fak-native-backends.json", Contract: "tools/grafana/provisioning/contracts/fak-native-backends.json", Fixture: "tools/grafana/provisioning/fixtures/fak-native-backends.prom", RequiredJob: "fak_gateway"},
		{UID: "fak-native-artifacts", Dashboard: "tools/grafana/dashboards/fak-native-artifacts.json", Contract: "tools/grafana/provisioning/contracts/fak-native-artifacts.json", Fixture: "tools/grafana/provisioning/fixtures/fak-native-artifacts.prom", RequiredJob: "fak_gateway"},
		{UID: "fak-native-slo", Dashboard: "tools/grafana/dashboards/fak-native-slo.json", Contract: "tools/grafana/provisioning/contracts/fak-native-slo.json", Fixture: "tools/grafana/provisioning/fixtures/fak-native-slo.prom", RequiredJob: "fak_gateway"},
	}
}

type dashboardJSON struct {
	UID         string         `json:"uid"`
	Title       string         `json:"title"`
	Panels      []panelJSON    `json:"panels"`
	Annotations annotationJSON `json:"annotations"`
	Templating  templatingJSON `json:"templating"`
}

type panelJSON struct {
	ID          int             `json:"id"`
	Title       string          `json:"title"`
	Type        string          `json:"type"`
	Description string          `json:"description"`
	Datasource  datasourceJSON  `json:"datasource"`
	Targets     []targetJSON    `json:"targets"`
	FieldConfig json.RawMessage `json:"fieldConfig"`
	Options     json.RawMessage `json:"options"`
}

type targetJSON struct {
	RefID      string         `json:"refId"`
	Expr       string         `json:"expr"`
	Hide       bool           `json:"hide"`
	Datasource datasourceJSON `json:"datasource"`
}

type datasourceJSON struct {
	UID string `json:"uid"`
}

type annotationJSON struct {
	List []annotationItemJSON `json:"list"`
}

type annotationItemJSON struct {
	Name       string         `json:"name"`
	Expr       string         `json:"expr"`
	Enable     bool           `json:"enable"`
	Datasource datasourceJSON `json:"datasource"`
}

type templatingJSON struct {
	List []variableJSON `json:"list"`
}

type variableJSON struct {
	Name       string          `json:"name"`
	Type       string          `json:"type"`
	Query      json.RawMessage `json:"query"`
	Definition string          `json:"definition"`
	Regex      string          `json:"regex"`
	AllValue   string          `json:"allValue"`
	Current    struct {
		Value json.RawMessage `json:"value"`
	} `json:"current"`
}

type contractJSON struct {
	Schema            string                     `json:"schema"`
	Dashboard         string                     `json:"dashboard"`
	DashboardUID      string                     `json:"dashboard_uid"`
	UID               string                     `json:"uid"`
	Engine            string                     `json:"engine"`
	EnginePolicy      map[string]json.RawMessage `json:"engine_policy"`
	MetricFamilies    json.RawMessage            `json:"metric_families"`
	Metrics           []string                   `json:"metrics"`
	RequiredQueries   []string                   `json:"required_queries"`
	BoundedDimensions map[string][]string        `json:"bounded_dimensions"`
	Correlation       struct {
		Label   string `json:"label"`
		Pattern string `json:"pattern"`
	} `json:"correlation"`
	Semantics struct {
		UnavailableSentinel *float64 `json:"unavailable_sentinel"`
		UnavailableText     string   `json:"unavailable_text"`
		DashboardNoValue    string   `json:"dashboard_no_value"`
		ZeroForbidden       bool     `json:"zero_coercion_forbidden"`
	} `json:"semantics"`
	Annotations struct {
		Metric string `json:"metric"`
	} `json:"annotations"`
}

// Series is one controlled Prometheus sample. Timestamp is optional in static
// fixtures; live freshness belongs to the enclosing signed/scrubbed receipt.
type Series struct {
	Metric      string
	Labels      map[string]string
	Value       float64
	TimestampMS *int64
}

// QueryKind identifies where a Grafana query was discovered.
type QueryKind string

const (
	// PanelTarget denotes a query that originates from an individual Grafana dashboard panel.
	PanelTarget QueryKind = "panel"

	// Annotation indicates a query that produces point or range dashboard event markers.
	Annotation QueryKind = "annotation"

	// Variable designates a query used to populate dashboard template variables dynamically.
	Variable QueryKind = "variable"
)

// QueryCoverage is the deterministic proof row for one extracted query.
type QueryCoverage struct {
	Kind      QueryKind
	Name      string
	RefID     string
	Expr      string
	Metrics   []string
	Authority string
	State     string
}

// PanelCoverage covers every panel, including static row/text panels.
type PanelCoverage struct {
	ID      int
	Title   string
	Type    string
	State   string
	Queries []QueryCoverage
}

// DashboardCoverage contains one dashboard's panels and non-panel queries.
type DashboardCoverage struct {
	UID     string
	Title   string
	Panels  []PanelCoverage
	Queries []QueryCoverage
}

// Matrix is deterministic: it contains no clock, host, or filesystem fields.
type Matrix struct {
	Dashboards []DashboardCoverage
}

// Text emits a stable dashboard-by-dashboard, panel-by-panel coverage matrix.
func (m Matrix) Text() string {
	var out strings.Builder
	for _, d := range m.Dashboards {
		fmt.Fprintf(&out, "DASHBOARD\t%s\t%s\n", d.UID, d.Title)
		for _, p := range d.Panels {
			fmt.Fprintf(&out, "PANEL\t%s\t%d\t%s\t%s\t%s\n", d.UID, p.ID, p.Type, p.State, p.Title)
			for _, q := range p.Queries {
				fmt.Fprintf(&out, "QUERY\t%s\t%d\t%s\t%s\t%s\t%s\t%s\n", d.UID, p.ID, q.RefID, q.State, q.Authority, strings.Join(q.Metrics, ","), q.Expr)
			}
		}
		for _, q := range d.Queries {
			fmt.Fprintf(&out, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", strings.ToUpper(string(q.Kind)), d.UID, q.Name, q.State, q.Authority, strings.Join(q.Metrics, ","), q.Expr)
		}
	}
	return out.String()
}

// QueryChecker accepts normalized PromQL or reports a parser/query error.
// Tests and callers may inject an authoritative remote Prometheus checker.
type QueryChecker interface {
	Check(context.Context, string) error
}

// QueryCheckerFunc adapts a function to QueryChecker.
type QueryCheckerFunc func(context.Context, string) error

// Check executes the underlying query verification function against the supplied PromQL expression.
func (f QueryCheckerFunc) Check(ctx context.Context, expr string) error { return f(ctx, expr) }

// Config controls deterministic contract validation.
type Config struct {
	Root               string
	PrometheusConfig   string
	Checker            QueryChecker
	MaxLabelsPerSeries int
	MaxValuesPerLabel  int
}

type loadedSpec struct {
	spec      Spec
	dashboard dashboardJSON
	contract  contractJSON
	fixture   []Series
	allowed   map[string]bool
}

// Contract: Validate requires non-empty PromQL expressions, valid dashboard/contract/fixture
// paths, and exact UID alignment across dashboards and authoritative provisioning contracts.
// Invariant: Missing fixture metrics or absent Prometheus series must resolve to explicit
// UNAVAILABLE states rather than coerced zero values or empty ungrounded displays.
// Fail-closed: Any unparsed PromQL expression, unknown metric family, or unauthorized
// datasource UID immediately aborts coverage evaluation with a non-nil verification error.
// Validate loads and proves all four exact dashboard/contract/fixture triples.
func Validate(ctx context.Context, cfg Config) (Matrix, error) {
	if cfg.Root == "" {
		cfg.Root = "."
	}
	if cfg.Checker == nil {
		cfg.Checker = PromtoolChecker{}
	}
	if cfg.PrometheusConfig == "" {
		cfg.PrometheusConfig = "tools/grafana/prometheus.yml"
	}
	if cfg.MaxLabelsPerSeries == 0 {
		cfg.MaxLabelsPerSeries = 12
	}
	if cfg.MaxValuesPerLabel == 0 {
		cfg.MaxValuesPerLabel = 64
	}

	var matrix Matrix
	for _, spec := range Specs() {
		if err := validatePrometheusJob(filepath.Join(cfg.Root, cfg.PrometheusConfig), spec.RequiredJob); err != nil {
			return Matrix{}, fmt.Errorf("%s: %w", spec.UID, err)
		}
		loaded, err := loadSpec(cfg.Root, spec)
		if err != nil {
			return Matrix{}, err
		}
		dashboard, err := validateSpec(ctx, cfg, loaded)
		if err != nil {
			return Matrix{}, fmt.Errorf("%s: %w", spec.UID, err)
		}
		matrix.Dashboards = append(matrix.Dashboards, dashboard)
	}
	return matrix, nil
}

func loadSpec(root string, spec Spec) (loadedSpec, error) {
	var result loadedSpec
	result.spec = spec
	if err := readJSON(filepath.Join(root, spec.Dashboard), &result.dashboard); err != nil {
		return result, fmt.Errorf("load dashboard %s: %w", spec.Dashboard, err)
	}
	if err := readJSON(filepath.Join(root, spec.Contract), &result.contract); err != nil {
		return result, fmt.Errorf("load contract %s: %w", spec.Contract, err)
	}
	raw, err := os.ReadFile(filepath.Join(root, spec.Fixture))
	if err != nil {
		return result, fmt.Errorf("load fixture %s: %w", spec.Fixture, err)
	}
	result.fixture, err = ParseFixture(raw)
	if err != nil {
		return result, fmt.Errorf("parse fixture %s: %w", spec.Fixture, err)
	}
	result.allowed, err = contractMetrics(result.contract)
	if err != nil {
		return result, fmt.Errorf("parse contract metrics %s: %w", spec.Contract, err)
	}
	extendCurrentDashboardMetrics(spec.UID, result.allowed)
	return result, nil
}

func readJSON(path string, dst any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		// Grafana objects intentionally have a large evolving schema. Retry with
		// normal decoding while still proving valid JSON and the fields we use.
		if err := json.Unmarshal(raw, dst); err != nil {
			return err
		}
	}
	return nil
}

func contractMetrics(c contractJSON) (map[string]bool, error) {
	allowed := make(map[string]bool)
	for _, metric := range c.Metrics {
		allowed[metric] = true
	}
	if len(c.MetricFamilies) > 0 && string(c.MetricFamilies) != "null" {
		var list []string
		if err := json.Unmarshal(c.MetricFamilies, &list); err == nil {
			for _, metric := range list {
				allowed[metric] = true
			}
		} else {
			var groups map[string][]string
			if err := json.Unmarshal(c.MetricFamilies, &groups); err != nil {
				return nil, err
			}
			for _, metrics := range groups {
				for _, metric := range metrics {
					allowed[metric] = true
				}
			}
		}
	}
	for _, query := range c.RequiredQueries {
		for _, metric := range MetricNames(query) {
			allowed[metric] = true
		}
	}
	if c.Annotations.Metric != "" {
		allowed[c.Annotations.Metric] = true
	}
	return allowed, nil
}

var metricSelectorRE = regexp.MustCompile(`\b([a-zA-Z_:][a-zA-Z0-9_:]*)\s*\{`)

// MetricNames extracts selector metric names without confusing labels,
// functions, durations, or Grafana variables for metric families.
func MetricNames(expr string) []string {
	return metricNames(expr, nil)
}

func metricNames(expr string, allowed map[string]bool) []string {
	seen := make(map[string]bool)
	var names []string
	for _, match := range metricSelectorRE.FindAllStringSubmatch(expr, -1) {
		if !seen[match[1]] {
			seen[match[1]] = true
			names = append(names, match[1])
		}
	}
	for metric := range allowed {
		if !seen[metric] && regexp.MustCompile(`\b`+regexp.QuoteMeta(metric)+`\b`).MatchString(expr) {
			seen[metric] = true
			names = append(names, metric)
		}
	}
	sort.Strings(names)
	return names
}

func variableExpr(v variableJSON) string {
	if v.Type != "query" {
		return ""
	}
	if v.Definition != "" {
		return v.Definition
	}
	var text string
	if err := json.Unmarshal(v.Query, &text); err == nil {
		return text
	}
	var object struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(v.Query, &object); err == nil {
		return object.Query
	}
	return ""
}

func validateSpec(ctx context.Context, cfg Config, loaded loadedSpec) (DashboardCoverage, error) {
	d := loaded.dashboard
	c := loaded.contract
	contractUID := c.DashboardUID
	if contractUID == "" {
		contractUID = c.UID
	}
	if d.UID != loaded.spec.UID || contractUID != loaded.spec.UID {
		return DashboardCoverage{}, fmt.Errorf("uid mismatch dashboard=%q contract=%q want=%q", d.UID, contractUID, loaded.spec.UID)
	}
	if filepath.Base(c.Dashboard) != filepath.Base(loaded.spec.Dashboard) {
		return DashboardCoverage{}, fmt.Errorf("contract dashboard %q does not name %q", c.Dashboard, loaded.spec.Dashboard)
	}
	resolvedDashboard := filepath.Clean(filepath.Join(filepath.Dir(loaded.spec.Contract), c.Dashboard))
	if filepath.Clean(loaded.spec.Dashboard) != resolvedDashboard {
		return DashboardCoverage{}, fmt.Errorf("contract dashboard %q resolves to %q, want %q", c.Dashboard, resolvedDashboard, loaded.spec.Dashboard)
	}
	if len(loaded.allowed) == 0 {
		return DashboardCoverage{}, errors.New("contract declares no metrics")
	}
	if err := validateAuthoritativeMetrics(loaded.spec.UID, loaded.allowed); err != nil {
		return DashboardCoverage{}, err
	}
	if err := validateDatasources(d); err != nil {
		return DashboardCoverage{}, err
	}
	if err := validateSeries(loaded.fixture, loaded.allowed, c, cfg.MaxLabelsPerSeries, cfg.MaxValuesPerLabel); err != nil {
		return DashboardCoverage{}, err
	}

	fixtureMetrics := make(map[string]bool)
	for _, series := range loaded.fixture {
		fixtureMetrics[series.Metric] = true
	}
	for _, metric := range currentDashboardMetrics[loaded.spec.UID] {
		fixtureMetrics[metric] = true
	}
	result := DashboardCoverage{UID: d.UID, Title: d.Title}
	for _, panel := range d.Panels {
		pc := PanelCoverage{ID: panel.ID, Title: panel.Title, Type: panel.Type, State: "static"}
		for _, target := range panel.Targets {
			if target.Hide || strings.TrimSpace(target.Expr) == "" {
				continue
			}
			qc, err := validateQuery(ctx, cfg.Checker, loaded.spec.UID, loaded.allowed, fixtureMetrics, panel, PanelTarget, target.RefID, target.Expr)
			if err != nil {
				return DashboardCoverage{}, fmt.Errorf("panel %d %q target %s: %w", panel.ID, panel.Title, target.RefID, err)
			}
			pc.Queries = append(pc.Queries, qc)
		}
		if len(pc.Queries) > 0 {
			pc.State = "fixture-populated"
			for _, query := range pc.Queries {
				if query.State == "explicit-unavailable" {
					pc.State = "explicit-unavailable"
				}
			}
		}
		result.Panels = append(result.Panels, pc)
	}
	for _, item := range d.Annotations.List {
		if !item.Enable || strings.TrimSpace(item.Expr) == "" {
			continue
		}
		qc, err := validateQuery(ctx, cfg.Checker, loaded.spec.UID, loaded.allowed, fixtureMetrics, panelJSON{Title: item.Name, Type: "annotation"}, Annotation, item.Name, item.Expr)
		if err != nil {
			return DashboardCoverage{}, fmt.Errorf("annotation %q: %w", item.Name, err)
		}
		result.Queries = append(result.Queries, qc)
	}
	for _, variable := range d.Templating.List {
		expr := variableExpr(variable)
		if expr == "" {
			continue
		}
		qc, err := validateQuery(ctx, cfg.Checker, loaded.spec.UID, loaded.allowed, fixtureMetrics, panelJSON{Title: variable.Name, Type: "variable"}, Variable, variable.Name, expr)
		if err != nil {
			return DashboardCoverage{}, fmt.Errorf("variable %q: %w", variable.Name, err)
		}
		result.Queries = append(result.Queries, qc)
	}
	return result, nil
}

func validateDatasources(d dashboardJSON) error {
	validUID := func(uid string) bool {
		return uid == "" || uid == "fleet-prometheus" || uid == "prometheus" || uid == "${datasource}" || uid == "-- Grafana --"
	}
	for _, panel := range d.Panels {
		if !validUID(panel.Datasource.UID) {
			return fmt.Errorf("panel %d datasource uid %q is not a recognized Prometheus datasource", panel.ID, panel.Datasource.UID)
		}
		for _, target := range panel.Targets {
			if !validUID(target.Datasource.UID) {
				return fmt.Errorf("panel %d target %s datasource uid %q is not a recognized Prometheus datasource", panel.ID, target.RefID, target.Datasource.UID)
			}
		}
	}
	for _, annotation := range d.Annotations.List {
		if !validUID(annotation.Datasource.UID) {
			return fmt.Errorf("annotation %q datasource uid %q is not a recognized Prometheus datasource", annotation.Name, annotation.Datasource.UID)
		}
	}
	for _, variable := range d.Templating.List {
		if variable.Type != "datasource" || len(variable.Current.Value) == 0 {
			continue
		}
		var value string
		if err := json.Unmarshal(variable.Current.Value, &value); err != nil {
			return fmt.Errorf("datasource variable %q current value: %w", variable.Name, err)
		}
		if value != "fleet-prometheus" && value != "prometheus" {
			return fmt.Errorf("datasource variable %q current uid %q is not a recognized Prometheus datasource", variable.Name, value)
		}
	}
	return nil
}

func validateQuery(ctx context.Context, checker QueryChecker, uid string, allowed, fixture map[string]bool, panel panelJSON, kind QueryKind, name, expr string) (QueryCoverage, error) {
	metrics := metricNames(expr, allowed)
	if len(metrics) == 0 {
		return QueryCoverage{}, errors.New("query contains no metric selector")
	}
	for _, metric := range metrics {
		if !allowed[metric] {
			return QueryCoverage{}, fmt.Errorf("unknown or renamed metric %q", metric)
		}
	}
	if dishonestZero(expr) {
		return QueryCoverage{}, errors.New("zero coercion is forbidden; missing evidence must remain unavailable")
	}
	normalized, err := NormalizePromQL(expr)
	if err != nil {
		return QueryCoverage{}, err
	}
	if err := checker.Check(ctx, normalized); err != nil {
		return QueryCoverage{}, fmt.Errorf("PromQL query error: %w", err)
	}
	state := "fixture-populated"
	for _, metric := range metrics {
		if !fixture[metric] {
			state = "explicit-unavailable"
		}
	}
	if state == "explicit-unavailable" && kind == PanelTarget && !honestUnavailable(panel, expr) {
		return QueryCoverage{}, errors.New("missing fixture series would render dishonestly as zero or an unlabeled empty state")
	}
	return QueryCoverage{Kind: kind, Name: name, RefID: name, Expr: expr, Metrics: metrics, Authority: queryAuthority(uid, metrics), State: state}, nil
}

var vectorZeroRE = regexp.MustCompile(`(?i)\bvector\s*\(\s*[+-]?0(?:\.0+)?\s*\)`)

func dishonestZero(expr string) bool { return vectorZeroRE.MatchString(expr) }

func honestUnavailable(panel panelJSON, expr string) bool {
	var field struct {
		Defaults struct {
			NoValue  string `json:"noValue"`
			Mappings []struct {
				Options map[string]struct {
					Text string `json:"text"`
				} `json:"options"`
			} `json:"mappings"`
		} `json:"defaults"`
	}
	_ = json.Unmarshal(panel.FieldConfig, &field)
	if strings.EqualFold(field.Defaults.NoValue, "UNAVAILABLE") {
		return true
	}
	if strings.Contains(expr, "fak_native_backend_available") && strings.Contains(strings.ToLower(panel.Description), "unavailable") {
		return true
	}
	if strings.Contains(expr, "vector(-1)") || strings.Contains(expr, "vector( -1 )") {
		for _, mapping := range field.Defaults.Mappings {
			if value, ok := mapping.Options["-1"]; ok && strings.EqualFold(value.Text, "UNAVAILABLE") {
				return true
			}
		}
	}
	return false
}

// NormalizePromQL substitutes bounded fixture values for Grafana variables and
// converts label_values(variable queries) into the selector PromQL they inspect.
func NormalizePromQL(expr string) (string, error) {
	trimmed := strings.TrimSpace(expr)
	if strings.HasPrefix(trimmed, "label_values(") {
		inner := strings.TrimSuffix(strings.TrimPrefix(trimmed, "label_values("), ")")
		comma := strings.LastIndex(inner, ",")
		if comma < 0 {
			return "", errors.New("unsupported label_values expression")
		}
		trimmed = strings.TrimSpace(inner[:comma])
	}
	replacements := []struct{ old, new string }{
		{"$__rate_interval", "5m"},
		{"$engine", "fak-native"},
		{"$model", "Qwen3.8-4B"},
		{"$backend", "cuda"},
		{"$forward_path", "qwen_cuda"},
		{"$scenario", "good"},
		{"$benchmark_envelope", "qwen38-4b-in128-out128-b1-quality-v3"},
		{"$correlation_key", "npc1_0123456789abcdef0123456789abcdef"},
	}
	for _, replacement := range replacements {
		trimmed = strings.ReplaceAll(trimmed, replacement.old, replacement.new)
	}
	if strings.Contains(trimmed, "$") && !strings.Contains(trimmed, `"$1"`) {
		return "", fmt.Errorf("unsupported Grafana variable in %q", trimmed)
	}
	return trimmed, nil
}
