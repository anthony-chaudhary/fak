package nativeperfcoverage

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// TestNormalizePromQLInvariants proves that NormalizePromQL performs required
// variable substitutions, extracts inner queries from label_values, and rejects
// unhandled Grafana variables.
func TestNormalizePromQLInvariants(t *testing.T) {
	cases := []struct {
		name    string
		input   string
		want    string
		wantErr bool
	}{
		{
			name:  "rate interval substitution",
			input: "rate(my_metric[$__rate_interval])",
			want:  "rate(my_metric[5m])",
		},
		{
			name:  "all standard template variables",
			input: "my_metric{engine=\"$engine\",model=\"$model\",backend=\"$backend\",forward_path=\"$forward_path\",scenario=\"$scenario\",benchmark_envelope=\"$benchmark_envelope\",correlation_key=\"$correlation_key\"}",
			want:  "my_metric{engine=\"fak-native\",model=\"Qwen3.8-4B\",backend=\"cuda\",forward_path=\"qwen_cuda\",scenario=\"good\",benchmark_envelope=\"qwen38-4b-in128-out128-b1-quality-v3\",correlation_key=\"npc1_0123456789abcdef0123456789abcdef\"}",
		},
		{
			name:  "label_values expression extraction",
			input: "label_values(fak_native_receipt_requests_total{engine=\"inkernel\"}, backend)",
			want:  "fak_native_receipt_requests_total{engine=\"inkernel\"}",
		},
		{
			name:    "label_values without comma fails",
			input:   "label_values(invalid_query)",
			wantErr: true,
		},
		{
			name:    "unsupported variable fails",
			input:   "sum(rate(unsupported_metric{tier=\"$unknown_variable\"}[5m]))",
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NormalizePromQL(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("NormalizePromQL(%q) expected error, got nil", tc.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("NormalizePromQL(%q) unexpected error: %v", tc.input, err)
			}
			if got != tc.want {
				t.Fatalf("NormalizePromQL(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

// TestDishonestZeroDetectionInvariants verifies that dishonest zero coercions are rejected
// while permitted negative or non-zero sentinels pass through.
func TestDishonestZeroDetectionInvariants(t *testing.T) {
	dishonestCases := []string{
		"vector(0)",
		"vector( 0 )",
		"vector(0.0)",
		"vector(0.000)",
		"vector(+0)",
		"vector(-0)",
		"vector(-0.0)",
		"sum(rate(reqs[5m])) or vector(0)",
		"VECTOR(0)",
	}

	for _, expr := range dishonestCases {
		if !dishonestZero(expr) {
			t.Errorf("dishonestZero(%q) = false, want true", expr)
		}
	}

	honestCases := []string{
		"vector(-1)",
		"vector( -1 )",
		"vector(1)",
		"vector(0.5)",
		"sum(rate(reqs[5m])) or vector(-1)",
		"max(fak_native_receipt_signal_supported{signal=\"queue\"})",
	}

	for _, expr := range honestCases {
		if dishonestZero(expr) {
			t.Errorf("dishonestZero(%q) = true, want false", expr)
		}
	}
}

// TestMetricNamesExtractionInvariants verifies that MetricNames extracts and sorts
// metric selectors from complex expressions without label contamination.
func TestMetricNamesExtractionInvariants(t *testing.T) {
	expr := `(sum(rate(metric_b{phase="queue"}[5m])) + sum(rate(metric_a{kind="kv"}[5m]))) or vector(-1)`
	names := MetricNames(expr)
	if len(names) != 2 {
		t.Fatalf("MetricNames(%q) length=%d, want 2", expr, len(names))
	}
	if names[0] != "metric_a" || names[1] != "metric_b" {
		t.Fatalf("MetricNames(%q) = %v, want [metric_a metric_b] (sorted)", expr, names)
	}

	// Unbraced metric recognized via allowed map
	allowed := map[string]bool{"metric_c": true}
	unbracedExpr := `rate(metric_c[5m])`
	extracted := metricNames(unbracedExpr, allowed)
	if len(extracted) != 1 || extracted[0] != "metric_c" {
		t.Fatalf("metricNames(%q) with allowed = %v, want [metric_c]", unbracedExpr, extracted)
	}
}

// TestHonestUnavailableInvariants verifies panel formatting checks for honest unavailable display.
func TestHonestUnavailableInvariants(t *testing.T) {
	// Honest case 1: noValue is UNAVAILABLE
	honestPanel1 := panelJSON{
		FieldConfig: json.RawMessage(`{"defaults":{"noValue":"UNAVAILABLE"}}`),
	}
	if !honestUnavailable(honestPanel1, "some_metric") {
		t.Error("honestUnavailable should be true when defaults.noValue is UNAVAILABLE")
	}

	// Honest case 2: description contains unavailable for fak_native_backend_available
	honestPanel2 := panelJSON{
		Description: "Shows backend status; UNAVAILABLE when offline",
	}
	if !honestUnavailable(honestPanel2, "fak_native_backend_available{backend=\"cuda\"}") {
		t.Error("honestUnavailable should be true for backend available with unavailable description")
	}

	// Honest case 3: vector(-1) with value mapping -1 -> UNAVAILABLE
	honestPanel3 := panelJSON{
		FieldConfig: json.RawMessage(`{"defaults":{"mappings":[{"options":{"-1":{"text":"UNAVAILABLE"}}}]}}`),
	}
	if !honestUnavailable(honestPanel3, "rate(foo[5m]) or vector(-1)") {
		t.Error("honestUnavailable should be true when -1 mapping specifies UNAVAILABLE")
	}

	// Dishonest case: no honest mapping or sentinel
	dishonestPanel := panelJSON{
		FieldConfig: json.RawMessage(`{"defaults":{"noValue":"0"}}`),
	}
	if honestUnavailable(dishonestPanel, "some_metric or vector(-1)") {
		t.Error("honestUnavailable should be false when defaults.noValue is 0")
	}
}

// TestSpecsCoverageInvariants verifies completeness and uniqueness of Specs entries.
func TestSpecsCoverageInvariants(t *testing.T) {
	specs := Specs()
	if len(specs) != 4 {
		t.Fatalf("Specs() returned %d specs, want 4", len(specs))
	}

	seenUID := make(map[string]bool)
	for _, s := range specs {
		if s.UID == "" {
			t.Errorf("spec has empty UID: %+v", s)
		}
		if seenUID[s.UID] {
			t.Errorf("duplicate spec UID: %s", s.UID)
		}
		seenUID[s.UID] = true

		if s.Dashboard == "" || s.Contract == "" || s.Fixture == "" {
			t.Errorf("spec %s has missing paths: %+v", s.UID, s)
		}
		if s.RequiredJob != "fak_gateway" {
			t.Errorf("spec %s RequiredJob = %q, want fak_gateway", s.UID, s.RequiredJob)
		}
	}
}

// TestQueryCheckerFuncInvariant verifies QueryCheckerFunc interface adaptation.
func TestQueryCheckerFuncInvariant(t *testing.T) {
	called := false
	testErr := errors.New("query check failure")
	checker := QueryCheckerFunc(func(_ context.Context, expr string) error {
		called = true
		if expr == "bad_expr" {
			return testErr
		}
		return nil
	})

	ctx := context.Background()
	if err := checker.Check(ctx, "good_expr"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("expected checker function to be called")
	}

	if err := checker.Check(ctx, "bad_expr"); !errors.Is(err, testErr) {
		t.Fatalf("expected testErr, got: %v", err)
	}
}

// TestMatrixTextDeterministicInvariant verifies stable serialization of the coverage matrix.
func TestMatrixTextDeterministicInvariant(t *testing.T) {
	m := Matrix{
		Dashboards: []DashboardCoverage{
			{
				UID:   "dash-1",
				Title: "Dashboard 1",
				Panels: []PanelCoverage{
					{
						ID:    1,
						Title: "Panel 1",
						Type:  "stat",
						State: "fixture-populated",
						Queries: []QueryCoverage{
							{
								Kind:      PanelTarget,
								Name:      "A",
								RefID:     "A",
								Expr:      "rate(foo[5m])",
								Metrics:   []string{"foo"},
								Authority: "internal/foo",
								State:     "fixture-populated",
							},
						},
					},
				},
				Queries: []QueryCoverage{
					{
						Kind:      Variable,
						Name:      "backend",
						Expr:      "label_values(foo, backend)",
						Metrics:   []string{"foo"},
						Authority: "internal/foo",
						State:     "fixture-populated",
					},
				},
			},
		},
	}

	text1 := m.Text()
	text2 := m.Text()
	if text1 != text2 {
		t.Fatal("Matrix.Text() is not deterministic across invocations")
	}

	if !strings.Contains(text1, "DASHBOARD\tdash-1\tDashboard 1\n") {
		t.Errorf("missing DASHBOARD header in: %s", text1)
	}
	if !strings.Contains(text1, "PANEL\tdash-1\t1\tstat\tfixture-populated\tPanel 1\n") {
		t.Errorf("missing PANEL line in: %s", text1)
	}
	if !strings.Contains(text1, "QUERY\tdash-1\t1\tA\tfixture-populated\tinternal/foo\tfoo\trate(foo[5m])\n") {
		t.Errorf("missing QUERY line in: %s", text1)
	}
	if !strings.Contains(text1, "VARIABLE\tdash-1\tbackend\tfixture-populated\tinternal/foo\tfoo\tlabel_values(foo, backend)\n") {
		t.Errorf("missing VARIABLE line in: %s", text1)
	}
}
