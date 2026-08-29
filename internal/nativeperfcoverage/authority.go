package nativeperfcoverage

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/nativeperfslo"
)

const (
	authorityFixture = "controlled-fixture"
	authorityGateway = "internal/gateway"
	authorityNative  = "internal/nativeperf"
	authoritySLO     = "internal/nativeperfslo"
)

// currentDashboardMetrics is the bounded producer-backed metric inventory used
// by the rewritten dashboards. The provisioned contracts still authorize the
// controlled legacy fixtures, so validation extends rather than replaces that
// fixture allowlist.
var currentDashboardMetrics = map[string][]string{
	"fak-native-kernel-performance": {
		"fak_gateway_inference_cached_prompt_hit_ratio",
		"fak_gateway_inference_output_tokens_per_second",
		"fak_gateway_inference_tpot_seconds_bucket",
		"fak_gateway_inference_ttft_seconds_bucket",
		"fak_native_receipt_bytes_total",
		"fak_native_receipt_latest_age_seconds",
		"fak_native_receipt_latest_stale",
		"fak_native_receipt_phase_seconds_total",
		"fak_native_receipt_requests_total",
		"fak_native_receipt_signal_supported",
		"fak_native_receipt_unsupported_total",
		"fak_native_slo_state",
	},
	"fak-native-backends": {
		"fak_native_receipt_bytes_total",
		"fak_native_receipt_latest_age_seconds",
		"fak_native_receipt_latest_stale",
		"fak_native_receipt_phase_seconds_total",
		"fak_native_receipt_requests_total",
		"fak_native_receipt_signal_supported",
		"fak_native_receipt_unsupported_total",
	},
	"fak-native-artifacts": {
		"fak_native_receipt_latest_stale",
		"fak_native_receipt_requests_total",
	},
}

func extendCurrentDashboardMetrics(uid string, allowed map[string]bool) {
	for _, metric := range currentDashboardMetrics[uid] {
		allowed[metric] = true
	}
}

// validateAuthoritativeMetrics binds current dashboard queries to bounded
// producer inventories while retaining the controlled fixture metrics declared
// by the provisioned contracts.
func validateAuthoritativeMetrics(uid string, allowed map[string]bool) error {
	for _, metric := range currentDashboardMetrics[uid] {
		if !allowed[metric] {
			return fmt.Errorf("current producer metric %q is absent from the dashboard authority", metric)
		}
	}
	if uid != "fak-native-slo" {
		return nil
	}
	result := nativeperfslo.Result{
		At:       time.Unix(1, 0).UTC(),
		Envelope: nativeperfslo.Envelope{ModuleRev: "internal/nativeperf@r1+g000000000", Benchmark: "qwen38-fixture", Model: "Qwen3.8-4B", Backend: "cuda"},
		State:    nativeperfslo.StateGood,
		Values: map[nativeperfslo.Objective]nativeperfslo.Value{
			nativeperfslo.TTFT: {Available: true, Value: 1},
		},
		Ratios: map[nativeperfslo.Objective]nativeperfslo.Value{
			nativeperfslo.TTFT: {Available: true, Value: 1},
		},
		Violations: map[nativeperfslo.Objective]bool{},
	}
	series, err := ParseFixture([]byte(nativeperfslo.RenderPrometheus(result)))
	if err != nil {
		return fmt.Errorf("parse authoritative SLO producer output: %w", err)
	}
	produced := make(map[string]bool)
	for _, sample := range series {
		produced[sample.Metric] = true
	}
	for _, required := range []string{"fak_native_slo_state", "fak_native_slo_value", "fak_native_slo_ratio", "fak_native_slo_violation"} {
		if !allowed[required] || !produced[required] {
			return fmt.Errorf("SLO producer metric %q is absent from contract or authoritative output", required)
		}
	}
	// Lifecycle annotations are a distinct controlled fixture until their
	// bounded producer is wired; do not relabel them as renderer output.
	if !allowed["fak_native_lifecycle_event_info"] {
		return fmt.Errorf("SLO lifecycle annotation metric is absent from contract")
	}
	return nil
}

func queryAuthority(_ string, metrics []string) string {
	authorities := make(map[string]bool)
	for _, metric := range metrics {
		authority := authorityFixture
		switch {
		case strings.HasPrefix(metric, "fak_native_receipt_"):
			authority = authorityNative
		case strings.HasPrefix(metric, "fak_gateway_inference_"):
			authority = authorityGateway
		case strings.HasPrefix(metric, "fak_native_slo_"):
			authority = authoritySLO
		}
		authorities[authority] = true
	}
	var values []string
	for authority := range authorities {
		values = append(values, authority)
	}
	sort.Strings(values)
	return strings.Join(values, "+")
}
