package nativeperfcoverage

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/nativeperfbackend"
	"github.com/anthony-chaudhary/fak/internal/nativeperfslo"
)

const (
	authorityFixture  = "controlled-fixture"
	authorityBackend  = "internal/nativeperfbackend"
	authorityArtifact = "internal/nativeperfartifact"
	authoritySLO      = "internal/nativeperfslo"
)

// validateAuthoritativeMetrics binds dashboard contracts to exported producer
// inventories where those producers exist. Kernel overview metrics remain
// explicitly fixture-backed because its contract still separates existing and
// proposed families rather than exposing one Go inventory.
func validateAuthoritativeMetrics(uid string, allowed map[string]bool) error {
	switch uid {
	case "fak-native-backends":
		producer := make(map[string]bool)
		for _, metric := range nativeperfbackend.Metrics() {
			producer[metric.Name] = true
		}
		if err := requireSameMetricSet(allowed, producer); err != nil {
			return fmt.Errorf("backend producer contract drift: %w", err)
		}
	case "fak-native-slo":
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
	case "fak-native-artifacts":
		if len(allowed) != 1 || !allowed["fak_native_artifact_info"] {
			return fmt.Errorf("artifact index contract must contain only fak_native_artifact_info")
		}
	}
	return nil
}

func requireSameMetricSet(contract, producer map[string]bool) error {
	var missing, extra []string
	for metric := range producer {
		if !contract[metric] {
			missing = append(missing, metric)
		}
	}
	for metric := range contract {
		if !producer[metric] {
			extra = append(extra, metric)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)
	if len(missing) != 0 || len(extra) != 0 {
		return fmt.Errorf("missing=%v extra=%v", missing, extra)
	}
	return nil
}

func queryAuthority(uid string, metrics []string) string {
	authorities := make(map[string]bool)
	for _, metric := range metrics {
		authority := authorityFixture
		switch uid {
		case "fak-native-backends":
			authority = authorityBackend
		case "fak-native-artifacts":
			authority = authorityArtifact
		case "fak-native-slo":
			if metric != "fak_native_lifecycle_event_info" {
				authority = authoritySLO
			}
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
