package nativeperfcoverage

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

var (
	correlationKeyRE = regexp.MustCompile(`^npc1_[0-9a-f]{32}$`)
	forbiddenLabels  = map[string]bool{
		"request_id": true, "trace_id": true, "session_id": true,
		"prompt": true, "response": true, "user": true, "raw_path": true,
	}
)

func validateSeries(series []Series, allowed map[string]bool, contract contractJSON, maxLabels, maxValues int) error {
	values := make(map[string]map[string]bool)
	for _, sample := range series {
		if !allowed[sample.Metric] {
			return fmt.Errorf("fixture contains unknown or renamed metric %q", sample.Metric)
		}
		if len(sample.Labels) > maxLabels {
			return fmt.Errorf("metric %q has %d labels, bounded maximum is %d", sample.Metric, len(sample.Labels), maxLabels)
		}
		for label, value := range sample.Labels {
			if forbiddenLabels[label] {
				return fmt.Errorf("metric %q carries unbounded label %q", sample.Metric, label)
			}
			key := sample.Metric + "\x00" + label
			if values[key] == nil {
				values[key] = make(map[string]bool)
			}
			values[key][value] = true
			if len(values[key]) > maxValues {
				return fmt.Errorf("metric %q label %q has more than %d controlled values", sample.Metric, label, maxValues)
			}
		}
		if engine, ok := sample.Labels["engine"]; ok && engine != NativeEngine {
			return fmt.Errorf("metric %q names non-native engine %q", sample.Metric, engine)
		}
		if model, ok := sample.Labels["model"]; ok && !strings.HasPrefix(model, Qwen38Prefix) {
			return fmt.Errorf("metric %q names non-Qwen3.8 model %q", sample.Metric, model)
		}
		if family, ok := sample.Labels["model_family"]; ok && family != Qwen38Prefix && family != "other" {
			return fmt.Errorf("metric %q has unbounded model_family %q", sample.Metric, family)
		}
		if key, ok := sample.Labels["correlation_key"]; ok && !correlationKeyRE.MatchString(key) {
			return fmt.Errorf("metric %q has unbounded correlation_key %q", sample.Metric, key)
		}
		if err := validateBoundedDimensions(sample, contract.BoundedDimensions); err != nil {
			return err
		}
	}
	return nil
}

func validateBoundedDimensions(sample Series, dimensions map[string][]string) error {
	checks := map[string]string{
		"backend":      "backend",
		"model_family": "model_family",
		"reason":       "reason",
		"direction":    "direction",
		"graph_state":  "state",
	}
	switch sample.Metric {
	case "fak_native_backend_memory_bytes":
		checks["memory_kind"] = "kind"
	case "fak_native_backend_delay_seconds":
		checks["delay_kind"] = "kind"
	case "fak_native_backend_kernel_calls_total", "fak_native_backend_kernel_seconds_total":
		checks["kernel_family"] = "family"
	case "fak_native_backend_sync_events_total", "fak_native_backend_sync_seconds_total":
		checks["sync_kind"] = "kind"
	}
	keys := make([]string, 0, len(checks))
	for dimension := range checks {
		keys = append(keys, dimension)
	}
	sort.Strings(keys)
	for _, dimension := range keys {
		allowed, exists := dimensions[dimension]
		label := checks[dimension]
		value, present := sample.Labels[label]
		if !exists || !present {
			continue
		}
		if !contains(allowed, value) {
			return fmt.Errorf("metric %q label %q=%q is outside bounded %s values", sample.Metric, label, value, dimension)
		}
	}
	return nil
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
