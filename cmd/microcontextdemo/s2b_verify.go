package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

const s2bSchema = "fak-microcontext-kernel-prefix-ab/1"

func verifyS2BArtifact(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var report map[string]any
	if err := json.Unmarshal(b, &report); err != nil {
		return err
	}
	return verifyS2BReport(report)
}

func verifyS2BReport(report map[string]any) error {
	if text(report, "schema") != s2bSchema {
		return fmt.Errorf("schema: got %q", text(report, "schema"))
	}
	for _, key := range []string{"captured_at", "endpoint_mode", "cache_" + "implementation", "model", "model_artifact", "hardware", "forward_path", "reset_control", "warm_control", "unique_mutation", "tuned_sequential_baseline", "provider_native_batch_status", "claim_scope"} {
		if text(report, key) == "" {
			return fmt.Errorf("missing %s", key)
		}
	}
	if len(text(report, "base_fingerprint_sha256")) != 64 || text(report, "claim_verdict") != "net-true" || !boolean(report, "usage_"+"kernel_"+"reconciliation") {
		return fmt.Errorf("control/verdict boundary mismatch")
	}
	unique, err := object(report, "unique")
	if err != nil {
		return err
	}
	shared, err := object(report, "shared")
	if err != nil {
		return err
	}
	if err := checkS2BArm("unique", unique); err != nil {
		return err
	}
	if err := checkS2BArm("shared", shared); err != nil {
		return err
	}
	if text(unique, "prefix_mode") != "unique" || text(shared, "prefix_mode") != "shared" {
		return fmt.Errorf("prefix modes mismatch")
	}
	if number(shared, "kernel_"+"reused_"+"tokens_"+"delta") <= number(unique, "kernel_"+"reused_"+"tokens_"+"delta") {
		return fmt.Errorf("shared reuse did not exceed unique")
	}
	rateRatio := number(shared, "contexts_"+"per_"+"second") / number(unique, "contexts_"+"per_"+"second")
	if math.Abs(rateRatio-number(report, "shared_vs_unique_throughput_ratio")) > 1e-6 || math.Abs(number(unique, "wall_ms")/number(shared, "wall_ms")-number(report, "shared_vs_unique_wall_ratio")) > 1e-6 {
		return fmt.Errorf("derived ratio mismatch")
	}
	if number(report, "shared_"+"warm_"+"cached_"+"fraction") <= 0.9 {
		return fmt.Errorf("shared warm reuse fraction too low")
	}
	if excluded, ok := report["excluded_claims"].([]any); !ok || len(excluded) == 0 {
		return fmt.Errorf("excluded claims missing")
	}
	return nil
}

func checkS2BArm(name string, arm map[string]any) error {
	contexts := int(number(arm, "contexts"))
	rows, ok := arm["rows"].([]any)
	if !boolean(arm, "fresh_process") || contexts < 2 || int(number(arm, "physical_workers")) != 1 || int(number(arm, "completed")) != contexts || number(arm, "failed") != 0 || !ok || len(rows) != contexts {
		return fmt.Errorf("%s accounting/control mismatch", name)
	}
	if int(number(arm, "kernel_"+"turns_"+"delta")) != contexts || number(arm, "kernel_"+"prompt_"+"tokens_"+"delta") != number(arm, "prompt_tokens") || number(arm, "kernel_"+"reused_"+"tokens_"+"delta") != number(arm, "cached_prompt_tokens") {
		return fmt.Errorf("%s endpoint/usage mismatch", name)
	}
	var prompt, cached float64
	for _, raw := range rows {
		row, ok := raw.(map[string]any)
		if !ok || !boolean(row, "nonempty") {
			return fmt.Errorf("%s invalid row", name)
		}
		prompt += number(row, "prompt_tokens")
		cached += number(row, "cached_prompt_tokens")
	}
	if prompt != number(arm, "prompt_tokens") || cached != number(arm, "cached_prompt_tokens") {
		return fmt.Errorf("%s row totals mismatch", name)
	}
	return nil
}

func object(v map[string]any, key string) (map[string]any, error) {
	out, ok := v[key].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("missing object %s", key)
	}
	return out, nil
}
func text(v map[string]any, key string) string    { out, _ := v[key].(string); return out }
func number(v map[string]any, key string) float64 { out, _ := v[key].(float64); return out }
func boolean(v map[string]any, key string) bool   { out, _ := v[key].(bool); return out }
