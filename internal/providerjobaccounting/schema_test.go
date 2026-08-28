package providerjobaccounting

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

const (
	schemaName = "fak-provider-job-accounting/1"
	issueID    = "#9575"
)

var (
	contextBuckets = []int{35000, 64000, 128000, 200000}
	workloadRatios = []int{200, 300}
	rawCounterKeys = []string{
		"input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens",
		"reasoning_tokens", "tool_calls", "search_calls", "container_sessions",
		"request_attempts", "provider_reported_cost_usd",
	}
)

func TestSchemaDeclaresClosedIssue9575Contract(t *testing.T) {
	schema := readJSONObject(t, docsPath("standards", "provider-job-accounting-schema.json"))
	if got := schema["$schema"]; got != "https://json-schema.org/draft/2020-12/schema" {
		t.Fatalf("$schema = %v", got)
	}
	defs := object(t, schema, "$defs")
	common := object(t, defs, "common")
	properties := object(t, common, "properties")
	if got := object(t, properties, "schema")["const"]; got != schemaName {
		t.Fatalf("schema const = %v", got)
	}
	if got := object(t, properties, "issue")["const"]; got != issueID {
		t.Fatalf("issue const = %v", got)
	}

	envelope := object(t, defs, "workload_envelope")
	envelopeProperties := object(t, envelope, "properties")
	assertNumberEnum(t, object(t, envelopeProperties, "context_bucket_tokens"), "enum", contextBuckets)
	assertNumberEnum(t, object(t, envelopeProperties, "input_to_output_ratio"), "enum", workloadRatios)

	raw := object(t, defs, "raw_provider_counters")
	required := stringSlice(t, raw["required"])
	for _, key := range rawCounterKeys {
		if !slices.Contains(required, key) {
			t.Errorf("raw provider counters do not require %q", key)
		}
	}
	for _, defName := range []string{"official_terms_record", "raw_provider_counters", "local_job_outcome", "derived_accounting"} {
		if _, ok := defs[defName]; !ok {
			t.Errorf("schema missing required separation definition %q", defName)
		}
	}
}

func TestRepresentativeLedgersValidate(t *testing.T) {
	paths := []string{
		docsPath("standards", "fixtures", "provider-job-accounting-gpt56-sol-api.jsonl"),
		docsPath("standards", "fixtures", "provider-job-accounting-local.jsonl"),
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			records := readJSONL(t, path)
			if err := validateLedger(records); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGPT56SolOfficialTermsFixture(t *testing.T) {
	records := readJSONL(t, docsPath("standards", "fixtures", "provider-job-accounting-gpt56-sol-api.jsonl"))
	terms := object(t, records[0], "official_commercial_terms")
	if got := terms["checked_on"]; got != "2026-08-27" {
		t.Fatalf("checked_on = %v", got)
	}
	if got := terms["context_window_tokens"]; got != float64(1050000) {
		t.Fatalf("context window = %v", got)
	}
	if got := terms["max_output_tokens"]; got != float64(128000) {
		t.Fatalf("max output = %v", got)
	}
	assertRates(t, object(t, terms, "standard_rates"), []float64{4, .4, 5, 20})
	long := object(t, terms, "long_context")
	if got := long["threshold_input_tokens_exclusive"]; got != float64(272000) {
		t.Fatalf("long-context threshold = %v", got)
	}
	if got := long["applies_to_full_request"]; got != true {
		t.Fatalf("long-context full-request flag = %v", got)
	}
	assertRates(t, object(t, long, "rates"), []float64{8, .8, 10, 30})
	promotion := object(t, terms, "promotional_pricing")
	if got := promotion["available_through_at_least"]; got != "2026-11-21" {
		t.Fatalf("promotion date = %v", got)
	}
}

func TestAbsentProviderCountersStayNull(t *testing.T) {
	records := readJSONL(t, docsPath("standards", "fixtures", "provider-job-accounting-local.jsonl"))
	raw := object(t, records[0], "raw_provider_counters")
	for _, key := range rawCounterKeys {
		value, ok := raw[key]
		if !ok {
			t.Errorf("raw counter %q absent; want explicit null", key)
		} else if value != nil {
			t.Errorf("raw counter %q = %v; want null for unavailable local provider telemetry", key, value)
		}
	}
}

func TestGPTFixturePreservesObservedProviderCounters(t *testing.T) {
	records := readJSONL(t, docsPath("standards", "fixtures", "provider-job-accounting-gpt56-sol-api.jsonl"))
	var observation map[string]any
	for _, record := range records {
		if record["record_kind"] == "provider_counter_observation" {
			observation = record
			break
		}
	}
	if observation == nil {
		t.Fatal("GPT fixture lacks provider_counter_observation")
	}
	if got := observation["observation_scope"]; got != "counter_only_not_a_quality_qualified_job_receipt" {
		t.Fatalf("observation_scope = %v", got)
	}
	turns, ok := observation["turns"].([]any)
	if !ok || len(turns) != 2 {
		t.Fatalf("turns = %T %v, want two turns", observation["turns"], observation["turns"])
	}
	want := []struct{ input, cached, write float64 }{{28473, 3456, 0}, {39739, 3456, 0}}
	for i, value := range turns {
		turn, ok := value.(map[string]any)
		if !ok {
			t.Fatalf("turn %d is %T", i+1, value)
		}
		for key, expected := range map[string]float64{"input_tokens": want[i].input, "cached_input_tokens": want[i].cached, "cache_write_input_tokens": want[i].write} {
			if got, ok := number(turn[key]); !ok || got != expected {
				t.Fatalf("turn %d %s = %v, want %.0f", i+1, key, turn[key], expected)
			}
		}
		for _, key := range []string{"output_tokens", "reasoning_tokens", "outcome", "quality_result", "elapsed_seconds", "cost_usd"} {
			if value, present := turn[key]; !present || value != nil {
				t.Fatalf("turn %d %s = %v (present=%v), want explicit null", i+1, key, value, present)
			}
		}
	}
	unavailable := stringSlice(t, observation["unavailable_fields"])
	for _, field := range []string{"task_revision", "output_tokens", "reasoning_tokens", "outcome", "quality_result", "elapsed_seconds", "cost_usd", "realized_ratio"} {
		if !slices.Contains(unavailable, field) {
			t.Fatalf("unavailable_fields lacks %q: %v", field, unavailable)
		}
	}
	if _, exists := observation["workload_envelope"]; exists {
		t.Fatal("counter-only observation must not fabricate a workload envelope or task revision")
	}
	if _, exists := observation["derived_accounting"]; exists {
		t.Fatal("counter-only observation must not fabricate costs or a realized ratio")
	}
	next := object(t, observation, "next_evidence")
	if next["#9552"] == "" || next["#9578"] == "" {
		t.Fatalf("next evidence pointers incomplete: %v", next)
	}

	for name, mutate := range map[string]func(map[string]any){
		"unknown output cannot become a number": func(record map[string]any) {
			turn := record["turns"].([]any)[0].(map[string]any)
			turn["output_tokens"] = float64(1)
		},
		"unknown quality cannot be omitted": func(record map[string]any) {
			turn := record["turns"].([]any)[0].(map[string]any)
			delete(turn, "quality_result")
		},
	} {
		t.Run(name, func(t *testing.T) {
			mutated := clone(t, observation)
			mutate(mutated)
			if err := validateCounterObservation(mutated); err == nil {
				t.Fatal("validation unexpectedly accepted fabricated or omitted uncertainty")
			}
		})
	}
}

func TestValidationRejectsDishonestOrIncompleteRecords(t *testing.T) {
	base := readJSONL(t, docsPath("standards", "fixtures", "provider-job-accounting-gpt56-sol-api.jsonl"))[1]
	tests := []struct {
		name   string
		mutate func(map[string]any)
	}{
		{name: "missing raw counter", mutate: func(record map[string]any) {
			delete(record["raw_provider_counters"].(map[string]any), "cache_write_tokens")
		}},
		{name: "unsupported context bucket", mutate: func(record map[string]any) {
			record["workload_envelope"].(map[string]any)["context_bucket_tokens"] = float64(100000)
		}},
		{name: "qualified cost without quality pass", mutate: func(record map[string]any) {
			record["local_job_outcome"].(map[string]any)["quality_gate"].(map[string]any)["status"] = "fail"
		}},
		{name: "double counted meter total", mutate: func(record map[string]any) {
			record["derived_accounting"].(map[string]any)["attempted_job_cost_usd"] = 9.0
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			record := clone(t, base)
			tc.mutate(record)
			if err := validateLedger([]map[string]any{record}); err == nil {
				t.Fatal("validation unexpectedly accepted mutated record")
			}
		})
	}
}

func validateLedger(records []map[string]any) error {
	seen := make(map[string]struct{}, len(records))
	for i, record := range records {
		if record["schema"] != schemaName || record["issue"] != issueID {
			return fmt.Errorf("record %d has wrong schema or issue", i+1)
		}
		id, ok := record["record_id"].(string)
		if !ok || id == "" {
			return fmt.Errorf("record %d has no record_id", i+1)
		}
		if _, duplicate := seen[id]; duplicate {
			return fmt.Errorf("record %d duplicates record_id %q", i+1, id)
		}
		if correction := record["corrects_record_id"]; correction != nil {
			corrects, ok := correction.(string)
			if !ok {
				return fmt.Errorf("record %q has invalid correction reference", id)
			}
			if _, earlier := seen[corrects]; !earlier {
				return fmt.Errorf("record %q correction does not reference an earlier record", id)
			}
		}
		seen[id] = struct{}{}

		switch record["record_kind"] {
		case "official_commercial_terms":
			if _, ok := record["official_commercial_terms"].(map[string]any); !ok {
				return fmt.Errorf("terms record %q lacks official terms", id)
			}
		case "job_receipt":
			if err := validateJob(record); err != nil {
				return fmt.Errorf("job %q: %w", id, err)
			}
		case "provider_counter_observation":
			if err := validateCounterObservation(record); err != nil {
				return fmt.Errorf("provider counter observation %q: %w", id, err)
			}
		default:
			return fmt.Errorf("record %q has unsupported kind", id)
		}
	}
	return nil
}

func validateCounterObservation(record map[string]any) error {
	if record["observation_scope"] != "counter_only_not_a_quality_qualified_job_receipt" {
		return fmt.Errorf("wrong observation scope")
	}
	turns, ok := record["turns"].([]any)
	if !ok || len(turns) == 0 {
		return fmt.Errorf("missing observed turns")
	}
	for i, value := range turns {
		turn, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("turn %d is not an object", i+1)
		}
		for _, key := range []string{"input_tokens", "cached_input_tokens", "cache_write_input_tokens"} {
			if _, ok := number(turn[key]); !ok {
				return fmt.Errorf("turn %d lacks observed %s", i+1, key)
			}
		}
		for _, key := range []string{"output_tokens", "reasoning_tokens", "outcome", "quality_result", "elapsed_seconds", "cost_usd"} {
			if value, present := turn[key]; !present || value != nil {
				return fmt.Errorf("turn %d %s must be explicit null", i+1, key)
			}
		}
	}
	for _, forbidden := range []string{"workload_envelope", "local_job_outcome", "derived_accounting"} {
		if _, present := record[forbidden]; present {
			return fmt.Errorf("must not contain %s", forbidden)
		}
	}
	return nil
}

func validateJob(record map[string]any) error {
	envelope, ok := record["workload_envelope"].(map[string]any)
	if !ok {
		return fmt.Errorf("missing workload envelope")
	}
	if !containsJSONNumber(contextBuckets, envelope["context_bucket_tokens"]) {
		return fmt.Errorf("unsupported context bucket")
	}
	if !containsJSONNumber(workloadRatios, envelope["input_to_output_ratio"]) {
		return fmt.Errorf("unsupported workload ratio")
	}
	raw, ok := record["raw_provider_counters"].(map[string]any)
	if !ok {
		return fmt.Errorf("missing raw provider counters")
	}
	for _, key := range rawCounterKeys {
		if _, present := raw[key]; !present {
			return fmt.Errorf("raw provider counter %q must be present (null when unavailable)", key)
		}
	}
	outcome, ok := record["local_job_outcome"].(map[string]any)
	if !ok {
		return fmt.Errorf("missing local job outcome")
	}
	quality, ok := outcome["quality_gate"].(map[string]any)
	if !ok {
		return fmt.Errorf("missing quality gate")
	}
	derived, ok := record["derived_accounting"].(map[string]any)
	if !ok {
		return fmt.Errorf("missing derived accounting")
	}
	qualified := outcome["status"] == "completed" && quality["status"] == "pass"
	if qualified != (derived["quality_qualified_completed_job_cost_usd"] != nil && derived["qualified_wall_time_seconds"] != nil) {
		return fmt.Errorf("qualified cost and wall time must exist iff completion passes quality")
	}
	attempted, ok := number(derived["attempted_job_cost_usd"])
	if !ok {
		return fmt.Errorf("attempted job cost must be numeric")
	}
	for _, key := range []string{"meter_cost_usd", "phase_cost_attribution_usd"} {
		partition, ok := derived[key].(map[string]any)
		if !ok {
			return fmt.Errorf("missing %s", key)
		}
		total, complete := sumNumbers(partition)
		if complete && math.Abs(total-attempted) > 1e-9 {
			return fmt.Errorf("%s total %.12g != attempted job cost %.12g", key, total, attempted)
		}
	}
	return nil
}

func assertRates(t *testing.T, rates map[string]any, want []float64) {
	t.Helper()
	keys := []string{"input_per_mtok_usd", "cached_input_per_mtok_usd", "cache_write_per_mtok_usd", "output_per_mtok_usd"}
	for i, key := range keys {
		if got := rates[key]; got != want[i] {
			t.Errorf("%s = %v, want %v", key, got, want[i])
		}
	}
}

func assertNumberEnum(t *testing.T, object map[string]any, key string, want []int) {
	t.Helper()
	values, ok := object[key].([]any)
	if !ok || len(values) != len(want) {
		t.Fatalf("%s = %#v, want %v", key, object[key], want)
	}
	for i, value := range values {
		if value != float64(want[i]) {
			t.Fatalf("%s[%d] = %v, want %d", key, i, value, want[i])
		}
	}
}

func readJSONObject(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var value map[string]any
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return value
}

func readJSONL(t *testing.T, path string) []map[string]any {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var records []map[string]any
	scanner := bufio.NewScanner(file)
	for line := 1; scanner.Scan(); line++ {
		var record map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			t.Fatalf("decode %s:%d: %v", path, line, err)
		}
		records = append(records, record)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(records) == 0 {
		t.Fatalf("%s has no records", path)
	}
	return records
}

func object(t *testing.T, parent map[string]any, key string) map[string]any {
	t.Helper()
	value, ok := parent[key].(map[string]any)
	if !ok {
		t.Fatalf("%q is %T, want object", key, parent[key])
	}
	return value
}

func stringSlice(t *testing.T, value any) []string {
	t.Helper()
	values, ok := value.([]any)
	if !ok {
		t.Fatalf("value is %T, want array", value)
	}
	result := make([]string, len(values))
	for i, value := range values {
		var ok bool
		result[i], ok = value.(string)
		if !ok {
			t.Fatalf("array value %d is %T, want string", i, value)
		}
	}
	return result
}

func containsJSONNumber(values []int, value any) bool {
	n, ok := number(value)
	if !ok {
		return false
	}
	return slices.Contains(values, int(n)) && n == float64(int(n))
}

func number(value any) (float64, bool) {
	n, ok := value.(float64)
	return n, ok && n >= 0
}

func sumNumbers(values map[string]any) (float64, bool) {
	var total float64
	for _, value := range values {
		if value == nil {
			return 0, false
		}
		n, ok := number(value)
		if !ok {
			return 0, false
		}
		total += n
	}
	return total, true
}

func clone(t *testing.T, value map[string]any) map[string]any {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatal(err)
	}
	return result
}

func docsPath(parts ...string) string {
	return filepath.Join(append([]string{"..", "..", "docs"}, parts...)...)
}
