// Package nativeperfobscontract freezes the bounded observability contract for
// fak-native inference performance surfaces.
package nativeperfobscontract

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Invariant: Performance contracts must enforce the fak-native engine and reject simulated fallbacks.
// Guard: Any unbounded label dimension such as request IDs or prompts causes immediate validation failure.
// Precondition: Every registered surface must map to either an exported metric or a justified exclusion.
// Postcondition: Validated contract JSON produces canonical, deterministic formatting with stable ordering.

const (
	// Schema identifies the immutable canonical schema specification for native performance observability contracts.
	Schema = "fak-native-performance-observability-contract/1"
	// Engine defines the mandatory fak-native runtime engine identifier enforced during validation.
	Engine = "fak-native"

	// StatusExported denotes an execution surface whose performance signals are actively emitted as contracted metrics.
	StatusExported = "exported"
	// StatusExcluded denotes an execution surface that is deliberately excluded from metrics with formal architectural justification.
	StatusExcluded = "excluded"
)

// Metric defines one immutable telemetry signal with bounded labels and explicit cardinality budgets in the native performance contract.
// Invariant: All metric identifiers must adhere to the fak_native_ namespace prefix and declare a positive cardinality budget.
type Metric struct {
	Name              string   `json:"name"`
	Kind              string   `json:"kind"`
	Unit              string   `json:"unit"`
	Labels            []string `json:"labels"`
	CardinalityBudget int      `json:"cardinality_budget"`
	Provenance        string   `json:"provenance"`
	Freshness         string   `json:"freshness"`
	Unsupported       string   `json:"unsupported"`
}

// Surface maps one mandatory native execution stage to an exported metric signal or a formally justified exclusion.
// Invariant: Surfaces require non-empty producer paths and evidence links; UNKNOWN coverage status is strictly forbidden as debt.
type Surface struct {
	ID        string `json:"id"`
	Producer  string `json:"producer"`
	Status    string `json:"status"`
	Metric    string `json:"metric,omitempty"`
	Exclusion string `json:"exclusion,omitempty"`
	Evidence  string `json:"evidence"`
}

// Contract represents the complete, machine-readable coverage matrix for native inference performance observability.
// Invariant: A valid contract specifies the canonical schema, the fak-native engine, and Qwen3.8 as the default model.
type Contract struct {
	Schema       string    `json:"schema"`
	Engine       string    `json:"engine"`
	DefaultModel string    `json:"default_model"`
	Metrics      []Metric  `json:"metrics"`
	Surfaces     []Surface `json:"surfaces"`
}

// Frozen returns the canonical versioned native performance signal contract with deterministic metrics and execution surfaces.
// Invariant: Returned metrics enforce bounded label dimensions with zero unbound runtime identifiers or prompt strings.
// Postcondition: The resulting contract is fully populated and guaranteed to pass Validate without errors.
func Frozen() Contract {
	return Contract{
		Schema:       Schema,
		Engine:       Engine,
		DefaultModel: "Qwen3.8",
		Metrics: []Metric{
			metric("fak_native_model_load_seconds", "histogram", "seconds", []string{"engine", "model_family", "device_class", "outcome"}, 96, "internal/modelengine lifecycle receipt", "one sample per load attempt", "unsupported loaders emit no sample and the surface remains contract debt"),
			metric("fak_native_admission_wait_seconds", "histogram", "seconds", []string{"engine", "device_class", "queue_class", "outcome"}, 96, "internal/modelengine native scheduler admission", "one sample per admitted or rejected request", "missing queue timing is unsupported, never coerced to zero"),
			metric("fak_native_tokenization_seconds", "histogram", "seconds", []string{"engine", "model_family", "phase", "outcome"}, 64, "internal/tokenizer encode/decode operation", "one sample per tokenization operation", "tokenizers without timing emit no sample"),
			metric("fak_native_phase_seconds", "histogram", "seconds", []string{"engine", "model_family", "device_class", "phase", "outcome"}, 192, "internal/nativeperf receipt/profile phase timing", "one sample per completed phase", "absent phase timing is unsupported, never inferred from wall time"),
			metric("fak_native_tokens_total", "counter", "tokens", []string{"engine", "model_family", "phase", "outcome"}, 64, "internal/nativeperf receipt token counts", "updated when a phase receipt closes", "unknown token counts emit no sample"),
			metric("fak_native_kv_bytes", "gauge", "bytes", []string{"engine", "model_family", "device_class", "state"}, 128, "internal/modelperfobs and internal/xenginekv capacity snapshots", "refreshed at allocation, release, and observation snapshot", "unobservable allocator state is unsupported"),
			metric("fak_native_cache_events_total", "counter", "events", []string{"engine", "model_family", "cache", "result"}, 96, "internal/modelperfobs cache state and native receipt counters", "updated on each cache lookup result", "cache paths without hit/miss provenance emit no sample"),
			metric("fak_native_transfer_bytes_total", "counter", "bytes", []string{"engine", "device_class", "direction", "memory_class"}, 96, "internal/modelperfobs bandwidth/transfer observation", "updated when a native transfer completes", "driver-estimated bytes must be marked unsupported rather than synthesized"),
			metric("fak_native_kernel_launches_total", "counter", "launches", []string{"engine", "device_class", "kernel_family", "outcome"}, 160, "internal/metalgemm execution events", "updated for each observed fak-owned kernel launch", "backends without execution events emit no sample"),
			metric("fak_native_synchronization_seconds", "histogram", "seconds", []string{"engine", "device_class", "sync_scope", "outcome"}, 96, "native backend synchronization event", "one sample per observed synchronization", "implicit or unobservable synchronization emits no sample"),
			metric("fak_native_memory_bytes", "gauge", "bytes", []string{"engine", "device_class", "memory_class", "state"}, 128, "internal/modelperfobs memory and pressure snapshot", "refreshed at lifecycle and observation boundaries", "unavailable pressure or allocation data emits no sample"),
			metric("fak_native_output_seconds", "histogram", "seconds", []string{"engine", "model_family", "output_stage", "outcome"}, 96, "internal/modelengine output/decode consumer boundary", "one sample per completed output stage", "client/network time is excluded from native engine output"),
			metric("fak_native_benchmark_gate_total", "counter", "decisions", []string{"engine", "model_family", "gate", "verdict"}, 128, "internal/nativebench and internal/nativeperf gate receipt", "updated for each deterministic gate decision", "UNKNOWN verdict is debt and fails contract validation"),
			metric("fak_native_promotion_total", "counter", "decisions", []string{"engine", "model_family", "action", "verdict"}, 96, "native benchmark promotion/revert receipt", "updated for each promote, hold, or revert decision", "missing decision provenance is unsupported and must block promotion"),
		},
		Surfaces: []Surface{
			surface("model_loading", "internal/modelengine", "fak_native_model_load_seconds", "internal/modelengine/lifecycle.go"),
			surface("admission_queueing", "internal/modelengine", "fak_native_admission_wait_seconds", "internal/modelengine/nativesched.go"),
			surface("tokenization", "internal/tokenizer", "fak_native_tokenization_seconds", "internal/tokenizer/tokenizer.go"),
			surface("prefill", "internal/nativeperf", "fak_native_phase_seconds", "internal/nativeperf/receipt.go"),
			surface("decode", "internal/nativeperf", "fak_native_phase_seconds", "internal/nativeperf/receipt.go"),
			surface("kv_cache", "internal/modelperfobs", "fak_native_kv_bytes", "internal/modelperfobs/kv_capacity.go"),
			surface("cache", "internal/modelperfobs", "fak_native_cache_events_total", "internal/modelperfobs/cache_state.go"),
			surface("transfers", "internal/modelperfobs", "fak_native_transfer_bytes_total", "internal/modelperfobs/bandwidth.go"),
			surface("kernel_launches", "internal/metalgemm", "fak_native_kernel_launches_total", "internal/metalgemm/execution_events.go"),
			surface("synchronization", "internal/metalgemm", "fak_native_synchronization_seconds", "internal/metalgemm/execution_events.go"),
			surface("memory", "internal/modelperfobs", "fak_native_memory_bytes", "internal/modelperfobs/observation.go"),
			surface("output", "internal/modelengine", "fak_native_output_seconds", "internal/modelengine/pipeline.go"),
			surface("benchmark_gates", "internal/nativebench", "fak_native_benchmark_gate_total", "internal/nativebench/nativebench.go"),
			surface("promotion_revert", "internal/nativeperf", "fak_native_promotion_total", "internal/nativeperf/gate.go"),
		},
	}
}

func metric(name, kind, unit string, labels []string, budget int, provenance, freshness, unsupported string) Metric {
	return Metric{Name: name, Kind: kind, Unit: unit, Labels: labels, CardinalityBudget: budget, Provenance: provenance, Freshness: freshness, Unsupported: unsupported}
}

func surface(id, producer, metricName, evidence string) Surface {
	return Surface{ID: id, Producer: producer, Status: StatusExported, Metric: metricName, Evidence: evidence}
}

// Validate verifies that a candidate contract satisfies all structural, naming, cardinality, and engine-safety invariants.
// Precondition: Contract fields, metrics, and surfaces must be non-empty and well-formed.
// Invariant: Validation rejects non-native engines, missing execution surfaces, unbounded labels, and unknown coverage statuses.
// Postcondition: Returns nil if and only if every metric and surface complies with the frozen observability specification.
func Validate(c Contract) error {
	var problems []string
	if c.Schema != Schema {
		problems = append(problems, fmt.Sprintf("schema must be %q", Schema))
	}
	if c.Engine != Engine {
		problems = append(problems, fmt.Sprintf("engine must be %q", Engine))
	}
	if c.DefaultModel != "Qwen3.8" {
		problems = append(problems, "default_model must be Qwen3.8")
	}

	metrics := make(map[string]Metric, len(c.Metrics))
	for _, m := range c.Metrics {
		if m.Name == "" || metrics[m.Name].Name != "" {
			problems = append(problems, "metric names must be non-empty and unique")
		}
		metrics[m.Name] = m
		if !strings.HasPrefix(m.Name, "fak_native_") {
			problems = append(problems, m.Name+": metric must use fak_native_ prefix")
		}
		if m.Kind != "counter" && m.Kind != "gauge" && m.Kind != "histogram" {
			problems = append(problems, m.Name+": unsupported kind")
		}
		if m.Unit == "" || m.CardinalityBudget < 1 || m.Provenance == "" || m.Freshness == "" || m.Unsupported == "" {
			problems = append(problems, m.Name+": incomplete metric contract")
		}
		seen := map[string]bool{}
		for _, label := range m.Labels {
			if seen[label] {
				problems = append(problems, m.Name+": duplicate label "+label)
			}
			seen[label] = true
			if label == "request_id" || label == "prompt" || label == "path" || label == "host" || label == "model" {
				problems = append(problems, m.Name+": unbounded label "+label)
			}
		}
		if !seen["engine"] {
			problems = append(problems, m.Name+": engine label is required")
		}
	}

	required := []string{"model_loading", "admission_queueing", "tokenization", "prefill", "decode", "kv_cache", "cache", "transfers", "kernel_launches", "synchronization", "memory", "output", "benchmark_gates", "promotion_revert"}
	surfaces := make(map[string]Surface, len(c.Surfaces))
	for _, s := range c.Surfaces {
		if s.ID == "" || surfaces[s.ID].ID != "" {
			problems = append(problems, "surface IDs must be non-empty and unique")
		}
		surfaces[s.ID] = s
		if s.Producer == "" || s.Evidence == "" {
			problems = append(problems, s.ID+": producer and evidence are required")
		}
		switch s.Status {
		case StatusExported:
			if s.Metric == "" || metrics[s.Metric].Name == "" {
				problems = append(problems, s.ID+": exported surface must name a contracted metric")
			}
			if s.Exclusion != "" {
				problems = append(problems, s.ID+": exported surface cannot also be excluded")
			}
		case StatusExcluded:
			if s.Exclusion == "" {
				problems = append(problems, s.ID+": excluded surface needs justification")
			}
			if s.Metric != "" {
				problems = append(problems, s.ID+": excluded surface cannot name a metric")
			}
		default:
			problems = append(problems, s.ID+": status must be exported or excluded; UNKNOWN is debt")
		}
	}
	for _, id := range required {
		if surfaces[id].ID == "" {
			problems = append(problems, id+": required surface is missing")
		}
	}
	if len(problems) > 0 {
		sort.Strings(problems)
		return fmt.Errorf("native performance observability contract invalid:\n- %s", strings.Join(problems, "\n- "))
	}
	return nil
}

// JSON validates the candidate contract and serializes it into canonical indented JSON representation.
// Precondition: The candidate contract must successfully satisfy all Validate checks before serialization.
// Postcondition: Returns formatted JSON bytes or a non-nil validation or formatting error.
func JSON(c Contract) ([]byte, error) {
	if err := Validate(c); err != nil {
		return nil, err
	}
	return json.MarshalIndent(c, "", "  ")
}
