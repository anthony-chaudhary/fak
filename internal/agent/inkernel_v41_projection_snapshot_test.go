package agent

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

var v41ProjectionSnapshotKeys = []string{
	"head_projection_device_calls", "head_projection_host_calls", "head_projection_device_rows", "head_projection_host_rows",
	"head_projection_activation_upload_bytes", "head_projection_readback_bytes", "head_projection_nanos",
	"engram_projection_device_calls", "engram_projection_host_calls", "engram_projection_device_rows", "engram_projection_host_rows",
	"engram_projection_matmul_calls", "engram_projection_activation_upload_bytes", "engram_projection_readback_bytes",
	"engram_projection_host_weight_f32_bytes", "engram_projection_nanos",
}

func v41ProjectionSnapshotSource(t *testing.T, values map[string]map[string]int64) model.V41ExpertFaultAttribution {
	t.Helper()
	raw, err := json.Marshal(values)
	if err != nil {
		t.Fatal(err)
	}
	var source model.V41ExpertFaultAttribution
	if err := json.Unmarshal(raw, &source); err != nil {
		t.Fatal(err)
	}
	roundtrip, err := json.Marshal(source)
	if err != nil {
		t.Fatal(err)
	}
	var phases map[string]map[string]json.RawMessage
	if err := json.Unmarshal(roundtrip, &phases); err != nil {
		t.Fatal(err)
	}
	for phase, fields := range values {
		for key, want := range fields {
			var number *int64
			if err := json.Unmarshal(phases[phase][key], &number); err != nil {
				t.Fatalf("source %s.%s must be numeric: %v", phase, key, err)
			}
			if number == nil || *number != want {
				t.Fatalf("source %s.%s did not retain numeric fixture", phase, key)
			}
		}
	}
	return source
}

func v41ProjectionSnapshotPhases(t *testing.T, ledger MoEResidencyLedger) map[string]map[string]int64 {
	t.Helper()
	raw, err := json.Marshal(ledger)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		V41 struct {
			Scope       string                                `json:"scope"`
			Prefill     map[string]json.RawMessage            `json:"prefill"`
			Decode      map[string]json.RawMessage            `json:"decode"`
			Projections map[string]map[string]json.RawMessage `json:"projections"`
		} `json:"v41_phases"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.V41.Scope != "model_lifetime" {
		t.Fatal("projection source must inherit model_lifetime scope")
	}
	legacyJSON, err := json.Marshal(v41SnapshotFixture(0))
	if err != nil {
		t.Fatal(err)
	}
	var legacy map[string]json.RawMessage
	if err := json.Unmarshal(legacyJSON, &legacy); err != nil {
		t.Fatal(err)
	}
	if len(legacy) != 31 {
		t.Fatalf("legacy fixture field count=%d want 31", len(legacy))
	}
	for phase, fields := range map[string]map[string]json.RawMessage{"prefill": doc.V41.Prefill, "decode": doc.V41.Decode} {
		if len(fields) != 31 {
			t.Fatalf("legacy %s schema changed: fields=%d want 31", phase, len(fields))
		}
		for key := range legacy {
			if _, ok := fields[key]; !ok {
				t.Fatalf("legacy %s omits %s", phase, key)
			}
		}
	}
	if len(doc.V41.Projections) != 2 {
		t.Fatal("projection object must contain only prefill and decode")
	}
	result := map[string]map[string]int64{}
	for _, phase := range []string{"prefill", "decode"} {
		fields := doc.V41.Projections[phase]
		if len(fields) != len(v41ProjectionSnapshotKeys) {
			t.Fatalf("projection %s fields=%d want 16", phase, len(fields))
		}
		result[phase] = map[string]int64{}
		for _, key := range v41ProjectionSnapshotKeys {
			var number *int64
			if err := json.Unmarshal(fields[key], &number); err != nil {
				t.Fatalf("%s.%s must be numeric: %v", phase, key, err)
			}
			if number == nil {
				t.Fatalf("%s.%s must not be null", phase, key)
			}
			result[phase][key] = *number
		}
	}
	return result
}

func v41ProjectionSnapshotMutateReader(t *testing.T, value reflect.Value) int {
	t.Helper()
	if value.Kind() != reflect.Struct {
		t.Fatalf("projection snapshot must be a value struct, got %s", value.Kind())
	}
	count := 0
	for i := 0; i < value.NumField(); i++ {
		field := value.Field(i)
		if field.Kind() == reflect.Struct {
			count += v41ProjectionSnapshotMutateReader(t, field)
			continue
		}
		if field.CanSet() && field.Kind() >= reflect.Int && field.Kind() <= reflect.Int64 {
			field.SetInt(999999)
			count++
		} else {
			t.Fatalf("projection snapshot field %s must be numeric value", value.Type().Field(i).Name)
		}
	}
	return count
}

// fak-test:justify why=contract when=changed:internal/agent/**
// fak-test:runtime fast est=10ms lane=default
func TestV41PhaseAttributionProjectionOnlySourceRetained(t *testing.T) {
	for _, phase := range []string{"prefill", "decode"} {
		for _, key := range v41ProjectionSnapshotKeys {
			t.Run(phase+"/"+key, func(t *testing.T) {
				p := &InKernelPlanner{}
				p.foldV41PhasesLocked(v41ProjectionSnapshotSource(t, map[string]map[string]int64{phase: {key: 19}}))
				got := p.MoEResidencyStats()
				if got.V41Phases == nil {
					t.Fatal("projection-only source dropped")
				}
				if got.V41Phases.Scope != "model_lifetime" || got.Requests != 0 || got.Tokens != 0 || got.Last.Ring.Enabled || got.Checkpoint != nil {
					t.Fatal("projection-only source must remain independent of ring and checkpoint")
				}
				for name, values := range v41ProjectionSnapshotPhases(t, got) {
					for _, field := range v41ProjectionSnapshotKeys {
						var want int64
						if name == phase && field == key {
							want = 19
						}
						if values[field] != want {
							t.Fatalf("%s.%s=%d want %d", name, field, values[field], want)
						}
					}
				}
			})
		}
	}
}

// fak-test:justify why=contract when=changed:internal/agent/**
// fak-test:runtime fast est=1ms lane=default
func TestV41PhaseAttributionProjectionLatestAndReaderIsolation(t *testing.T) {
	p := &InKernelPlanner{}
	for _, base := range []int64{100, 700} {
		input := map[string]map[string]int64{}
		for i, phase := range []string{"prefill", "decode"} {
			input[phase] = map[string]int64{}
			for j, key := range v41ProjectionSnapshotKeys {
				input[phase][key] = base + int64(100*i+j+1)
			}
		}
		p.foldV41PhasesLocked(v41ProjectionSnapshotSource(t, input))
		got := p.MoEResidencyStats()
		if numbers := v41ProjectionSnapshotPhases(t, got); !reflect.DeepEqual(numbers, input) {
			t.Fatalf("latest projections=%v want source=%v", numbers, input)
		}
		fields := reflect.ValueOf(got.V41Phases).Elem()
		found := false
		for i := 0; i < fields.NumField(); i++ {
			if strings.Split(fields.Type().Field(i).Tag.Get("json"), ",")[0] == "projections" {
				if n := v41ProjectionSnapshotMutateReader(t, fields.Field(i)); n != 32 {
					t.Fatalf("projection snapshot numeric fields=%d want 32", n)
				}
				found = true
			}
		}
		if !found {
			t.Fatal("retained projection snapshot absent")
		}
		if numbers := v41ProjectionSnapshotPhases(t, p.MoEResidencyStats()); !reflect.DeepEqual(numbers, input) {
			t.Fatal("reader changed retained projection counters")
		}
	}
	before := v41ProjectionSnapshotPhases(t, p.MoEResidencyStats())
	p.foldV41PhasesLocked(model.V41ExpertFaultAttribution{})
	if got := v41ProjectionSnapshotPhases(t, p.MoEResidencyStats()); !reflect.DeepEqual(got, before) {
		t.Fatal("zero fold changed retained projections")
	}
	for _, active := range []string{"prefill", "decode", "legacy"} {
		input := map[string]map[string]int64{"prefill": {}, "decode": {}}
		for _, phase := range []string{"prefill", "decode"} {
			for _, key := range v41ProjectionSnapshotKeys {
				input[phase][key] = 0
			}
		}
		source := model.V41ExpertFaultAttribution{}
		if active == "legacy" {
			source.Prefill.Tokens = 2
		} else {
			input[active]["head_projection_host_calls"] = 41
			source = v41ProjectionSnapshotSource(t, input)
		}
		p.foldV41PhasesLocked(source)
		if numbers := v41ProjectionSnapshotPhases(t, p.MoEResidencyStats()); !reflect.DeepEqual(numbers, input) {
			t.Fatalf("%s replacement did not clear zero siblings", active)
		}
	}
	zero := &InKernelPlanner{}
	zero.foldV41PhasesLocked(model.V41ExpertFaultAttribution{})
	if zero.MoEResidencyStats().V41Phases != nil {
		t.Fatal("zero source must be inert")
	}
}
