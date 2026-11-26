package agent

// fak-test:runtime fast est=1s

import (
	"encoding/json"
	"testing"
)

func TestTimingsMetalPipelinesJSON(t *testing.T) {
	raw, err := json.Marshal(NewTimings(10, 0, 4, 0.01, 0.02))
	if err != nil {
		t.Fatal(err)
	}
	var plain map[string]json.RawMessage
	if err := json.Unmarshal(raw, &plain); err != nil {
		t.Fatal(err)
	}
	if _, ok := plain["metal_pipelines"]; ok {
		t.Fatalf("nil MetalPipelines serialized: %s", raw)
	}

	tm := NewTimings(10, 0, 4, 0.01, 0.02)
	tm.MetalPipelines = &MetalPipelineTimings{}
	raw, err = json.Marshal(tm)
	if err != nil {
		t.Fatal(err)
	}
	var withZero struct {
		MetalPipelines map[string]json.RawMessage `json:"metal_pipelines"`
	}
	if err := json.Unmarshal(raw, &withZero); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"compiled", "archive_hits", "archived", "load_ms"} {
		if v, ok := withZero.MetalPipelines[k]; !ok || string(v) != "0" {
			t.Fatalf("zero MetalPipelines key %q = %q (present %v): %s", k, v, ok, raw)
		}
	}
	if len(withZero.MetalPipelines) != 4 {
		t.Fatalf("metal_pipelines keys = %v, want exactly 4", withZero.MetalPipelines)
	}

	tm.MetalPipelines = &MetalPipelineTimings{Compiled: 2, ArchiveHits: 41, Archived: 42, LoadMS: 12.5}
	raw, err = json.Marshal(tm)
	if err != nil {
		t.Fatal(err)
	}
	var back Timings
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if back.MetalPipelines == nil || *back.MetalPipelines != *tm.MetalPipelines {
		t.Fatalf("round trip = %+v, want %+v", back.MetalPipelines, tm.MetalPipelines)
	}
}
