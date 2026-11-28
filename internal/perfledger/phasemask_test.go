package perfledger

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestPhaseMask(t *testing.T) {
	s, err := NewColumnStore(ColumnStoreLimits{})
	if err != nil {
		t.Fatal(err)
	}
	// Interleaved wall-clock windows: record 0 and 2 are "warmup" even though
	// record 2 has a later timestamp than record 1 ("steady").
	if err := s.IngestCategorical(0, "phase", "warmup"); err != nil {
		t.Fatal(err)
	}
	if err := s.Ingest(0, 0, 10, map[string]any{"latency_ms": 50.0}); err != nil {
		t.Fatal(err)
	}
	if err := s.IngestCategorical(1, "phase", "steady"); err != nil {
		t.Fatal(err)
	}
	if err := s.Ingest(1, 10, 20, map[string]any{"latency_ms": 10.0}); err != nil {
		t.Fatal(err)
	}
	if err := s.IngestCategorical(2, "phase", "warmup"); err != nil {
		t.Fatal(err)
	}
	if err := s.Ingest(2, 20, 30, map[string]any{"latency_ms": 70.0}); err != nil {
		t.Fatal(err)
	}

	mask, err := PhaseMask(s, "phase", "warmup")
	if err != nil {
		t.Fatalf("PhaseMask: %v", err)
	}
	if !reflect.DeepEqual(mask, []bool{true, false, true}) {
		t.Fatalf("warmup mask = %v, want [true false true]", mask)
	}

	vals, summary, err := PhaseScopedNumeric(s, "latency_ms", "phase", "warmup")
	if err != nil {
		t.Fatalf("PhaseScopedNumeric: %v", err)
	}
	if !reflect.DeepEqual(vals, []float64{50.0, 70.0}) {
		t.Fatalf("warmup vals = %v, want [50 70]", vals)
	}
	if summary.Matched != 2 || summary.Count != 2 || summary.Mean != 60.0 {
		t.Fatalf("unexpected summary: %+v", summary)
	}

	// Absent phase value returns all-false mask and zero summary.
	emptyVals, emptySummary, err := PhaseScopedNumeric(s, "latency_ms", "phase", "cooldown")
	if err != nil {
		t.Fatalf("PhaseScopedNumeric absent: %v", err)
	}
	if len(emptyVals) != 0 || emptySummary.Matched != 0 {
		t.Fatalf("expected empty summary, got vals=%v summary=%+v", emptyVals, emptySummary)
	}
}
