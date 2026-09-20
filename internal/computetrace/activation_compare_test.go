package computetrace

// activation_compare_test.go -- the #13324 witness for bounded activation-sample
// capture and first-divergence comparison.
//
// The existing recorder captures event metadata (shape/dtype/provenance) but no
// activation VALUES, so a worker diagnosing a V4.1 divergence can see that a
// stage ran and that its shapes are compatible, yet cannot compare a successful
// intermediate against an independent reference. This leaf adds an opt-in,
// strictly bounded sample record and a pure comparator that reports the FIRST
// divergent (layer, token, stage) identity plus the maximum sampled error, or an
// explicit INCOMPARABLE verdict when the traces are not comparable.
//
// The contract is deliberately fail-closed: an unaccounted drop, a mismatched
// input digest, a missing sample, a duplicate identity or an incompatible shape
// must never be reported as agreement.

import (
	"math"
	"strings"
	"testing"
)

// mustReader adapts a JSON literal to an io.Reader for the legacy-artifact case.
func mustReader(s string) *strings.Reader { return strings.NewReader(s) }

// sampleEvent builds a comparable event carrying one bounded sample record.
func sampleEvent(layer, token int, stage string, values []float32) Event {
	return Event{
		RunID:            "run",
		RequestID:        "req",
		Operation:        "matmul",
		Phase:            stage,
		Layer:            layer,
		Token:            token,
		InputDType:       "f32",
		Shapes:           [][]int{{len(values)}},
		InputDigest:      Digest("input", "w"),
		WeightDigest:     Digest("weight", "w"),
		SampleIndices:    make([]int, len(values)),
		SampleValues:     values,
		ProvenanceDigest: Digest("prov"),
	}
}

func indexedSample(layer, token int, stage string, values []float32) Event {
	e := sampleEvent(layer, token, stage, values)
	for i := range e.SampleIndices {
		e.SampleIndices[i] = i
	}
	return e
}

func collect(events ...Event) Artifact {
	a := Artifact{Schema: Schema, RetainedSampleValues: 0}
	for _, e := range events {
		a.Events = append(a.Events, e)
		a.RetainedSampleValues += uint64(len(e.SampleValues))
	}
	return a
}

// TestActivationTraceCompare is the named #13324 witness.
func TestActivationTraceCompare(t *testing.T) {
	t.Run("identical fixtures compare equal", func(t *testing.T) {
		ref := collect(
			indexedSample(0, 0, "q_proj", []float32{1, 2, 3}),
			indexedSample(0, 1, "q_proj", []float32{4, 5, 6}),
		)
		got := collect(
			indexedSample(0, 0, "q_proj", []float32{1, 2, 3}),
			indexedSample(0, 1, "q_proj", []float32{4, 5, 6}),
		)
		res := CompareActivationTraces(ref, got, ActivationTolerance{Atol: 1e-6, Rtol: 1e-6})
		if res.Verdict != TraceEqual {
			t.Fatalf("identical traces: verdict=%s divergent=%+v err=%v", res.Verdict, res.Divergent, res.MaxAbsError)
		}
		if res.ComparedSamples != 6 {
			t.Fatalf("compared %d samples, want 6", res.ComparedSamples)
		}
	})

	t.Run("a perturbation at the second stage is identified", func(t *testing.T) {
		ref := collect(
			indexedSample(0, 0, "q_proj", []float32{1, 2, 3}),
			indexedSample(1, 0, "o_proj", []float32{10, 20, 30}),
		)
		got := collect(
			indexedSample(0, 0, "q_proj", []float32{1, 2, 3}),
			indexedSample(1, 0, "o_proj", []float32{10, 20, 31}),
		)
		res := CompareActivationTraces(ref, got, ActivationTolerance{Atol: 1e-6, Rtol: 1e-6})
		if res.Verdict != TraceDivergent {
			t.Fatalf("perturbed trace: verdict=%s, want DIVERGENT", res.Verdict)
		}
		if res.Divergent.Layer != 1 || res.Divergent.Token != 0 || res.Divergent.Stage != "o_proj" {
			t.Fatalf("first divergence = %+v, want layer=1 token=0 stage=o_proj", res.Divergent)
		}
		if res.DivergentIndex != 2 {
			t.Fatalf("divergent sample index = %d, want 2", res.DivergentIndex)
		}
		if math.Abs(res.MaxAbsError-1) > 1e-9 {
			t.Fatalf("max abs error = %v, want 1", res.MaxAbsError)
		}
	})

	t.Run("NaN and Inf cannot pass", func(t *testing.T) {
		ref := collect(indexedSample(0, 0, "q_proj", []float32{1, 0, 3}))
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			got := collect(indexedSample(0, 0, "q_proj", []float32{1, bad, 3}))
			res := CompareActivationTraces(ref, got, ActivationTolerance{Atol: 1, Rtol: 1})
			if res.Verdict != TraceDivergent {
				t.Fatalf("non-finite %v: verdict=%s, want DIVERGENT", bad, res.Verdict)
			}
			if !math.IsNaN(res.MaxAbsError) && !math.IsInf(res.MaxAbsError, 0) {
				t.Fatalf("non-finite sample reported finite max error %v", res.MaxAbsError)
			}
		}
	})

	t.Run("mixed input digests are INCOMPARABLE", func(t *testing.T) {
		ref := collect(indexedSample(0, 0, "q_proj", []float32{1, 2, 3}))
		got := collect(indexedSample(0, 0, "q_proj", []float32{1, 2, 3}))
		got.Events[0].InputDigest = Digest("input", "different")
		res := CompareActivationTraces(ref, got, ActivationTolerance{Atol: 1e-6, Rtol: 1e-6})
		if res.Verdict != TraceIncomparable {
			t.Fatalf("mixed input digest: verdict=%s, want INCOMPARABLE", res.Verdict)
		}
	})

	t.Run("missing or duplicate keys are INCOMPARABLE", func(t *testing.T) {
		ref := collect(
			indexedSample(0, 0, "q_proj", []float32{1, 2, 3}),
			indexedSample(1, 0, "o_proj", []float32{4, 5, 6}),
		)
		missing := collect(indexedSample(0, 0, "q_proj", []float32{1, 2, 3}))
		if res := CompareActivationTraces(ref, missing, ActivationTolerance{Atol: 1e-6, Rtol: 1e-6}); res.Verdict != TraceIncomparable {
			t.Fatalf("missing key: verdict=%s, want INCOMPARABLE", res.Verdict)
		}
		dup := collect(
			indexedSample(0, 0, "q_proj", []float32{1, 2, 3}),
			indexedSample(0, 0, "q_proj", []float32{1, 2, 3}),
		)
		if res := CompareActivationTraces(ref, dup, ActivationTolerance{Atol: 1e-6, Rtol: 1e-6}); res.Verdict != TraceIncomparable {
			t.Fatalf("duplicate key: verdict=%s, want INCOMPARABLE", res.Verdict)
		}
	})

	t.Run("dropped samples make comparison INCOMPARABLE", func(t *testing.T) {
		ref := collect(indexedSample(0, 0, "q_proj", []float32{1, 2, 3}))
		got := collect(indexedSample(0, 0, "q_proj", []float32{1, 2, 3}))
		got.DroppedSampleValues = 1
		res := CompareActivationTraces(ref, got, ActivationTolerance{Atol: 1e-6, Rtol: 1e-6})
		if res.Verdict != TraceIncomparable {
			t.Fatalf("dropped samples: verdict=%s, want INCOMPARABLE", res.Verdict)
		}
	})

	t.Run("incompatible shapes are INCOMPARABLE", func(t *testing.T) {
		ref := collect(indexedSample(0, 0, "q_proj", []float32{1, 2, 3}))
		got := collect(indexedSample(0, 0, "q_proj", []float32{1, 2}))
		res := CompareActivationTraces(ref, got, ActivationTolerance{Atol: 1e-6, Rtol: 1e-6})
		if res.Verdict != TraceIncomparable {
			t.Fatalf("incompatible shape: verdict=%s, want INCOMPARABLE", res.Verdict)
		}
	})
}

// TestActivationSampleBoundsAndCounters pins the recorder-side bounds and the
// counters that qualify a trace as comparable at all.
func TestActivationSampleBoundsAndCounters(t *testing.T) {
	t.Run("per-event sample cap drops excess values", func(t *testing.T) {
		r := New(64)
		large := make([]float32, MaxActivationSamplesPerEvent+5)
		r.Record(Event{Operation: "matmul", SampleIndices: make([]int, len(large)), SampleValues: large})
		a := r.Artifact()
		if got := len(a.Events[0].SampleValues); got != MaxActivationSamplesPerEvent {
			t.Fatalf("retained %d samples, want the %d cap", got, MaxActivationSamplesPerEvent)
		}
		if a.DroppedSampleValues != 5 {
			t.Fatalf("DroppedSampleValues=%d, want 5", a.DroppedSampleValues)
		}
	})

	t.Run("recorder total cap drops further samples", func(t *testing.T) {
		per := MaxActivationSamplesPerEvent
		n := MaxActivationSamplesTotal/per + 1 // one event past the total sample budget
		// Size the event limit above n so the SAMPLE cap, not the event cap, is
		// the binding constraint under test.
		r := New(n + 8)
		for i := 0; i < n; i++ {
			vals := make([]float32, per)
			r.Record(Event{Operation: "matmul", Layer: i, SampleIndices: make([]int, per), SampleValues: vals})
		}
		a := r.Artifact()
		if a.RetainedSampleValues != MaxActivationSamplesTotal {
			t.Fatalf("RetainedSampleValues=%d, want %d", a.RetainedSampleValues, MaxActivationSamplesTotal)
		}
		if a.DroppedSampleValues == 0 {
			t.Fatal("total sample cap exceeded but DroppedSampleValues=0")
		}
	})

	t.Run("snapshots own deep copies of samples", func(t *testing.T) {
		r := New(4)
		vals := []float32{1, 2, 3}
		idx := []int{0, 1, 2}
		r.Record(Event{Operation: "matmul", SampleIndices: idx, SampleValues: vals})
		vals[0], idx[0] = 99, 99 // caller mutates after Record
		a := r.Artifact()
		if a.Events[0].SampleValues[0] != 1 || a.Events[0].SampleIndices[0] != 0 {
			t.Fatalf("artifact aliased caller sample memory: %+v", a.Events[0])
		}
		a.Events[0].SampleValues[1] = 77 // artifact mutation must not affect recorder
		if r.Artifact().Events[0].SampleValues[1] != 2 {
			t.Fatal("artifact mutation leaked back into the recorder")
		}
	})

	t.Run("disabled recorder stays zero-allocation and counts nothing", func(t *testing.T) {
		r := New(0)
		r.Record(Event{Operation: "matmul", SampleValues: []float32{1, 2, 3}})
		a := r.Artifact()
		if len(a.Events) != 0 || a.Dropped != 0 || a.RetainedSampleValues != 0 || a.DroppedSampleValues != 0 {
			t.Fatalf("disabled recorder changed: %+v", a)
		}
	})

	t.Run("historical v1 artifact without sample fields still reads", func(t *testing.T) {
		legacy := `{"schema":"fak.compute_trace.v1","events":[{"sequence":1,"operation":"matmul","status":"ok","provenance_digest":"sha256:x"}],"dropped_events":0,"observer_overhead_ns":0}`
		a, err := Read(mustReader(legacy))
		if err != nil {
			t.Fatalf("historical artifact failed to read: %v", err)
		}
		if len(a.Events) != 1 || a.RetainedSampleValues != 0 || len(a.Events[0].SampleValues) != 0 {
			t.Fatalf("historical artifact misread: %+v", a)
		}
		// A trace with no activation evidence is not comparable agreement.
		res := CompareActivationTraces(a, a, ActivationTolerance{Atol: 1e-6, Rtol: 1e-6})
		if res.Verdict != TraceIncomparable {
			t.Fatalf("sampleless trace: verdict=%s, want INCOMPARABLE", res.Verdict)
		}
	})
}
