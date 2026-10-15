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
	"encoding/json"
	"errors"
	"math"
	"slices"
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

// fak-test:runtime fast est=10ms
func TestActivationTraceMalformedIndices(t *testing.T) {
	decode := func(indices string) Artifact {
		t.Helper()
		a, err := Read(strings.NewReader(`{"schema":"fak.compute_trace.v1","events":[{
			"layer":4,"token":2,"phase":"q_proj","shapes":[[3]],
			"input_digest":"input","weight_digest":"weight",
			"sample_indices":` + indices + `,"sample_values":[1,2,3]}]}`))
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	for _, tc := range []struct{ name, indices string }{
		{"short", "[0]"},
		{"overlong", "[0,1,2,3]"},
		{"duplicate", "[0,0,2]"},
	} {
		for _, peer := range []struct{ name, indices string }{
			{"explicit", "[0,1,2]"},
			{"implicit", "null"},
		} {
			for _, side := range []string{"reference", "actual"} {
				t.Run(tc.name+"/"+peer.name+"/"+side, func(t *testing.T) {
					ref, got := decode(tc.indices), decode(peer.indices)
					if side == "actual" {
						ref, got = got, ref
					}
					if r := CompareActivationTraces(ref, got, ActivationTolerance{}); r.Verdict != TraceIncomparable {
						t.Fatalf("strict verdict=%s, want %s", r.Verdict, TraceIncomparable)
					}
					r := FirstDivergence(ref, got, DivergenceTolerance{}, ActivationTolerance{})
					if r.Verdict != TraceIncomparable || r.Strict.Verdict != TraceIncomparable || r.FirstDivergent != nil {
						t.Fatalf("first-divergence verdict=%s strict=%s first=%v", r.Verdict, r.Strict.Verdict, r.FirstDivergent)
					}
				})
			}
		}
	}
	for _, indices := range []string{"[0,1,2]", "null"} {
		ref, got := decode("null"), decode(indices)
		if r := CompareActivationTraces(ref, got, ActivationTolerance{}); r.Verdict != TraceEqual {
			t.Fatalf("legacy positional control verdict=%s", r.Verdict)
		}
		got.Events[0].SampleValues[2]++
		r := CompareActivationTraces(ref, got, ActivationTolerance{})
		if r.Verdict != TraceDivergent || r.Divergent != (ActivationIdentity{Layer: 4, Token: 2, Stage: "q_proj"}) || r.DivergentIndex != 2 {
			t.Fatalf("valid divergence control: verdict=%s identity=%v index=%d", r.Verdict, r.Divergent, r.DivergentIndex)
		}
	}
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

// TestActivationTraceTolerance rejects invalid comparison parameters before
// consuming any samples and preserves the existing finite tolerance formula.
// fak-test:runtime fast est=1ms
func TestActivationTraceTolerance(t *testing.T) {
	t.Run("invalid tolerances are incomparable", func(t *testing.T) {
		invalid := []struct {
			name string
			tol  ActivationTolerance
		}{
			{"nan_atol", ActivationTolerance{Atol: math.NaN()}},
			{"nan_rtol", ActivationTolerance{Rtol: math.NaN()}},
			{"positive_inf_atol", ActivationTolerance{Atol: math.Inf(1)}},
			{"positive_inf_rtol", ActivationTolerance{Rtol: math.Inf(1)}},
			{"negative_inf_atol", ActivationTolerance{Atol: math.Inf(-1)}},
			{"negative_inf_rtol", ActivationTolerance{Rtol: math.Inf(-1)}},
			{"negative_atol", ActivationTolerance{Atol: -1}},
			{"negative_rtol", ActivationTolerance{Rtol: -1}},
			{"negative_subnormal_atol", ActivationTolerance{Atol: -math.SmallestNonzeroFloat64}},
			{"negative_subnormal_rtol", ActivationTolerance{Rtol: -math.SmallestNonzeroFloat64}},
			{"negative_atol_compensated", ActivationTolerance{Atol: -1, Rtol: 100}},
			{"negative_rtol_compensated", ActivationTolerance{Atol: 100, Rtol: -1}},
		}
		pairs := []struct {
			name     string
			ref, got float32
		}{
			{"equal", 1, 1},
			{"unequal", 1, 2},
			{"zero_actual", 1, 0},
		}
		for _, tc := range invalid {
			t.Run(tc.name, func(t *testing.T) {
				for _, pair := range pairs {
					t.Run(pair.name, func(t *testing.T) {
						ref := collect(indexedSample(0, 0, "q_proj", []float32{pair.ref}))
						got := collect(indexedSample(0, 0, "q_proj", []float32{pair.got}))
						res := CompareActivationTraces(ref, got, tc.tol)
						if res.Verdict != TraceIncomparable || res.ComparedSamples != 0 || res.Reason == "" {
							t.Fatalf("invalid tolerance %+v: result=%+v, want INCOMPARABLE with reason and zero compared samples", tc.tol, res)
						}
					})
				}
			})
		}
	})

	t.Run("valid tolerances preserve comparison", func(t *testing.T) {
		negativeZero := math.Copysign(0, -1)
		cases := []struct {
			name     string
			tol      ActivationTolerance
			ref, got float32
			want     Verdict
		}{
			{"zero_equal", ActivationTolerance{}, 1, 1, TraceEqual},
			{"zero_unequal", ActivationTolerance{}, 1, 2, TraceDivergent},
			{"negative_zero_equal", ActivationTolerance{Atol: negativeZero, Rtol: negativeZero}, 1, 1, TraceEqual},
			{"negative_zero_unequal", ActivationTolerance{Atol: negativeZero, Rtol: negativeZero}, 1, 2, TraceDivergent},
			{"atol_boundary", ActivationTolerance{Atol: 1}, 1, 2, TraceEqual},
			{"atol_below_boundary", ActivationTolerance{Atol: math.Nextafter(1, 0)}, 1, 2, TraceDivergent},
			{"rtol_actual_boundary", ActivationTolerance{Rtol: 0.25}, 3, 4, TraceEqual},
			{"rtol_reverse_pair", ActivationTolerance{Rtol: 0.25}, 4, 3, TraceDivergent},
			{"rtol_negative_actual", ActivationTolerance{Rtol: 0.25}, -3, -4, TraceEqual},
			{"combined_boundary", ActivationTolerance{Atol: 0.5, Rtol: 0.125}, 3, 4, TraceEqual},
			{"positive_subnormal_atol", ActivationTolerance{Atol: math.SmallestNonzeroFloat64}, 0, math.SmallestNonzeroFloat32, TraceDivergent},
			{"positive_subnormal_rtol", ActivationTolerance{Rtol: math.SmallestNonzeroFloat64}, 1, 2, TraceDivergent},
			{"maximum_finite_atol", ActivationTolerance{Atol: math.MaxFloat64}, -math.MaxFloat32, math.MaxFloat32, TraceEqual},
			{"maximum_finite_rtol_zero_actual", ActivationTolerance{Rtol: math.MaxFloat64}, 1, 0, TraceDivergent},
			{"finite_limit_overflow", ActivationTolerance{Rtol: math.MaxFloat64}, 1, 2, TraceEqual},
			{"nan_reference", ActivationTolerance{Atol: math.MaxFloat64}, float32(math.NaN()), 1, TraceDivergent},
			{"nan_actual", ActivationTolerance{Atol: math.MaxFloat64}, 1, float32(math.NaN()), TraceDivergent},
			{"positive_inf_reference", ActivationTolerance{Atol: math.MaxFloat64}, float32(math.Inf(1)), 1, TraceDivergent},
			{"positive_inf_actual", ActivationTolerance{Atol: math.MaxFloat64}, 1, float32(math.Inf(1)), TraceDivergent},
			{"negative_inf_reference", ActivationTolerance{Atol: math.MaxFloat64}, float32(math.Inf(-1)), 1, TraceDivergent},
			{"negative_inf_actual", ActivationTolerance{Atol: math.MaxFloat64}, 1, float32(math.Inf(-1)), TraceDivergent},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				ref := collect(indexedSample(0, 0, "q_proj", []float32{tc.ref}))
				got := collect(indexedSample(0, 0, "q_proj", []float32{tc.got}))
				res := CompareActivationTraces(ref, got, tc.tol)
				if res.Verdict != tc.want || res.ComparedSamples != 1 || res.Reason != "" {
					t.Fatalf("valid tolerance %+v: result=%+v, want %s with one compared sample and no refusal reason", tc.tol, res, tc.want)
				}
			})
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

func activationCaptureFixture(layers int) Artifact {
	a := Artifact{Schema: Schema, InputTokens: []int{7, 3, 9}, ActivationCapture: &ActivationCapture{
		ID: "step-current", Kind: ActivationCaptureIncrementalStep, Outcome: ActivationCaptureCommitted,
		Token: 2, ExpectedLayers: layers,
		ExpectedStages: []string{"q_latent", "kv_latent", "attention_projected", "moe_sum", "residual_streams"},
	}}
	for layer := 0; layer < layers; layer++ {
		for _, stage := range a.ActivationCapture.ExpectedStages {
			e := indexedSample(layer, 2, stage, []float32{1, 2})
			e.RunID, e.Operation = a.ActivationCapture.ID, "v41.activation"
			a.Events = append(a.Events, e)
			a.RetainedSampleValues += uint64(len(e.SampleValues))
		}
	}
	return a
}

// fak-test:justify why=regression when=changed:internal/computetrace/**
// fak-test:runtime fast est=20ms lane=default
func TestActivationCaptureValidation(t *testing.T) {
	if err := ValidateActivationCapture(activationCaptureFixture(40)); err != nil {
		t.Fatalf("complete 40-layer capture rejected: %v", err)
	}
	if err := ValidateActivationCapture(collect(indexedSample(0, 0, "q_proj", []float32{1}))); err != nil {
		t.Fatalf("legacy numerical artifact rejected: %v", err)
	}
	cases := []struct {
		name    string
		corrupt func(*Artifact)
	}{
		{"aborted", func(a *Artifact) { a.ActivationCapture.Outcome = ActivationCaptureAborted }},
		{"stale_token", func(a *Artifact) { a.ActivationCapture.Token-- }},
		{"wrong_event_capture", func(a *Artifact) { a.Events[0].RunID = "older-step" }},
		{"unknown_kind", func(a *Artifact) { a.ActivationCapture.Kind = "unknown" }},
		{"dropped_event", func(a *Artifact) { a.Dropped++ }},
		{"dropped_sample", func(a *Artifact) { a.DroppedSampleValues++ }},
		{"missing_stage", func(a *Artifact) { a.Events = a.Events[1:]; a.RetainedSampleValues -= 2 }},
		{"duplicate_stage", func(a *Artifact) { a.Events[1] = a.Events[0] }},
		{"false_stage_declaration", func(a *Artifact) { a.ActivationCapture.ExpectedStages[0] = "unknown" }},
		{"wrong_layers", func(a *Artifact) { a.ActivationCapture.ExpectedLayers++ }},
		{"missing_samples", func(a *Artifact) {
			a.Events[0].SampleValues = nil
			a.Events[0].SampleIndices = nil
			a.RetainedSampleValues -= 2
		}},
		{"nonfinite_sample", func(a *Artifact) { a.Events[0].SampleValues[0] = float32(math.NaN()) }},
		{"invalid_sample_index", func(a *Artifact) { a.Events[0].SampleIndices[0] = 2 }},
		{"sample_accounting", func(a *Artifact) { a.RetainedSampleValues++ }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			bad := activationCaptureFixture(2)
			tc.corrupt(&bad)
			if err := ValidateActivationCapture(bad); !errors.Is(err, ErrActivationCaptureInvalid) {
				t.Fatalf("invalid capture error = %v, want sentinel", err)
			}
			for _, side := range []string{"reference", "actual"} {
				t.Run(side, func(t *testing.T) {
					ref, got := bad, activationCaptureFixture(2)
					if side == "actual" {
						ref, got = got, ref
					}
					if r := CompareActivationTraces(ref, got, ActivationTolerance{}); r.Verdict != TraceIncomparable || r.ComparedSamples != 0 {
						t.Fatalf("invalid %s admitted: verdict=%s samples=%d", side, r.Verdict, r.ComparedSamples)
					}
					r := FirstDivergence(ref, got, DivergenceTolerance{}, ActivationTolerance{})
					if r.Verdict != TraceIncomparable || r.FirstDivergent != nil {
						t.Fatalf("invalid %s established first divergence: verdict=%s first=%v", side, r.Verdict, r.FirstDivergent)
					}
				})
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/computetrace/**
// fak-test:runtime fast est=20ms lane=default
func TestActivationCaptureReferenceProjection(t *testing.T) {
	makePair := func() (Artifact, Artifact) {
		actual := activationCaptureFixture(40)
		ref := Artifact{Schema: Schema, InputTokens: slices.Clone(actual.InputTokens)}
		for token := range ref.InputTokens {
			for _, e := range actual.Events {
				e.Token, e.RunID = token, "full-forward-reference"
				e.InputDigest, e.WeightDigest = "reference-input", "reference-weights"
				e.SampleValues = slices.Clone(e.SampleValues)
				e.SampleIndices = slices.Clone(e.SampleIndices)
				ref.Events = append(ref.Events, e)
				ref.RetainedSampleValues += uint64(len(e.SampleValues))
			}
		}
		return ref, actual
	}
	ref, actual := makePair()
	projected, err := ProjectActivationReference(ref, actual)
	if err != nil {
		t.Fatal(err)
	}
	if projected.ActivationCapture != nil || !slices.Equal(projected.InputTokens, ref.InputTokens) || len(projected.Events) != 40*5 || projected.RetainedSampleValues != 40*5*2 {
		t.Fatal("projection changed the full input identity, fabricated capture metadata, or lost target coverage")
	}
	for _, e := range projected.Events {
		if e.Token != 2 || e.InputDigest != "reference-input" || e.WeightDigest != "reference-weights" || e.RunID != "full-forward-reference" {
			t.Fatal("projection rewrote absolute token or reference provenance")
		}
	}
	projected.InputTokens[0]++
	projected.Events[0].SampleValues[0]++
	projected.Events[0].SampleIndices[0]++
	projected.Events[0].Shapes[0][0]++
	if ref.InputTokens[0] != 7 || ref.Events[40*5*2].SampleValues[0] != 1 || ref.Events[40*5*2].SampleIndices[0] != 0 || ref.Events[40*5*2].Shapes[0][0] != 2 {
		t.Fatal("projection aliases the source reference")
	}
	cases := []struct {
		name    string
		corrupt func(*Artifact, *Artifact)
	}{
		{"earlier_prefix_same_last_token", func(r, a *Artifact) { r.InputTokens[0]++ }},
		{"missing_actual_capture", func(r, a *Artifact) { a.ActivationCapture = nil }},
		{"aborted_actual", func(r, a *Artifact) { a.ActivationCapture.Outcome = ActivationCaptureAborted }},
		{"missing_reference_stage", func(r, a *Artifact) { r.Events = r.Events[:len(r.Events)-1]; r.RetainedSampleValues -= 2 }},
		{"duplicate_reference_stage", func(r, a *Artifact) { r.Events[len(r.Events)-1] = r.Events[len(r.Events)-2] }},
		{"dropped_reference", func(r, a *Artifact) { r.Dropped++ }},
		{"aborted_reference_capture", func(r, a *Artifact) {
			*r = activationCaptureFixture(40)
			r.ActivationCapture.Outcome = ActivationCaptureAborted
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ref, actual := makePair()
			tc.corrupt(&ref, &actual)
			if _, err := ProjectActivationReference(ref, actual); !errors.Is(err, ErrActivationCaptureInvalid) {
				t.Fatalf("projection error=%v, want sentinel", err)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestActivationCaptureReadEnvelope(t *testing.T) {
	const legacy = "{\"schema\":\"fak.compute_trace.v1\",\"events\":[],\"future_field\":true}"
	if _, err := Read(strings.NewReader(legacy + " \n\t")); err != nil {
		t.Fatalf("unknown legacy field or trailing whitespace rejected: %v", err)
	}
	for _, suffix := range []string{"{}", "[]", "]", "}", "garbage"} {
		t.Run(suffix, func(t *testing.T) {
			if _, err := Read(strings.NewReader(legacy + suffix)); err == nil {
				t.Fatal("trailing JSON or garbage accepted as one artifact")
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestActivationCaptureTokenPresence(t *testing.T) {
	a := activationCaptureFixture(1)
	a.InputTokens, a.ActivationCapture.Token = []int{0}, 0
	for i := range a.Events {
		a.Events[i].Token = 0
		a.Events[i].SampleValues[0] = 0
	}
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	explicit, err := Read(strings.NewReader(string(raw)))
	if err != nil || ValidateActivationCapture(explicit) != nil {
		t.Fatal("explicit absolute token zero rejected")
	}
	for _, missing := range []bool{true, false} {
		name := "null"
		if missing {
			name = "absent"
		}
		t.Run(name, func(t *testing.T) {
			var envelope map[string]json.RawMessage
			if err := json.Unmarshal(raw, &envelope); err != nil {
				t.Fatal(err)
			}
			var capture map[string]json.RawMessage
			if err := json.Unmarshal(envelope["activation_capture"], &capture); err != nil {
				t.Fatal(err)
			}
			if missing {
				delete(capture, "token")
			} else {
				capture["token"] = json.RawMessage("null")
			}
			envelope["activation_capture"], err = json.Marshal(capture)
			if err != nil {
				t.Fatal(err)
			}
			bad, err := json.Marshal(envelope)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := Read(strings.NewReader(string(bad)))
			if err == nil {
				err = ValidateActivationCapture(decoded)
			}
			if !errors.Is(err, ErrActivationCaptureInvalid) {
				t.Fatalf("missing explicit absolute token error=%v, want sentinel", err)
			}
		})
	}

	reference := explicit
	reference.ActivationCapture = nil
	if _, err := ProjectActivationReference(reference, explicit); err != nil {
		t.Fatal("explicit zero legacy reference rejected by Step admission")
	}
	for _, field := range []string{"input_tokens", "sample_values", "sample_indices"} {
		for _, side := range []string{"actual", "reference"} {
			t.Run("null_element_"+side+"_"+field, func(t *testing.T) {
				var envelope map[string]json.RawMessage
				if err := json.Unmarshal(raw, &envelope); err != nil {
					t.Fatal(err)
				}
				if side == "reference" {
					delete(envelope, "activation_capture")
				}
				if field == "input_tokens" {
					envelope[field] = json.RawMessage("[null]")
				} else {
					var events []map[string]json.RawMessage
					if err := json.Unmarshal(envelope["events"], &events); err != nil {
						t.Fatal(err)
					}
					events[0][field] = json.RawMessage("[null,1]")
					if field == "sample_values" {
						events[0][field] = json.RawMessage("[null,2]")
					}
					envelope["events"], err = json.Marshal(events)
					if err != nil {
						t.Fatal(err)
					}
				}
				data, err := json.Marshal(envelope)
				if err != nil {
					t.Fatal(err)
				}
				decoded, err := Read(strings.NewReader(string(data)))
				if side == "reference" && err != nil {
					t.Fatalf("legacy numerical reader rejected reference before Step admission: %v", err)
				}
				if err == nil {
					if side == "reference" {
						_, err = ProjectActivationReference(decoded, explicit)
					} else {
						err = ValidateActivationCapture(decoded)
					}
				}
				if !errors.Is(err, ErrActivationCaptureInvalid) {
					t.Fatalf("null %s %s element error=%v, want sentinel", side, field, err)
				}
			})
		}
	}
}
