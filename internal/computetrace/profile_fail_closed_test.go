package computetrace

import (
	"encoding/json"
	"math"
	"testing"
)

// These are synthetic bounded records, not independent activation or checkpoint
// witnesses. They qualify refusal policy only.
func TestActivationProfileFailClosed(t *testing.T) {
	one := func(stage string, value float32) Artifact {
		return collect(indexedSample(0, 0, stage, []float32{value}))
	}
	clamped := func(value float32) Artifact {
		a := one("a", value)
		if n := a.ClampNonFinite(); n != 1 {
			t.Fatalf("clamped %d samples, want 1", n)
		}
		encoded, err := json.Marshal(a)
		if err != nil {
			t.Fatal(err)
		}
		var decoded Artifact
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Events[0].Status != NonFiniteClampedStatus {
			t.Fatalf("clamped status lost in JSON: %+v", decoded)
		}
		return decoded
	}
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	for _, tc := range []struct {
		name     string
		ref, got Artifact
		tol      ActivationTolerance
	}{
		{"clamped_nan_both", clamped(nan), clamped(nan), ActivationTolerance{}},
		{"clamped_nan_vs_inf", clamped(nan), clamped(inf), ActivationTolerance{}},
		{"clamped_opposite_inf", clamped(negInf), clamped(inf), ActivationTolerance{}},
		{"clamped_reference", clamped(nan), one("a", math.MaxFloat32), ActivationTolerance{}},
		{"clamped_actual", one("a", math.MaxFloat32), clamped(inf), ActivationTolerance{}},
		{"clamped_excessive_finite_tolerance", clamped(negInf), clamped(inf), ActivationTolerance{Atol: math.MaxFloat64}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			strict := CompareActivationTraces(tc.ref, tc.got, tc.tol)
			if strict.Verdict != TraceIncomparable || strict.ComparedSamples != 0 || strict.Reason == "" {
				t.Errorf("strict clamped comparison = %+v, want INCOMPARABLE before measuring clamped values", strict)
			}
			p := ActivationErrorProfile(tc.ref, tc.got, tc.tol)
			if p.Verdict != TraceIncomparable || p.ComparedSamples != 0 || p.FirstDivergent != nil || p.Reason == "" || len(p.Rows) != 1 || p.Rows[0].Refusals != 1 || p.Rows[0].Reason == "" || p.Rows[0].Comparable || p.Rows[0].Match {
				t.Fatalf("clamped profile = %+v, want INCOMPARABLE with one refused, unmeasured row", p)
			}
		})
	}
	for _, tc := range []struct {
		name             string
		dropRef, dropGot uint64
	}{
		{"dropped_event_reference", 1, 0},
		{"dropped_event_actual", 0, 1},
		{"dropped_event_both", 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ref, got := one("a", 0), one("a", 0)
			ref.Dropped, got.Dropped = tc.dropRef, tc.dropGot
			strict := CompareActivationTraces(ref, got, ActivationTolerance{})
			if strict.Verdict != TraceIncomparable || strict.ComparedSamples != 0 || strict.Reason == "" {
				t.Errorf("strict dropped-event comparison = %+v, want INCOMPARABLE before measuring", strict)
			}
			p := ActivationErrorProfile(ref, got, ActivationTolerance{})
			if p.Verdict != TraceIncomparable || !p.Incomplete || p.Reason == "" || p.ComparedSamples != 1 {
				t.Fatalf("dropped-event profile = %+v, want INCOMPARABLE with retained sample measured", p)
			}
		})
	}
	t.Run("recorder_event_limit_self_profile", func(t *testing.T) {
		r := NewWithSampleBudget(1, 8)
		r.Record(indexedSample(0, 0, "a", []float32{0}))
		r.Record(indexedSample(0, 0, "b", []float32{99}))
		a := r.Artifact()
		if a.Dropped != 1 || a.DroppedSampleValues != 0 || len(a.Events) != 1 {
			t.Fatalf("fixture does not exercise event-only loss: %+v", a)
		}
		strict := CompareActivationTraces(a, a, ActivationTolerance{})
		if strict.Verdict != TraceIncomparable || strict.ComparedSamples != 0 || strict.Reason == "" {
			t.Errorf("strict truncated self-comparison = %+v, want INCOMPARABLE", strict)
		}
		p := ActivationErrorProfile(a, a, ActivationTolerance{})
		if p.Verdict != TraceIncomparable || !p.Incomplete || p.ComparedSamples != 1 || p.Reason == "" {
			t.Fatalf("truncated self-profile = %+v, want INCOMPARABLE", p)
		}
	})
	for _, extra := range []Event{
		indexedSample(0, 0, "b", []float32{999}),
		indexedSample(0, 1, "a", []float32{999}),
	} {
		t.Run("extra_"+stageOf(extra)+"_"+itoa(extra.Token), func(t *testing.T) {
			ref := one("a", 0)
			got := collect(indexedSample(0, 0, "a", []float32{0}), extra)
			p := ActivationErrorProfile(ref, got, ActivationTolerance{})
			if p.Verdict != TraceIncomparable || p.ExtraIdentities != 1 || p.ComparedSamples != 1 || p.Reason == "" || p.FirstDivergent != nil {
				t.Fatalf("extra-identity profile = %+v, want INCOMPARABLE", p)
			}
		})
	}
}

func TestActivationProfileDivergencePrecedence(t *testing.T) {
	for _, mode := range []string{"extra", "dropped_event", "dropped_sample", "clamped_other_row", "missing"} {
		t.Run(mode, func(t *testing.T) {
			ref := collect(indexedSample(0, 0, "a", []float32{0}))
			got := collect(indexedSample(0, 0, "a", []float32{1}))
			switch mode {
			case "extra":
				got.Events = append(got.Events, indexedSample(0, 0, "b", []float32{2}))
			case "dropped_event":
				got.Dropped = 1
			case "dropped_sample":
				got.DroppedSampleValues = 1
			case "clamped_other_row":
				ref.Events = append(ref.Events, indexedSample(0, 0, "b", []float32{float32(math.NaN())}))
				got.Events = append(got.Events, indexedSample(0, 0, "b", []float32{float32(math.NaN())}))
				ref.ClampNonFinite()
				got.ClampNonFinite()
			case "missing":
				ref.Events = append(ref.Events, indexedSample(0, 0, "b", []float32{2}))
			}
			p := ActivationErrorProfile(ref, got, ActivationTolerance{})
			if p.Verdict != TraceDivergent || p.FirstDivergent == nil || *p.FirstDivergent != (ActivationStage{Layer: 0, Stage: "a"}) || p.Rows[0].Mismatches != 1 {
				t.Fatalf("profile lost measured divergence beside unproven data: %+v", p)
			}
		})
	}
}

func TestActivationImporterUndumpedStageUnproven(t *testing.T) {
	dir := writeDump(t, []int{7}, []dumpTensor{
		{name: "qr-0", ne: [4]int64{1, 1, 1, 1}, dumped: true},
		{name: "attn_out-0", ne: [4]int64{1, 1, 1, 1}, dumped: false},
	})
	ref, err := ConvertLlamaCppDump(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(ref.Events) != 1 || ref.Events[0].Phase != "q_latent" || ref.Dropped != 0 || ref.DroppedSampleValues != 0 {
		t.Fatalf("fixture did not expose importer omission: %+v", ref)
	}
	got := collect(indexedSample(0, 0, "q_latent", []float32{0}), indexedSample(0, 0, "attention_projected", []float32{999}))
	got.InputTokens = []int{7}
	if err := AdoptActivationDigests(&ref, got); err != nil {
		t.Fatal(err)
	}
	p := ActivationErrorProfile(ref, got, ActivationTolerance{})
	if p.Verdict != TraceIncomparable || p.ExtraIdentities != 1 || p.ComparedSamples != 1 || p.Reason == "" {
		t.Fatalf("undumped reference stage profile = %+v, want INCOMPARABLE", p)
	}
}

func TestActivationImporterWeightAdoptionIsNotIndependentProof(t *testing.T) {
	dir := writeDump(t, []int{7}, []dumpTensor{{name: "qr-0", ne: [4]int64{1, 1, 1, 1}, dumped: true}})
	ref, err := ConvertLlamaCppDump(dir)
	if err != nil {
		t.Fatal(err)
	}
	if ref.Events[0].WeightDigest != "" {
		t.Fatalf("external fixture unexpectedly proves weights: %+v", ref)
	}
	got := collect(indexedSample(0, 0, "q_latent", []float32{0}))
	got.InputTokens = []int{7}
	got.Events[0].WeightDigest = "caller-asserted-weights-only"
	if err := AdoptActivationDigests(&ref, got); err != nil {
		t.Fatal(err)
	}
	if ref.Events[0].WeightDigest != got.Events[0].WeightDigest {
		t.Fatal("adoption did not copy the caller assertion")
	}
	if p := ActivationErrorProfile(ref, got, ActivationTolerance{}); p.Verdict != TraceEqual {
		t.Fatalf("changed existing digest-adoption policy: %+v", p)
	}
	t.Log("Synthetic equality after copying a weight assertion is not an independently qualified checkpoint or activation witness")
}

func TestActivationStrictPrecedencePreserved(t *testing.T) {
	ref := collect(indexedSample(0, 0, "a", []float32{0}), indexedSample(0, 0, "b", []float32{float32(math.NaN())}))
	got := collect(indexedSample(0, 0, "a", []float32{1}), indexedSample(0, 0, "b", []float32{float32(math.NaN())}))
	ref.ClampNonFinite()
	got.ClampNonFinite()
	cmp := CompareActivationTraces(ref, got, ActivationTolerance{})
	if cmp.Verdict != TraceDivergent || cmp.Divergent.Stage != "a" || cmp.ComparedSamples != 1 {
		t.Fatalf("strict comparator no longer stops at first measured divergence: %+v", cmp)
	}
	got.Dropped = 1
	cmp = CompareActivationTraces(ref, got, ActivationTolerance{})
	if cmp.Verdict != TraceIncomparable || cmp.ComparedSamples != 0 || cmp.Reason == "" {
		t.Fatalf("strict incomplete-artifact check no longer precedes measurement: %+v", cmp)
	}
}
